import json
import sqlite3
import threading
import time
from pathlib import Path

from core.models import SavedChat

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    messages   TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    scope      TEXT
);
"""

# Columns added after the first release; applied to pre-existing databases
# on open so upgrading in place never needs a manual migration. The same
# mechanism ingestion/registry_db.py uses.
_MIGRATIONS = [
    ("scope", "TEXT"),
]

# save(scope=...) has to tell "no scope" (search everything) apart from
# "do not touch the stored scope". None is a real value here, so the
# default cannot be None.
_KEEP = object()

TITLE_MAX = 40


def chat_title(messages: list[dict]) -> str:
    """A short title from the first user message, or a placeholder.

    Whitespace is collapsed and the text is truncated so long questions
    don't overflow the sidebar.
    """
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(message.get("content", "").split())
            if text:
                return text[:TITLE_MAX] + "…" if len(text) > TITLE_MAX else text
    return "New chat"


class ChatStore:
    """SQLite persistence for conversation history.

    Separate from the ingestion registry so chat history and document state
    stay decoupled. Only the UI thread touches it, but it mirrors Registry's
    lock-guarded, single-connection pattern for consistency and safety.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        existing = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(chats)")}
        for name, ddl in _MIGRATIONS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE chats ADD COLUMN {name} {ddl}"
                )

    def _row_to_chat(self, row) -> SavedChat:
        return SavedChat(
            chat_id=row["chat_id"],
            title=row["title"],
            messages=json.loads(row["messages"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            scope=json.loads(row["scope"]) if row["scope"] else None,
        )

    def save(self, chat_id: str, title: str, messages: list[dict],
             scope=_KEEP) -> None:
        """Insert a new chat or update an existing one in place.

        created_at is preserved across updates (ON CONFLICT keeps it); only
        the title, messages, and updated_at move.

        scope defaults to _KEEP, meaning "leave the stored scope alone". An
        answer landing from a background thread saves a chat that may not
        be on screen, and must not stamp the visible conversation's scope
        onto it.
        """
        now = time.time()
        payload = json.dumps(messages, ensure_ascii=False)
        keep_scope = scope is _KEEP
        encoded = None if keep_scope or scope is None else json.dumps(scope)
        with self._lock:
            self._conn.execute(
                "INSERT INTO chats "
                "(chat_id, title, messages, created_at, updated_at, scope) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "title = excluded.title, messages = excluded.messages, "
                "updated_at = excluded.updated_at, "
                "scope = CASE WHEN ? THEN chats.scope ELSE excluded.scope END",
                (chat_id, title, payload, now, now, encoded, keep_scope),
            )
            self._conn.commit()

    def get(self, chat_id: str) -> SavedChat | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._row_to_chat(row) if row else None

    def all(self) -> list[SavedChat]:
        """Saved chats, most recently updated first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_chat(r) for r in rows]

    def delete(self, chat_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM chats WHERE chat_id = ?", (chat_id,)
            )
            self._conn.commit()
