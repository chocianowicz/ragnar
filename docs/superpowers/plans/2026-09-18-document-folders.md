# Document Folders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user group documents into folders and scope an answer to one folder, with each chat remembering the folders it was scoped to.

**Architecture:** Additive only. `registry.db` gains a `folders` table and a `documents.folder_id` column; `chats.db` gains a `scope` column. All folder logic that can be pure is pure and lives in a new `ui/folders.py`, tested without Streamlit. Nothing in `retrieval/` or `generation/` is touched — the UI already builds a `doc_ids` filter and folders only change which boxes are ticked.

**Tech Stack:** Python 3.11, SQLite (stdlib `sqlite3`), Streamlit, pytest. Tests run in the container.

**Spec:** `docs/superpowers/specs/2026-09-18-document-folders-design.md`

---

## Before you start

Run the suite once so you know it was green before you touched it:

```bash
cd /Users/denis/Documents/AI/GitHub/ragnar
docker compose exec -T app python -m pytest tests/ -q
```

Expected: `331 passed, 30 skipped`. If it is not green, stop and say so.

**Run tests this way throughout.** The app runs in Docker and the repo is bind-mounted, so your edits are live inside the container with no rebuild. A bare `pytest` on the host will fail on missing dependencies.

**Never edit `data/registry.db` or `data/chats.db` by hand.** They hold the user's real corpus. Every test builds its own database in `tmp_path`.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `core/models.py` | `Document` and `Folder` dataclasses | Modify |
| `ingestion/registry_db.py` | folders table, `folder_id` column, folder CRUD | Modify |
| `history/chat_store.py` | migration mechanism, `scope` column | Modify |
| `ui/folders.py` | name validation, `folder_state`, scope save/restore | **Create** |
| `ui/panels/documents.py` | folder grouping, row picker, folder CRUD UI | Modify |
| `ui/app.py` | wire scope save/restore into the chat lifecycle | Modify |
| `tests/test_folders.py` | pure logic in `ui/folders.py` | **Create** |
| `tests/test_registry.py` | folder CRUD and migration | Modify |
| `tests/test_chat_store.py` | scope column and round-trip | Modify |

Tasks 1-7 are pure data and logic, each independently testable. Tasks 8-10 are Streamlit wiring and are not unit-testable; they are verified by hand at the end.

---

### Task 1: `Folder` model and `Document.folder_id`

**Files:**
- Modify: `core/models.py:41-47`

- [ ] **Step 1: Add the field and the new dataclass**

In `core/models.py`, add `folder_id` to `Document`:

```python
@dataclass
class Document:
    doc_id: str
    filename: str
    status: IngestStatus = IngestStatus.QUEUED
    error: str | None = None
    chunk_count: int = 0
    folder_id: str | None = None   # None = Unfiled
```

And add a `Folder` dataclass beside it:

```python
@dataclass
class Folder:
    """A named group of documents.

    Identity is folder_id, not name: the name is a label the user renames
    freely, and a saved chat scope stores the id so a rename never has to
    reach into chats.db.
    """
    folder_id: str
    name: str
    created_at: float
```

`folder_id` is last and defaulted, so every existing positional construction of `Document` in the tests keeps working.

- [ ] **Step 2: Verify nothing broke**

Run: `docker compose exec -T app python -m pytest tests/ -q`
Expected: `331 passed, 30 skipped` — unchanged.

- [ ] **Step 3: Commit**

```bash
git add core/models.py
git commit -m "feat: a Folder model, and a folder on Document"
```

---

### Task 2: Registry migration and `_write_many`

**Files:**
- Modify: `ingestion/registry_db.py:9-28` (SCHEMA, `_MIGRATIONS`), `:80-92` (`_migrate`, `_write`)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_folder_id_is_added_to_a_database_from_before_the_column(tmp_path):
    """A registry created by an earlier release gains the column on open."""
    import sqlite3
    path = tmp_path / "registry.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE documents ("
        " doc_id TEXT PRIMARY KEY, filename TEXT NOT NULL,"
        " status TEXT NOT NULL, error TEXT,"
        " chunk_count INTEGER NOT NULL DEFAULT 0,"
        " bytes INTEGER NOT NULL DEFAULT 0,"
        " added_at REAL NOT NULL DEFAULT (julianday('now')));"
    )
    conn.execute(
        "INSERT INTO documents (doc_id, filename, status) VALUES (?, ?, ?)",
        ("old", "prior.pdf", "done"),
    )
    conn.commit()
    conn.close()

    registry = Registry(path)

    assert registry.get("old").folder_id is None
    assert registry.folders() == []


def test_write_many_is_atomic(tmp_path):
    """Both statements land, or neither does."""
    registry = Registry(tmp_path / "registry.db")
    registry.add("a", "one.pdf")

    with pytest.raises(sqlite3.OperationalError):
        registry._write_many([
            ("UPDATE documents SET filename = ? WHERE doc_id = ?", ("x", "a")),
            ("UPDATE nonexistent SET k = 1", ()),
        ])

    assert registry.get("a").filename == "one.pdf"
```

Add `import sqlite3` and `import pytest` at the top of the file if they are not already there.

- [ ] **Step 2: Run it and watch it fail**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q -k "folder_id_is_added or write_many"`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'folders'`

- [ ] **Step 3: Implement**

In `ingestion/registry_db.py`, extend `SCHEMA` with the new table:

```python
CREATE TABLE IF NOT EXISTS folders (
    folder_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL
);
```

Add the column to `_MIGRATIONS`:

```python
_MIGRATIONS = [
    ("bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("started_at", "REAL"),
    ("finished_at", "REAL"),
    ("folder_id", "TEXT"),
]
```

Add `folder_id` to `_row_to_doc`:

```python
            folder_id=row["folder_id"],
```

Add the multi-statement writer beside `_write`:

```python
    def _write_many(self, statements: list[tuple[str, tuple]]) -> None:
        """Several statements, one lock and one commit.

        _write runs a single statement, which is enough for every other
        mutation. Deleting a folder is two — unfile its documents, then
        drop the row — and they must not be separable, or a crash between
        them leaves documents pointing at a folder that no longer exists.
        """
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
```

Note the `executescript` in `__init__` runs `SCHEMA`, so the `folders` table is created on open for new and existing databases alike; `_MIGRATIONS` only handles columns on `documents`.

- [ ] **Step 4: Run the tests**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q`
Expected: PASS (`folders()` is still missing — add the stub in Task 3; if the first test fails only on `registry.folders()`, proceed to Task 3 and re-run both together).

- [ ] **Step 5: Commit**

```bash
git add ingestion/registry_db.py tests/test_registry.py
git commit -m "feat: a folders table, and a multi-statement registry write"
```

---

### Task 3: Folder CRUD on the Registry

**Files:**
- Modify: `ingestion/registry_db.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_folders_round_trip(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    folder_id = registry.create_folder("Acme Corp")

    folders = registry.folders()

    assert [f.name for f in folders] == ["Acme Corp"]
    assert folders[0].folder_id == folder_id


def test_folders_are_listed_alphabetically(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    for name in ("Zeta", "alpha", "Mid"):
        registry.create_folder(name)

    assert [f.name for f in registry.folders()] == ["alpha", "Mid", "Zeta"]


def test_moving_a_document_sets_its_folder(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    registry.add("a", "nda.pdf")
    folder_id = registry.create_folder("Acme Corp")

    registry.set_folder("a", folder_id)
    assert registry.get("a").folder_id == folder_id

    registry.set_folder("a", None)
    assert registry.get("a").folder_id is None


def test_renaming_keeps_the_id_and_the_documents(tmp_path):
    """Identity is the id, so a rename cannot orphan anything."""
    registry = Registry(tmp_path / "registry.db")
    folder_id = registry.create_folder("Acme Corp")
    registry.add("a", "nda.pdf")
    registry.set_folder("a", folder_id)

    registry.rename_folder(folder_id, "Acme Corporation")

    assert [f.name for f in registry.folders()] == ["Acme Corporation"]
    assert registry.get("a").folder_id == folder_id


def test_deleting_a_folder_returns_its_documents_to_unfiled(tmp_path):
    """Deleting a folder is filing, never data loss."""
    registry = Registry(tmp_path / "registry.db")
    folder_id = registry.create_folder("Acme Corp")
    for doc_id in ("a", "b"):
        registry.add(doc_id, f"{doc_id}.pdf")
        registry.set_folder(doc_id, folder_id)

    registry.delete_folder(folder_id)

    assert registry.folders() == []
    assert registry.get("a").folder_id is None
    assert registry.get("b").folder_id is None
    assert registry.get("a").filename == "a.pdf"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'create_folder'`

- [ ] **Step 3: Implement**

Add `import uuid` at the top of `ingestion/registry_db.py`, import `Folder` from `core.models`, and add these methods to `Registry`:

```python
    def folders(self) -> list[Folder]:
        """Every folder, by name, case-insensitively."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM folders ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [Folder(folder_id=r["folder_id"], name=r["name"],
                       created_at=r["created_at"]) for r in rows]

    def create_folder(self, name: str) -> str:
        """Insert a folder and return its id.

        The name is validated by ui.folders.validate_name before this is
        called; the UNIQUE constraint here is the backstop, not the error
        path.
        """
        folder_id = str(uuid.uuid4())
        self._write(
            "INSERT INTO folders (folder_id, name, created_at) "
            "VALUES (?, ?, ?)",
            (folder_id, name, time.time()),
        )
        return folder_id

    def rename_folder(self, folder_id: str, name: str) -> None:
        self._write(
            "UPDATE folders SET name = ? WHERE folder_id = ?",
            (name, folder_id),
        )

    def delete_folder(self, folder_id: str) -> None:
        """Drop the folder; its documents fall back to Unfiled."""
        self._write_many([
            ("UPDATE documents SET folder_id = NULL WHERE folder_id = ?",
             (folder_id,)),
            ("DELETE FROM folders WHERE folder_id = ?", (folder_id,)),
        ])

    def set_folder(self, doc_id: str, folder_id: str | None) -> None:
        self._write(
            "UPDATE documents SET folder_id = ? WHERE doc_id = ?",
            (folder_id, doc_id),
        )
```

- [ ] **Step 4: Run the tests**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q`
Expected: PASS, including Task 2's tests.

- [ ] **Step 5: Commit**

```bash
git add ingestion/registry_db.py tests/test_registry.py
git commit -m "feat: create, rename, delete and fill folders"
```

---

### Task 4: A re-uploaded document inherits its folder

**Files:**
- Modify: `ingestion/registry_db.py` (`add`)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
def test_a_reuploaded_file_inherits_its_folder(tmp_path):
    """doc_id is a content hash, so correcting a typo mints a new id.

    Without this the document would silently leave the folder the user
    filed it in, at exactly the moment they expect it to stay put.
    """
    registry = Registry(tmp_path / "registry.db")
    folder_id = registry.create_folder("Acme Corp")
    registry.add("hash-v1", "nda.pdf")
    registry.set_folder("hash-v1", folder_id)

    registry.add("hash-v2", "nda.pdf")

    assert registry.get("hash-v2").folder_id == folder_id


def test_a_new_filename_starts_unfiled(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    folder_id = registry.create_folder("Acme Corp")
    registry.add("a", "nda.pdf")
    registry.set_folder("a", folder_id)

    registry.add("b", "different.pdf")

    assert registry.get("b").folder_id is None
```

- [ ] **Step 2: Run and watch it fail**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q -k reuploaded`
Expected: FAIL — `assert None == '<uuid>'`

- [ ] **Step 3: Implement**

Replace `Registry.add`:

```python
    def add(self, doc_id: str, filename: str, size_bytes: int = 0) -> None:
        """Queue a document, inheriting the folder of its previous version.

        doc_id is a hash of the file's bytes, so re-uploading a corrected
        document arrives as a new, unrelated id. Matching on filename keeps
        it in the folder the user filed the old one in; they can always
        move it afterwards.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT folder_id FROM documents "
                "WHERE filename = ? AND folder_id IS NOT NULL "
                "ORDER BY added_at DESC LIMIT 1",
                (filename,),
            ).fetchone()
            inherited = row["folder_id"] if row else None
            self._conn.execute(
                "INSERT OR IGNORE INTO documents "
                "(doc_id, filename, status, bytes, folder_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, IngestStatus.QUEUED.value, size_bytes,
                 inherited),
            )
            self._conn.commit()
```

The lookup and the insert share one lock acquisition, so the worker thread cannot file a document between them.

- [ ] **Step 4: Run the whole registry file**

Run: `docker compose exec -T app python -m pytest tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion/registry_db.py tests/test_registry.py
git commit -m "feat: a re-uploaded document keeps the folder it was filed in"
```

---

### Task 5: Chat store migration and the `scope` column

**Files:**
- Modify: `history/chat_store.py:9-17` (SCHEMA), `:44-51` (`__init__`), `:53-80` (`_row_to_chat`, `save`), `core/models.py` (`SavedChat`)
- Test: `tests/test_chat_store.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_scope_is_added_to_a_database_from_before_the_column(tmp_path):
    import sqlite3
    path = tmp_path / "chats.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE chats ("
        " chat_id TEXT PRIMARY KEY, title TEXT NOT NULL,"
        " messages TEXT NOT NULL, created_at REAL NOT NULL,"
        " updated_at REAL NOT NULL);"
    )
    conn.execute(
        "INSERT INTO chats VALUES (?, ?, ?, ?, ?)",
        ("old", "Prior chat", "[]", 1.0, 1.0),
    )
    conn.commit()
    conn.close()

    store = ChatStore(path)

    # A chat saved before this feature has no scope, which restores as
    # everything rather than as nothing.
    assert store.get("old").scope is None


def test_a_scope_round_trips(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    scope = {"v": 1, "folders": ["9f2c"], "docs": ["a1b2"]}

    store.save("c1", "Acme", [], scope=scope)

    assert store.get("c1").scope == scope


def test_saving_without_a_scope_keeps_the_stored_one(tmp_path):
    """commit() lands an answer for a chat that may not be on screen.

    It must not overwrite that chat's saved scope with whatever the
    visible conversation happens to be scoped to.
    """
    store = ChatStore(tmp_path / "chats.db")
    scope = {"v": 1, "folders": ["9f2c"], "docs": []}
    store.save("c1", "Acme", [], scope=scope)

    store.save("c1", "Acme", [{"role": "user", "content": "hi"}])

    assert store.get("c1").scope == scope
    assert len(store.get("c1").messages) == 1
```

- [ ] **Step 2: Run and watch them fail**

Run: `docker compose exec -T app python -m pytest tests/test_chat_store.py -q`
Expected: FAIL — `TypeError: save() got an unexpected keyword argument 'scope'`

- [ ] **Step 3: Implement**

Add `scope` to `SavedChat` in `core/models.py`, last and defaulted:

```python
    scope: dict | None = None
```

In `history/chat_store.py`, add the migration machinery mirroring `registry_db.py`, a module-level sentinel, and the scope handling:

```python
# Columns added after the first release; applied to pre-existing databases
# on open so upgrading in place never needs a manual migration. The same
# mechanism ingestion/registry_db.py uses.
_MIGRATIONS = [
    ("scope", "TEXT"),
]

# save(scope=...) distinguishes "no scope" (search everything) from "do not
# touch the stored scope". None is a real value here, so the default cannot
# be None.
_KEEP = object()
```

In `__init__`, call `self._migrate()` inside the existing lock, right after `executescript`:

```python
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
```

Add the column to `SCHEMA` too, so a fresh database has it without relying on the migration:

```python
    updated_at REAL NOT NULL,
    scope      TEXT
```

Read it in `_row_to_chat`:

```python
            scope=json.loads(row["scope"]) if row["scope"] else None,
```

And rewrite `save`:

```python
    def save(self, chat_id: str, title: str, messages: list[dict],
             scope=_KEEP) -> None:
        """Insert a new chat or update an existing one in place.

        created_at is preserved across updates (ON CONFLICT keeps it); only
        the title, messages, and updated_at move.

        scope defaults to _KEEP, meaning "leave the stored scope alone".
        An answer landing from a background thread saves a chat that may
        not be on screen, and must not stamp the visible conversation's
        scope onto it.
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
```

- [ ] **Step 4: Run the tests**

Run: `docker compose exec -T app python -m pytest tests/test_chat_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add history/chat_store.py core/models.py tests/test_chat_store.py
git commit -m "feat: a chat remembers the scope it was asked under"
```

---

### Task 6: Folder name validation

**Files:**
- Create: `ui/folders.py`
- Test: `tests/test_folders.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_folders.py`:

```python
import pytest

from core.models import Document, Folder
from ui.folders import (
    UNFILED, FolderNameError, validate_name, folder_state,
    scope_from_selection, selection_from_scope,
)


def _folders(*names):
    return [Folder(folder_id=f"id-{n}", name=n, created_at=0.0)
            for n in names]


def test_a_valid_name_is_trimmed():
    assert validate_name("  Acme Corp  ", _folders()) == "Acme Corp"


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_an_empty_name_is_rejected(name):
    with pytest.raises(FolderNameError):
        validate_name(name, _folders())


def test_an_overlong_name_is_rejected():
    with pytest.raises(FolderNameError):
        validate_name("x" * 61, _folders())


def test_a_name_of_exactly_the_limit_is_allowed():
    assert validate_name("x" * 60, _folders()) == "x" * 60


def test_a_duplicate_is_rejected_regardless_of_case():
    with pytest.raises(FolderNameError, match="already"):
        validate_name("acme corp", _folders("Acme Corp"))


@pytest.mark.parametrize("name", ["Unfiled", "unfiled", "UNFILED"])
def test_unfiled_is_reserved(name):
    """It renders folder_id IS NULL, so a real folder of that name would
    show two identical rows with different contents."""
    with pytest.raises(FolderNameError, match="reserved"):
        validate_name(name, _folders())


def test_renaming_a_folder_to_its_own_name_is_allowed():
    existing = _folders("Acme Corp")
    assert validate_name("Acme Corp", existing,
                         allow_id="id-Acme Corp") == "Acme Corp"
```

- [ ] **Step 2: Run and watch it fail**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.folders'`

- [ ] **Step 3: Implement**

Create `ui/folders.py`:

```python
"""Folder logic, kept out of the Streamlit panel so it can be tested.

Everything here is a pure function over Documents and Folders. The panel
renders what these return; the registry stores what they validate.
"""
from __future__ import annotations

from core.models import Document, Folder

# Unfiled is not a row. It is how the panel renders folder_id IS NULL, and
# how a saved scope names that group — real ids are uuid4, so the literal
# cannot collide with one.
UNFILED = "unfiled"

NAME_MAX = 60


class FolderNameError(ValueError):
    """A folder name the user has to fix before anything is written."""


def validate_name(name: str, existing: list[Folder],
                  allow_id: str | None = None) -> str:
    """The name to store, or raise with a message fit to show the user.

    allow_id is the folder being renamed, which is allowed to keep its own
    name — otherwise renaming a folder to what it is already called would
    collide with itself.
    """
    cleaned = name.strip()
    if not cleaned:
        raise FolderNameError("A folder needs a name.")
    if len(cleaned) > NAME_MAX:
        raise FolderNameError(f"Keep it to {NAME_MAX} characters or fewer.")
    if cleaned.casefold() == UNFILED.casefold():
        raise FolderNameError(
            "\"Unfiled\" is reserved — it is where documents sit when they "
            "are in no folder."
        )
    for folder in existing:
        if folder.folder_id == allow_id:
            continue
        if folder.name.casefold() == cleaned.casefold():
            raise FolderNameError(f"There is already a folder called "
                                  f"\"{folder.name}\".")
    return cleaned
```

- [ ] **Step 4: Run the tests**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q`
Expected: PASS (the imports of `folder_state` and the scope helpers still fail — write them in Tasks 7 and 8 before re-running the whole file, or comment them out of the import until then).

- [ ] **Step 5: Commit**

```bash
git add ui/folders.py tests/test_folders.py
git commit -m "feat: folder name rules, with Unfiled reserved"
```

---

### Task 7: `folder_state`

**Files:**
- Modify: `ui/folders.py`
- Test: `tests/test_folders.py`

- [ ] **Step 1: Write the failing tests**

```python
def _docs(*specs):
    """specs are (doc_id, folder_id) pairs."""
    return [Document(doc_id=d, filename=f"{d}.pdf", folder_id=f)
            for d, f in specs]


def test_folder_state_all_checked():
    docs = _docs(("a", "f1"), ("b", "f1"))
    assert folder_state(docs, {"a", "b"}) == (True, "2/2")


def test_folder_state_some_checked():
    """Streamlit has no indeterminate checkbox, so the count carries it."""
    docs = _docs(("a", "f1"), ("b", "f1"), ("c", "f1"))
    assert folder_state(docs, {"a"}) == (False, "1/3")


def test_folder_state_none_checked():
    docs = _docs(("a", "f1"), ("b", "f1"))
    assert folder_state(docs, set()) == (False, "0/2")


def test_an_empty_folder_is_vacuously_checked():
    """It contributes no ids, so it cannot change an answer either way."""
    assert folder_state([], set()) == (True, "0/0")
```

- [ ] **Step 2: Run and watch it fail**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q -k folder_state`
Expected: FAIL — `ImportError: cannot import name 'folder_state'`

- [ ] **Step 3: Implement**

```python
def folder_state(docs: list[Document],
                 checked_ids: set[str]) -> tuple[bool, str]:
    """(is_checked, "3/12") for one folder's documents.

    Streamlit checkboxes have no indeterminate state, so a partly-ticked
    folder reads as unchecked and the count carries the truth. An empty
    folder satisfies "all checked" vacuously and contributes no ids, so it
    renders checked and changes nothing.
    """
    total = len(docs)
    checked = sum(1 for d in docs if d.doc_id in checked_ids)
    return checked == total, f"{checked}/{total}"
```

- [ ] **Step 4: Run the tests**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q -k folder_state`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/folders.py tests/test_folders.py
git commit -m "feat: a folder's checkbox state and count"
```

---

### Task 8: Saving and restoring a scope

**Files:**
- Modify: `ui/folders.py`
- Test: `tests/test_folders.py`

This is the task the spec is most careful about. Read its "Saved scope" section before writing code.

- [ ] **Step 1: Write the failing tests**

```python
def test_everything_checked_saves_as_all():
    docs = _docs(("a", "f1"), ("b", None))
    assert scope_from_selection(docs, {"a", "b"}) == {"v": 1, "all": True}


def test_a_whole_folder_saves_as_its_id():
    docs = _docs(("a", "f1"), ("b", "f1"), ("c", "f2"))
    scope = scope_from_selection(docs, {"a", "b"})
    assert scope == {"v": 1, "folders": ["f1"], "docs": []}


def test_unfiled_saves_as_the_sentinel():
    docs = _docs(("a", None), ("b", None), ("c", "f1"))
    scope = scope_from_selection(docs, {"a", "b"})
    assert scope == {"v": 1, "folders": [UNFILED], "docs": []}


def test_a_partial_folder_saves_its_documents():
    docs = _docs(("a", "f1"), ("b", "f1"))
    scope = scope_from_selection(docs, {"a"})
    assert scope == {"v": 1, "folders": [], "docs": ["a"]}


def test_nothing_checked_saves_as_an_empty_scope():
    docs = _docs(("a", "f1"))
    assert scope_from_selection(docs, set()) == {"v": 1, "folders": [],
                                                 "docs": []}


def test_all_restores_everything():
    docs = _docs(("a", "f1"), ("b", None))
    assert selection_from_scope({"v": 1, "all": True}, docs) == {"a", "b"}


def test_no_scope_restores_everything():
    """A chat saved before this feature had no scope, and meant everything."""
    docs = _docs(("a", "f1"), ("b", None))
    assert selection_from_scope(None, docs) == {"a", "b"}


def test_an_empty_scope_restores_nothing():
    """The regression that would reverse the refusal guarantee.

    ui/app.py sends an explicit empty doc_ids list, and store.py treats it
    as "refuse" rather than "search everything". An empty saved scope has
    to keep meaning exactly that.
    """
    docs = _docs(("a", "f1"), ("b", None))
    scope = {"v": 1, "folders": [], "docs": []}
    assert selection_from_scope(scope, docs) == set()


def test_a_folder_that_gained_a_document_restores_with_it():
    """Membership is resolved at restore time, not frozen at save time."""
    scope = {"v": 1, "folders": ["f1"], "docs": []}
    docs = _docs(("a", "f1"), ("new", "f1"))
    assert selection_from_scope(scope, docs) == {"a", "new"}


def test_unfiled_gains_documents_the_same_way():
    """Unfiled is a peer of every other folder, so it must behave like one."""
    scope = {"v": 1, "folders": [UNFILED], "docs": []}
    docs = _docs(("a", None), ("new", None), ("c", "f1"))
    assert selection_from_scope(scope, docs) == {"a", "new"}


def test_a_deleted_folder_is_dropped_and_the_rest_kept():
    scope = {"v": 1, "folders": ["gone", "f1"], "docs": ["also-gone"]}
    docs = _docs(("a", "f1"))
    assert selection_from_scope(scope, docs) == {"a"}
```

- [ ] **Step 2: Run and watch them fail**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q -k "scope or restore"`
Expected: FAIL — `ImportError: cannot import name 'scope_from_selection'`

- [ ] **Step 3: Implement**

```python
def _group(docs: list[Document]) -> dict[str, list[Document]]:
    """Documents by folder id, with Unfiled under the sentinel."""
    grouped: dict[str, list[Document]] = {}
    for doc in docs:
        grouped.setdefault(doc.folder_id or UNFILED, []).append(doc)
    return grouped


def scope_from_selection(docs: list[Document],
                         checked_ids: set[str]) -> dict:
    """What to store for a chat asked under this selection.

    Whole folders are stored by id so they pick up documents added later.
    Anything else is stored as the document ids themselves.

    "Everything checked" is stored as {"all": True} rather than as every
    folder, so it stays distinguishable from an empty selection — which
    means "refuse", not "search everything".
    """
    checked = {d.doc_id for d in docs if d.doc_id in checked_ids}
    if len(checked) == len(docs) and docs:
        return {"v": 1, "all": True}

    folders, loose = [], set(checked)
    for folder_id, group in _group(docs).items():
        if group and all(d.doc_id in checked for d in group):
            folders.append(folder_id)
            loose -= {d.doc_id for d in group}
    return {"v": 1, "folders": sorted(folders), "docs": sorted(loose)}


def selection_from_scope(scope: dict | None,
                         docs: list[Document]) -> set[str]:
    """Which documents to tick when a chat is reopened.

    None is a chat saved before scopes existed and means everything.
    {"all": True} means everything. Anything else is resolved against the
    documents that exist NOW, so a folder picks up what was added to it
    since — and an empty scope stays empty, which is a refusal.
    """
    if scope is None or scope.get("all"):
        return {d.doc_id for d in docs}

    wanted = set(scope.get("folders") or [])
    selected = {d.doc_id for d in docs if (d.folder_id or UNFILED) in wanted}
    known = {d.doc_id for d in docs}
    return selected | (set(scope.get("docs") or []) & known)
```

- [ ] **Step 4: Run the whole file**

Run: `docker compose exec -T app python -m pytest tests/test_folders.py -q`
Expected: PASS, every test in the file.

- [ ] **Step 5: Run everything**

Run: `docker compose exec -T app python -m pytest tests/ -q`
Expected: all green, with more tests than the 331 you started with.

- [ ] **Step 6: Commit**

```bash
git add ui/folders.py tests/test_folders.py
git commit -m "feat: save and restore a chat's folder scope"
```

---

### Task 9: Folders in the Documents panel

**Files:**
- Modify: `ui/panels/documents.py:98-155` (`_status_strip_body`)

No unit tests: this is Streamlit rendering. It is verified by hand in Task 11.

- [ ] **Step 1: Group the document list by folder**

Replace the document loop in `_status_strip_body`. Keep the existing upload, ETA and viewer code untouched.

```python
    docs = svc["registry"].all()
    all_folders = svc["registry"].folders()

    def _select_all_changed():
        value = st.session_state.get("select_all_docs", True)
        for d in docs:
            st.session_state[f"sel_{d.doc_id}"] = value

    if docs:
        st.checkbox("Select all", value=True, key="select_all_docs",
                    on_change=_select_all_changed)
        st.caption(
            "Unchecked documents and folders are excluded from answers — "
            "only checked ones are searched."
        )

    checked_ids = {
        d.doc_id for d in docs
        if st.session_state.get(f"sel_{d.doc_id}", True)
    }
    grouped = folders.group_for_display(docs, all_folders)

    for folder_id, label, group in grouped:
        _render_folder(svc, folder_id, label, group, checked_ids, all_folders)
```

- [ ] **Step 2: Add the display grouping helper to `ui/folders.py`**

```python
def group_for_display(docs: list[Document],
                      all_folders: list[Folder]) -> list:
    """(folder_id, label, documents) in the order the panel draws them.

    Named folders alphabetically, then Unfiled last — it is where things
    land rather than somewhere the user chose, so it belongs at the bottom.
    Empty named folders are included; a user who just made one needs to see
    it to file into it.
    """
    by_folder = _group(docs)
    rows = [(f.folder_id, f.name, by_folder.get(f.folder_id, []))
            for f in all_folders]
    unfiled = by_folder.get(UNFILED, [])
    if unfiled or not all_folders:
        rows.append((UNFILED, "Unfiled", unfiled))
    return rows
```

Add a test for it in `tests/test_folders.py`:

```python
def test_unfiled_is_drawn_last():
    docs = _docs(("a", "id-Acme"), ("b", None))
    rows = group_for_display(docs, _folders("Acme"))
    assert [label for _, label, _ in rows] == ["Acme", "Unfiled"]


def test_an_empty_folder_is_still_drawn():
    rows = group_for_display([], _folders("Acme"))
    assert [label for _, label, _ in rows] == ["Acme", "Unfiled"]
```

- [ ] **Step 3: Write the folder and document rows**

Add to `ui/panels/documents.py`:

```python
def _render_folder(svc, folder_id, label, group, checked_ids,
                   all_folders) -> None:
    """One folder header and its documents."""
    is_checked, count = folders.folder_state(group, checked_ids)

    def _folder_changed(folder_id=folder_id, group=group):
        value = st.session_state.get(f"folder_sel_{folder_id}", True)
        for d in group:
            st.session_state[f"sel_{d.doc_id}"] = value

    col_check, col_name, col_count = st.columns([0.6, 4.4, 1])
    with col_check:
        st.checkbox(f"Include {label}", value=is_checked,
                   key=f"folder_sel_{folder_id}",
                   on_change=_folder_changed,
                   label_visibility="collapsed")
    with col_name:
        st.markdown(f"**{label}**")
    with col_count:
        st.caption(count)

    for doc in group:
        _render_document(svc, doc, all_folders)
```

`_render_document` is the body of today's `for doc in docs:` loop, moved
into a function and given one new column for the folder picker:

```python
def _render_document(svc, doc, all_folders) -> None:
    icon = STATUS_ICONS[doc.status.value]
    col_check, col_view, col_folder, col_remove = st.columns(
        [0.6, 3.0, 1.4, 1])
    with col_check:
        st.checkbox(f"Include {doc.filename}", value=True,
                    key=f"sel_{doc.doc_id}", label_visibility="collapsed")
    with col_view:
        if st.button(f"{icon} {doc.filename}", key=f"view_{doc.doc_id}",
                     use_container_width=True):
            show_key = f"show_md_{doc.doc_id}"
            st.session_state[show_key] = not st.session_state.get(
                show_key, False)
    with col_folder:
        _folder_picker(svc, doc, all_folders)
    with col_remove:
        ...  # unchanged from today
```

The picker, which is how a document is filed:

```python
NEW_FOLDER = "New folder…"


def _folder_picker(svc, doc, all_folders) -> None:
    """The control that files a document.

    Includes a New folder… entry so the first document can be filed
    without hunting for the panel button first — on a fresh install there
    are no folders to pick from at all.
    """
    names = ["Unfiled"] + [f.name for f in all_folders] + [NEW_FOLDER]
    current = next((f.name for f in all_folders
                    if f.folder_id == doc.folder_id), "Unfiled")
    picked = st.selectbox(
        f"Folder for {doc.filename}", names, index=names.index(current),
        key=f"folder_of_{doc.doc_id}", label_visibility="collapsed",
    )
    if picked == current:
        return
    if picked == NEW_FOLDER:
        st.session_state["new_folder_for"] = doc.doc_id
        st.rerun()
    target = next((f.folder_id for f in all_folders if f.name == picked),
                  None)
    svc["registry"].set_folder(doc.doc_id, target)
    st.rerun()
```

Add `from ui import folders` to the imports.

- [ ] **Step 4: Check it renders**

Reload `http://localhost:8501` and open Documents. Every document should
appear under **Unfiled** with a picker beside it. Nothing else should have
moved.

- [ ] **Step 5: Commit**

```bash
git add ui/panels/documents.py ui/folders.py tests/test_folders.py
git commit -m "feat: group the document list by folder, and file from each row"
```

---

### Task 10: Folder create, rename and delete

**Files:**
- Modify: `ui/panels/documents.py`

- [ ] **Step 1: Add the controls under the list**

```python
def _render_folder_controls(svc, all_folders) -> None:
    """New / Rename / Delete, under the list.

    Rename and Delete act on a chosen folder rather than a selected row:
    the checkboxes mean "search this", and reusing them to mean "act on
    this" is exactly the overloading this design avoids.
    """
    with st.form("new_folder", clear_on_submit=True):
        cols = st.columns([4, 1])
        with cols[0]:
            name = st.text_input("New folder", placeholder="e.g. Acme Corp",
                                 label_visibility="collapsed")
        with cols[1]:
            submitted = st.form_submit_button("Add", use_container_width=True)
    if submitted:
        try:
            svc["registry"].create_folder(
                folders.validate_name(name, all_folders))
        except folders.FolderNameError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    if not all_folders:
        return

    names = [f.name for f in all_folders]
    chosen = st.selectbox("Folder", names, key="folder_admin_target",
                          label_visibility="collapsed")
    target = next(f for f in all_folders if f.name == chosen)

    cols = st.columns(2)
    with cols[0]:
        new_name = st.text_input("Rename to", value=target.name,
                                 key=f"rename_{target.folder_id}",
                                 label_visibility="collapsed")
        if st.button("Rename", use_container_width=True):
            try:
                svc["registry"].rename_folder(
                    target.folder_id,
                    folders.validate_name(new_name, all_folders,
                                          allow_id=target.folder_id))
            except folders.FolderNameError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    with cols[1]:
        held = sum(1 for d in svc["registry"].all()
                   if d.folder_id == target.folder_id)
        if st.button("Delete", use_container_width=True,
                     help=f"{held} document(s) move to Unfiled"):
            svc["registry"].delete_folder(target.folder_id)
            st.rerun()
```

Delete needs no confirmation: the spec makes it non-destructive, and the
help text says where the documents go.

Call it at the end of `_status_strip_body`:

```python
    _render_folder_controls(svc, all_folders)
```

- [ ] **Step 2: Handle the New folder… picker entry**

At the top of `_render_folder_controls`, before the form:

```python
    pending = st.session_state.pop("new_folder_for", None)
    if pending:
        st.info("Name the new folder below — the document will move into it.")
```

and after a successful create, if `pending` was set, call
`svc["registry"].set_folder(pending, new_id)`.

- [ ] **Step 3: Verify by hand**

- Create "Acme Corp" → it appears, empty
- Try to create "acme corp" → an error, no second folder
- Try to create "Unfiled" → a reserved-name error
- File a document into it with the row picker → it moves
- Rename it → documents stay in it
- Delete it → documents appear under Unfiled, still indexed

- [ ] **Step 4: Commit**

```bash
git add ui/panels/documents.py
git commit -m "feat: create, rename and delete folders from the panel"
```

---

### Task 11: Remember the scope per chat

**Files:**
- Modify: `ui/app.py:149-159` (selection), `:177` (save), `ui/panels/chats.py:33-35` (open)

- [ ] **Step 1: Save the scope with the chat**

In `ui/app.py`, where the question is saved (line ~177):

```python
    svc["chats"].save(st.session_state.current_chat_id,
                      chat_title(st.session_state.messages),
                      st.session_state.messages,
                      scope=folders.scope_from_selection(
                          all_docs, set(selected_doc_ids)))
```

Add `from ui import folders` to the imports. Leave `commit()` alone — it
omits `scope`, which now means "keep what is stored", which is what an
answer landing for an off-screen chat must do.

- [ ] **Step 2: Restore the scope when a chat is opened**

In `ui/panels/chats.py`, where a chat is opened:

```python
                if st.button(label, key=f"open_chat_{chat.chat_id}",
                             use_container_width=True):
                    st.session_state.messages = chat.messages
                    st.session_state.current_chat_id = chat.chat_id
                    ticked = folders.selection_from_scope(
                        chat.scope, svc["registry"].all())
                    for doc in svc["registry"].all():
                        st.session_state[f"sel_{doc.doc_id}"] = (
                            doc.doc_id in ticked)
                    st.rerun()
```

Add `from ui import folders`.

- [ ] **Step 3: Verify by hand**

- Ask a question with only "Acme Corp" ticked
- Start a new chat → everything is ticked again
- Reopen the Acme chat → only Acme is ticked
- Add a document to Acme, reopen the chat → the new document is ticked too
- Untick everything, ask a question → it refuses; reopen that chat → still
  nothing ticked, and it still refuses

That last one is the regression the spec cares most about.

- [ ] **Step 4: Run everything**

Run: `docker compose exec -T app python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/panels/chats.py
git commit -m "feat: a chat reopens scoped to the folders it was asked under"
```

---

### Task 12: Update the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document folders**

The README describes the Documents panel as a flat list. Add a short
section after "What it looks like" covering: folders group documents,
ticking a folder scopes the answer, a chat reopens with its scope, and
deleting a folder returns its documents to Unfiled rather than removing
them.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: folders in the Documents panel"
```

---

## Done when

- `docker compose exec -T app python -m pytest tests/ -q` is green
- The hand-checks in Tasks 10 and 11 all pass
- `git log --oneline` shows one commit per task
- The user has seen it working before anything is pushed
