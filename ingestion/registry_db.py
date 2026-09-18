import sqlite3
import threading
import uuid
import time
from pathlib import Path

from core.models import Document, Folder, IngestStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    started_at  REAL,
    finished_at REAL,
    added_at    REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS folders (
    folder_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL
);
"""

# Columns added after the first release; applied to pre-existing databases
# on open so upgrading in place never needs a manual migration.
_MIGRATIONS = [
    ("bytes", "INTEGER NOT NULL DEFAULT 0"),
    ("started_at", "REAL"),
    ("finished_at", "REAL"),
    ("folder_id", "TEXT"),
]


def estimate_eta(done: list[tuple[int, float]], remaining_bytes: int,
                 remaining_count: int, current_elapsed: float) -> float | None:
    """Seconds until the queue drains, or None if not yet estimable.

    `done` is (bytes, processing_seconds) for each finished document.

    Prefer a bytes/second throughput (robust when the remaining documents
    differ wildly in size), but fall back to a per-document average while
    byte sizes are unknown - older queue entries predate size tracking.
    `current_elapsed` discounts time already spent on the in-progress
    document so the estimate counts down instead of stalling.
    """
    if remaining_count <= 0:
        return None

    byte_done = [(b, d) for b, d in done if b and b > 0 and d > 0]
    eta: float | None = None

    if remaining_bytes > 0 and byte_done:
        total_bytes = sum(b for b, _ in byte_done)
        total_secs = sum(d for _, d in byte_done)
        eta = remaining_bytes / (total_bytes / total_secs)
    else:
        timed = [d for _, d in done if d > 0]
        if timed:
            eta = (sum(timed) / len(timed)) * remaining_count

    if eta is None:
        return None
    return max(eta - current_elapsed, 0.0)


class Registry:
    """SQLite document registry.

    Also the sole communication channel between the ingestion worker thread
    and the Streamlit UI, so every method must be thread-safe.
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
                    self._conn.execute("PRAGMA table_info(documents)")}
        for name, ddl in _MIGRATIONS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE documents ADD COLUMN {name} {ddl}"
                )

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _write_many(self, statements: list[tuple[str, tuple]]) -> None:
        """Several statements, one lock and one commit.

        _write runs a single statement, which is enough for every other
        mutation. Deleting a folder is two - unfile its documents, then
        drop the row - and they must not be separable, or a failure
        between them leaves documents pointing at a folder that is gone.
        """
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _row_to_doc(self, row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            status=IngestStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
            folder_id=row["folder_id"],
        )

    def add(self, doc_id: str, filename: str, size_bytes: int = 0) -> None:
        """Queue a document, inheriting the folder of its previous version.

        doc_id is a hash of the file's bytes, so re-uploading a corrected
        document arrives as a new, unrelated id. Matching on filename keeps
        it in the folder the user filed the old one in; they can always
        move it afterwards.

        The lookup and the insert share one lock acquisition - _write would
        take the same non-reentrant lock again - so the worker thread
        cannot file a document between them.
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

    def get(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def all(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY added_at"
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def next_queued(self) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE status = ? "
                "ORDER BY added_at LIMIT 1",
                (IngestStatus.QUEUED.value,),
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def mark_processing(self, doc_id: str) -> None:
        # Stamp the start and clear any prior finish time so a re-run (e.g. a
        # retry) is timed from scratch rather than showing a stale duration.
        self._write(
            "UPDATE documents SET status = ?, started_at = ?, "
            "finished_at = NULL WHERE doc_id = ?",
            (IngestStatus.PROCESSING.value, time.time(), doc_id),
        )

    def mark_done(self, doc_id: str, chunk_count: int) -> None:
        self._write(
            "UPDATE documents SET status = ?, error = NULL, chunk_count = ?, "
            "finished_at = ? WHERE doc_id = ?",
            (IngestStatus.DONE.value, chunk_count, time.time(), doc_id),
        )

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._write(
            "UPDATE documents SET status = ?, error = ?, finished_at = ? "
            "WHERE doc_id = ?",
            (IngestStatus.FAILED.value, error, time.time(), doc_id),
        )

    def requeue(self, doc_id: str) -> None:
        """Put an existing document back in the queue (e.g. to re-chunk it)."""
        self._write(
            "UPDATE documents SET status = ?, error = NULL WHERE doc_id = ?",
            (IngestStatus.QUEUED.value, doc_id),
        )

    def remove(self, doc_id: str) -> None:
        self._write("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def reset_stale_processing(self) -> int:
        """Recover documents wedged by a crash mid-ingest."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET status = ? WHERE status = ?",
                (IngestStatus.QUEUED.value, IngestStatus.PROCESSING.value),
            )
            self._conn.commit()
            return cursor.rowcount

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def ingest_eta(self) -> tuple[int, int, float | None]:
        """(processing count, queued count, estimated seconds remaining).

        The estimate is None until at least one document has finished with a
        measured duration this run; see estimate_eta for the method.
        """
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, bytes, started_at, finished_at FROM documents"
            ).fetchall()

        processing = sum(1 for r in rows
                         if r["status"] == IngestStatus.PROCESSING.value)
        queued = sum(1 for r in rows
                     if r["status"] == IngestStatus.QUEUED.value)

        done = [
            (r["bytes"] or 0, r["finished_at"] - r["started_at"])
            for r in rows
            if r["started_at"] is not None and r["finished_at"] is not None
            and r["finished_at"] > r["started_at"]
        ]
        remaining_bytes = sum(
            r["bytes"] or 0 for r in rows
            if r["status"] in (IngestStatus.QUEUED.value,
                               IngestStatus.PROCESSING.value)
        )
        in_flight = [r["started_at"] for r in rows
                     if r["status"] == IngestStatus.PROCESSING.value
                     and r["started_at"] is not None]
        current_elapsed = (now - min(in_flight)) if in_flight else 0.0

        eta = estimate_eta(done, remaining_bytes, processing + queued,
                           current_elapsed)
        return processing, queued, eta
