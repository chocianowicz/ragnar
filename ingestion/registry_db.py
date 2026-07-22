import sqlite3
import threading
from pathlib import Path

from core.models import Document, IngestStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    added_at    REAL NOT NULL DEFAULT (julianday('now'))
);
"""


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
            self._conn.commit()

    def _row_to_doc(self, row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            status=IngestStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
        )

    def add(self, doc_id: str, filename: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO documents (doc_id, filename, status) "
                "VALUES (?, ?, ?)",
                (doc_id, filename, IngestStatus.QUEUED.value),
            )
            self._conn.commit()

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

    def _set_status(self, doc_id: str, status: IngestStatus,
                    error: str | None = None,
                    chunk_count: int | None = None) -> None:
        with self._lock:
            if chunk_count is None:
                self._conn.execute(
                    "UPDATE documents SET status = ?, error = ? "
                    "WHERE doc_id = ?",
                    (status.value, error, doc_id),
                )
            else:
                self._conn.execute(
                    "UPDATE documents SET status = ?, error = ?, "
                    "chunk_count = ? WHERE doc_id = ?",
                    (status.value, error, chunk_count, doc_id),
                )
            self._conn.commit()

    def mark_processing(self, doc_id: str) -> None:
        self._set_status(doc_id, IngestStatus.PROCESSING)

    def mark_done(self, doc_id: str, chunk_count: int) -> None:
        self._set_status(doc_id, IngestStatus.DONE, chunk_count=chunk_count)

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._set_status(doc_id, IngestStatus.FAILED, error=error)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            self._conn.commit()

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
