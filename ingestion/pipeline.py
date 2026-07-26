from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestResult:
    chunk_count: int
    markdown: str


class Pipeline:
    """Orchestrates parse → chunk → embed → store for one document."""

    def __init__(self, parser, chunker, embedder, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def set_chunker(self, chunker) -> None:
        """Swap the chunker used by future ingest() calls.

        Safe to call from the UI thread while the background worker thread
        reads self._chunker in ingest() — a bare attribute reassignment is
        atomic under the GIL. Already-ingested documents are unaffected;
        this only changes how documents ingested after the call are chunked.
        """
        self._chunker = chunker

    def ingest(self, path: Path, doc_id: str) -> IngestResult:
        parsed = self._parser.parse(path)
        chunks = self._chunker.chunk(parsed, doc_id, path.name)

        # Replace wholesale so stale and fresh chunks never coexist.
        self._store.delete_by_doc(doc_id)

        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._store.upsert(chunks, vectors)

        return IngestResult(chunk_count=len(chunks),
                            markdown=parsed.markdown)
