from pathlib import Path


class Pipeline:
    """Orchestrates parse → chunk → embed → store for one document."""

    def __init__(self, parser, chunker, embedder, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(self, path: Path, doc_id: str) -> int:
        parsed = self._parser.parse(path)
        chunks = self._chunker.chunk(parsed, doc_id, path.name)
        if not chunks:
            return 0

        # Replace wholesale so stale and fresh chunks never coexist.
        self._store.delete_by_doc(doc_id)

        vectors = self._embedder.embed([c.text for c in chunks])
        self._store.upsert(chunks, vectors)
        return len(chunks)
