from core.models import Chunk, SearchResult


class FakeEmbedder:
    """Deterministic embeddings without a server."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((hash(t) >> (i * 4)) % 10) for i in range(self.dim)]
            for t in texts
        ]


class FakeStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.deleted: list[str] = []

    def ensure_collection(self):
        pass

    def upsert(self, chunks, vectors):
        self.chunks.extend(chunks)

    def search(self, vector, limit, doc_ids=None):
        pool = self.chunks
        if doc_ids is not None:
            pool = [c for c in pool if c.doc_id in doc_ids]
        return [SearchResult(chunk=c, score=1.0) for c in pool[:limit]]

    def delete_by_doc(self, doc_id):
        self.deleted.append(doc_id)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]


class FakeParser:
    def __init__(self, blocks=None, fail=False):
        self.blocks = blocks or []
        self.fail = fail

    def parse(self, path):
        if self.fail:
            raise ValueError("unreadable file")
        from ingestion.parser import ParsedDocument
        return ParsedDocument(markdown="# doc", blocks=self.blocks,
                              page_count=1)
