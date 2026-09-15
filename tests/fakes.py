"""Shared test doubles. Anything used by more than one test module
lives here."""
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

    def search(self, vector, limit, doc_ids=None, text=None):
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


class ScriptedLLM:
    """Returns the next scripted reply, recording what it was asked."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def generate(self, system, user, **kwargs):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


class FailingLLM:
    def generate(self, system, user, **kwargs):
        raise RuntimeError("ollama is down")


class RecordingStore:
    """Returns a fixed pool, remembering every query text it was given."""

    def __init__(self, results=None, by_text=None):
        self._results = results or []
        self._by_text = by_text or {}
        self.queries = []
        self.is_hybrid = True

    def search(self, vector, limit, doc_ids=None, text=None):
        self.queries.append(text)
        return self._by_text.get(text, self._results)[:limit]


class PassThroughReranker:
    """Scores by position, so ordering is predictable."""

    def __init__(self, score=0.9):
        self.score = score
        self.queries = []

    def rerank(self, query, candidates, top_k):
        self.queries.append(query)
        return [SearchResult(chunk=c.chunk, score=self.score)
                for c in candidates][:top_k]
