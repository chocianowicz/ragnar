from core.models import SearchResult


class Search:
    def __init__(self, embedder, store, candidates: int = 25):
        self._embedder = embedder
        self._store = store
        self._candidates = candidates

    def find(self, question: str) -> list[SearchResult]:
        vector = self._embedder.embed([question])[0]
        return self._store.search(vector, limit=self._candidates)
