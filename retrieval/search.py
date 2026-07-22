from dataclasses import dataclass, field

from core.models import SearchResult

RELATED_COUNT = 3


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    related: list[SearchResult] = field(default_factory=list)
    refused: bool = False


class Search:
    """Retrieve, rerank, then apply the similarity floor.

    When nothing clears the floor the LLM is never called — that is what
    makes the refusal trustworthy, since there is no opportunity for the
    model to improvise.
    """

    def __init__(self, embedder, store, reranker=None, candidates: int = 25,
                 top_k: int = 5, score_floor: float = 0.0):
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        self._score_floor = score_floor

    def find(self, question: str) -> SearchOutcome:
        vector = self._embedder.embed([question])[0]
        candidates = self._store.search(vector, limit=self._candidates)

        if not candidates:
            return SearchOutcome(refused=True)

        if self._reranker is not None:
            ranked = self._reranker.rerank(question, candidates, self._top_k)
        else:
            ranked = candidates[: self._top_k]

        kept = [r for r in ranked if r.score >= self._score_floor]

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True
            )

        return SearchOutcome(results=kept)
