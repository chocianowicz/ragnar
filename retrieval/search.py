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
        # Public so the UI can adjust it per-query without rebuilding Search.
        self.score_floor = score_floor

    def find(self, question: str,
             doc_ids: list[str] | None = None,
             score_floor: float | None = None,
             use_reranker: bool = True) -> SearchOutcome:
        """Retrieve, optionally rerank, then apply the similarity floor.

        doc_ids=None searches the whole corpus; a list scopes retrieval to
        just those documents (e.g. a user-selected subset in the UI).

        score_floor overrides self.score_floor for this one call, so the UI
        can pass a per-request value without mutating shared state.

        use_reranker=False skips the cross-encoder entirely (much faster).
        The floor is not applied in that mode: raw vector-similarity scores
        are on a different, uncalibrated scale, so refusal then happens only
        when retrieval finds nothing at all.
        """
        floor = self.score_floor if score_floor is None else score_floor
        vector = self._embedder.embed([question])[0]
        candidates = self._store.search(
            vector, limit=self._candidates, doc_ids=doc_ids
        )

        if not candidates:
            return SearchOutcome(refused=True)

        if not (use_reranker and self._reranker is not None):
            return SearchOutcome(results=candidates[: self._top_k])

        ranked = self._reranker.rerank(question, candidates, self._top_k)
        kept = [r for r in ranked if r.score >= floor]

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True
            )

        return SearchOutcome(results=kept)
