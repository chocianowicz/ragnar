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
             candidates: int | None = None,
             use_reranker: bool = True) -> SearchOutcome:
        """Retrieve, optionally rerank, then apply the similarity floor.

        doc_ids=None searches the whole corpus; a list scopes retrieval to
        just those documents (e.g. a user-selected subset in the UI).

        score_floor and candidates override the instance defaults for this
        one call, so the UI can pass per-request values without mutating
        shared state.

        Widening `candidates` is the blunt lever for a question whose
        answer the embedder ranks poorly. It costs linearly: the reranker
        scores every candidate, measured at roughly 1.3s each on CPU, so
        50 candidates doubles retrieval latency. It also cannot rescue an
        answer the embedder ranks far down — an exact identifier in a large
        reference table ranked 291st on the real corpus, because a code
        carries almost no signal a dense vector can use. That case needs
        lexical matching, not a longer list.

        use_reranker=False skips the cross-encoder entirely (much faster).
        The floor is not applied in that mode: raw vector-similarity scores
        are on a different, uncalibrated scale, so refusal then happens only
        when retrieval finds nothing at all.
        """
        floor = self.score_floor if score_floor is None else score_floor
        limit = self._candidates if candidates is None else candidates
        vector = self._embedder.embed([question])[0]
        pool = self._store.search(vector, limit=limit, doc_ids=doc_ids)

        if not pool:
            return SearchOutcome(refused=True)

        if not (use_reranker and self._reranker is not None):
            return SearchOutcome(results=pool[: self._top_k])

        ranked = self._reranker.rerank(question, pool, self._top_k)
        kept = [r for r in ranked if r.score >= floor]

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True
            )

        return SearchOutcome(results=kept)
