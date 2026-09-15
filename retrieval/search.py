import time
from dataclasses import dataclass, field

from core.models import SearchResult

RELATED_COUNT = 3


@dataclass
class SearchTrace:
    """What retrieval actually did, for the UI to show.

    A question takes tens of seconds on local hardware. Without this the
    user watches a spinner and has no idea whether anything was found, how
    close to the floor it came, or which stage is slow — and when the
    answer is a refusal, no way to tell "nothing matched" from "matches
    were found but scored too low", which are different problems with
    different fixes.
    """
    candidates: int = 0        # fetched from the store
    reranked: int = 0          # scored by the cross-encoder
    kept: int = 0              # cleared the floor
    floor: float = 0.0
    best_score: float | None = None
    hybrid: bool = False
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    related: list[SearchResult] = field(default_factory=list)
    refused: bool = False
    trace: SearchTrace = field(default_factory=SearchTrace)


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
             use_reranker: bool = True,
             on_step=None) -> SearchOutcome:
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

        on_step, if given, is called with a short label as each stage
        starts, so a caller can show progress during a wait measured in
        tens of seconds rather than leaving the user with a bare spinner.
        """
        step = on_step or (lambda _label: None)
        floor = self.score_floor if score_floor is None else score_floor
        limit = self._candidates if candidates is None else candidates
        hybrid = bool(getattr(self._store, "is_hybrid", False))
        trace = SearchTrace(floor=floor, hybrid=hybrid)

        step("Reading the question")
        clock = time.perf_counter()
        vector = self._embedder.embed([question])[0]
        trace.seconds["embed"] = time.perf_counter() - clock

        step("Searching your documents" + (" (meaning and wording)"
                                           if hybrid else ""))
        clock = time.perf_counter()
        # The question goes to the store as text as well as a vector: on a
        # hybrid collection that adds the lexical half, which is what
        # finds exact identifiers the embedder ranks nowhere near the top.
        pool = self._store.search(vector, limit=limit, doc_ids=doc_ids,
                                  text=question)
        trace.seconds["search"] = time.perf_counter() - clock
        trace.candidates = len(pool)

        if not pool:
            return SearchOutcome(refused=True, trace=trace)

        if not (use_reranker and self._reranker is not None):
            trace.kept = min(len(pool), self._top_k)
            return SearchOutcome(results=pool[: self._top_k], trace=trace)

        step(f"Re-checking the {len(pool)} closest passages")
        clock = time.perf_counter()
        ranked = self._reranker.rerank(question, pool, self._top_k)
        trace.seconds["rerank"] = time.perf_counter() - clock
        trace.reranked = len(pool)
        trace.best_score = ranked[0].score if ranked else None

        kept = [r for r in ranked if r.score >= floor]
        trace.kept = len(kept)

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True, trace=trace,
            )

        return SearchOutcome(results=kept, trace=trace)
