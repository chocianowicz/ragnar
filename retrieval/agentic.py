"""Agentic retrieval: rewrite the query, search several ways, follow up.

Wraps Search and uses only its public seams — retrieve() for a raw
candidate pool, narrow() to rerank and apply the floor. It does not
re-implement either.

The shape is deliberately different from the obvious one. Retrieval is
cheap (a vector search is milliseconds) and reranking is expensive
(~1.3s per candidate on CPU, and it dominates every question). So extra
queries are used to widen the *pool*, and the pool is reranked once,
against the question the user actually asked. Running each variant through
a full rerank instead would multiply the slowest stage by the number of
variants, which is what makes naive agentic RAG unusable locally.

That also removes a subtler problem: scores from different queries are not
comparable, so fusing reranked lists lets a chunk cross the floor on a
phrasing the model invented rather than on the user's question. Reranking
the union once means every surviving chunk cleared the floor against the
real question.

One invariant is preserved throughout: the model never sees retrieved
content that has not cleared the floor. Query rewriting and variant
generation see only the user's question, never a document. Multi-hop and
self-correction do see excerpts — so they run only after a first pass has
produced results that cleared the floor, where the model was going to be
called anyway.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

from core.models import SearchResult
from generation.agentic_prompts import (
    build_rewrite_prompt, build_multi_query_prompt, build_hop_prompt,
    build_correction_prompt, parse_tagged,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticTrace:
    """What the agent did, for the UI to show beside the retrieval trace."""
    rewritten_query: str | None = None
    queries: list[str] = field(default_factory=list)
    hops: int = 0
    self_corrected: bool = False
    pool_size: int = 0
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Every field, for the UI and the chat history.

        dataclasses.asdict rather than a hand-written literal: the
        previous version named each field, so adding one meant the UI
        quietly never showed it.
        """
        return dataclasses.asdict(self)


def _fuse(results: list[SearchResult], cap: int) -> list[SearchResult]:
    """Best-scoring instance of each chunk, highest first, capped.

    Used for both the multi-query union and each widening hop: the same
    three lines were written out in both places, and a difference between
    them would have been a silent behaviour change.
    """
    best: dict[str, SearchResult] = {}
    for result in results:
        current = best.get(result.chunk.key())
        if current is None or result.score > current.score:
            best[result.chunk.key()] = result
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:cap]


class AgenticSearch:
    """Search with optional query expansion and follow-up retrieval.

    Every stage is off unless switched on. They cost LLM calls on top of a
    question that is already slow, and none of them is validated against a
    golden set — see eval/README.md.
    """

    def __init__(self, base_search, llm, *, max_hops: int = 2,
                 variants: int = 3):
        self._search = base_search
        self._llm = llm
        self._max_hops = max_hops
        self._variants = variants

    def find(self, question: str, *,
             doc_ids: list[str] | None = None,
             score_floor: float | None = None,
             candidates: int | None = None,
             use_reranker: bool = True,
             rewrite: bool = False,
             multi_query: bool = False,
             multi_hop: bool = False,
             self_correct: bool = False,
             on_step=None):
        """Returns (SearchOutcome, AgenticTrace).

        Every stage flag defaults to off, so this is exactly a base search
        until asked for more.
        """
        step = on_step or (lambda _label: None)
        trace = AgenticTrace()
        limit = candidates if candidates is not None else None

        search_query = question
        if rewrite:
            step("Rewriting the question for search")
            search_query = self._rewrite(question, trace)
            trace.rewritten_query = (
                search_query if search_query != question else None)

        queries = [search_query]
        if multi_query:
            step("Thinking of other ways to ask")
            # Deduplicated against the query already being searched, not
            # just against each other: a model asked for rephrasings will
            # often hand back the original among them, and searching the
            # same text twice costs a round trip for nothing.
            seen = {search_query.lower().rstrip("?.")}
            for variant in self._variant_queries(question, trace):
                key = variant.lower().rstrip("?.")
                if key not in seen:
                    seen.add(key)
                    queries.append(variant)
        trace.queries = list(queries)

        # Cheap stage: one vector/lexical search per phrasing.
        step(f"Searching your documents ({len(queries)} phrasings)"
             if len(queries) > 1 else "Searching your documents")
        pool = self._pool(queries, doc_ids, limit)
        trace.pool_size = len(pool)
        if not pool:
            from retrieval.search import SearchOutcome
            return SearchOutcome(refused=True), trace

        # Expensive stage, run once, against the user's actual question.
        outcome = self._search.narrow(
            question, pool, score_floor=score_floor,
            use_reranker=use_reranker, step=step)

        # Follow-ups read retrieved text, so they only run once something
        # has cleared the floor — the model is never shown content that
        # did not.
        if outcome.results and (multi_hop or self_correct):
            outcome = self._follow_up(
                question, outcome, pool, doc_ids, limit, score_floor,
                use_reranker, multi_hop, self_correct, trace, step)

        return outcome, trace

    # ------------------------------------------------------------------

    def _ask(self, system: str, user: str, trace: AgenticTrace) -> str | None:
        """One LLM call, counted, and never fatal to the search."""
        try:
            trace.llm_calls += 1
            return self._llm.generate(system, user).strip()
        except Exception as exc:
            logger.warning("agentic stage failed: %s", exc)
            trace.notes.append("a reasoning step failed and was skipped")
            return None

    def _rewrite(self, question: str, trace: AgenticTrace) -> str:
        system, user = build_rewrite_prompt(question)
        rewritten = self._ask(system, user, trace)
        # A rewrite that collapses to almost nothing has lost the question;
        # keep the original rather than search for a fragment.
        if rewritten and len(rewritten) >= max(8, len(question) // 4):
            return rewritten
        return question

    def _variant_queries(self, question: str,
                         trace: AgenticTrace) -> list[str]:
        system, user = build_multi_query_prompt(question, self._variants)
        raw = self._ask(system, user, trace)
        if not raw:
            return []
        seen, out = set(), []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•0123456789. ").strip()
            key = line.lower().rstrip("?.")
            if line and key not in seen:
                seen.add(key)
                out.append(line)
        return out[: self._variants]

    def _pool(self, queries: list[str], doc_ids, limit) -> list[SearchResult]:
        """Union of the candidate pools for each phrasing.

        Sequential on purpose. The obvious move is a thread pool, but these
        calls share one embedder connection and one store client, and the
        work here is milliseconds — the time in a question is reranking,
        which happens once, after this.
        """
        found: list[SearchResult] = []
        for query in queries:
            found += self._search.retrieve(query, doc_ids=doc_ids,
                                           limit=limit)
        # Keep the pool the same size the reranker would have seen anyway,
        # so extra phrasings improve what is in the pool without making the
        # expensive stage any slower.
        cap = limit if limit is not None else self._search.candidates
        return _fuse(found, cap)

    def _follow_up(self, question, outcome, pool, doc_ids, limit,
                   score_floor, use_reranker, multi_hop, self_correct,
                   trace, step):
        """Ask whether anything is missing, and search again if so."""
        for hop in range(self._max_hops if multi_hop else 0):
            excerpts = [(r.chunk.citation_label(), r.chunk.text)
                        for r in outcome.results]
            step(f"Checking for gaps (follow-up {hop + 1})")
            raw = self._ask(*build_hop_prompt(question, excerpts), trace)
            if not raw or parse_tagged(raw, "Sufficient", "yes").lower() == "yes":
                break
            follow_up = parse_tagged(raw, "FollowUp", "none")
            if follow_up.lower() in ("none", "", "n/a"):
                break

            trace.hops += 1
            trace.queries.append(follow_up)
            outcome = self._widen(question, follow_up, pool, doc_ids, limit,
                                  score_floor, use_reranker, step)
            if not outcome.results:
                break

        if self_correct and outcome.results:
            step("Reviewing the draft answer")
            outcome = self._correct(question, outcome, pool, doc_ids, limit,
                                    score_floor, use_reranker, trace, step)
        return outcome

    def _widen(self, question, extra_query, pool, doc_ids, limit,
               score_floor, use_reranker, step):
        """Rerank the pool plus one query's extra candidates.

        Returns a new pool rather than growing the caller's: the previous
        version extended the list it was passed, which is invisible from
        the signature and compounds over hops.
        """
        extra = self._search.retrieve(extra_query, doc_ids=doc_ids,
                                      limit=limit)
        cap = limit if limit is not None else self._search.candidates
        widened = _fuse(list(pool) + extra, cap)
        return self._search.narrow(question, widened,
                                   score_floor=score_floor,
                                   use_reranker=use_reranker, step=step)

    def _correct(self, question, outcome, pool, doc_ids, limit, score_floor,
                 use_reranker, trace, step):
        from generation.answerer import build_excerpts
        from generation.prompts import SYSTEM_PROMPT, build_user_prompt

        excerpts = build_excerpts(outcome.results)
        draft = self._ask(SYSTEM_PROMPT,
                          build_user_prompt(question, excerpts), trace)
        if not draft:
            return outcome

        raw = self._ask(*build_correction_prompt(question, excerpts, draft),
                        trace)
        if not raw or parse_tagged(raw, "Complete", "yes").lower() == "yes":
            return outcome

        improvement = parse_tagged(raw, "Improvement", "none")
        if improvement.lower() in ("none", "", "n/a"):
            return outcome

        trace.self_corrected = True
        trace.queries.append(improvement)
        widened = self._widen(question, improvement, pool, doc_ids, limit,
                              score_floor, use_reranker, step)
        return widened if widened.results else outcome
