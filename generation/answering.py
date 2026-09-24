"""The whole answer, start to finish, with nothing from Streamlit in it.

This lived inside ui/app.py, reachable only through AppTest. It shipped
once with a name missing from the imports and the entire unit suite passed
while every answer failed. Services are injected so it can be driven by
fakes; the caller supplies a `job` to report into, which is the only
mutable thing it touches.

Runs on a background thread that outlives the script run which started it,
so it must never call st.*.
"""
from __future__ import annotations

from dataclasses import dataclass

from generation import broaden, followup
from generation.answerer import (
    AnswerMode, build_citations, citation_labels, classify, declined,
    no_results_message,
)
from generation.guards import aggregation_refusal
from generation.prompts import NO_ANSWER


@dataclass(frozen=True)
class Settings:
    """Per-question settings, as chosen in the Settings panel.

    Frozen and passed by value: two people using the app must never be
    able to change each other's choices, which is why none of this is
    written back onto a shared service.
    """
    model: str
    temperature: float
    floor: float
    candidates: int
    use_reranker: bool
    follow_up: bool
    multi_query: bool
    multi_hop: bool
    self_correct: bool
    # Off unless asked for, so anything that builds Settings without
    # knowing about it (the eval harness, tests) keeps the old refusals.
    broaden: bool = False

    @property
    def wants_extra_stages(self) -> bool:
        return any((self.multi_query, self.multi_hop,
                    self.self_correct))


def answer(job, question: str, *, history: list[dict],
           doc_ids: list[str] | None, settings: Settings,
           search, answerer, agentic, llm, publish) -> None:
    """Retrieve, decide, and stream an answer into `job`.

    Reports progress by assigning to job.status and appends text as it
    arrives.
    """
    # "Remember context" governs every use of the conversation, not just
    # follow-up resolution: with it off, the answer model sees no earlier
    # messages and no broader-subject link can come from the chat. The
    # one switch the user sees is the whole of what the chat can do.
    if not settings.follow_up:
        history = []

    search_question, resolved = question, False
    if history and settings.follow_up:
        job.status = "Working out what the question refers to"
        search_question, resolved = followup.resolve(
            llm, question, history, model=settings.model)

    def step(label: str) -> None:
        job.status = label

    common = dict(doc_ids=doc_ids, score_floor=settings.floor,
                  candidates=settings.candidates,
                  use_reranker=settings.use_reranker, on_step=step)

    if settings.wants_extra_stages and agentic is not None:
        outcome, agentic_trace = agentic.find(
            search_question, multi_query=settings.multi_query,
            multi_hop=settings.multi_hop,
            self_correct=settings.self_correct, **common)
    else:
        outcome = search.find(search_question, **common)
        agentic_trace = None

    mode = classify(question, outcome.refused, outcome.results)
    job.trace = outcome.trace.as_dict()
    job.trace["resolved_question"] = search_question if resolved else None
    if agentic_trace is not None:
        job.trace["agentic"] = agentic_trace.as_dict()

    if mode is AnswerMode.NO_RESULTS:
        # Nothing retrieved at all means an empty scope, not a gap a
        # broader subject could fill.
        if outcome.trace.candidates and _indirect(
                job, question, search_question, history=history,
                settings=settings, search=search, answerer=answerer,
                llm=llm, publish=publish, common=common):
            return
        # outcome.related is what retrieval found and the floor rejected.
        # Naming those documents is the difference between "the index is
        # empty on this" and "the index covers this area but not your
        # question" — which is what the user needs to know next.
        job.append(no_results_message(outcome.related))
        job.trace["related"] = citation_labels(outcome.related)
        return
    if mode is AnswerMode.AGGREGATION_REFUSED:
        job.append(aggregation_refusal(outcome.results))
        return

    job.citations = build_citations(outcome.results, publish=publish)
    job.status = "Writing the answer"

    if not _stream(job, answerer.stream(question, outcome.results,
                                        model=settings.model,
                                        temperature=settings.temperature,
                                        history=history)):
        return

    # The passages looked relevant but did not answer. Not an answer, so
    # no sources: they did not produce this.
    job.chunks.clear()
    job.citations = []
    if _indirect(job, question, search_question, history=history,
                 settings=settings, search=search, answerer=answerer,
                 llm=llm, publish=publish, common=common):
        return
    job.append(no_results_message(outcome.results, model_declined=True))
    job.trace["model_declined"] = True
    job.trace["related"] = citation_labels(outcome.results)


def _stream(job, pieces) -> bool:
    """Stream an answer into `job`; True if the model declined instead.

    Holds the opening back until it is clear whether this is an answer or
    the model declining, so the sentinel never appears on screen. It is
    the first thing emitted when it is emitted at all, so a short buffer
    settles it. On a decline nothing has been appended.
    """
    buffer, deciding = "", True
    for piece in pieces:
        if deciding:
            buffer += piece
            if declined(buffer):
                return True
            if len(buffer.strip()) < len(NO_ANSWER):
                continue          # still could go either way
            deciding = False
            job.append(buffer)
            continue
        job.append(piece)

    if declined(buffer):
        return True
    if deciding:
        job.append(buffer)        # stream ended inside the buffer
    return False


def _indirect(job, question: str, search_question: str, *, history,
              settings: Settings, search, answerer, llm, publish,
              common: dict) -> bool:
    """Try to answer through a broader subject; True if it did.

    Runs only after a refusal, and leaves `job` untouched apart from the
    trace when it cannot answer, so the caller's refusal goes out as it
    would have. See generation/broaden.py for what makes a link count.
    """
    if not settings.broaden:
        return False

    job.status = "Looking for a broader subject the documents cover"
    bridge, found = broaden.attempt(search_question, history, llm=llm,
                                    search=search, typed=question,
                                    model=settings.model, **common)
    if bridge is None:
        job.trace["broader_attempt"] = found
        return False

    job.citations = build_citations(found.results + bridge.link_results,
                                    publish=publish)
    job.status = "Writing the answer"
    if _stream(job, answerer.stream_indirect(
            question, bridge, found.results, model=settings.model,
            temperature=settings.temperature, history=history)):
        job.citations = []
        job.trace["broader_attempt"] = (
            f"searched for “{bridge.broader_question}”, but the passages "
            f"found do not answer it")
        return False

    job.trace["indirect"] = bridge.as_dict()
    job.trace["indirect"]["search"] = found.trace.as_dict()
    return True
