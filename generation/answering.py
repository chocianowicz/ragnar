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

from generation import followup
from generation.answerer import (
    AnswerMode, build_citations, citation_labels, classify, declined,
    NO_RESULTS_MESSAGE,
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
    rewrite: bool
    multi_query: bool
    multi_hop: bool
    self_correct: bool

    @property
    def wants_extra_stages(self) -> bool:
        return any((self.rewrite, self.multi_query,
                    self.multi_hop, self.self_correct))


def answer(job, question: str, *, history: list[dict],
           doc_ids: list[str] | None, settings: Settings,
           search, answerer, agentic, llm, publish) -> None:
    """Retrieve, decide, and stream an answer into `job`.

    Reports progress by assigning to job.status and appends text as it
    arrives.
    """
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
            search_question, rewrite=settings.rewrite,
            multi_query=settings.multi_query, multi_hop=settings.multi_hop,
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
        job.append(NO_RESULTS_MESSAGE)
        job.trace["related"] = citation_labels(outcome.related)
        return
    if mode is AnswerMode.AGGREGATION_REFUSED:
        job.append(aggregation_refusal(outcome.results))
        return

    job.citations = build_citations(outcome.results, publish=publish)
    job.status = "Writing the answer"

    # Hold the opening back until it is clear whether this is an answer or
    # the model declining, so the sentinel never appears on screen. It is
    # the first thing emitted when it is emitted at all, so a short buffer
    # settles it.
    buffer, deciding = "", True
    for piece in answerer.stream(question, outcome.results,
                                 model=settings.model,
                                 temperature=settings.temperature,
                                 history=history):
        if deciding:
            buffer += piece
            if declined(buffer):
                break
            if len(buffer.strip()) < len(NO_ANSWER):
                continue          # still could go either way
            deciding = False
            job.append(buffer)
            continue
        job.append(piece)

    if declined(buffer):
        # The passages looked relevant but did not answer. Not an answer,
        # so no sources: they did not produce this.
        job.chunks.clear()
        job.citations = []
        job.append(NO_RESULTS_MESSAGE)
        job.trace["model_declined"] = True
        job.trace["related"] = citation_labels(outcome.results)
    elif deciding:
        job.append(buffer)        # stream ended inside the buffer
