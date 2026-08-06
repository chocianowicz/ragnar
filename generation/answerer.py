from dataclasses import dataclass, field
from enum import Enum

from core.models import SearchResult
from generation.guards import should_refuse_aggregation
from generation.prompts import (
    SYSTEM_PROMPT, build_user_prompt,
    build_history_summary_prompt,
)

NO_RESULTS_MESSAGE = (
    "I could not find anything relevant in the indexed documents."
)


def citation_labels(results: list[SearchResult]) -> list[str]:
    """Deduplicated citation labels from search results, first-seen order.

    Citations come from retrieved chunk metadata, never from model prose —
    the model can't cite a document that wasn't actually retrieved.
    """
    seen: list[str] = []
    for result in results:
        label = result.chunk.citation_label()
        if label not in seen:
            seen.append(label)
    return seen


def build_excerpts(results: list[SearchResult]) -> list[tuple[str, str]]:
    """(citation_label, text) pairs for the prompt builder."""
    return [(r.chunk.citation_label(), r.chunk.text) for r in results]


class AnswerMode(Enum):
    NO_RESULTS = "no_results"
    AGGREGATION_REFUSED = "aggregation_refused"
    ANSWER = "answer"


def classify(question: str, refused: bool,
             results: list[SearchResult]) -> AnswerMode:
    """The single source of truth for the refuse / guard / answer decision.

    Both the UI and the eval harness route through this so the policy —
    "no candidates cleared the floor" vs "aggregation over a table we can't
    safely compute" vs "answer normally" — can never drift between them.
    """
    if refused:
        return AnswerMode.NO_RESULTS
    if should_refuse_aggregation(question, results):
        return AnswerMode.AGGREGATION_REFUSED
    return AnswerMode.ANSWER


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False


class Answerer:
    def __init__(self, llm):
        self._llm = llm

    def summarize_history(self, history: list[dict]) -> str | None:
        """Condense previous Q&A into 2-3 sentences, or None if too short.

        One turn or an empty history has nothing useful to summarize — the
        model handles a single follow-up fine without a summary.
        """
        if len(history) < 2:
            return None
        system_prompt, user_prompt = build_history_summary_prompt(history)
        try:
            return self._llm.generate(system_prompt, user_prompt)
        except Exception:
            return None

    def answer(self, question: str, results: list[SearchResult], *,
               model: str | None = None,
               temperature: float | None = None,
               history: list[dict] | None = None,
               context_summary: str | None = None) -> Answer:
        if not results:
            # Skip the model entirely — a refusal it cannot embellish.
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        if context_summary is None:
            context_summary = self.summarize_history(history or [])
        text = self._llm.generate(
            SYSTEM_PROMPT,
            build_user_prompt(question, build_excerpts(results),
                              context_summary=context_summary),
            model=model, temperature=temperature,
            history=history,
        )

        return Answer(text=text, citations=citation_labels(results))

    def stream(self, question: str, results: list[SearchResult], *,
               model: str | None = None, temperature: float | None = None,
               history: list[dict] | None = None,
               context_summary: str | None = None):
        """Yield answer-text deltas for the UI's st.write_stream.

        Citations are not part of the stream — they come from
        citation_labels(results) and are known before generation starts.
        model/temperature are threaded through per call so a shared Answerer
        instance never has to mutate the underlying LLM's state.
        """
        if context_summary is None:
            context_summary = self.summarize_history(history or [])
        yield from self._llm.stream(
            SYSTEM_PROMPT,
            build_user_prompt(question, build_excerpts(results),
                              context_summary=context_summary),
            model=model, temperature=temperature,
            history=history,
        )
