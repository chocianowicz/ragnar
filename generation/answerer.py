from dataclasses import dataclass, field
from enum import Enum

from core.models import SearchResult
from generation.guards import should_refuse_aggregation
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

NO_RESULTS_MESSAGE = (
    "I could not find anything relevant in the indexed documents."
)

# How many prior messages are replayed to the model. Six is three
# exchanges, which covers the follow-up chains people actually write
# ("...and the base year?", "...what about Iceland?") without letting the
# prompt grow without bound as a conversation goes on. Anything older is
# dropped rather than summarised: summarising costs a model call on every
# turn, and the question sent to retrieval is already resolved against
# history before it gets here (see generation/followup.py).
HISTORY_TURNS = 6


def recent_history(history: list[dict] | None) -> list[dict]:
    """The last few turns, reduced to what a chat model accepts.

    Messages carry citations and a retrieval trace for the UI; sending
    those to the model would be noise at best. Only role and content go.
    """
    if not history:
        return []
    return [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in history[-HISTORY_TURNS:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


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


def build_citations(results: list[SearchResult]) -> list[dict]:
    """Citations with enough metadata for the UI to show and open a source.

    Deduplicated by label, first-seen order, same as citation_labels — but
    carrying the passage that was actually used, so "show me where this
    came from" is answered from what the model was given rather than from a
    fresh lookup that might return something else.

    Plain dicts, deliberately: these are written straight into the chat
    history, which is persisted as JSON. A dataclass here would serialize
    only with a custom encoder, and would come back as a dict on load
    anyway — so the two paths would disagree about the type.
    """
    seen: set[str] = set()
    citations: list[dict] = []
    for result in results:
        chunk = result.chunk
        label = chunk.citation_label()
        if label in seen:
            continue
        seen.add(label)
        citations.append({
            "label": label,
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page": chunk.page,
            "sheet": chunk.sheet,
            "chunk_index": chunk.chunk_index,
            "score": round(float(result.score), 4),
            "text": chunk.text,
        })
    return citations


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

    def answer(self, question: str, results: list[SearchResult], *,
               model: str | None = None,
               temperature: float | None = None,
               history: list[dict] | None = None) -> Answer:
        if not results:
            # Skip the model entirely — a refusal it cannot embellish.
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        text = self._llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, build_excerpts(results)),
            model=model, temperature=temperature,
            history=recent_history(history),
        )

        return Answer(text=text, citations=citation_labels(results))

    def stream(self, question: str, results: list[SearchResult], *,
               model: str | None = None, temperature: float | None = None,
               history: list[dict] | None = None):
        """Yield answer-text deltas for the UI's st.write_stream.

        Citations are not part of the stream — they come from
        citation_labels(results) and are known before generation starts.
        model/temperature are threaded through per call so a shared Answerer
        instance never has to mutate the underlying LLM's state.
        """
        yield from self._llm.stream(
            SYSTEM_PROMPT, build_user_prompt(question, build_excerpts(results)),
            model=model, temperature=temperature,
            history=recent_history(history),
        )
