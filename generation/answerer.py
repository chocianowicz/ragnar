from dataclasses import dataclass, field
from enum import Enum

from core.models import SearchResult
from generation import injection
from generation.guards import should_refuse_aggregation
from generation.prompts import (
    INDIRECT_SYSTEM, NO_ANSWER, SYSTEM_PROMPT, build_indirect_prompt,
    build_user_prompt,
)

# The opening sentence of every refusal. "Nothing relevant" would
# contradict the near-miss sentence that can follow it — retrieval often
# does find relevant material that simply does not answer the question.
NO_RESULTS_MESSAGE = (
    "I could not find an answer to this in the indexed documents."
)

# How many of the closest documents a refusal names. Three is enough to
# say "the corpus covers this area, just not your question" and few enough
# that the refusal still reads as a refusal rather than a results page.
NEAR_MISS_LIMIT = 3

# Reranker scores are the cross-encoder's probability that a passage is
# relevant (retrieval/reranker.py). Measured on the real corpus, a question
# with no bearing on it at all pegs its entire candidate list below 0.001,
# while a genuine near miss reaches 0.012-0.31. Requiring real positive
# signal keeps the refusal from manufacturing a connection the reranker did
# not find — which would be worse than saying nothing, because the user goes
# and reads the documents it named. The threshold is a property of the
# model's output scale, not of the configured floor, so it survives
# recalibration. (Those figures were measured as 0.5002 and 0.503-0.578
# while a second sigmoid squeezed every score; this is the same cut-off.)
NEAR_MISS_MIN = 0.04


def _closest_labels(candidates: list[SearchResult]) -> list[str]:
    """One citation label per document, best-scoring page, best first."""
    best: dict[str, SearchResult] = {}
    for result in candidates:
        if result.score < NEAR_MISS_MIN:
            continue
        seen = best.get(result.chunk.filename)
        if seen is None or result.score > seen.score:
            best[result.chunk.filename] = result
    ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
    return [r.chunk.citation_label() for r in ranked[:NEAR_MISS_LIMIT]]


def _join(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def no_results_message(candidates: list[SearchResult], *,
                       model_declined: bool = False) -> str:
    """The refusal, naming the documents that came closest.

    A bare "nothing relevant" is wrong whenever retrieval found coherent
    material that was simply about something else — on the real corpus a
    question about one member state's target returns the bloc-wide target,
    which is topical and silent on the country. Read as "nothing relevant",
    that looks like a broken index rather than an honest gap.

    Only filenames and page numbers are used. Naming a document cannot be
    mistaken for an answer, whereas quoting one would be an answer that
    never cleared the floor. Nothing here calls the model, so a refusal
    stays deterministic — the property the floor exists to provide.

    `model_declined` distinguishes the two refusals, which fail at
    different stages and must not describe each other: below the floor
    nothing was close enough to read, while a declined answer means the
    model read passages that cleared the floor and found no answer in them.
    """
    labels = _closest_labels(candidates)
    if not labels:
        return NO_RESULTS_MESSAGE
    listed = _join(labels)
    if model_declined:
        return (f"{NO_RESULTS_MESSAGE} The closest passages were in "
                f"{listed}, but they do not answer it.")
    return (f"{NO_RESULTS_MESSAGE} The closest passages were in {listed}, "
            f"but none matched closely enough to answer from.")


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


def declined(text: str) -> bool:
    """Whether the model reported that the excerpts do not answer.

    Retrieval clearing the floor only means the passages looked relevant;
    the model reading them can still find no answer in them. That is a
    refusal too, and it must not be dressed up as an answer with five
    sources under it — the sources did not produce it.
    """
    return text.strip().startswith(NO_ANSWER)


def strip_sentinel(text: str) -> str:
    """The user-facing form of a declined answer."""
    return NO_RESULTS_MESSAGE if declined(text) else text


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


def build_citations(results: list[SearchResult], publish=None) -> list[dict]:
    """Citations with enough metadata for the UI to show and open a source.

    Deduplicated by label, first-seen order, same as citation_labels — but
    carrying the passage that was actually used, so "show me where this
    came from" is answered from what the model was given rather than from a
    fresh lookup that might return something else.

    `publish(doc_id, filename) -> str | None` makes the original file
    reachable and returns its URL. Injected rather than imported so this
    module stays free of UI concerns, and called once here rather than on
    every redraw.

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
            # Instruction-shaped text in the passage, if any. Empty for
            # almost every citation; when it is not, the UI says so.
            "flags": injection.flag(chunk.text),
            "url": publish(chunk.doc_id, chunk.filename) if publish else None,
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

        if declined(text):
            # No citations: nothing here was answered from them. This also
            # keeps the eval harness honest, which would otherwise score a
            # declined answer as a successful one.
            return Answer(
                text=no_results_message(results, model_declined=True),
                refused=True)

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

    def stream_indirect(self, question: str, bridge,
                        results: list[SearchResult], *,
                        model: str | None = None,
                        temperature: float | None = None,
                        history: list[dict] | None = None):
        """stream(), for an answer reached through a broader subject.

        `bridge` is a generation.broaden.Bridge. Its link passages, when
        the link came from the documents, go in beside the broader
        subject's, so the model sees every passage it is cited as using.
        """
        excerpts = build_excerpts(list(results) + list(bridge.link_results))
        yield from self._llm.stream(
            INDIRECT_SYSTEM,
            build_indirect_prompt(question, bridge.broader_question,
                                  bridge.link, bridge.source, excerpts),
            model=model, temperature=temperature,
            history=recent_history(history),
        )
