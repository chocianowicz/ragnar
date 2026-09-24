"""Answering through a broader subject the documents do cover.

On the real corpus, "what is Poland's net zero target?" is refused: the
documents give the EU-wide target and never name Poland, so nothing clears
the floor against a question about Poland. Yet the conversation may already
have established that Poland is an EU member state, and with that one link
the EU target is a real, sourced answer, as long as it is presented as
what it is.

The model may *propose* the broader subject and the link between them, but
its proposal is never evidence. The link has to be established by one of
two things, checked here rather than taken on the model's word:

- the conversation: the model must quote it, and the quote must appear
  verbatim in a message of this chat or in the question itself;
- the documents: the link, searched on its own, must clear the floor.

If neither holds, the refusal stands. The broader question also goes
through the ordinary search and floor, so the answer itself is only ever
built from passages that cleared it. Everything here runs only after a
refusal, so a question the documents answer directly pays nothing for it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.models import SearchResult
from generation.agentic_prompts import parse_tagged

logger = logging.getLogger(__name__)

BROADEN_SYSTEM = """\
A document search found nothing about the subject of a question. You
propose a broader subject the documents might cover instead, and the fact
that links the two.

Answer in exactly this form, three lines:
Broader: <the same question, asking for the same thing, but about the
larger group, region or category the subject belongs to, or none>
Link: <one sentence stating that the subject belongs to it, or none>
Quote: <the exact words from the conversation or the question that state
the link, copied character for character, or none>

Rules:
- Keep the language of the question.
- Only propose a group the subject genuinely belongs to.
- The broader question asks what the question asks, nothing else. Do not
  switch to another topic from the conversation.
- Quote only words that actually appear above, and only a statement: a
  question someone asked establishes nothing. If nothing there states the
  link, write "Quote: none". The link will be checked, so do not invent one.
- If there is no sensible broader subject, write "Broader: none".
"""

# How much conversation the proposal sees. The link, when the chat
# establishes it, is usually in the last exchange or two; the same window
# the follow-up resolver uses.
CONTEXT_TURNS = 4

# A quote shorter than this proves nothing: "yes" appears in every chat.
MIN_QUOTE_CHARS = 8

# Passages from the link search carried into the answer and its sources.
LINK_RESULTS = 2

_NONE = ("", "none", "n/a")


@dataclass
class Proposal:
    broader: str
    link: str
    quote: str | None


@dataclass
class Bridge:
    """How the question was connected to what the documents cover."""
    broader_question: str
    link: str
    source: str                          # "conversation" | "documents"
    quote: str | None = None
    speaker: str | None = None           # "user" | "assistant", chat only
    link_results: list[SearchResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        """For the trace, which is persisted as JSON: labels, not chunks."""
        return {
            "broader_question": self.broader_question,
            "link": self.link,
            "source": self.source,
            "quote": self.quote,
            "speaker": self.speaker,
            "link_citations": [r.chunk.citation_label()
                               for r in self.link_results],
        }


def _transcript(history: list[dict]) -> str:
    lines = []
    for message in history[-CONTEXT_TURNS:]:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        who = "User" if role == "user" else "Assistant"
        lines.append(f"{who}: {str(message.get('content', ''))[:800]}")
    return "\n".join(lines)


def _normalise(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def propose(llm, question: str, history: list[dict] | None,
            model: str | None = None) -> Proposal | None:
    """The model's suggestion, or None. Never fatal to the answer."""
    transcript = _transcript(history or [])
    user = (f"Conversation:\n{transcript or '(none)'}\n\n"
            f"Question: {question}")
    try:
        raw = llm.generate(BROADEN_SYSTEM, user, model=model)
    except Exception as exc:
        logger.warning("broader-subject proposal failed: %s", exc)
        return None

    broader = parse_tagged(raw or "", "Broader", "none")
    link = parse_tagged(raw or "", "Link", "none")
    quote = parse_tagged(raw or "", "Quote", "none").strip().strip("\"'“”„")
    if broader.lower() in _NONE or link.lower() in _NONE:
        return None
    if _normalise(broader).rstrip("?.") == _normalise(question).rstrip("?."):
        return None               # not broader, just the question again
    return Proposal(broader=broader, link=link,
                    quote=None if quote.lower() in _NONE else quote)


def _stated(needle: str, text: str) -> bool:
    """Whether `needle` occurs in `text` as part of a statement.

    Words inside a question establish nothing: "Is Poland in the EU?"
    contains "Poland in the EU" and proves the opposite of settled. So an
    occurrence in a sentence ending in a question mark counts only as a
    premise stated up front: it opens the sentence and the question follows
    it, as in "Poland is in the EU, so what is its target?".
    """
    start = text.find(needle)
    while start != -1:
        rest = text[start + len(needle):]
        end = re.search(r"[.!?\n]", rest)
        if not (end and end.group() == "?"):
            return True
        before = re.split(r"[.!?\n]", text[:start])[-1]
        premise = not before.strip(" \"'“”„-—:")
        if premise and rest[:end.start()].strip(" ,;:—-"):
            return True
        start = text.find(needle, start + 1)
    return False


def quoted_in(quote: str | None, question: str,
              history: list[dict] | None) -> str | None:
    """Who stated the quoted words, if anyone in this chat did.

    Returns "user" or "assistant", or None when the words are not there
    as a statement. This check is the reason a quote is asked for at all:
    whether the conversation established something is decided by a
    substring match, not by the model saying so. `question` must be what
    the user typed. A rewritten follow-up is model output, and matching
    against it would let the model quote itself.
    """
    if not quote:
        return None
    needle = _normalise(quote).rstrip(".!")
    if len(needle) < MIN_QUOTE_CHARS or needle.endswith("?"):
        return None
    for message in reversed(history or []):
        role = message.get("role")
        if role in ("user", "assistant") and \
                _stated(needle, _normalise(str(message.get("content", "")))):
            return role
    if _stated(needle, _normalise(question)):
        return "user"
    return None


def attempt(question: str, history: list[dict] | None, *, llm, search,
            typed: str | None = None, model: str | None = None,
            **search_kwargs):
    """(Bridge, SearchOutcome) for an indirect answer, or (None, note).

    `question` is what gets broadened, normally the resolved follow-up;
    `typed` is what the user actually wrote, the only form of the question
    a quote may be matched against. `note` says why no indirect answer was
    possible, for the trace: a refusal is more useful when it says what
    was also tried.
    """
    proposal = propose(llm, question, history, model=model)
    if proposal is None:
        return None, "no broader subject to try"

    outcome = search.find(proposal.broader, **search_kwargs)
    if outcome.refused or not outcome.results:
        return None, (f"searched for “{proposal.broader}”, "
                      f"but nothing cleared the floor")

    speaker = quoted_in(proposal.quote,
                        question if typed is None else typed, history)
    if speaker is not None:
        return Bridge(broader_question=proposal.broader, link=proposal.link,
                      source="conversation", quote=proposal.quote,
                      speaker=speaker), outcome

    if not search_kwargs.get("use_reranker", True):
        # Without the reranker there is no floor (retrieval/search.py), so
        # a search for the link returns something whatever the documents
        # say, and any link the model proposed would pass as proven.
        return None, (f"found “{proposal.broader}”, but this chat does not "
                      f"establish that {proposal.link.rstrip('.')}, and with re-ranking "
                      f"off the documents cannot be checked for it")

    linked = search.find(proposal.link, **search_kwargs)
    if linked.refused or not linked.results:
        return None, (f"found “{proposal.broader}”, but neither this chat "
                      f"nor the documents establish that {proposal.link.rstrip('.')}")
    return Bridge(broader_question=proposal.broader, link=proposal.link,
                  source="documents",
                  link_results=linked.results[:LINK_RESULTS]), outcome
