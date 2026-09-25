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

Answer in exactly this form, five lines:
Subject: <the specific thing the question is about, as written in it>
Group: <the larger group, region or category the subject belongs to, or
none>
Broader: <the same question, asking for the same thing, but about the
group, or none>
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
- The subject is the thing (a country, organisation, company, product),
  never the topic asked about it. The group is a thing of the same kind
  that contains it, never a topic.

Example. Question: "What is Bavaria's renewable energy target?", after the
user said "Bavaria is a state of Germany."
Subject: Bavaria
Group: Germany
Broader: What is Germany's renewable energy target?
Link: Bavaria is a state of Germany.
Quote: Bavaria is a state of Germany
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

SUBJECT_SYSTEM = """\
You name the specific subject a question asks about: a country, region,
organisation, company, sector, product or person. Not the topic, the
subject: in "what is Norway's 2030 target?" it is Norway, not the target.

Answer in exactly this form, two lines:
Subject: <the subject, as written in the question, or none>
English: <the same subject in English, or none>

Write "none" when the question has no specific subject, such as a
general or definitional question.

Examples.

Question: What is Bavaria's renewable energy target?
Subject: Bavaria
English: Bavaria

Question: jaki jest cel emisyjny Niemiec?
Subject: Niemiec
English: Germany

Question: What does net zero mean?
Subject: none
English: none
"""


@dataclass
class Proposal:
    broader: str
    link: str
    quote: str | None
    subject: str | None = None
    group: str | None = None


@dataclass
class Bridge:
    """How the question was connected to what the documents cover."""
    broader_question: str
    link: str
    source: str                          # "conversation" | "documents"
    quote: str | None = None
    speaker: str | None = None           # "user" | "assistant", chat only
    link_results: list[SearchResult] = field(default_factory=list)
    subject: str | None = None           # what the documents do not cover
    group: str | None = None             # what they cover instead

    def as_dict(self) -> dict:
        """For the trace, which is persisted as JSON: labels, not chunks."""
        return {
            "subject": self.subject,
            "group": self.group,
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

    subject = parse_tagged(raw or "", "Subject", "none")
    group = parse_tagged(raw or "", "Group", "none")
    broader = parse_tagged(raw or "", "Broader", "none")
    link = parse_tagged(raw or "", "Link", "none")
    quote = parse_tagged(raw or "", "Quote", "none").strip().strip("\"'“”„")
    if broader.lower() in _NONE or link.lower() in _NONE:
        return None
    if _normalise(broader).rstrip("?.") == _normalise(question).rstrip("?."):
        return None               # not broader, just the question again
    return Proposal(broader=broader, link=link,
                    quote=None if quote.lower() in _NONE else quote,
                    subject=None if subject.lower() in _NONE else subject,
                    group=None if group.lower() in _NONE else group)


def subject(llm, question: str, model: str | None = None) -> list[str]:
    """The question's specific subject, as written and in English.

    Empty when the question has none, or when the model fails: a general
    question is answered normally, and a failed check must not cost the
    user an answer the passages may well support.
    """
    try:
        raw = llm.generate(SUBJECT_SYSTEM, f"Question: {question}",
                           model=model)
    except Exception as exc:
        logger.warning("subject check failed: %s", exc)
        return []
    names = []
    for tag in ("Subject", "English"):
        value = parse_tagged(raw or "", tag, "none")
        # A model that folds both onto one line ("Polski / English:
        # Poland") still yields both names.
        for part in re.split(r"\s*/\s*(?:english:)?\s*", value,
                             flags=re.IGNORECASE):
            name = part.strip().strip("\"'“”„")
            if name and name.lower() not in _NONE and name not in names:
                names.append(name)
    return names


def _stems(name: str) -> list[str]:
    """Word stems that identify `name` in running text.

    One letter comes off longer words, so "Poland" also finds "Poland's"
    and "Germany" finds "German", without "Pol" matching "policy". Words
    under four letters ("EU", "UK") are kept whole and matched as words.
    """
    stems = []
    for word in re.findall(r"\w+", name.casefold()):
        if word in _GENERIC:
            continue
        stems.append(word[:-1] if len(word) > 5 else word)
    return stems


# Words that name the kind of thing rather than the thing: "the Republic
# of Korea" is identified by "Korea", and "republic" appears everywhere.
_GENERIC = {"the", "of", "and", "republic", "kingdom", "state", "states",
            "united", "federal", "government", "company", "sector"}


def mentioned(names: list[str], results: list[SearchResult]) -> bool:
    """Whether any passage mentions any of `names`, decided in code.

    This is the check that keeps the model from answering a question about
    one subject out of passages about another: with the re-ranker off,
    five passages always reach it, whatever they are about. A name counts
    as mentioned if every one of its identifying stems starts a word in
    some passage.
    """
    texts = [r.chunk.text.casefold() for r in results]
    for name in names:
        stems = _stems(name)
        if not stems:
            continue
        for text in texts:
            if all(re.search(rf"\b{re.escape(stem)}", text)
                   for stem in stems):
                return True
    return False


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
            subject_name: str | None = None, **search_kwargs):
    """(Bridge, SearchOutcome) for an indirect answer, or (None, note).

    `question` is what gets broadened, normally the resolved follow-up;
    `typed` is what the user actually wrote, the only form of the question
    a quote may be matched against. `subject_name` is the subject already
    found missing, when it is known. `note` says why no indirect answer was
    possible, for the trace: a refusal is more useful when it says what
    was also tried.
    """
    proposal = propose(llm, question, history, model=model)
    if proposal is None:
        return None, "no broader subject to try"
    proposal.subject = subject_name or proposal.subject

    outcome = search.find(proposal.broader, **search_kwargs)
    if outcome.refused or not outcome.results:
        return None, (f"searched for “{proposal.broader}”, "
                      f"but nothing cleared the floor")
    # With no floor the search returns five passages whatever they are
    # about, so they only count as covering the group if they name it.
    if proposal.group and not mentioned([proposal.group], outcome.results):
        return None, (f"searched for “{proposal.broader}”, but the passages "
                      f"found do not mention {proposal.group}")

    speaker = quoted_in(proposal.quote,
                        question if typed is None else typed, history)
    if speaker is not None:
        return Bridge(broader_question=proposal.broader, link=proposal.link,
                      source="conversation", quote=proposal.quote,
                      speaker=speaker, subject=proposal.subject,
                      group=proposal.group), outcome

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
                  link_results=linked.results[:LINK_RESULTS],
                  subject=proposal.subject, group=proposal.group), outcome
