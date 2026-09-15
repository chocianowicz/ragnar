"""Turning a follow-up into a question that can be searched on its own.

Giving the model the conversation fixes half the problem — it can then
tell what "that" refers to. It does not fix the other half: retrieval runs
before the model, on the text of the question, and "And what is the base
year for that?" is a poor search query no matter how much context the
model gets afterwards. On the real corpus that exact follow-up, asked
after a question about Norway, retrieved base years from Iceland, Georgia
and Saudi Arabia, because nothing in the query said Norway.

So the question is resolved against the conversation first, and the
standalone form is what gets searched. The user still sees, and the model
still answers, what they actually typed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RESOLVE_SYSTEM = """\
You rewrite a follow-up question so it can be understood on its own,
without the conversation.

Rules:
- Replace pronouns and references ("that", "it", "there", "the same") with
  what they refer to, taken from the conversation.
- Change nothing else. Keep the wording, the language, and every name,
  number and code exactly as written.
- If the question already stands on its own, repeat it back unchanged.
- Output only the question. No explanation, no quotes.
"""

# How much conversation the resolver sees. It needs the referent, which is
# almost always in the previous exchange; more than that is prompt weight
# for no gain, and makes it likelier to latch onto a stale topic.
CONTEXT_TURNS = 4


def _transcript(history: list[dict]) -> str:
    lines = []
    for message in history[-CONTEXT_TURNS:]:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        who = "User" if role == "user" else "Assistant"
        # The assistant's own answers can be long; the referent is
        # invariably near the start.
        lines.append(f"{who}: {str(message.get('content', ''))[:500]}")
    return "\n".join(lines)


def resolve(llm, question: str, history: list[dict] | None,
            model: str | None = None) -> tuple[str, bool]:
    """(question to search, whether it was rewritten).

    Falls back to the question as typed whenever anything is missing or
    goes wrong: a failed resolution should cost the follow-up its context,
    not cost the user their answer.
    """
    transcript = _transcript(history or [])
    if not transcript:
        return question, False

    try:
        resolved = llm.generate(
            RESOLVE_SYSTEM,
            f"Conversation:\n{transcript}\n\nFollow-up question: {question}",
            model=model,
        ).strip()
    except Exception as exc:
        logger.warning("follow-up resolution failed: %s", exc)
        return question, False

    # A resolution should expand a question, not replace it. Something far
    # shorter than what was asked has lost it rather than clarified it.
    if not resolved or len(resolved) < max(8, len(question) // 2):
        return question, False
    if resolved.strip().lower() == question.strip().lower():
        return question, False
    return resolved, True
