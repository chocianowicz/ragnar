# The model says this, exactly, when the excerpts do not answer the
# question. A fixed token rather than free prose because the app has to act
# on it — sources are hidden when the answer is not built from them — and
# "say so plainly" produces different wording every time, in whichever
# language the question was asked. Matching that by heuristic would be
# guesswork; matching one token is not.
NO_ANSWER = "NO_ANSWER_IN_EXCERPTS"

SYSTEM_PROMPT = """\
You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- Reply with a complete sentence. Be concise and factual; do not speculate \
or embellish.
- Never write citations or source references yourself. They are added \
separately, from the excerpts actually used.
- If, and only if, the excerpts do not contain the answer, reply with \
exactly NO_ANSWER_IN_EXCERPTS and no other text. Do not guess.
"""


def build_user_prompt(question: str, excerpts: list[tuple[str, str]]) -> str:
    """excerpts: list of (source_label, text)."""
    blocks = "\n\n".join(
        f"[{label}]\n{text}" for label, text in excerpts
    )
    return f"Excerpts:\n\n{blocks}\n\nQuestion: {question}"
