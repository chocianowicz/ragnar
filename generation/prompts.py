SYSTEM_PROMPT = """\
You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- Be concise and factual. Do not speculate or embellish.
- Do not write citations yourself; they are attached automatically.
"""


def build_user_prompt(question: str, excerpts: list[tuple[str, str]]) -> str:
    """excerpts: list of (source_label, text)."""
    blocks = "\n\n".join(
        f"[{label}]\n{text}" for label, text in excerpts
    )
    return f"Excerpts:\n\n{blocks}\n\nQuestion: {question}"
