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
- The conversation so far is only for understanding what the question \
refers to. Never take a fact from it, including from your own earlier \
answers: every fact in the answer must be in the excerpts.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- Reply with a complete sentence. Be concise and factual; do not speculate \
or embellish.
- Never write citations or source references yourself. They are added \
separately, from the excerpts actually used.
- If, and only if, the excerpts do not contain the answer, reply with \
exactly NO_ANSWER_IN_EXCERPTS and no other text. Do not guess.
"""


INDIRECT_SYSTEM = """\
You answer a question the document excerpts do not address directly,
through a broader subject they do cover. You are given the question, a
link that connects its subject to the broader subject, and excerpts about
the broader subject.

Rules:
- Use ONLY the excerpts and the link. Never use outside knowledge.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- First say plainly that the documents do not state this for the subject \
of the question specifically. Then give what the excerpts say about the \
broader subject, then say how the link makes it apply.
- Never claim or imply the documents mention the original subject.
- Be concise and factual; do not speculate or embellish.
- Never write citations or source references yourself.
- If, and only if, the excerpts do not answer the question for the broader \
subject either, reply with exactly NO_ANSWER_IN_EXCERPTS and no other text.
"""

LINK_SOURCES = {
    "conversation": "stated earlier in this conversation",
    "documents": "stated in the documents",
}


def build_indirect_prompt(question: str, broader_question: str, link: str,
                          link_source: str,
                          excerpts: list[tuple[str, str]]) -> str:
    """excerpts: list of (source_label, text), link passages included."""
    blocks = "\n\n".join(
        f"[{label}]\n{text}" for label, text in excerpts
    )
    return (f"Link ({LINK_SOURCES.get(link_source, link_source)}): {link}\n"
            f"Broader question searched: {broader_question}\n\n"
            f"Excerpts:\n\n{blocks}\n\nQuestion: {question}")


def build_user_prompt(question: str, excerpts: list[tuple[str, str]]) -> str:
    """excerpts: list of (source_label, text)."""
    blocks = "\n\n".join(
        f"[{label}]\n{text}" for label, text in excerpts
    )
    return f"Excerpts:\n\n{blocks}\n\nQuestion: {question}"
