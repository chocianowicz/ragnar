"""Prompts for the agentic retrieval stages.

Each returns (system, user). Outputs are deliberately terse and tagged, so
parsing does not depend on the model being chatty in a particular way.
"""

REWRITE_SYSTEM = """\
You rewrite a user's question into a better search query for a document
retrieval system.

Rules:
- Keep every proper noun, number, code and identifier exactly as written.
  They are what the lexical search matches on; changing them loses the
  document.
- Expand abbreviations and resolve vague references where you can.
- Preserve the original language.
- Output only the rewritten query. No explanation, no quotes.
"""

MULTI_QUERY_SYSTEM = """\
You generate alternative phrasings of a question, to widen a document
search that may miss the answer on wording alone.

Rules:
- Vary the wording, not the meaning.
- Keep every identifier, code and number exactly as written.
- Preserve the original language.
- Output one query per line. No numbering, no bullets, no other text.
"""

HOP_SYSTEM = """\
You judge whether retrieved excerpts are enough to answer a question, and
if not, what single follow-up search would help.

Answer in exactly this form, two lines:
Sufficient: yes|no
FollowUp: <one search query, or none>

Say "yes" whenever the excerpts contain the answer, even partially. Only
ask for a follow-up when something specific and nameable is missing.
"""

CORRECTION_SYSTEM = """\
You review a draft answer against the excerpts it was built from.

Answer in exactly this form, two lines:
Complete: yes|no
Improvement: <one search query that would fill the gap, or none>

Judge only whether the excerpts support the answer. Do not add knowledge
of your own. Say "yes" unless something the question asked for is
genuinely absent.
"""


def build_rewrite_prompt(question: str) -> tuple[str, str]:
    return REWRITE_SYSTEM, f"Question: {question}"


def build_multi_query_prompt(question: str, count: int) -> tuple[str, str]:
    return (MULTI_QUERY_SYSTEM,
            f"Generate {count} alternative phrasings.\n\nQuestion: {question}")


def _excerpt_block(excerpts: list[tuple[str, str]], limit: int = 600) -> str:
    return "\n\n".join(f"[{label}]\n{text[:limit]}" for label, text in excerpts)


def build_hop_prompt(question: str,
                     excerpts: list[tuple[str, str]]) -> tuple[str, str]:
    return (HOP_SYSTEM,
            f"Question: {question}\n\nExcerpts:\n\n{_excerpt_block(excerpts)}")


def build_correction_prompt(question: str, excerpts: list[tuple[str, str]],
                            draft: str) -> tuple[str, str]:
    return (CORRECTION_SYSTEM,
            f"Question: {question}\n\nExcerpts:\n\n"
            f"{_excerpt_block(excerpts)}\n\nDraft answer:\n{draft}")


def parse_tagged(text: str, tag: str, default: str = "") -> str:
    """Read `Tag: value` from a line, case-insensitively."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith(f"{tag.lower()}:"):
            return line.split(":", 1)[1].strip()
    return default
