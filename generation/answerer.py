from dataclasses import dataclass, field

from core.models import SearchResult
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

NO_RESULTS_MESSAGE = (
    "I could not find anything relevant in the indexed documents."
)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False


class Answerer:
    def __init__(self, llm):
        self._llm = llm

    def answer(self, question: str,
               results: list[SearchResult]) -> Answer:
        if not results:
            # Skip the model entirely — a refusal it cannot embellish.
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        excerpts = [(r.chunk.citation_label(), r.chunk.text) for r in results]
        text = self._llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, excerpts)
        )

        # Citations come from retrieved metadata, not model prose.
        seen: list[str] = []
        for result in results:
            label = result.chunk.citation_label()
            if label not in seen:
                seen.append(label)

        return Answer(text=text, citations=seen)
