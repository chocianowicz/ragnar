import pytest
from generation.llm import OllamaLLM


@pytest.mark.integration
def test_llm_answers_from_context_only():
    import os
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        "Answer only from the excerpt. If absent, say you don't know.",
        "Excerpt: The contract number is SC-4471.\n\n"
        "Question: What is the contract number?",
    )

    assert "SC-4471" in reply


@pytest.mark.integration
def test_llm_answers_in_the_language_of_the_question():
    import os
    from generation.prompts import SYSTEM_PROMPT
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        SYSTEM_PROMPT,
        "Excerpts:\n\n[a.pdf, p. 1]\nThe contract number is SC-4471.\n\n"
        "Question: Jaki jest numer kontraktu?",
    )

    assert "SC-4471" in reply
    # crude Polish-output check: at least one Polish-specific character
    # or common Polish word
    assert any(t in reply.lower() for t in
               ["numer", "kontrakt", "wynosi", "to "])
