import pytest
from generation.llm import OllamaLLM


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class StubClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append(json)
        return StubResponse(self._payload)


def test_temperature_defaults_to_zero():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client)

    llm.generate("sys", "user")

    assert client.calls[0]["options"]["temperature"] == 0.0


def test_temperature_is_settable_and_sent_in_request():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "m", client=client)

    llm.temperature = 0.7
    llm.generate("sys", "user")

    assert client.calls[0]["options"]["temperature"] == 0.7


def test_per_call_model_and_temperature_override_instance_defaults():
    client = StubClient({"message": {"content": "ok"}})
    llm = OllamaLLM("http://x", "default-model", client=client)

    llm.generate("sys", "user", model="other-model", temperature=0.9)

    assert client.calls[0]["model"] == "other-model"
    assert client.calls[0]["options"]["temperature"] == 0.9
    # Instance defaults are untouched — nothing was mutated.
    assert llm.model == "default-model"
    assert llm.temperature == 0.0


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
