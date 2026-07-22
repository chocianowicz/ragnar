import pytest
from retrieval.embedder import OllamaEmbedder


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


def test_embedder_sends_all_texts_in_one_batch():
    client = StubClient({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls[0]["input"] == ["a", "b"]
    assert len(client.calls) == 1


def test_embedder_returns_empty_for_no_input():
    client = StubClient({"embeddings": []})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)
    assert embedder.embed([]) == []


@pytest.mark.integration
def test_embedder_against_real_ollama_returns_1024_dims():
    import os
    embedder = OllamaEmbedder(os.environ["OLLAMA_BASE_URL"], "bge-m3")
    vectors = embedder.embed(["kontrakt serwisowy", "service contract"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
