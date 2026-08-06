import httpx


class OllamaEmbedder:
    """Embeds text via Ollama's /api/embed endpoint.

    The client is injectable so unit tests can stub HTTP without a server.
    """

    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 120.0, keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Keeps bge-m3 resident between turns instead of relying on
        # Ollama's 5-minute default TTL — see OllamaLLM.keep_alive.
        self.keep_alive = keep_alive
        self._client = client or httpx.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts,
                  "keep_alive": self.keep_alive},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
