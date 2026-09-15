import httpx


class OllamaEmbedder:
    """Embeds text via Ollama's /api/embed endpoint.

    The client is injectable so unit tests can stub HTTP without a server.
    """

    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 120.0, batch_size: int = 64,
                 keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Ingestion embeds a whole document at once, and a 200-page PDF is
        # on the order of a thousand chunks. Sending those as one request
        # puts the entire document behind a single timeout: it either all
        # arrives or the whole parse is wasted. Batching bounds each
        # request instead, and a query (one text) still costs one call.
        self.batch_size = batch_size
        # Same reason as OllamaLLM.keep_alive: the embedding model is
        # needed for every question and must not be evicted between them.
        self.keep_alive = keep_alive
        self._client = client or httpx.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch,
                      "keep_alive": self.keep_alive},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])

        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding count mismatch: asked for {len(texts)} vectors, "
                f"got {len(vectors)}"
            )
        return vectors
