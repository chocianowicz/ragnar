import json

import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 300.0, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._client = client or httpx.Client()

    def _payload(self, system: str, user: str, stream: bool,
                 model: str | None, temperature: float | None) -> dict:
        # model/temperature default to the instance values but can be
        # overridden per call, so a shared LLM instance stays safe when
        # different requests want different settings.
        return {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "options": {
                "temperature": self.temperature if temperature is None
                else temperature,
            },
        }

    def generate(self, system: str, user: str, *, model: str | None = None,
                 temperature: float | None = None) -> str:
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=self._payload(system, user, False, model, temperature),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, system: str, user: str, *, model: str | None = None,
               temperature: float | None = None):
        """Yields token deltas. Used by the UI."""
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(system, user, True, model, temperature),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("done"):
                    break
                yield payload["message"]["content"]
