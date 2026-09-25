import logging
import os

import httpx

from core.models import SearchResult

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# bge-reranker-v2-m3 accepts 8192 tokens, and CrossEncoder defaults to that
# maximum unless told otherwise — so every pair was being scored at 8192
# even though chunking targets 500 tokens. On the real corpus that cost
# ~4.5x: 25 candidates took ~130s at 8192 and ~30s at 512, on the same
# chunks, on CPU. 512 covers a target-size chunk with room to spare.
#
# It does not cover an oversized one. Table chunks are grouped by row count
# with no size budget (ingestion/tables.py), so a wide table can produce
# chunks several times the prose target — those get truncated here. That is
# a chunking problem, not a reranking one, and truncating is the right
# behaviour until it is fixed.
MAX_LENGTH = 512

# host_server.py on the Mac, where the cross-encoder runs on the GPU. Docker
# on macOS cannot reach Metal: measured on an M5 Pro, 30 candidates took
# 10.34s in the container and 1.48s on the host GPU. Unset keeps scoring
# in-process.
RERANKER_URL = os.environ.get("RERANKER_URL", "").rstrip("/")
RERANK_TIMEOUT = 180.0


class BGEReranker:
    """Cross-encoder reranker.

    With a url (RERANKER_URL), the model call goes to host_server.py and
    runs on the host's GPU. Only the model call moves: sorting and the
    top_k cut below run here on both paths, so they cannot drift apart. If
    the host service is unreachable, scoring falls back to the in-process
    model with a warning. That's slower, but the same scores to ~1e-6.
    """

    def __init__(self, model_name: str = MODEL_NAME, model=None,
                 max_length: int = MAX_LENGTH, url: str | None = None,
                 client: httpx.Client | None = None):
        self._model_name = model_name
        self._model = model
        self._max_length = max_length
        self._url = (RERANKER_URL if url is None else url).rstrip("/")
        self._client = client or httpx.Client()

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name,
                                       max_length=self._max_length)
        return self._model

    def _remote_scores(self, query: str, texts: list[str]) -> list[float] | None:
        if not self._url:
            return None
        try:
            response = self._client.post(
                f"{self._url}/rerank", timeout=RERANK_TIMEOUT,
                json={"query": query, "texts": texts,
                      "max_length": self._max_length})
            response.raise_for_status()
            scores = response.json()["scores"]
            # Validated here, not at the point of use: a wrong-shaped payload
            # has to be a fallback, not an exception. A string is the case
            # that matters — iterating it would yield characters, and every
            # candidate would score identically and silently.
            if not isinstance(scores, list) or len(scores) != len(texts):
                raise ValueError(
                    f"expected {len(texts)} scores, got "
                    f"{type(scores).__name__} of "
                    f"{len(scores) if hasattr(scores, '__len__') else '?'}")
            return [float(s) for s in scores]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            log.warning("host reranker at %s failed (%s); scoring in-process",
                        self._url, exc)
            return None

    def rerank(self, query: str, candidates: list[SearchResult],
               top_k: int) -> list[SearchResult]:
        if not candidates:
            return []

        texts = [c.chunk.text for c in candidates]
        raw_scores = self._remote_scores(query, texts)
        if raw_scores is None:
            raw_scores = self._ensure_model().predict(
                [(query, t) for t in texts])

        # Already probabilities in (0, 1): CrossEncoder.predict applies a
        # sigmoid itself for a one-label model like this one. Applying a
        # second one here squeezed every score into 0.50-0.73 until
        # 2026-09-25, and the floors were set on that scale.
        rescored = [
            SearchResult(chunk=c.chunk, score=float(s))
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
