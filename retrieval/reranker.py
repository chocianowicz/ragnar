import math

from core.models import SearchResult

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


class BGEReranker:
    """Cross-encoder reranker running on CPU inside the container.

    If it proves too slow, the ONNX export of the same model is a drop-in
    replacement that removes the torch dependency entirely.
    """

    def __init__(self, model_name: str = MODEL_NAME, model=None,
                 max_length: int = MAX_LENGTH):
        self._model_name = model_name
        self._model = model
        self._max_length = max_length

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name,
                                       max_length=self._max_length)
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult],
               top_k: int) -> list[SearchResult]:
        if not candidates:
            return []

        model = self._ensure_model()
        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores = model.predict(pairs)

        # The cross-encoder emits logits, not probabilities. Sigmoid maps
        # them to (0, 1) so a single interpretable floor can be configured.
        rescored = [
            SearchResult(chunk=c.chunk, score=1 / (1 + math.exp(-float(s))))
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
