import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path) as fh:
            self._raw = yaml.safe_load(fh)

        self.ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.qdrant_url = os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )

    @property
    def llm_model(self) -> str:
        return self._raw["models"]["llm"]

    @property
    def embedding_model(self) -> str:
        return self._raw["models"]["embedding"]

    @property
    def embedding_dim(self) -> int:
        return self._raw["models"]["embedding_dim"]

    @property
    def reranker_model(self) -> str:
        return self._raw["models"]["reranker"]

    @property
    def collection(self) -> str:
        return self._raw["storage"]["collection"]

    @property
    def data_dir(self) -> Path:
        return Path(self._raw["storage"]["data_dir"])

    @property
    def candidates(self) -> int:
        return self._raw["retrieval"]["candidates"]

    @property
    def top_k(self) -> int:
        return self._raw["retrieval"]["top_k"]

    @property
    def score_floor(self) -> float:
        return self._raw["retrieval"]["score_floor"]

    @property
    def vector_floor(self) -> float:
        return self._raw["retrieval"]["vector_floor"]

    @property
    def chunking(self) -> dict:
        return self._raw["chunking"]
