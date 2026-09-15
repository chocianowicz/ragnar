import os
from pathlib import Path

import yaml


# (section, key, type) for every setting that has no default. Checked once
# at startup: every one of these used to be a bare lookup evaluated lazily,
# so a typo in config.yaml surfaced as a KeyError inside a chat turn,
# rendered as a Streamlit traceback, rather than as the configuration
# problem it is. Types are coerced too - `score_floor: "0.55"` is valid
# YAML and compares wrongly against a float for the whole run.
REQUIRED = [
    ("models", "llm", str),
    ("models", "embedding", str),
    ("models", "embedding_dim", int),
    ("models", "reranker", str),
    ("chunking", "strategy", str),
    ("retrieval", "candidates", int),
    ("retrieval", "top_k", int),
    ("retrieval", "score_floor", float),
    ("storage", "data_dir", str),
    ("storage", "collection", str),
]


class ConfigError(ValueError):
    """config.yaml is missing something, or has it in the wrong shape."""


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path, encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh)

        if not isinstance(self._raw, dict):
            raise ConfigError(f"{path} is empty or not a mapping")
        self._validate(path)

        self.ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.qdrant_url = os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )

    def _validate(self, path: str | Path) -> None:
        problems: list[str] = []
        for section, key, kind in REQUIRED:
            block = self._raw.get(section)
            if not isinstance(block, dict):
                problems.append(f"{section}: missing section")
                continue
            if key not in block:
                problems.append(f"{section}.{key}: missing")
                continue
            try:
                # bool is an int subclass; nothing here wants one, and
                # `top_k: true` should be an error rather than 1.
                if isinstance(block[key], bool):
                    raise TypeError
                block[key] = kind(block[key])
            except (TypeError, ValueError):
                problems.append(
                    f"{section}.{key}: expected {kind.__name__}, "
                    f"got {block[key]!r}"
                )
        if problems:
            raise ConfigError(
                f"{path} is not usable:\n  " + "\n  ".join(problems)
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
    def agentic(self) -> dict:
        """Optional extra retrieval stages. Every one defaults to off: each
        costs LLM calls on a question that is already slow, and none is
        validated against a golden set yet."""
        return self._raw.get("agentic") or {}

    @property
    def hybrid(self) -> bool:
        """Whether new collections pair the dense vector with a lexical one."""
        return bool(self._raw["retrieval"].get("hybrid", True))

    @property
    def reranker_max_length(self) -> int:
        """Token window per rerank pair. See retrieval/reranker.py."""
        return self._raw["models"].get("reranker_max_length", 512)

    @property
    def embedding_batch(self) -> int:
        """Texts per /api/embed request during ingestion."""
        return self._raw["models"].get("embedding_batch", 64)

    @property
    def upsert_batch(self) -> int:
        """Points per Qdrant upsert request during ingestion."""
        return self._raw["storage"].get("upsert_batch", 128)

    @property
    def chunking(self) -> dict:
        return self._raw["chunking"]
