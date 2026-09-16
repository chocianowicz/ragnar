import pytest
import yaml

from core.config import Config, ConfigError

VALID = {
    "models": {
        "llm": "qwen2.5:14b",
        "embedding": "bge-m3",
        "embedding_dim": 1024,
        "reranker": "BAAI/bge-reranker-v2-m3",
    },
    "chunking": {"strategy": "structural", "target_tokens": 500},
    "retrieval": {"candidates": 25, "top_k": 5, "score_floor": 0.55},
    "storage": {"data_dir": "/app/data", "collection": "documents"},
}


def _write(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    cfg = Config(_write(tmp_path, VALID))

    assert cfg.llm_model == "qwen2.5:14b"
    assert cfg.top_k == 5
    assert cfg.score_floor == 0.55


def test_the_shipped_config_is_valid():
    """Guards against the validator and config.yaml drifting apart."""
    assert Config("config.yaml").embedding_dim == 1024


def test_missing_key_is_reported_by_name(tmp_path):
    raw = {**VALID, "retrieval": {"candidates": 25, "top_k": 5}}

    with pytest.raises(ConfigError, match=r"retrieval\.score_floor: missing"):
        Config(_write(tmp_path, raw))


def test_missing_section_is_reported(tmp_path):
    raw = {k: v for k, v in VALID.items() if k != "storage"}

    with pytest.raises(ConfigError, match="storage: missing section"):
        Config(_write(tmp_path, raw))


def test_every_problem_is_reported_at_once(tmp_path):
    """One startup failure listing everything beats fixing typos one run
    at a time."""
    raw = {**VALID, "retrieval": {"candidates": 25}}

    with pytest.raises(ConfigError) as exc:
        Config(_write(tmp_path, raw))

    assert "retrieval.top_k: missing" in str(exc.value)
    assert "retrieval.score_floor: missing" in str(exc.value)


def test_numeric_string_is_coerced_not_rejected(tmp_path):
    """`score_floor: "0.55"` is valid YAML and used to compare wrongly
    against a float for the whole run."""
    raw = {**VALID, "retrieval": {**VALID["retrieval"], "score_floor": "0.55"}}

    cfg = Config(_write(tmp_path, raw))

    assert cfg.score_floor == 0.55
    assert isinstance(cfg.score_floor, float)


def test_uncoercible_value_is_rejected(tmp_path):
    raw = {**VALID, "retrieval": {**VALID["retrieval"], "top_k": "five"}}

    with pytest.raises(ConfigError, match=r"retrieval\.top_k: expected int"):
        Config(_write(tmp_path, raw))


def test_boolean_is_not_silently_read_as_a_number(tmp_path):
    raw = {**VALID, "retrieval": {**VALID["retrieval"], "top_k": True}}

    with pytest.raises(ConfigError, match=r"retrieval\.top_k"):
        Config(_write(tmp_path, raw))


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty or not a mapping"):
        Config(path)


def test_batch_sizes_fall_back_to_defaults_when_absent(tmp_path):
    cfg = Config(_write(tmp_path, VALID))

    assert cfg.embedding_batch == 64
    assert cfg.upsert_batch == 128


def test_keep_alive_defaults_and_is_settable(tmp_path):
    """Holding a 14 GB model through idle time is a trade the deployment
    should be able to make, not one baked into the code."""
    assert Config(_write(tmp_path, VALID)).keep_alive == "10m"

    raw = {**VALID, "models": {**VALID["models"], "keep_alive": "2m"}}
    assert Config(_write(tmp_path, raw)).keep_alive == "2m"
