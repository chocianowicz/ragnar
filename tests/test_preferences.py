import json

from ui.preferences import Preferences


def test_an_absent_file_falls_back_to_the_config_default(tmp_path):
    prefs = Preferences(tmp_path / "settings.json")

    assert prefs.get("candidates", 25) == 25


def test_a_stored_value_overrides_the_default(tmp_path):
    path = tmp_path / "settings.json"
    Preferences(path).set("candidates", 50)

    assert Preferences(path).get("candidates", 25) == 50


def test_settings_survive_a_restart(tmp_path):
    """The point of the file: an operator tunes a running instance once,
    not on every browser session."""
    path = tmp_path / "settings.json"
    first = Preferences(path)
    first.set("candidates", 40)
    first.set("use_reranker", False)

    reopened = Preferences(path)

    assert reopened.get("candidates", 25) == 40
    assert reopened.get("use_reranker", True) is False


def test_a_corrupt_file_does_not_stop_the_app(tmp_path):
    """config.yaml on its own is a complete configuration; a bad
    preferences file must degrade to it rather than take the app down."""
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert Preferences(path).get("candidates", 25) == 25


def test_a_file_that_is_not_an_object_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert Preferences(path).get("candidates", 25) == 25


def test_a_hand_edited_string_is_coerced_to_the_default_s_type(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"candidates": "40"}), encoding="utf-8")

    assert Preferences(path).get("candidates", 25) == 40


def test_an_uncoercible_value_falls_back_rather_than_breaking(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"candidates": "lots"}), encoding="utf-8")

    assert Preferences(path).get("candidates", 25) == 25


def test_writing_an_unchanged_value_does_not_rewrite_the_file(tmp_path):
    path = tmp_path / "settings.json"
    prefs = Preferences(path)
    prefs.set("candidates", 30)
    before = path.stat().st_mtime_ns

    prefs.set("candidates", 30)

    assert path.stat().st_mtime_ns == before
