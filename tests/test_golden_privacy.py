"""Private golden-set cases must never land in a tracked file.

The repo is public. Cases about personal documents belong in the
gitignored eval/<name>.private.yaml, which load_golden() appends.
"""
from pathlib import Path

import yaml

from eval.run_eval import load_golden, private_companion

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"


def _tracked_golden_files():
    return [p for p in sorted(EVAL.glob("*.yaml"))
            if not p.name.endswith(".private.yaml")]


def test_no_tracked_golden_file_holds_a_private_case():
    leaks = []
    for path in _tracked_golden_files():
        for entry in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            if isinstance(entry, dict) and entry.get("private"):
                leaks.append(f"{path.name}: {entry.get('question', entry)}")
    assert not leaks, (
        "private cases belong in eval/*.private.yaml, not a tracked file:\n"
        + "\n".join(leaks))


def test_private_companions_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "eval/*.private.yaml" in ignore


def test_load_golden_appends_the_private_companion(tmp_path):
    public = tmp_path / "golden.yaml"
    public.write_text('- question: "public"\n', encoding="utf-8")
    private_companion(public).write_text(
        '- question: "secret"\n  private: true\n', encoding="utf-8")

    assert [e["question"] for e in load_golden(public)] == ["public", "secret"]


def test_load_golden_without_a_companion_is_just_the_public_set(tmp_path):
    public = tmp_path / "golden.yaml"
    public.write_text('- question: "public"\n', encoding="utf-8")

    assert [e["question"] for e in load_golden(public)] == ["public"]
