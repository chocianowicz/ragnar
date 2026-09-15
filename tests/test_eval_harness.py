import pytest

from core.models import Chunk, SearchResult
from eval.run_eval import parse_floors, check_corpus, load_golden


class FakeStore:
    """Stands in for QdrantStore during the pre-run corpus check."""

    def __init__(self, filenames, dim=1024, collection="documents"):
        self.dim = dim
        self.collection = collection
        self._filenames = filenames

    def search(self, vector, limit, doc_ids=None):
        return [
            SearchResult(
                chunk=Chunk(doc_id="d", filename=name, text="", chunk_index=0),
                score=1.0,
            )
            for name in self._filenames
        ]


def test_floor_sweep_includes_the_shipped_value():
    """The old sweep stepped by 0.1 and so could never produce the 0.55 the
    config actually ships - a grid that cannot contain the answer."""
    assert 0.55 in parse_floors("0.40:0.80:0.05")


def test_floor_sweep_is_inclusive_of_the_stop_value():
    assert parse_floors("0.5:0.7:0.1") == [0.5, 0.6, 0.7]


@pytest.mark.parametrize("spec", ["0.1:0.2", "a:b:c", "0.5:0.4:0.1",
                                  "0.1:0.2:0", ""])
def test_malformed_floor_spec_is_rejected(spec):
    with pytest.raises(SystemExit):
        parse_floors(spec)


def test_corpus_check_passes_when_expected_sources_are_indexed():
    golden = [{"question": "q", "expected_sources": ["a.pdf"],
               "out_of_corpus": False}]

    check_corpus(FakeStore(["a.pdf", "b.pdf"]), golden)


def test_corpus_check_fails_when_expected_sources_are_missing():
    """Without this, every in-corpus case refuses, the run reports a
    plausible refusal_accuracy, and the number means nothing."""
    golden = [{"question": "q", "expected_sources": ["missing.pdf"],
               "out_of_corpus": False}]

    with pytest.raises(SystemExit, match="missing.pdf"):
        check_corpus(FakeStore(["other.pdf"]), golden)


def test_corpus_check_ignores_out_of_corpus_entries():
    """Out-of-corpus questions are supposed to have no source."""
    golden = [{"question": "q", "expected_sources": [], "out_of_corpus": True}]

    check_corpus(FakeStore([]), golden)


def test_empty_golden_set_is_rejected(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="nothing to evaluate"):
        load_golden(path)


def test_golden_set_round_trips_non_ascii_questions(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text(
        '- question: "Jaka jest najwyższa łączna kwota?"\n'
        '  expected_sources: []\n'
        '  out_of_corpus: true\n',
        encoding="utf-8",
    )

    assert load_golden(path)[0]["question"].endswith("kwota?")
