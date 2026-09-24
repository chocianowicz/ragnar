from core.models import Chunk, SearchResult
from eval.check_golden import check, parse_verdict


def test_verdict_json_is_found_inside_fences_and_prose():
    text = 'Sure.\n```json\n{"verdict": "ok", "note": "fine"}\n```'
    assert parse_verdict(text)["verdict"] == "ok"
    assert parse_verdict("no json here") == {}


class FakeSearch:
    def __init__(self, filenames):
        self.filenames, self.calls = filenames, []

    def find(self, question, doc_ids=None, score_floor=None):
        self.calls.append(doc_ids)
        results = [SearchResult(chunk=Chunk(doc_id="d", filename=f, text="t",
                                            chunk_index=0), score=1.0)
                   for f in self.filenames]
        return type("Outcome", (), {"results": results})()


class FakeJudge:
    def __init__(self, model):
        self.model, self.prompts = model, []

    def generate(self, system, user):
        self.prompts.append(user)
        return '{"verdict": "ok"}'


def run(entry, filenames, private=frozenset()):
    cloud, local = FakeJudge("cloud"), FakeJudge("local")
    search = FakeSearch(filenames)
    result = check(entry, search, {"a.pdf": {"id-a"}, "tax.pdf": {"id-t"}},
                   set(private), cloud, local)
    return result, search, cloud, local


def test_in_corpus_evidence_is_scoped_to_the_expected_document():
    entry = {"question": "q", "expected_answer": "x",
             "expected_sources": ["a.pdf"]}
    result, search, cloud, local = run(entry, ["a.pdf"])
    assert search.calls == [["id-a"]]
    assert result["judge"] == "cloud" and not local.prompts


def test_private_entries_are_judged_locally():
    entry = {"question": "q", "expected_answer": "x",
             "expected_sources": ["tax.pdf"], "private": True}
    result, _, cloud, _ = run(entry, ["tax.pdf"], {"tax.pdf"})
    assert result["judge"] == "local" and not cloud.prompts


def test_an_out_of_corpus_probe_that_retrieves_private_text_stays_local():
    entry = {"question": "q", "expected_sources": [], "out_of_corpus": True}
    result, search, cloud, _ = run(entry, ["a.pdf", "tax.pdf, p. 1"],
                                   {"tax.pdf"})
    assert search.calls == [None]
    assert result["judge"] == "local" and not cloud.prompts


def test_an_unindexed_source_is_reported_not_judged():
    entry = {"question": "q", "expected_answer": "x",
             "expected_sources": ["missing.pdf"]}
    result, _, cloud, local = run(entry, [])
    assert result["verdict"] == "source_not_indexed"
    assert not cloud.prompts and not local.prompts
