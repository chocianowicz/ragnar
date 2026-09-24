import json
import math

import pytest

pytest.importorskip("ragas")

from eval import run_eval
from eval.metrics import (aggregation_guard, citation_precision,
                          false_refusal_rate, missed_refusal_rate,
                          multi_hop_citation_accuracy,
                          multi_hop_retrieval_recall)
from eval.run_ragas import summarize, tasks
from eval.stats import metric_ci, paired_diff, read_jsonl, row_key


def case(question="q", *, refused=False, answer="a", contexts=("c",),
         sources=("a.pdf",), context_sources=("a.pdf, p. 1",),
         out_of_corpus=False, **extra):
    return {"question": question, "expected_answer": "x",
            "expected_sources": list(sources), "out_of_corpus": out_of_corpus,
            "refused": refused, "answer": "" if refused else answer,
            "citations": [] if refused else list(context_sources),
            "contexts": list(contexts),
            "context_sources": list(context_sources), **extra}


# --- deterministic metrics -------------------------------------------------

def test_refusal_split_separates_the_two_failures():
    cases = [case("in", refused=True),                       # false refusal
             case("in2"),
             case("out", out_of_corpus=True, sources=()),        # missed
             case("out2", out_of_corpus=True, sources=(), refused=True)]
    assert false_refusal_rate(cases) == 0.5
    assert missed_refusal_rate(cases) == 0.5


def test_citation_precision_penalises_citing_everything():
    c = case(context_sources=("a.pdf, p. 1", "b.pdf", "c.pdf", "d.pdf"))
    assert citation_precision([c]) == 0.25


def test_multi_hop_needs_every_source_cited_and_retrieved():
    both = case(sources=("Table T", "Passage P"), multihop=True,
                context_sources=("Table T", "Passage P"))
    one = case(sources=("Table T", "Passage P"), multihop=True,
               context_sources=("Table T",))
    assert multi_hop_citation_accuracy([both, one]) == 0.5
    assert multi_hop_retrieval_recall([both, one]) == 0.5


def test_multi_hop_accepts_a_sub_page_of_the_source():
    c = case(sources=("Alpine skiing",), multihop=True,
             context_sources=("Alpine skiing – Men's super-G",))
    assert multi_hop_retrieval_recall([c]) == 1.0


def test_aggregation_guard_counts_catches_and_false_positives():
    cases = [case(aggregation="true", mode="aggregation_refused"),
             case(aggregation="true", mode="answer"),
             case(aggregation="false", mode="aggregation_refused")]
    assert aggregation_guard(cases) == {"caught": "1/2",
                                        "false_positive": "1/1"}


# --- error bars -------------------------------------------------------------

def mean(values):
    return sum(values) / len(values)


def test_bootstrap_range_brackets_the_value_and_is_reproducible():
    values = [0, 1] * 50
    ci = metric_ci(mean, values)
    assert ci["lo"] < ci["value"] == 0.5 < ci["hi"]
    assert metric_ci(mean, values) == ci


def test_identical_runs_differ_by_nothing():
    rows = {str(i): {"s": i % 2} for i in range(40)}
    d = paired_diff(lambda rs: mean([r["s"] for r in rs]), rows, rows)
    assert d["value"] == 0 and not d["significant"]


def test_a_consistent_gain_is_significant():
    a = {str(i): {"s": 0} for i in range(40)}
    b = {str(i): {"s": 1} for i in range(40)}
    d = paired_diff(lambda rs: mean([r["s"] for r in rs]), a, b)
    assert d["value"] == 1 and d["significant"]


# --- crash safety -----------------------------------------------------------

def test_a_torn_last_line_is_dropped_not_fatal(tmp_path):
    sink = tmp_path / "a.jsonl"
    sink.write_text('{"_provenance": {"config": 1}}\n{"question": "q1"}\n{"quest')
    prov, rows = read_jsonl(sink)
    assert prov == {"config": 1} and rows == [{"question": "q1"}]


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """run_cases with every model and store stubbed, and a controllable
    run_one: raise `crash_at` to simulate a crash on that question."""
    golden = tmp_path / "g.yaml"
    golden.write_text("\n".join(
        f"- question: q{i}\n  expected_sources: [a.pdf]\n"
        f"  out_of_corpus: false" for i in range(5)))
    for name in ("OllamaEmbedder", "QdrantStore", "BGEReranker", "Search",
                 "OllamaLLM", "Answerer"):
        monkeypatch.setattr(run_eval, name, lambda *a, **k: object())
    monkeypatch.setattr(run_eval, "check_corpus", lambda *a: None)
    state = {"crash_at": None, "ran": []}

    def run_one(entry, *a):
        if entry["question"] == state["crash_at"]:
            raise KeyboardInterrupt
        state["ran"].append(entry["question"])
        return {**entry, "latency_s": 0.0}

    monkeypatch.setattr(run_eval, "run_one", run_one)
    return golden, state


def test_a_crashed_run_resumes_without_repeating_cases(fake_pipeline,
                                                        tmp_path):
    golden, state = fake_pipeline
    sink = tmp_path / "answers.jsonl"

    state["crash_at"] = "q3"
    with pytest.raises(KeyboardInterrupt):
        run_eval.run_cases(golden_path=golden, sink=sink)
    assert [r["question"] for r in read_jsonl(sink)[1]] == ["q0", "q1", "q2"]

    state["crash_at"], state["ran"] = None, []
    cases = run_eval.run_cases(golden_path=golden, sink=sink)
    assert state["ran"] == ["q3", "q4"]
    assert [c["question"] for c in cases] == [f"q{i}" for i in range(5)]
    assert len(read_jsonl(sink)[1]) == 5


def test_resuming_under_a_different_config_is_refused(fake_pipeline,
                                                      tmp_path):
    golden, _ = fake_pipeline
    sink = tmp_path / "answers.jsonl"
    run_eval.run_cases(golden_path=golden, sink=sink)
    with pytest.raises(SystemExit, match="different config"):
        run_eval.run_cases(golden_path=golden, sink=sink,
                           stages={"multi_hop": True})


# --- what reaches the judge -------------------------------------------------

def test_private_cases_and_private_contexts_never_reach_the_judge():
    cases = [
        case("mine", sources=("tax.pdf",), private=True,
             context_sources=("tax.pdf, p. 1",)),
        case("leaks", context_sources=("a.pdf", "tax.pdf, p. 2")),
        case("public"),
    ]
    todo, skipped = tasks(cases)
    assert [k for k, _, _ in todo] == [row_key(cases[2])]
    assert skipped == {"private": 1, "private_context": 1}


def test_refused_answerable_cases_get_context_recall_only():
    todo, skipped = tasks([case(refused=True),
                           case("oos", out_of_corpus=True, sources=())])
    assert [kind for _, kind, _ in todo] == ["refused"]
    assert skipped == {"should_refuse": 1}


def test_a_follow_up_gets_a_rewrite_check_and_is_judged_as_searched():
    c = case("And its ISBN?", resolved_question="What is the guide's ISBN?",
             history=[{"role": "user", "content": "Who edited the guide?"},
                      {"role": "assistant", "content": "Mike and Andy."}])
    todo, _ = tasks([c])
    kinds = {kind: sample for _, kind, sample in todo}
    assert kinds["answered"].user_input == "What is the guide's ISBN?"
    assert "Who edited the guide?" in kinds["rewrite"].user_input
    assert kinds["rewrite"].response == "What is the guide's ISBN?"


def test_noise_sensitivity_only_when_asked():
    assert "noise" not in {k for _, k, _ in tasks([case()])[0]}
    assert "noise" in {k for _, k, _ in tasks([case()], noise=True)[0]}


def test_a_partial_ragas_file_still_summarises(tmp_path):
    path = tmp_path / "ragas.jsonl"
    rows = [{"_provenance": {"judge": "j"}},
            {"key": "1", "kind": "answered", "faithfulness": 1.0},
            {"key": "2", "kind": "answered", "faithfulness": None},
            {"key": "3", "kind": "answered", "faithfulness": 0.0}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{\"key\"")
    summary = summarize(path)
    f = summary["kinds"]["answered"]["faithfulness"]
    assert f["value"] == 0.5 and f["failed"] == 1
    assert not math.isnan(f["lo"])


def test_compare_keeps_a_follow_ups_two_ragas_rows_apart(tmp_path):
    from eval.stats import compare
    rows = [{"key": "k", "kind": "answered", "faithfulness": 1.0},
            {"key": "k", "kind": "rewrite", "rewrite_preserves_intent": 0.0}]
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    diffs = compare(path, path)
    assert diffs["faithfulness"]["value"] == 0
    assert diffs["rewrite_preserves_intent"]["value"] == 0
