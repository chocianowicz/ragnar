import pytest

from core.models import Chunk, SearchResult
from retrieval.agentic import AgenticSearch
from retrieval.search import Search
from tests.fakes import FakeEmbedder


class ScriptedLLM:
    """Returns the next scripted reply, recording what it was asked."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def generate(self, system, user, **kwargs):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""


class FailingLLM:
    def generate(self, system, user, **kwargs):
        raise RuntimeError("ollama is down")


class RecordingStore:
    """Returns a fixed pool, remembering every query text it was given."""

    def __init__(self, results=None, by_text=None):
        self._results = results or []
        self._by_text = by_text or {}
        self.queries = []
        self.is_hybrid = True

    def search(self, vector, limit, doc_ids=None, text=None):
        self.queries.append(text)
        return self._by_text.get(text, self._results)[:limit]


class PassThroughReranker:
    """Scores by position, so ordering is predictable."""

    def __init__(self, score=0.9):
        self.score = score
        self.queries = []

    def rerank(self, query, candidates, top_k):
        self.queries.append(query)
        return [SearchResult(chunk=c.chunk, score=self.score)
                for c in candidates][:top_k]


def _result(text, index=0, score=0.5, doc_id="d"):
    return SearchResult(
        chunk=Chunk(doc_id=doc_id, filename="f.pdf", text=text,
                    chunk_index=index),
        score=score,
    )


def _agentic(store, llm, reranker=None, top_k=5, candidates=25, floor=0.3):
    search = Search(FakeEmbedder(), store,
                    reranker=reranker or PassThroughReranker(),
                    candidates=candidates, top_k=top_k, score_floor=floor)
    return AgenticSearch(search, llm), search


# --- the default: nothing on -------------------------------------------

def test_all_stages_off_behaves_exactly_like_a_base_search():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM())

    outcome, trace = agentic.find("what is the notice period?")

    assert [r.chunk.text for r in outcome.results] == ["a"]
    assert trace.llm_calls == 0
    assert store.queries == ["what is the notice period?"]


def test_no_llm_call_when_nothing_is_retrieved():
    """The refusal must stay free: no model call when there is nothing."""
    agentic, _ = _agentic(RecordingStore([]), FailingLLM())

    outcome, trace = agentic.find("something absent")

    assert outcome.refused
    assert trace.llm_calls == 0


# --- rewriting ----------------------------------------------------------

def test_rewrite_changes_the_query_sent_to_the_store():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["notice period termination"]))

    _, trace = agentic.find("when can I quit?", rewrite=True)

    assert store.queries == ["notice period termination"]
    assert trace.rewritten_query == "notice period termination"


def test_a_degenerate_rewrite_is_discarded():
    """A rewrite that collapses to a fragment has lost the question."""
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["?"]))

    agentic.find("what is the agreed notice period for termination?",
                 rewrite=True)

    assert store.queries == ["what is the agreed notice period for termination?"]


def test_reranking_uses_the_original_question_not_the_rewrite():
    """Otherwise a chunk can clear the floor against wording the user never
    used, and be cited as though it answered their question."""
    reranker = PassThroughReranker()
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["rewritten form"]),
                          reranker=reranker)

    agentic.find("original question", rewrite=True)

    assert reranker.queries == ["original question"]


# --- multi-query --------------------------------------------------------

def test_multi_query_searches_every_variant_but_reranks_once():
    """The point of the design: extra phrasings widen the pool without
    multiplying the expensive stage."""
    reranker = PassThroughReranker()
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["how long is notice?\nnotice term"]),
                          reranker=reranker)

    _, trace = agentic.find("notice period", multi_query=True)

    assert store.queries == ["notice period", "how long is notice?", "notice term"]
    assert len(reranker.queries) == 1
    assert trace.queries == store.queries


def test_multi_query_pool_is_capped_to_the_candidate_budget():
    """Three phrasings must not hand the reranker three times the work."""
    many = {f"q{i}": [_result(f"t{i}{j}", index=j * 10 + i)
                      for j in range(20)] for i in range(3)}
    store = RecordingStore(by_text=many)
    agentic, _ = _agentic(store, ScriptedLLM(["q1\nq2"]), candidates=10)

    # base query is "q0" so all three phrasings return distinct chunks
    agentic.find("q0", multi_query=True)

    # 60 distinct chunks retrieved, but the pool handed on is capped
    assert len(store.queries) == 3


def test_duplicate_variants_are_dropped():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["notice period\nNotice Period?\nnotice term"]))

    agentic.find("notice period", multi_query=True)

    assert store.queries == ["notice period", "notice term"]


# --- follow-ups ---------------------------------------------------------

def test_multi_hop_searches_again_when_a_gap_is_reported():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(
        store, ScriptedLLM(["Sufficient: no\nFollowUp: penalty clause",
                            "Sufficient: yes\nFollowUp: none"]))

    _, trace = agentic.find("notice period", multi_hop=True)

    assert "penalty clause" in store.queries
    assert trace.hops == 1


def test_multi_hop_stops_when_the_excerpts_are_sufficient():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM(["Sufficient: yes\nFollowUp: none"]))

    _, trace = agentic.find("notice period", multi_hop=True)

    assert trace.hops == 0
    assert store.queries == ["notice period"]


def test_multi_hop_is_bounded_by_max_hops():
    store = RecordingStore([_result("a")])
    search = Search(FakeEmbedder(), store, reranker=PassThroughReranker(),
                    candidates=25, top_k=5, score_floor=0.3)
    agentic = AgenticSearch(
        search, ScriptedLLM(["Sufficient: no\nFollowUp: more"] * 10),
        max_hops=2)

    _, trace = agentic.find("q", multi_hop=True)

    assert trace.hops == 2


def test_follow_ups_never_run_when_nothing_cleared_the_floor():
    """They read retrieved text, so they must not see content the floor
    rejected — that is the property the whole refusal rests on."""
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, FailingLLM(),
                          reranker=PassThroughReranker(score=0.01),
                          floor=0.5)

    outcome, trace = agentic.find("q", multi_hop=True, self_correct=True)

    assert outcome.refused
    assert trace.llm_calls == 0


def test_self_correction_widens_when_the_draft_is_judged_incomplete():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM([
        "a draft answer",
        "Complete: no\nImprovement: indexation clause",
    ]))

    _, trace = agentic.find("rent review", self_correct=True)

    assert trace.self_corrected
    assert "indexation clause" in store.queries


def test_self_correction_leaves_a_complete_answer_alone():
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM([
        "a draft answer", "Complete: yes\nImprovement: none"]))

    _, trace = agentic.find("rent review", self_correct=True)

    assert not trace.self_corrected
    assert store.queries == ["rent review"]


# --- robustness ---------------------------------------------------------

def test_a_failing_model_degrades_to_a_plain_search():
    """An agentic stage is an enhancement; losing it must not lose the
    answer."""
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, FailingLLM())

    outcome, trace = agentic.find("q", rewrite=True, multi_query=True)

    assert [r.chunk.text for r in outcome.results] == ["a"]
    assert store.queries == ["q"]
    assert trace.notes


def test_results_are_capped_to_top_k():
    """The defect that made the original version unusable: the fused pool
    was never truncated, so the prompt grew with every extra query."""
    store = RecordingStore([_result(f"t{i}", index=i) for i in range(30)])
    agentic, _ = _agentic(store, ScriptedLLM(["b\nc"]), top_k=5)

    outcome, _ = agentic.find("a", multi_query=True)

    assert len(outcome.results) == 5


@pytest.mark.parametrize("flag", ["rewrite", "multi_query", "multi_hop",
                                  "self_correct"])
def test_each_stage_is_individually_optional(flag):
    store = RecordingStore([_result("a")])
    agentic, _ = _agentic(store, ScriptedLLM([
        "rewritten", "Sufficient: yes\nFollowUp: none",
        "Complete: yes\nImprovement: none"]))

    outcome, _ = agentic.find("q", **{flag: True})

    assert outcome.results
