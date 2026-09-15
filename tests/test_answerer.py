import pytest
from core.models import Chunk, SearchResult
from generation.answerer import (
    Answerer, AnswerMode, build_excerpts, citation_labels, classify,
)


class StubLLM:
    def __init__(self, reply="Contract number is SC-4471."):
        self.reply = reply
        self.prompts = []
        self.opts = []
        self.histories = []

    def generate(self, system, user, *, model=None, temperature=None,
                 history=None):
        self.prompts.append((system, user))
        self.opts.append((model, temperature))
        self.histories.append(history)
        return self.reply

    def stream(self, system, user, *, model=None, temperature=None,
               history=None):
        self.prompts.append((system, user))
        self.opts.append((model, temperature))
        self.histories.append(history)
        yield self.reply


def _result(filename, page, text, score=0.9, is_table=False, is_summary=False):
    return SearchResult(
        chunk=Chunk(doc_id="d1", filename=filename, text=text,
                    chunk_index=0, page=page, is_table=is_table,
                    is_summary=is_summary),
        score=score,
    )


def test_citations_derive_from_chunks_not_model_output():
    llm = StubLLM(reply="The answer is in doc_that_does_not_exist.pdf")
    answerer = Answerer(llm)

    answer = answerer.answer("q", [_result("real.pdf", 4, "text")])

    assert answer.citations == ["real.pdf, p. 4"]


def test_citations_are_deduplicated_by_file_and_page():
    answerer = Answerer(StubLLM())
    results = [
        _result("a.pdf", 1, "one"),
        _result("a.pdf", 1, "two"),
        _result("a.pdf", 2, "three"),
    ]

    answer = answerer.answer("q", results)

    assert answer.citations == ["a.pdf, p. 1", "a.pdf, p. 2"]


def test_empty_results_refuse_without_calling_the_model():
    llm = StubLLM()
    answerer = Answerer(llm)

    answer = answerer.answer("q", [])

    assert answer.refused is True
    assert answer.citations == []
    assert llm.prompts == []


def test_context_includes_source_labels_for_each_chunk():
    llm = StubLLM()
    Answerer(llm).answer("q", [_result("a.pdf", 7, "body text")])

    _system, user = llm.prompts[0]
    assert "a.pdf, p. 7" in user
    assert "body text" in user


def test_citation_labels_dedupes_preserving_first_seen_order():
    results = [
        _result("b.pdf", 2, "x"),
        _result("a.pdf", 1, "y"),
        _result("b.pdf", 2, "z"),  # duplicate label, must not reappear
    ]
    assert citation_labels(results) == ["b.pdf, p. 2", "a.pdf, p. 1"]


def test_citation_labels_empty_for_no_results():
    assert citation_labels([]) == []


def test_build_excerpts_pairs_labels_with_text():
    assert build_excerpts([_result("a.pdf", 3, "body")]) == [
        ("a.pdf, p. 3", "body")
    ]


def test_stream_yields_answer_and_uses_source_labels():
    llm = StubLLM(reply="streamed answer")
    chunks = list(Answerer(llm).stream("q", [_result("a.pdf", 7, "body text")]))

    assert "".join(chunks) == "streamed answer"
    _system, user = llm.prompts[0]
    assert "a.pdf, p. 7" in user and "body text" in user


def test_per_request_model_and_temperature_are_threaded_to_the_llm():
    llm = StubLLM()
    Answerer(llm).answer("q", [_result("a.pdf", 1, "x")],
                         model="llama3", temperature=0.7)
    assert llm.opts[0] == ("llama3", 0.7)

    list(Answerer(llm).stream("q", [_result("a.pdf", 1, "x")],
                              model="qwen", temperature=0.2))
    assert llm.opts[1] == ("qwen", 0.2)


# --- classify: the shared refuse / guard / answer policy ---------------------

def test_classify_no_results_when_search_refused():
    assert classify("anything", True, []) is AnswerMode.NO_RESULTS


def test_classify_answers_normal_prose_question():
    results = [_result("a.pdf", 1, "the contract number is SC-4471")]
    assert classify("what is the contract number?", False, results) \
        is AnswerMode.ANSWER


def test_classify_refuses_aggregation_over_table_heavy_results():
    results = [_result("a.pdf", 1, "| x | y |", is_table=True),
               _result("a.pdf", 2, "| a | b |", is_table=True)]
    assert classify("what is the total?", False, results) \
        is AnswerMode.AGGREGATION_REFUSED


def test_classify_defers_aggregation_when_a_summary_was_retrieved():
    # A precomputed aggregate summary means the total is already a ready fact.
    results = [_result("a.pdf", 1, "| x | y |", is_table=True),
               _result("a.pdf", 1, "Aggregate: total 600", is_summary=True)]
    assert classify("what is the total?", False, results) is AnswerMode.ANSWER


def test_build_citations_carries_navigation_metadata():
    from generation.answerer import build_citations
    from core.models import Chunk, SearchResult

    results = [SearchResult(
        chunk=Chunk(doc_id="abc", filename="umowa.pdf", text="the passage",
                    chunk_index=7, page=4),
        score=0.81,
    )]

    [citation] = build_citations(results)

    assert citation["label"] == "umowa.pdf, p. 4"
    assert citation["doc_id"] == "abc"
    assert citation["page"] == 4
    assert citation["chunk_index"] == 7
    assert citation["text"] == "the passage"
    assert citation["score"] == 0.81


def test_build_citations_are_json_serialisable():
    """They go straight into the chat history, which is persisted as JSON.
    A dataclass here would need a custom encoder and would come back as a
    dict anyway, so the two paths would disagree about the type."""
    import json
    from generation.answerer import build_citations
    from core.models import Chunk, SearchResult

    citations = build_citations([SearchResult(
        chunk=Chunk(doc_id="d", filename="f.xlsx", text="t", chunk_index=0,
                    sheet="Q1"),
        score=0.5,
    )])

    assert json.loads(json.dumps(citations)) == citations


def test_build_citations_deduplicates_by_label():
    from generation.answerer import build_citations
    from core.models import Chunk, SearchResult

    same = [SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=f"passage {i}",
                    chunk_index=i, page=2),
        score=0.9 - i / 10,
    ) for i in range(3)]

    citations = build_citations(same)

    assert len(citations) == 1
    # first-seen wins, matching citation_labels; results arrive sorted so
    # that is also the best-scoring one here
    assert citations[0]["text"] == "passage 0"


def test_conversation_history_reaches_the_model():
    """Without this, a follow-up like "and the base year?" is answered with
    no idea what it refers to."""
    llm = StubLLM()
    history = [
        {"role": "user", "content": "What is Norway's target?"},
        {"role": "assistant", "content": "70-75% below 1990.",
         "citations": [{"label": "x"}]},
    ]

    Answerer(llm).answer("And the base year?",
                         [_result("f.pdf", 1, "text")], history=history)

    assert llm.histories[0] == [
        {"role": "user", "content": "What is Norway's target?"},
        {"role": "assistant", "content": "70-75% below 1990."},
    ]


def test_streaming_also_carries_history():
    llm = StubLLM()
    history = [{"role": "user", "content": "earlier question"}]

    list(Answerer(llm).stream("follow up", [_result("f.pdf", 1, "t")],
                              history=history))

    assert llm.histories[0] == [{"role": "user", "content": "earlier question"}]


def test_a_refusal_still_never_calls_the_model():
    """History must not create a path where an empty result set reaches the
    model anyway."""
    llm = StubLLM()

    answer = Answerer(llm).answer("q", [], history=[
        {"role": "user", "content": "earlier"}])

    assert answer.refused
    assert llm.prompts == []
