import pytest
from core.models import Chunk, SearchResult
from generation.answerer import Answerer


class StubLLM:
    def __init__(self, reply="Contract number is SC-4471."):
        self.reply = reply
        self.prompts = []

    def generate(self, system, user):
        self.prompts.append((system, user))
        return self.reply


def _result(filename, page, text, score=0.9):
    return SearchResult(
        chunk=Chunk(doc_id="d1", filename=filename, text=text,
                    chunk_index=0, page=page),
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
