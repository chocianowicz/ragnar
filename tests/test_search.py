from core.models import Chunk, SearchResult
from retrieval.search import Search
from tests.fakes import FakeEmbedder


class StubStore:
    def __init__(self, results):
        self._results = results

    def search(self, vector, limit):
        return self._results[:limit]


class StubReranker:
    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, candidates, top_k):
        scored = [
            SearchResult(chunk=c.chunk, score=s)
            for c, s in zip(candidates, self._scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _result(text):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=0.5,
    )


def test_results_below_floor_are_rejected():
    search = Search(
        FakeEmbedder(), StubStore([_result("irrelevant")]),
        reranker=StubReranker([0.05]), candidates=25, top_k=5,
        score_floor=0.3,
    )

    outcome = search.find("unrelated question")

    assert outcome.refused is True
    assert outcome.results == []
    assert len(outcome.related) == 1


def test_results_above_floor_are_returned():
    search = Search(
        FakeEmbedder(), StubStore([_result("relevant")]),
        reranker=StubReranker([0.9]), candidates=25, top_k=5,
        score_floor=0.3,
    )

    outcome = search.find("good question")

    assert outcome.refused is False
    assert len(outcome.results) == 1


def test_only_results_above_floor_survive_a_mixed_batch():
    search = Search(
        FakeEmbedder(),
        StubStore([_result("a"), _result("b"), _result("c")]),
        reranker=StubReranker([0.9, 0.1, 0.5]),
        candidates=25, top_k=5, score_floor=0.3,
    )

    outcome = search.find("q")

    assert [round(r.score, 1) for r in outcome.results] == [0.9, 0.5]


def test_empty_corpus_refuses_without_error():
    search = Search(FakeEmbedder(), StubStore([]),
                    reranker=StubReranker([]), score_floor=0.3)

    outcome = search.find("q")

    assert outcome.refused is True
    assert outcome.related == []
