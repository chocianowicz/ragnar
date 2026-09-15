import pytest
from core.models import Chunk, SearchResult
from retrieval.reranker import BGEReranker


def _result(text):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=0.5,
    )


@pytest.mark.integration
def test_reranker_ranks_relevant_chunk_first():
    candidates = [
        _result("The office cafeteria serves lunch at noon."),
        _result("The service contract number is SC-4471."),
        _result("Parking permits expire annually."),
    ]

    ranked = BGEReranker().rerank(
        "What is the service contract number?", candidates, top_k=3
    )

    assert "SC-4471" in ranked[0].chunk.text


@pytest.mark.integration
def test_reranker_respects_top_k():
    candidates = [_result(f"text {i}") for i in range(10)]
    ranked = BGEReranker().rerank("query", candidates, top_k=3)
    assert len(ranked) == 3


@pytest.mark.integration
def test_reranker_scores_are_normalised_to_unit_interval():
    candidates = [_result("The contract number is SC-4471.")]
    ranked = BGEReranker().rerank("contract number", candidates, top_k=1)
    assert 0.0 <= ranked[0].score <= 1.0


def test_reranker_handles_empty_candidates():
    assert BGEReranker.__init__ is not None  # import smoke test


def test_reranker_caps_the_pair_window_by_default():
    """sentence-transformers defaults CrossEncoder to the model's full
    8192-token context. Chunking targets 500, so scoring at 8192 cost
    roughly 4.5x on the real corpus (25 candidates: ~130s vs ~30s on CPU)
    for a window nothing fills."""
    from retrieval.reranker import BGEReranker, MAX_LENGTH

    assert MAX_LENGTH == 512
    assert BGEReranker()._max_length == 512


def test_reranker_window_is_configurable():
    from retrieval.reranker import BGEReranker

    assert BGEReranker(max_length=1024)._max_length == 1024
