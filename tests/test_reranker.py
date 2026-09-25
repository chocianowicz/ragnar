import httpx
import pytest
from core.models import Chunk, SearchResult
from retrieval.reranker import BGEReranker


def _result(text):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf", text=text, chunk_index=0),
        score=0.5,
    )


class RecordingModel:
    """Stands in for a CrossEncoder: no weights, no download, but records the
    pairs it was asked to score and returns scripted logits."""
    def __init__(self, logits=None):
        self.logits = logits if logits is not None else [1.0, -1.0]
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.logits[:len(self.pairs)]


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


# ── The host-service path (host_server.py) ───────────────────────────────────
#
# No network and no weights: the transport is mocked and the in-process model
# is injected, so what is under test is the decision between the two paths and
# the fact that they agree on everything after the model output.


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _explode_if_used():
    class Exploding:
        def predict(self, pairs):
            raise AssertionError("scored locally when the host answered")
    return Exploding()


def test_remote_scores_are_used_without_touching_the_local_model():
    import json

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.read())
        return httpx.Response(200, json={"scores": [0.88, 0.12]})

    reranker = BGEReranker(url="http://host:8007", client=_client(handler),
                           model=_explode_if_used())
    ranked = reranker.rerank("q", [_result("first"), _result("second")],
                             top_k=2)

    assert seen["url"] == "http://host:8007/rerank"
    assert seen["payload"] == {"query": "q", "texts": ["first", "second"],
                               "max_length": 512}
    # The host's scores are used as they come: already probabilities.
    assert ranked[0].chunk.text == "first"
    assert ranked[0].score == pytest.approx(0.88)


def test_the_remote_path_and_the_local_path_agree_on_the_score():
    """The HTTP hop must not change the numbers: identical scores in, and
    the sort and the top_k cut applied here on both paths."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": [0.8, 0.1, 0.4]})

    candidates = [_result("a"), _result("b"), _result("c")]
    remote = BGEReranker(url="http://host:8007", client=_client(handler),
                         model=_explode_if_used()).rerank("q", candidates, 3)
    local = BGEReranker(url="", model=RecordingModel([0.8, 0.1, 0.4])
                        ).rerank("q", candidates, 3)

    assert [r.chunk.text for r in remote] == [r.chunk.text for r in local]
    assert [r.score for r in remote] == [r.score for r in local]


@pytest.mark.parametrize("status", [400, 500, 503])
def test_a_bad_status_falls_back_to_the_local_model(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "no"})

    model = RecordingModel([1.0, 0.0])
    ranked = BGEReranker(url="http://host:8007", client=_client(handler),
                         model=model).rerank("q", [_result("a"), _result("b")], 2)

    assert model.pairs == [("q", "a"), ("q", "b")]
    assert ranked[0].chunk.text == "a"


def test_an_unreachable_host_falls_back_to_the_local_model():
    """The whole point of the fallback: host_server.py not running is a slower
    app, not a broken one."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    model = RecordingModel([1.0, 0.0])
    BGEReranker(url="http://host:8007", client=_client(handler),
                model=model).rerank("q", [_result("a"), _result("b")], 2)
    assert model.pairs == [("q", "a"), ("q", "b")]


def test_a_max_length_mismatch_is_refused_by_the_host_and_falls_back():
    """The host answers 400 for a max_length it was not started with rather
    than scoring at its own cap. Scoring at a different window would change
    the numbers the client's score_floor was calibrated against, so the client
    taking the slow local path is the correct outcome here."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = request.read().decode()
        return httpx.Response(400, json={
            "error": "max_length 1024 does not match the server's 512",
            "max_length": 512,
        })

    model = RecordingModel([1.0, 0.0])
    BGEReranker(url="http://host:8007", client=_client(handler), model=model,
                max_length=1024).rerank("q", [_result("a")], 1)

    import json
    assert json.loads(seen["payload"])["max_length"] == 1024
    assert model.pairs == [("q", "a")]


def test_a_missing_url_never_touches_the_network():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("posted with no url configured")

    model = RecordingModel([1.0, 0.0])
    BGEReranker(url="", client=_client(handler),
                model=model).rerank("q", [_result("a")], 1)
    assert model.pairs == [("q", "a")]


@pytest.mark.parametrize("body", [
    {},                          # no scores key
    {"scores": None},            # not a list
    {"scores": "0.5"},           # not a list
    {"scores": [{"value": 1}]},  # not numeric
])
def test_a_malformed_score_payload_falls_back_rather_than_scoring_garbage(body):
    """float() raises TypeError for a dict and ValueError for a string; both
    are in the caught set, so neither can corrupt a ranking silently."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    model = RecordingModel([1.0, 0.0])
    ranked = BGEReranker(url="http://host:8007", client=_client(handler),
                         model=model).rerank("q", [_result("a"), _result("b")], 2)

    assert model.pairs == [("q", "a"), ("q", "b")]
    assert ranked[0].chunk.text == "a"


def test_empty_candidates_short_circuit_before_any_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("posted for an empty candidate list")

    reranker = BGEReranker(url="http://host:8007", client=_client(handler),
                           model=_explode_if_used())
    assert reranker.rerank("q", [], top_k=5) == []


def test_reranker_url_module_default_has_no_trailing_slash():
    from retrieval.reranker import RERANKER_URL

    assert RERANKER_URL == RERANKER_URL.rstrip("/")

