"""The HTTP path to host_server.py, from the client side.

No network and no Docling: requests go through httpx.MockTransport and the
fallback is a fake parser, so what is under test is exactly the part that
decides between the host and in-process CPU.
"""
import json
from pathlib import Path

import httpx
import pytest

from ingestion.parser import (Block, DoclingParser, ParsedDocument,
                              RemoteParser, build_parser, from_dict, to_dict,
                              PARSER_URL)

FIXTURE = Path("tests/fixtures/sample.pdf")


def _parsed() -> ParsedDocument:
    return ParsedDocument(
        markdown="# doc",
        blocks=[
            Block(text="prose", page=1),
            Block(text="| a | b |", page=2, is_table=True),
            Block(text="total = 3", page=2, is_table=False, is_summary=True),
            Block(text="| x | y |", page=None, is_table=True, sheet="Q1"),
        ],
        page_count=2,
        low_confidence=False,
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeFallback:
    """Parses locally, recording that it was reached for."""

    def __init__(self, parsed=None):
        self.parsed = parsed or ParsedDocument(markdown="local", page_count=1)
        self.calls = []

    def parse(self, path):
        self.calls.append(path)
        return self.parsed


def _explode_if_called():
    class Exploding:
        def parse(self, path):
            raise AssertionError("fell back when the host had answered")
    return Exploding()


def test_to_dict_from_dict_round_trip_preserves_every_block_field():
    """Every field the chunker reads has to survive JSON, including the two
    easy to lose: a summary is not a table, and a spreadsheet block has no
    page number to carry."""
    original = _parsed()
    assert from_dict(to_dict(original)) == original


def test_round_trip_carries_low_confidence():
    parsed = _parsed()
    parsed.low_confidence = True
    assert from_dict(to_dict(parsed)).low_confidence is True


def test_remote_parser_sends_the_bytes_and_rebuilds_the_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["filename"] = request.headers.get("X-Filename")
        seen["body"] = request.read()
        return httpx.Response(200, json=to_dict(_parsed()))

    parser = RemoteParser(url="http://host:8007", client=_client(handler),
                          fallback=_explode_if_called())
    parsed = parser.parse(FIXTURE)

    assert seen["url"] == "http://host:8007/parse"
    # Bytes, not a path: the host and the container do not have to agree on
    # what the data directory is called.
    assert seen["body"] == FIXTURE.read_bytes()
    assert seen["filename"] == "sample.pdf"
    assert parsed == _parsed()


@pytest.mark.parametrize("status", [400, 500, 503])
def test_remote_parser_falls_back_on_a_bad_status(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "no"})

    fallback = FakeFallback()
    parser = RemoteParser(url="http://host:8007", client=_client(handler),
                          fallback=fallback)
    assert parser.parse(FIXTURE) == fallback.parsed
    assert fallback.calls == [FIXTURE]


def test_remote_parser_falls_back_when_the_host_is_unreachable():
    """The whole point of the fallback: host_server.py not running is a
    slower app, not a broken one."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    fallback = FakeFallback()
    parser = RemoteParser(url="http://host:8007", client=_client(handler),
                          fallback=fallback)
    assert parser.parse(FIXTURE) == fallback.parsed


@pytest.mark.parametrize("payload", [
    {"markdown": "x"},                              # missing keys
    {"markdown": "x", "blocks": [{"text": "t", "nope": 1}],
     "page_count": 1, "low_confidence": False},     # unknown block field
    {"markdown": None, "blocks": [], "page_count": 1, "low_confidence": False},
])
def test_remote_parser_falls_back_on_a_malformed_payload(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    fallback = FakeFallback()
    parser = RemoteParser(url="http://host:8007", client=_client(handler),
                          fallback=fallback)
    assert parser.parse(FIXTURE).markdown == "local"


def test_remote_parser_falls_back_on_a_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>502</html>")

    fallback = FakeFallback()
    parser = RemoteParser(url="http://host:8007", client=_client(handler),
                          fallback=fallback)
    assert parser.parse(FIXTURE).markdown == "local"


def test_a_missing_url_never_touches_the_network():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("posted with no url configured")

    fallback = FakeFallback()
    parser = RemoteParser(url="", client=_client(handler), fallback=fallback)
    assert parser.parse(FIXTURE).markdown == "local"


def test_build_parser_picks_remote_when_the_url_is_set(monkeypatch):
    import ingestion.parser as module

    monkeypatch.setattr(module, "PARSER_URL", "http://host:8007")
    assert isinstance(build_parser(), RemoteParser)


def test_build_parser_stays_local_when_the_url_is_unset(monkeypatch):
    import ingestion.parser as module

    monkeypatch.setattr(module, "PARSER_URL", "")
    assert isinstance(build_parser(), DoclingParser)


def test_the_module_default_url_has_no_trailing_slash():
    """RERANKER_URL/PARSER_URL are stripped at import so `{url}/parse` never
    becomes `//parse` when .env carries a trailing slash."""
    assert PARSER_URL == PARSER_URL.rstrip("/")

