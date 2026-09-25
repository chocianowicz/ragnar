"""host_server.py's handlers, driven over loopback with fake heavy parts.

The model and Docling are monkeypatched out (`_rerank`, `_parser`), so these
tests never download weights, never touch a GPU, and never leave the machine —
but they do go through the real HTTP status codes, header parsing and body
caps, because that plumbing is what the fallback logic on the client keys on.
"""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from ingestion.parser import Block, DoclingParser, ParsedDocument
import host_server as server_module
from host_server import (MAX_PARSE_BODY_BYTES, MAX_RERANK_BODY_BYTES, Handler,
                         _bad_rerank_payload, _filename_from)


@pytest.fixture
def live_server():
    """A real server on a loopback ephemeral port, torn down after the test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def fake_reranker(monkeypatch):
    """Stands in for the cross-encoder, recording the pairs it was given."""
    calls = []

    def fake(query, texts):
        calls.append((query, list(texts)))
        return [float(i) for i in range(len(texts))]

    monkeypatch.setattr(server_module, "_rerank", fake)
    monkeypatch.setattr(server_module, "_reranker", object())
    return calls


@pytest.fixture
def fake_parser(monkeypatch):
    """Stands in for Docling, recording the temp path it was handed."""
    seen = {}

    class Parsing:
        def parse(self, path):
            seen["path"] = path
            seen["bytes"] = path.read_bytes()
            seen["name"] = path.name
            seen["exists_during"] = path.exists()
            return ParsedDocument(
                markdown="# remote",
                blocks=[Block(text="prose", page=1)],
                page_count=1,
            )

    monkeypatch.setattr(server_module, "_parser", Parsing())
    return seen


def post(port, path, body=b"", headers=None):
    # Content-Length has to be set explicitly: the server sizes the body from
    # the header, deliberately, so it can refuse an oversized one before
    # reading it.
    sent = {"Content-Length": str(len(body))}
    sent.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", path, body=body, headers=sent)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def rerank_body(**over):
    payload = {"query": "q", "texts": ["a", "b"], "max_length": 512}
    payload.update(over)
    return json.dumps(payload).encode()


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_reports_which_halves_are_loaded(live_server, monkeypatch):
    monkeypatch.setattr(server_module, "_reranker", None)
    monkeypatch.setattr(server_module, "_parser", None)
    status, body = get(live_server, "/health")
    assert status == 200
    assert body == {"status": "ok", "reranker_loaded": False,
                    "parser_loaded": False}

    monkeypatch.setattr(server_module, "_reranker", object())
    monkeypatch.setattr(server_module, "_parser", object())
    _, body = get(live_server, "/health")
    assert body["reranker_loaded"] and body["parser_loaded"]


def test_unknown_paths_are_404(live_server):
    assert get(live_server, "/nope")[0] == 404
    assert post(live_server, "/nope", b"{}")[0] == 404


# ── /rerank ──────────────────────────────────────────────────────────────────

def test_rerank_returns_raw_logits(live_server, fake_reranker):
    status, body = post(live_server, "/rerank", rerank_body(),
                        {"Content-Type": "application/json"})
    assert status == 200
    # Raw logits, not sigmoid-ed: the client owns that transform.
    assert body == {"scores": [0.0, 1.0]}
    assert fake_reranker == [("q", ["a", "b"])]


def test_rerank_accepts_a_matching_max_length(live_server, fake_reranker):
    status, _ = post(live_server, "/rerank", rerank_body(max_length=512),
                     {"Content-Type": "application/json"})
    assert status == 200


def test_rerank_refuses_a_different_max_length_without_scoring(
        live_server, fake_reranker):
    """Refused, not served at the server's own cap: scoring a chunk at a
    window the client did not ask for would change numbers its score_floor was
    calibrated against. The 400 is what makes the client fall back instead."""
    status, body = post(live_server, "/rerank", rerank_body(max_length=1024),
                        {"Content-Type": "application/json"})

    assert status == 400
    assert body["max_length"] == 512
    assert fake_reranker == []          # the model was never asked


def test_rerank_tolerates_a_client_that_omits_max_length(live_server,
                                                         fake_reranker):
    """Older clients predate the field; the server's own cap applies and the
    answer is still usable."""
    body = json.dumps({"query": "q", "texts": ["a"]}).encode()
    status, response = post(live_server, "/rerank", body,
                            {"Content-Type": "application/json"})
    assert status == 200
    assert response == {"scores": [0.0]}


def test_rerank_answers_an_empty_candidate_list_without_scoring(
        live_server, fake_reranker):
    status, body = post(live_server, "/rerank", rerank_body(texts=[]),
                        {"Content-Type": "application/json"})
    assert status == 200
    assert body == {"scores": []}
    assert fake_reranker == []


@pytest.mark.parametrize("body", [
    b"not json",
    b"[1, 2, 3]",                                   # not an object
    b'{"texts": ["a"]}',                            # no query
    b'{"query": 5, "texts": ["a"]}',                # query not a string
    b'{"query": "q"}',                              # no texts
    b'{"query": "q", "texts": "a"}',                # texts not a list
    b'{"query": "q", "texts": [1, 2]}',             # texts not strings
    b'{"query": "q", "texts": ["a"], "max_length": "512"}',
])
def test_rerank_rejects_bad_payloads(live_server, fake_reranker, body):
    status, response = post(live_server, "/rerank", body,
                            {"Content-Type": "application/json"})
    assert status == 400
    assert "error" in response
    assert fake_reranker == []


def test_rerank_refuses_when_no_model_is_loaded(live_server, monkeypatch):
    """--no-reranker, or a failed load: answer 503 so the client falls back
    rather than waiting on an empty GPU."""
    monkeypatch.setattr(server_module, "_reranker", None)
    status, body = post(live_server, "/rerank", rerank_body(),
                        {"Content-Type": "application/json"})
    assert status == 503
    assert "error" in body


def test_rerank_oversized_body_is_refused(live_server, fake_reranker):
    """Refused on the header, before the body is read: an oversized request is
    answered without draining what the client is still uploading."""
    conn = http.client.HTTPConnection("127.0.0.1", live_server, timeout=10)
    try:
        conn.putrequest("POST", "/rerank")
        conn.putheader("Content-Length", str(MAX_RERANK_BODY_BYTES + 1))
        conn.endheaders()
        # No body is sent at all: the answer must not depend on receiving it.
        response = conn.getresponse()
        assert response.status == 400
        assert "bytes" in json.loads(response.read())["error"]
    finally:
        conn.close()
    assert fake_reranker == []


def test_rerank_reports_a_model_failure_as_500(live_server, monkeypatch):
    """The client falls back on a bad status, so a real failure has to surface
    as one instead of a 200 with wrong numbers in it."""
    def boom(query, texts):
        raise RuntimeError("MPS backend fell over")

    monkeypatch.setattr(server_module, "_rerank", boom)
    monkeypatch.setattr(server_module, "_reranker", object())
    status, body = post(live_server, "/rerank", rerank_body(),
                        {"Content-Type": "application/json"})
    assert status == 500
    assert "MPS backend fell over" in body["error"]


# ── /parse ───────────────────────────────────────────────────────────────────

def test_parse_round_trips_a_document(live_server, fake_parser):
    status, body = post(live_server, "/parse", b"%PDF-1.4 fake bytes",
                        {"X-Filename": "contract.pdf"})

    assert status == 200
    from ingestion.parser import from_dict
    parsed = from_dict(body)
    assert parsed.markdown == "# remote"
    assert parsed.blocks == [Block(text="prose", page=1)]


def test_parse_writes_the_bytes_to_a_file_doclings_suffix_rules_accept(
        live_server, fake_parser):
    """Docling dispatches on the suffix, and only the suffix can come from the
    request — hence the temp file being named `{hash}-{filename}`."""
    post(live_server, "/parse", b"sheet bytes", {"X-Filename": "budget.xlsx"})

    assert fake_parser["bytes"] == b"sheet bytes"
    assert fake_parser["exists_during"] is True
    assert fake_parser["name"].endswith("-budget.xlsx")


def test_parse_removes_the_temp_file_afterwards(live_server, fake_parser):
    """A long-running host must not accumulate every book it has parsed."""
    post(live_server, "/parse", b"%PDF-1.4", {"X-Filename": "a.pdf"})
    path = fake_parser["path"]

    # rmtree runs in the handler's finally; give it a moment to land.
    for _ in range(100):
        if not path.exists():
            break
        import time
        time.sleep(0.01)
    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize("filename", [None, "", "noextension", "script.sh",
                                      "page.html"])
def test_parse_falls_back_to_pdf_for_an_unusable_filename(
        live_server, fake_parser, filename):
    """An unset, empty, extensionless or unsupported X-Filename still parses:
    the request is refused a usable suffix, not refused service."""
    headers = {"X-Filename": filename} if filename is not None else {}
    status, _ = post(live_server, "/parse", b"%PDF-1.4", headers)
    assert status == 200
    assert fake_parser["name"].endswith("upload.pdf")


def test_parse_keeps_only_the_last_component_of_the_filename(
        live_server, fake_parser):
    """A traversal attempt must not steer the temp file out of its directory."""
    post(live_server, "/parse", b"%PDF-1.4",
         {"X-Filename": "../../../../tmp/evil.pdf"})

    assert fake_parser["name"].endswith("-evil.pdf")
    assert fake_parser["path"].parent.name.startswith("ragnar-parse-")


def test_parse_cap_is_large_enough_for_a_book():
    """The whole point of sending a whole file in one request, which is what
    lets the container send bytes instead of a path it cannot share."""
    assert MAX_PARSE_BODY_BYTES >= 200 * 1024 * 1024


def test_parse_refuses_an_empty_body(live_server, fake_parser):
    status, body = post(live_server, "/parse", b"", {"X-Filename": "a.pdf"})
    assert status == 400
    assert "bytes" in body["error"]


def test_parse_refuses_before_reading_an_oversized_body(live_server,
                                                        fake_parser):
    conn = http.client.HTTPConnection("127.0.0.1", live_server, timeout=10)
    try:
        conn.putrequest("POST", "/parse")
        conn.putheader("Content-Length", str(MAX_PARSE_BODY_BYTES + 1))
        conn.putheader("X-Filename", "book.pdf")
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 400
        assert "bytes" in json.loads(response.read())["error"]
    finally:
        conn.close()


def test_parse_refuses_when_no_parser_is_loaded(live_server, monkeypatch):
    monkeypatch.setattr(server_module, "_parser", None)
    status, body = post(live_server, "/parse", b"%PDF-1.4",
                        {"X-Filename": "a.pdf"})
    assert status == 503
    assert "error" in body


def test_parse_reports_a_parser_failure_as_500(live_server, monkeypatch):
    class Broken:
        def parse(self, path):
            raise RuntimeError("not a PDF")

    monkeypatch.setattr(server_module, "_parser", Broken())
    status, body = post(live_server, "/parse", b"garbage",
                        {"X-Filename": "a.pdf"})
    assert status == 500
    assert "not a PDF" in body["error"]


from host_server import (MAX_PARSE_BODY_BYTES, MAX_RERANK_BODY_BYTES, Handler,
                         _bad_rerank_payload, _filename_from)



# ── The pure pieces, without a socket ────────────────────────────────────────

@pytest.mark.parametrize("payload,ok", [
    ({"query": "q", "texts": ["a"]}, True),
    ({"query": "q", "texts": [], "max_length": 512}, True),
    ({"query": "q", "texts": ["a"], "max_length": None}, True),
    ({"query": "q", "texts": ["a"], "max_length": 512}, True),
    ({"query": "q", "texts": ["a"], "max_length": 512.0}, False),
    (None, False),
    ("string", False),
    ({"texts": ["a"]}, False),
])
def test_bad_rerank_payload(payload, ok):
    assert (_bad_rerank_payload(payload) is None) is ok


@pytest.mark.parametrize("header,expected", [
    ("report.pdf", "report.pdf"),
    ("data.xlsx", "data.xlsx"),
    ("notes.md", "notes.md"),
    ("memo.docx", "memo.docx"),
    ("REPORT.PDF", "REPORT.PDF"),
    ("page.html", "upload.pdf"),      # unsupported suffix
    ("noextension", "upload.pdf"),
    (None, "upload.pdf"),
    ("", "upload.pdf"),
    ("sub/dir/report.pdf", "report.pdf"),
    ("../../etc/passwd", "upload.pdf"),   # no usable suffix left
])
def test_filename_from(header, expected):
    assert _filename_from(header) == expected


def test_load_parser_builds_a_docling_parser(monkeypatch):
    """No weights are downloaded here: Docling loads its models on the first
    convert(), so this only asserts the wiring."""
    monkeypatch.setattr(server_module, "_parser", None)
    server_module.load_parser()
    assert isinstance(server_module._parser, DoclingParser)


def test_the_parse_lock_is_held_for_the_whole_parse(monkeypatch):
    """One Docling converter per instance, and a converter is not documented
    as safe to call from several threads, so a second parse must wait."""
    import threading as threading_module

    order = []

    class Slow:
        def parse(self, path):
            order.append(("start", threading_module.current_thread().name))
            import time
            time.sleep(0.05)
            order.append(("end", threading_module.current_thread().name))
            return ParsedDocument(markdown="x", page_count=1)

    monkeypatch.setattr(server_module, "_parser", Slow())

    threads = [threading_module.Thread(target=server_module._parse,
                                       args=(__file__,)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every parse is bracketed start->end with no interleaving.
    assert [kind for kind, _ in order] == ["start", "end"] * 3


def test_the_module_imports_and_defines_every_name_a_handler_needs():
    """The other tests monkeypatch `_rerank` and `_parser` out, which is what
    made this worth asserting: an edit once truncated the globals block, leaving
    `_parse_lock` undefined — the whole suite stayed green while a real /parse
    answered 500. This runs the module fresh and touches each name."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import host_server as h; "
         "h._reranker_lock; h._parse_lock; "
         "assert h.load_parser and h.load_reranker and h.Handler; "
         "assert h.DEFAULT_PORT == 8007 and h.MAX_LENGTH == 512; "
         "assert h._filename_from('a.pdf') == 'a.pdf'; "
         "print('ok')"],
        cwd=str(Path(server_module.__file__).parent),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_the_module_does_not_load_the_models():
    """`--no-reranker --no-parser` has to be a usable server on a machine with
    no weights cached, so importing must not construct anything heavy."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, host_server; "
         "print('sentence_transformers' in sys.modules)"],
        cwd=str(Path(server_module.__file__).parent),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"

