"""Host-side GPU service for the two workloads Docker on macOS cannot run fast.

Docker on macOS runs a Linux VM with no access to Metal or the Neural Engine,
so anything inside the app container is CPU-only by construction. Measured on
an M5 Pro:

    reranking (30 candidates)     container CPU 10.34s    host GPU (MPS) 1.48s
    Docling parse (300 pages)     container CPU ~30min    host GPU      minutes

This serves both from the host so the container keeps its reproducible,
pinned environment while the expensive tensor math runs on hardware the
container cannot reach — the same arrangement Ollama already uses here
(host.docker.internal:11434).

Endpoints, one process:

    POST /rerank   {query, texts, max_length}  -> {"scores": [raw logits]}
    POST /parse    raw file bytes, X-Filename  -> to_dict(ParsedDocument)
    GET  /health                               -> which halves are loaded

Returns raw logits, not finished scores. BGEReranker applies the sigmoid and
the top_k cut itself, and keeping that math on the client side means the HTTP
path and the in-process path run identical code from the model output onward.

Every client falls back to in-process CPU when this is unreachable, so the app
never breaks while this is down — it only gets slower.

Run it on the HOST (not in Docker), from anywhere:

    .venv/bin/python host_server.py
    .venv/bin/python host_server.py --device cpu --port 8008
    .venv/bin/python host_server.py --no-parser      # reranking only

Then point the container at it (docker-compose.yml already does):

    RERANKER_URL=http://host.docker.internal:8007
    PARSER_URL=http://host.docker.internal:8007
"""
import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Run as a script from the repo root makes that the script directory, which
# is on sys.path already — this also covers being launched from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# One number, not two: the host service and the in-container fallback must
# truncate identically, or the same chunk scores differently depending on
# where it was reranked and the floors calibrated against one stop applying
# to the other. See reranker.py's MAX_LENGTH for why any cap exists at all.
from ingestion.parser import DoclingParser, to_dict
from retrieval.reranker import MAX_LENGTH

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8007
# ~30 chunks of table/prose text, generously. A book is nowhere near this.
MAX_RERANK_BODY_BYTES = 32 * 1024 * 1024
# A whole PDF arrives in one request, so this has to fit a large one.
MAX_PARSE_BODY_BYTES = 200 * 1024 * 1024

# Docling dispatches on the suffix, and the temp file's name comes from the
# request, so this is also what keeps a request from naming it a format
# outside the ones the app accepts.
ALLOWED_SUFFIXES = {".pdf", ".xlsx", ".docx", ".md"}

_reranker = None
_parser = None
# GPU work serialises in the driver regardless, and concurrent MPS forward
# passes from several threads are not reliably safe, so inference is
# serialised here and the wins come from the device, not from overlapping
# requests. The parser holds one Docling converter per instance, and a
# converter is not documented as safe to call from several threads either.
_reranker_lock = threading.Lock()
_parse_lock = threading.Lock()


def load_reranker(model_name: str, device: str,
                  max_length: int = MAX_LENGTH) -> None:
    global _reranker
    from sentence_transformers import CrossEncoder
    logger.info("loading %s on %s (max_length=%d) ...",
                model_name, device, max_length)
    # max_length must match retrieval/reranker.py's cap, or the host service
    # and the in-container fallback score the same chunk differently and the
    # floors calibrated against one stop applying to the other.
    _reranker = CrossEncoder(model_name, device=device, max_length=max_length)
    # Warm up: the first forward pass compiles Metal kernels and allocates
    # buffers, and is several times slower than steady state. Doing it here
    # means the first real query does not eat that cost.
    _reranker.predict([("warmup query", "warmup document text")])
    logger.info("reranker ready on %s", device)


def load_parser() -> None:
    """Construct the parser /parse will use.

    Loading the model weights themselves is left to Docling, which does it on
    the first convert(). That is deliberate: a host that cannot parse yet must
    still be a host that started, because /health and /rerank are useful
    without it.
    """
    global _parser
    _parser = DoclingParser()
    logger.info("parser ready")


def _rerank(query: str, texts: list[str]) -> list[float]:
    with _reranker_lock:
        raw = _reranker.predict([(query, t) for t in texts])
    return [float(s) for s in raw]


def _parse(path: Path):
    with _parse_lock:
        return _parser.parse(path)


def _bad_rerank_payload(payload) -> str | None:
    """Why this /rerank payload is unusable, or None if it is fine."""
    if not isinstance(payload, dict):
        return "body must be a JSON object"
    if not isinstance(payload.get("query"), str):
        return "query must be a string"
    texts = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return "texts must be a list of strings"
    max_length = payload.get("max_length")
    if max_length is not None and not isinstance(max_length, int):
        return "max_length must be an integer"
    return None


def _filename_from(header: str | None) -> str:
    """The request's X-Filename, reduced to something safe to write.

    The client sends a bare filename. Only the last component of it is kept,
    so an odd or hostile header cannot steer the temp file out of its
    directory, and a suffix outside ALLOWED_SUFFIXES is replaced rather than
    handed to Docling.
    """
    suffix = Path(header or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return "upload.pdf"
    return Path(header).name


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body_length(self, cap: int) -> int | None:
        """The request body's length if it is within the cap, else None.

        A request that is refused also closes the connection: the response is
        sent without draining a body the client may still be uploading, and on
        a keep-alive connection those unread bytes would be read as the next
        request's method line.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            self.close_connection = True
            return None
        if length <= 0 or length > cap:
            self._send(400, {"error": f"body must be 1..{cap} bytes"})
            self.close_connection = True
            return None
        return length

    def do_GET(self):
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "status": "ok",
            "reranker_loaded": _reranker is not None,
            "parser_loaded": _parser is not None,
        })

    def do_POST(self):
        if self.path == "/rerank":
            self._rerank_post()
        elif self.path == "/parse":
            self._parse_post()
        else:
            self._send(404, {"error": "not found"})

    def _rerank_post(self):
        length = self._body_length(MAX_RERANK_BODY_BYTES)
        if length is None:
            return

        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError as exc:
            self._send(400, {"error": f"bad JSON: {exc}"})
            return

        problem = _bad_rerank_payload(payload)
        if problem:
            self._send(400, {"error": f"bad request: {problem}"})
            return

        if _reranker is None:
            self._send(503, {"error": "no reranker loaded on the host"})
            return

        # A request asking for a different window is refused rather than
        # served at this one. Scoring it at a window it did not ask for would
        # silently change numbers the client calibrated its floors against;
        # the 400 makes the client fall back in-process instead, which is
        # slower but scores what it meant to.
        requested = payload.get("max_length")
        if requested is not None and requested != MAX_LENGTH:
            self._send(400, {
                "error": f"max_length {requested} does not match the server's "
                         f"{MAX_LENGTH}; restart with --max-length {requested}",
                "max_length": MAX_LENGTH,
            })
            return

        texts = payload["texts"]
        if not texts:
            self._send(200, {"scores": []})
            return

        try:
            scores = _rerank(payload["query"], texts)
        except Exception as exc:                       # noqa: BLE001
            # A failure here has to be visible. The client answers a bad
            # status by scoring in-process, so without this the only trace
            # would be a slower answer.
            logger.exception("rerank failed")
            self._send(500, {"error": str(exc)})
            return

        self._send(200, {"scores": scores})


# passes from several threads are not reliably safe, so inference is

    def _parse_post(self):
        length = self._body_length(MAX_PARSE_BODY_BYTES)
        if length is None:
            return

        if _parser is None:
            self._send(503, {"error": "no parser loaded on the host"})
            return

        filename = _filename_from(self.headers.get("X-Filename"))
        body = self.rfile.read(length)
        temp_dir = Path(tempfile.mkdtemp(prefix="ragnar-parse-"))
        # Named by content hash as well as by filename: two clients can
        # legitimately send the same filename at the same time, and only the
        # suffix is load-bearing (Docling dispatches on it), so a shared
        # directory must not become a shared file.
        path = temp_dir / f"{hashlib.sha256(body).hexdigest()[:12]}-{filename}"
        try:
            path.write_bytes(body)
            parsed = _parse(path)
        except Exception as exc:                       # noqa: BLE001
            logger.exception("parse failed on %s", filename)
            self._send(500, {"error": str(exc)})
        else:
            self._send(200, to_dict(parsed))
        finally:
            # The bytes were the client's, and the original already lives in
            # data/originals on the caller's side — nothing here outlives the
            # request, or a long run would fill the temp volume with books.
            shutil.rmtree(temp_dir, ignore_errors=True)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 keeps this off the LAN; Docker still "
                             "reaches it via host.docker.internal.")
    parser.add_argument("--device", default="mps",
                        help="mps (Apple GPU), cpu, or cuda")
    parser.add_argument("--model", default=None,
                        help="defaults to config.yaml's models.reranker")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="token cap per (query, chunk) pair. Must match "
                             "retrieval/reranker.py's MAX_LENGTH or the two "
                             "paths score differently.")
    parser.add_argument("--no-reranker", action="store_true",
                        help="skip the cross-encoder; /rerank then answers 503 "
                             "and clients score in-process.")
    parser.add_argument("--no-parser", action="store_true",
                        help="skip Docling; /parse then answers 503 and "
                             "clients parse in-process.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.no_parser:
        load_parser()
    if not args.no_reranker:
        # config.yaml, not a second copy of the name: the host service must
        # score with the same model the client would have used in-process.
        model_name = args.model
        if model_name is None:
            from core.config import Config
            model_name = Config().reranker_model
        load_reranker(model_name, args.device, args.max_length)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

# serialised here and the wins come from the device, not from overlapping
# requests. The parser holds one Docling converter per instance and a
# converter is not documented as safe to call from several threads either.
_reranker_lock = threading.Lock()
_parse_lock = threading.Lock()
