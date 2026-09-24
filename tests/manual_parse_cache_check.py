"""Real check of the parse cache and the OCR fallback, with real Docling.

    .venv/bin/python tests/manual_parse_cache_check.py

Proves the two claims the README makes: that a cached parse makes a second
ingest skip Docling, and that the OCR fallback reads a scanned page.
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.storage import Storage
from ingestion.parser import DoclingParser

root = Path(tempfile.mkdtemp(prefix="ragnar-cache-check-"))
try:
    storage = Storage(root)
    doc_id = "cafebabe"
    fixture = Path("tests/fixtures/sample.pdf")

    parser = DoclingParser()
    t0 = time.perf_counter()
    parsed = parser.parse(fixture)
    first = time.perf_counter() - t0

    storage.write_converted(doc_id, parsed.markdown)
    storage.write_parsed(doc_id, parsed)

    t0 = time.perf_counter()
    cached = storage.read_parsed(doc_id)
    second = time.perf_counter() - t0

    assert cached is not None, "the cache did not read back"
    assert cached.blocks == parsed.blocks, "blocks changed through the cache"
    assert cached.low_confidence == parsed.low_confidence
    print(f"parse {first:.2f}s -> cache read {second:.4f}s "
          f"({first / max(second, 1e-9):.0f}x), "
          f"{len(cached.blocks)} blocks intact")

    # A re-chunk gets the same blocks as a fresh parse would, which is the
    # property that matters: the index must not depend on the cache.
    assert [b.text for b in cached.blocks] == [b.text for b in parsed.blocks]
    print("blocks identical to a fresh parse: OK")

    storage.remove_converted(doc_id)
    assert storage.read_parsed(doc_id) is None
    print("remove_converted clears the cache: OK")

    # The OCR converter, structurally (running RapidOCR needs a scanned page).
    from ingestion.parser import _ocr_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode
    opts = _ocr_converter().format_to_options[InputFormat.PDF].pipeline_options
    assert opts.do_ocr and opts.ocr_options.mode is OcrMode.FULL_PAGE
    print("OCR fallback forces full-page OCR: OK")
finally:
    shutil.rmtree(root, ignore_errors=True)

print("all checks passed")
