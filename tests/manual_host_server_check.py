"""Real end-to-end check: the actual clients against the actual host server.

Run the host server first, then this. Asserts and prints, so it fails loudly
if the wiring drifts.
"""
import sys
from pathlib import Path

# The repo root, ahead of everything else. The shared virtualenv carries an
# editable install of the *other* checkout (the venv lives there), so without
# this a script run from inside tests/ would import that checkout's ingestion
# package instead of this one's.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from core.models import Chunk, SearchResult
from retrieval.reranker import BGEReranker
from ingestion.parser import RemoteParser, from_dict

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8007"
health = httpx.get(f"{BASE}/health", timeout=30).json()
print("health:", health)

if health["reranker_loaded"]:
    rr = BGEReranker(url=BASE)
    cands = [SearchResult(chunk=Chunk(doc_id="d", filename="f.pdf",
                                      text=f"noise {i}", chunk_index=i),
                          score=0.5) for i in range(5)]
    cands[2] = SearchResult(chunk=Chunk(doc_id="d", filename="f.pdf",
                                        text="The service contract number "
                                             "is SC-4471.", chunk_index=2),
                            score=0.5)
    ranked = rr.rerank("What is the service contract number?", cands, top_k=2)
    print("rerank over http:", [round(r.score, 4) for r in ranked])
    assert "SC-4471" in ranked[0].chunk.text, "the relevant chunk did not win"
    print("rerank: OK (sigmoid applied client-side, best chunk first)")

if health["parser_loaded"]:
    fixture = Path("tests/fixtures/sample.pdf")
    parsed = RemoteParser(url=BASE).parse(fixture)
    assert "SC-4471" in parsed.markdown, "markdown did not survive the hop"
    pages = sorted({b.page for b in parsed.blocks if b.page})
    print("parse over http: blocks", len(parsed.blocks), "pages", pages)
    assert pages == [1, 2]
    print("parse: OK (round-tripped through from_dict)")

print("all checks passed")
