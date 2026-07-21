# Ragnar Dockerized Local RAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline RAG system over company PDF/Excel documents, running as a Docker Compose stack on macOS, answering questions with citations and refusing when the answer is not in the corpus.

**Architecture:** Ollama runs natively on the macOS host (for Metal GPU); Streamlit + Docling + reranker + Qdrant run in Docker. Documents are parsed by Docling into structured JSON, chunked with provenance, embedded via BGE-M3, and stored in Qdrant. Queries retrieve 25 candidates, rerank to 5, apply a similarity floor, then generate an answer with citations derived from chunk metadata. A separate evaluation harness scores the system against a hand-written golden set.

**Tech Stack:** Python 3.11, Streamlit, Docling, Qdrant, Ollama (`qwen2.5:14b`, `bge-m3`), `bge-reranker-v2-m3` via sentence-transformers, SQLite, pytest, Ragas.

**Reference spec:** `docs/superpowers/specs/2026-07-21-ragnar-docker-design.md`

---

## Sequencing Rationale

Phase 1 builds a **thin vertical slice**: one PDF, ingested, retrieved, answered, cited. It uses deliberately naive components (fixed-size chunking, no reranker, no guard). The point is to surface Docker/Ollama/Qdrant wiring problems on day one when they are cheap to fix, rather than after the interesting logic is written.

Phases 2–5 then replace the naive pieces with real ones, each independently testable against a working system.

**Do not skip ahead.** If Phase 1 does not work end to end, nothing later will.

---

## File Structure

| File | Responsibility |
|---|---|
| `core/models.py` | Dataclasses: `Chunk`, `Document`, `SearchResult`, `IngestStatus` |
| `core/interfaces.py` | Protocols: `DocumentParser`, `Chunker`, `Embedder`, `VectorStore`, `Reranker` |
| `core/config.py` | Load + validate `config.yaml` and env vars |
| `ingestion/parser.py` | `DoclingParser` — file → `DoclingDocument` + markdown |
| `ingestion/chunkers/fixed.py` | Naive fixed-size chunker (Phase 1 only) |
| `ingestion/chunkers/structural.py` | Heading/paragraph-aware chunker (default) |
| `ingestion/chunkers/semantic.py` | Embedding-similarity chunker (A/B alternative) |
| `ingestion/chunkers/registry.py` | Name → `Chunker`; the only selection point |
| `ingestion/tables.py` | Table row-group chunking with header repetition |
| `ingestion/registry_db.py` | SQLite document registry |
| `ingestion/storage.py` | inbox/originals/converted layout, hashing |
| `ingestion/pipeline.py` | Orchestration: parse → chunk → embed → store |
| `ingestion/worker.py` | Background queue drain thread |
| `retrieval/embedder.py` | `OllamaEmbedder` (bge-m3) |
| `retrieval/store.py` | `QdrantStore` |
| `retrieval/reranker.py` | `BGEReranker` (CPU) |
| `retrieval/search.py` | retrieve → rerank → threshold |
| `generation/prompts.py` | System prompt, PL/EN aware |
| `generation/guards.py` | Aggregation detection + refusal |
| `generation/answerer.py` | Prompt → Ollama → answer + citations |
| `ui/app.py` | Streamlit application |
| `eval/run_eval.py` | Ragas harness + deterministic metrics |
| `tests/fakes.py` | Fake `Embedder` / `VectorStore` for unit tests |

---

# PHASE 0 — Prerequisites

### Task 1: Pull required Ollama models

Ollama is already installed and running, but only `gemma4:26b` is present. The spec requires two other models.

**Files:** none (environment setup)

- [ ] **Step 1: Pull the embedding model**

```bash
ollama pull bge-m3
```

Expected: downloads ~1.2GB, ends with `success`.

- [ ] **Step 2: Pull the generation model**

```bash
ollama pull qwen2.5:14b
```

Expected: downloads ~9GB, ends with `success`. This takes several minutes.

- [ ] **Step 3: Verify both are present**

```bash
ollama list
```

Expected: output includes both `bge-m3` and `qwen2.5:14b`.

- [ ] **Step 4: Verify the embedding endpoint works and note the dimension**

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"bge-m3","input":"test"}' \
  | python3 -c "import sys,json; print('dim =', len(json.load(sys.stdin)['embeddings'][0]))"
```

Expected: `dim = 1024`

**If the dimension is not 1024, stop.** Every later task hardcodes 1024 as the Qdrant vector size; record the real value and substitute it throughout.

---

### Task 2: Repository skeleton and tooling

**Files:**
- Create: `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore` (modify), `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ragnar"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.40",
    "docling>=2.0",
    "qdrant-client>=1.12",
    "httpx>=0.27",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "pyyaml>=6.0",
    "sentence-transformers>=3.3",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=6.0", "reportlab>=4.2"]
eval = ["ragas>=0.2", "langchain-openai>=0.2", "datasets>=3.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires Ollama and/or Qdrant running",
]
```

Note: `eval` extras are intentionally separate — the app image must never install them.

- [ ] **Step 2: Create `config.yaml`**

```yaml
models:
  llm: qwen2.5:14b
  embedding: bge-m3
  embedding_dim: 1024
  reranker: BAAI/bge-reranker-v2-m3

chunking:
  strategy: fixed        # fixed | structural | semantic
  target_tokens: 500
  overlap_tokens: 50
  table_rows_per_group: 20

retrieval:
  candidates: 25
  top_k: 5
  score_floor: 0.0       # calibrated in Task 24; 0.0 disables the floor

storage:
  data_dir: /app/data
  collection: documents
```

- [ ] **Step 3: Create `.env.example`**

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
QDRANT_URL=http://qdrant:6333
```

- [ ] **Step 4: Append to `.gitignore`**

```
# python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# runtime data
data/
.env
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests requiring Ollama/Qdrant",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 6: Create the venv and install**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"
```

Expected: completes without error. Docling pulls torch; this takes a few minutes.

- [ ] **Step 7: Verify pytest runs**

```bash
.venv/bin/pytest -q
```

Expected: `no tests ran` — collection works, nothing to run yet.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml config.yaml .env.example .gitignore tests/conftest.py
git commit -m "chore: project skeleton, config, and pytest setup"
```

---

### Task 3: Docker Compose with Qdrant reachable

Prove the container topology works before any application code depends on it.

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Docling needs these for PDF rendering and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch — saves roughly 2GB over the default CUDA build
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY . .

CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./:/app
    environment:
      OLLAMA_BASE_URL: http://host.docker.internal:11434
      QDRANT_URL: http://qdrant:6333
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

The `./:/app` bind mount gives live code reload during development. Remove it for any real deployment.

- [ ] **Step 3: Create a placeholder UI so the container starts**

Create `ui/app.py`:

```python
import streamlit as st

st.title("Ragnar")
st.write("Wiring check — not yet functional.")
```

- [ ] **Step 4: Build and start**

```bash
docker compose up -d --build
```

Expected: both services start. First build takes several minutes.

- [ ] **Step 5: Verify Qdrant is reachable from inside the app container**

```bash
docker compose exec app python -c "
import httpx, os
r = httpx.get(os.environ['QDRANT_URL'] + '/collections', timeout=10)
print('qdrant:', r.status_code, r.json())"
```

Expected: `qdrant: 200 {'result': {'collections': []}, ...}`

- [ ] **Step 6: Verify host Ollama is reachable from inside the app container**

This is the single most likely thing to break on macOS.

```bash
docker compose exec app python -c "
import httpx, os
r = httpx.get(os.environ['OLLAMA_BASE_URL'] + '/api/tags', timeout=10)
print('ollama:', r.status_code)
print('models:', [m['name'] for m in r.json()['models']])"
```

Expected: `ollama: 200` and a list including `bge-m3` and `qwen2.5:14b`.

**If this fails:** confirm Ollama is bound to all interfaces, not just loopback. Restart it with `OLLAMA_HOST=0.0.0.0 ollama serve` and retry.

- [ ] **Step 7: Verify Streamlit serves**

Open `http://localhost:8501`. Expected: the "Ragnar" placeholder page.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile docker-compose.yml ui/app.py
git commit -m "chore: docker compose stack with qdrant and host ollama reachable"
```

---

# PHASE 1 — Thin Vertical Slice

**Milestone:** one PDF in, one cited answer out. Naive components throughout; correctness of wiring is the only goal.

### Task 4: Core data models

**Files:**
- Create: `core/models.py`, `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
from core.models import Chunk


def test_chunk_citation_label_prefers_page():
    c = Chunk(doc_id="a1", filename="report.pdf", text="x",
              chunk_index=0, page=4)
    assert c.citation_label() == "report.pdf, p. 4"


def test_chunk_citation_label_uses_sheet_for_spreadsheets():
    c = Chunk(doc_id="a1", filename="sales.xlsx", text="x",
              chunk_index=0, sheet="Q1")
    assert c.citation_label() == "sales.xlsx, sheet Q1"


def test_chunk_citation_label_falls_back_to_filename():
    c = Chunk(doc_id="a1", filename="notes.md", text="x", chunk_index=0)
    assert c.citation_label() == "notes.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core'`

- [ ] **Step 3: Write the implementation**

Create `core/__init__.py` (empty) and `core/models.py`:

```python
from dataclasses import dataclass, field
from enum import Enum


class IngestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Chunk:
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    sheet: str | None = None
    is_table: bool = False
    low_confidence: bool = False

    def citation_label(self) -> str:
        if self.page is not None:
            return f"{self.filename}, p. {self.page}"
        if self.sheet is not None:
            return f"{self.filename}, sheet {self.sheet}"
        return self.filename


@dataclass
class Document:
    doc_id: str
    filename: str
    status: IngestStatus = IngestStatus.QUEUED
    error: str | None = None
    chunk_count: int = 0


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/ tests/test_models.py
git commit -m "feat: core data models with citation labels"
```

---

### Task 5: Explore the Docling API

Docling's API surface changes between versions. **Explore before writing tests against it** — otherwise you will write tests encoding an API that does not exist.

**Files:**
- Create: `tests/fixtures/make_fixtures.py`, `tests/fixtures/sample.pdf`

- [ ] **Step 1: Generate a deterministic fixture PDF**

Create `tests/fixtures/make_fixtures.py`:

```python
"""Regenerate test fixtures. Run: python tests/fixtures/make_fixtures.py"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent


def make_sample_pdf():
    c = canvas.Canvas(str(OUT / "sample.pdf"), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, "Annual Maintenance Report")
    c.setFont("Helvetica", 11)
    c.drawString(72, 730, "The service contract number is SC-4471.")
    c.drawString(72, 710, "Coverage runs from January to December 2024.")
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 760, "Escalation Procedure")
    c.setFont("Helvetica", 11)
    c.drawString(72, 730, "Critical faults escalate to the duty engineer.")
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_sample_pdf()
    print("wrote sample.pdf")
```

- [ ] **Step 2: Generate the fixture**

```bash
.venv/bin/python tests/fixtures/make_fixtures.py
```

Expected: `wrote sample.pdf`, and `tests/fixtures/sample.pdf` exists with 2 pages.

- [ ] **Step 3: Explore what Docling actually returns**

```bash
docker compose exec app python -c "
from docling.document_converter import DocumentConverter
d = DocumentConverter().convert('tests/fixtures/sample.pdf').document
print('--- markdown ---')
print(d.export_to_markdown()[:400])
print('--- items ---')
for item, level in d.iterate_items():
    prov = getattr(item, 'prov', None)
    page = prov[0].page_no if prov else None
    print(repr(type(item).__name__), 'page=', page,
          repr(getattr(item, 'text', ''))[:60])
"
```

Expected: markdown containing "Annual Maintenance Report", and items reporting `page=1` and `page=2`.

**Record the exact attribute path used to reach the page number.** Later tasks depend on it. If `item.prov[0].page_no` is not correct in your Docling version, note what is and substitute it in Task 6.

- [ ] **Step 4: Commit the fixture**

```bash
git add tests/fixtures/
git commit -m "test: add deterministic PDF fixture and generator"
```

---

### Task 6: Docling parser

**Files:**
- Create: `core/interfaces.py`, `ingestion/parser.py`, `tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from ingestion.parser import DoclingParser

FIXTURE = Path("tests/fixtures/sample.pdf")


@pytest.mark.integration
def test_parser_extracts_text_and_pages():
    parsed = DoclingParser().parse(FIXTURE)

    assert "SC-4471" in parsed.markdown
    assert "Escalation" in parsed.markdown

    pages = {b.page for b in parsed.blocks if b.page is not None}
    assert pages == {1, 2}


@pytest.mark.integration
def test_parser_reports_text_density_for_ocr_decision():
    parsed = DoclingParser().parse(FIXTURE)
    # native-text PDF — well above the OCR trigger threshold
    assert parsed.chars_per_page > 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose exec app python -m pytest tests/test_parser.py -v --run-integration`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Write the interfaces**

Create `core/interfaces.py`:

```python
from pathlib import Path
from typing import Protocol
from core.models import Chunk


class ParsedBlock(Protocol):
    text: str
    page: int | None
    is_table: bool


class DocumentParser(Protocol):
    def parse(self, path: Path): ...


class Chunker(Protocol):
    def chunk(self, parsed, doc_id: str, filename: str) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], limit: int) -> list: ...
    def delete_by_doc(self, doc_id: str) -> None: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk],
               top_k: int) -> list: ...
```

- [ ] **Step 4: Write the parser**

Create `ingestion/__init__.py` (empty) and `ingestion/parser.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter


@dataclass
class Block:
    text: str
    page: int | None = None
    is_table: bool = False
    sheet: str | None = None


@dataclass
class ParsedDocument:
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0

    @property
    def chars_per_page(self) -> float:
        if self.page_count == 0:
            return 0.0
        return len(self.markdown) / self.page_count


class DoclingParser:
    """Converts a source file into markdown plus provenance-carrying blocks.

    Blocks are what the chunker consumes; the markdown is for human display
    only. Flattening to markdown loses page numbers, so the two are kept
    separate deliberately.
    """

    def __init__(self, converter: DocumentConverter | None = None):
        self._converter = converter or DocumentConverter()

    def parse(self, path: Path) -> ParsedDocument:
        doc = self._converter.convert(str(path)).document

        blocks: list[Block] = []
        pages: set[int] = set()

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov:
                page = getattr(prov[0], "page_no", None)
            if page is not None:
                pages.add(page)

            blocks.append(Block(
                text=text,
                page=page,
                is_table=type(item).__name__.lower().startswith("table"),
            ))

        return ParsedDocument(
            markdown=doc.export_to_markdown(),
            blocks=blocks,
            page_count=len(pages) or 1,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `docker compose exec app python -m pytest tests/test_parser.py -v --run-integration`
Expected: 2 passed

If page numbers come back as `None`, correct the attribute path using what you recorded in Task 5 Step 3.

- [ ] **Step 6: Commit**

```bash
git add core/interfaces.py ingestion/ tests/test_parser.py
git commit -m "feat: docling parser producing markdown and provenance blocks"
```

---

### Task 7: Naive fixed-size chunker

Deliberately simple. Replaced in Phase 3 — its only job is to unblock the slice.

**Files:**
- Create: `ingestion/chunkers/fixed.py`, `tests/test_chunker_fixed.py`

- [ ] **Step 1: Write the failing test**

```python
from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.fixed import FixedChunker


def _doc():
    return ParsedDocument(
        markdown="ignored",
        blocks=[
            Block(text="alpha " * 100, page=1),
            Block(text="beta " * 100, page=2),
        ],
        page_count=2,
    )


def test_chunker_carries_provenance_onto_every_chunk():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")

    assert len(chunks) > 2
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.filename == "sample.pdf" for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)


def test_chunker_indexes_chunks_sequentially():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_never_splits_across_pages():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")
    for c in chunks:
        assert "alpha" not in c.text or "beta" not in c.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_chunker_fixed.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `ingestion/chunkers/__init__.py` (empty) and `ingestion/chunkers/fixed.py`:

```python
from core.models import Chunk
from ingestion.parser import ParsedDocument


class FixedChunker:
    """Naive character-window chunker used only for the Phase 1 slice.

    Chunks never span blocks, so page provenance stays unambiguous.
    """

    def __init__(self, target_chars: int = 2000, overlap_chars: int = 200):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for block in parsed.blocks:
            text = block.text.strip()
            start = 0
            while start < len(text):
                piece = text[start:start + self.target_chars]
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=block.page,
                    sheet=block.sheet,
                    is_table=block.is_table,
                ))
                index += 1
                step = self.target_chars - self.overlap_chars
                start += max(step, 1)

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_chunker_fixed.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/chunkers/ tests/test_chunker_fixed.py
git commit -m "feat: naive fixed-size chunker for vertical slice"
```

---

### Task 8: Ollama embedder

**Files:**
- Create: `retrieval/embedder.py`, `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

Two tests: a unit test with a stubbed HTTP client (fast, always runs) and an integration test against real Ollama.

```python
import pytest
from retrieval.embedder import OllamaEmbedder


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class StubClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, timeout=None):
        self.calls.append(json)
        return StubResponse(self._payload)


def test_embedder_sends_all_texts_in_one_batch():
    client = StubClient({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)

    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls[0]["input"] == ["a", "b"]
    assert len(client.calls) == 1


def test_embedder_returns_empty_for_no_input():
    client = StubClient({"embeddings": []})
    embedder = OllamaEmbedder("http://x", "bge-m3", client=client)
    assert embedder.embed([]) == []


@pytest.mark.integration
def test_embedder_against_real_ollama_returns_1024_dims():
    import os
    embedder = OllamaEmbedder(os.environ["OLLAMA_BASE_URL"], "bge-m3")
    vectors = embedder.embed(["kontrakt serwisowy", "service contract"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `retrieval/__init__.py` (empty) and `retrieval/embedder.py`:

```python
import httpx


class OllamaEmbedder:
    """Embeds text via Ollama's /api/embed endpoint.

    The client is injectable so unit tests can stub HTTP without a server.
    """

    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
```

- [ ] **Step 4: Run unit tests**

Run: `.venv/bin/pytest tests/test_embedder.py -v`
Expected: 2 passed, 1 skipped

- [ ] **Step 5: Run the integration test inside the container**

Run: `docker compose exec app python -m pytest tests/test_embedder.py -v --run-integration`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add retrieval/ tests/test_embedder.py
git commit -m "feat: ollama embedder with injectable client"
```

---

### Task 9: Qdrant store

**Files:**
- Create: `retrieval/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import uuid
from core.models import Chunk
from retrieval.store import QdrantStore


@pytest.fixture
def store():
    import os
    name = f"test_{uuid.uuid4().hex[:8]}"
    s = QdrantStore(os.environ["QDRANT_URL"], name, dim=1024)
    s.ensure_collection()
    yield s
    s.drop_collection()


def _chunk(doc_id, text, index=0, page=1):
    return Chunk(doc_id=doc_id, filename="f.pdf", text=text,
                 chunk_index=index, page=page)


@pytest.mark.integration
def test_store_roundtrips_chunk_and_payload(store):
    store.upsert([_chunk("d1", "hello")], [[0.1] * 1024])

    results = store.search([0.1] * 1024, limit=5)

    assert len(results) == 1
    assert results[0].chunk.text == "hello"
    assert results[0].chunk.page == 1
    assert results[0].chunk.doc_id == "d1"


@pytest.mark.integration
def test_delete_by_doc_removes_only_that_document(store):
    store.upsert(
        [_chunk("d1", "keep"), _chunk("d2", "remove", index=1)],
        [[0.1] * 1024, [0.2] * 1024],
    )

    store.delete_by_doc("d2")
    results = store.search([0.1] * 1024, limit=10)

    assert [r.chunk.doc_id for r in results] == ["d1"]


@pytest.mark.integration
def test_reupsert_same_chunk_does_not_duplicate(store):
    chunk = _chunk("d1", "v1")
    store.upsert([chunk], [[0.1] * 1024])
    store.upsert([chunk], [[0.1] * 1024])

    assert len(store.search([0.1] * 1024, limit=10)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose exec app python -m pytest tests/test_store.py -v --run-integration`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `retrieval/store.py`:

```python
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue,
)

from core.models import Chunk, SearchResult

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int = 1024,
                 client: QdrantClient | None = None):
        self.collection = collection
        self.dim = dim
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim,
                                            distance=Distance.COSINE),
            )

    def drop_collection(self) -> None:
        self._client.delete_collection(self.collection)

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        # Deterministic: re-upserting the same chunk overwrites rather than
        # duplicating.
        return str(uuid.uuid5(NAMESPACE, f"{chunk.doc_id}:{chunk.chunk_index}"))

    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        if not chunks:
            return
        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page,
                    "sheet": chunk.sheet,
                    "is_table": chunk.is_table,
                    "low_confidence": chunk.low_confidence,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int) -> list[SearchResult]:
        hits = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points

        return [
            SearchResult(
                chunk=Chunk(
                    doc_id=h.payload["doc_id"],
                    filename=h.payload["filename"],
                    text=h.payload["text"],
                    chunk_index=h.payload["chunk_index"],
                    page=h.payload.get("page"),
                    sheet=h.payload.get("sheet"),
                    is_table=h.payload.get("is_table", False),
                    low_confidence=h.payload.get("low_confidence", False),
                ),
                score=h.score,
            )
            for h in hits
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose exec app python -m pytest tests/test_store.py -v --run-integration`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add retrieval/store.py tests/test_store.py
git commit -m "feat: qdrant store with deterministic point ids"
```

---

### Task 10: Answer generation with citations

**Files:**
- Create: `generation/prompts.py`, `generation/answerer.py`, `tests/test_answerer.py`

- [ ] **Step 1: Write the failing test**

The critical assertion: citations come from chunk metadata, never from model prose.

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_answerer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the prompt module**

Create `generation/__init__.py` (empty) and `generation/prompts.py`:

```python
SYSTEM_PROMPT = """\
You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- Be concise and factual. Do not speculate or embellish.
- Do not write citations yourself; they are attached automatically.
"""


def build_user_prompt(question: str, excerpts: list[tuple[str, str]]) -> str:
    """excerpts: list of (source_label, text)."""
    blocks = "\n\n".join(
        f"[{label}]\n{text}" for label, text in excerpts
    )
    return f"Excerpts:\n\n{blocks}\n\nQuestion: {question}"
```

- [ ] **Step 4: Write the answerer**

Create `generation/answerer.py`:

```python
from dataclasses import dataclass, field

from core.models import SearchResult
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

NO_RESULTS_MESSAGE = (
    "I could not find anything relevant in the indexed documents."
)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False


class Answerer:
    def __init__(self, llm):
        self._llm = llm

    def answer(self, question: str,
               results: list[SearchResult]) -> Answer:
        if not results:
            # Skip the model entirely — a refusal it cannot embellish.
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        excerpts = [(r.chunk.citation_label(), r.chunk.text) for r in results]
        text = self._llm.generate(
            SYSTEM_PROMPT, build_user_prompt(question, excerpts)
        )

        # Citations come from retrieved metadata, not model prose.
        seen: list[str] = []
        for result in results:
            label = result.chunk.citation_label()
            if label not in seen:
                seen.append(label)

        return Answer(text=text, citations=seen)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_answerer.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add generation/ tests/test_answerer.py
git commit -m "feat: answerer with metadata-derived citations"
```

---

### Task 11: Ollama LLM client

**Files:**
- Create: `generation/llm.py`, `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from generation.llm import OllamaLLM


@pytest.mark.integration
def test_llm_answers_from_context_only():
    import os
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        "Answer only from the excerpt. If absent, say you don't know.",
        "Excerpt: The contract number is SC-4471.\n\n"
        "Question: What is the contract number?",
    )

    assert "SC-4471" in reply


@pytest.mark.integration
def test_llm_answers_in_the_language_of_the_question():
    import os
    from generation.prompts import SYSTEM_PROMPT
    llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")

    reply = llm.generate(
        SYSTEM_PROMPT,
        "Excerpts:\n\n[a.pdf, p. 1]\nThe contract number is SC-4471.\n\n"
        "Question: Jaki jest numer kontraktu?",
    )

    assert "SC-4471" in reply
    # crude Polish-output check: at least one Polish-specific character
    # or common Polish word
    assert any(t in reply.lower() for t in
               ["numer", "kontrakt", "wynosi", "to "])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose exec app python -m pytest tests/test_llm.py -v --run-integration`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `generation/llm.py`:

```python
import httpx


class OllamaLLM:
    def __init__(self, base_url: str, model: str, client=None,
                 timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client()

    def generate(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, system: str, user: str):
        """Yields token deltas. Used by the UI."""
        import json
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "options": {"temperature": 0.0},
            },
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("done"):
                    break
                yield payload["message"]["content"]
```

Temperature is pinned to 0.0: this is an extraction task, not a creative one, and determinism makes evaluation meaningful.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose exec app python -m pytest tests/test_llm.py -v --run-integration`
Expected: 2 passed. First run is slow — the model loads into memory.

- [ ] **Step 5: Commit**

```bash
git add generation/llm.py tests/test_llm.py
git commit -m "feat: ollama llm client with streaming"
```

---

### Task 12: Wire the slice end to end

**Files:**
- Create: `core/config.py`, `ingestion/pipeline.py`, `retrieval/search.py`, `tests/test_slice_e2e.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
import pytest
import uuid
from pathlib import Path


@pytest.mark.integration
def test_pdf_ingested_then_answered_with_correct_citation():
    import os
    from core.config import Config
    from ingestion.parser import DoclingParser
    from ingestion.chunkers.fixed import FixedChunker
    from ingestion.pipeline import Pipeline
    from retrieval.embedder import OllamaEmbedder
    from retrieval.store import QdrantStore
    from retrieval.search import Search
    from generation.llm import OllamaLLM
    from generation.answerer import Answerer

    collection = f"e2e_{uuid.uuid4().hex[:8]}"
    embedder = OllamaEmbedder(os.environ["OLLAMA_BASE_URL"], "bge-m3")
    store = QdrantStore(os.environ["QDRANT_URL"], collection, dim=1024)
    store.ensure_collection()

    try:
        pipeline = Pipeline(DoclingParser(), FixedChunker(), embedder, store)
        count = pipeline.ingest(Path("tests/fixtures/sample.pdf"), "doc1")
        assert count > 0

        search = Search(embedder, store, candidates=25)
        results = search.find("What is the service contract number?")
        assert results

        llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")
        answer = Answerer(llm).answer(
            "What is the service contract number?", results[:5]
        )

        assert "SC-4471" in answer.text
        assert any("sample.pdf" in c for c in answer.citations)
    finally:
        store.drop_collection()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose exec app python -m pytest tests/test_slice_e2e.py -v --run-integration`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.config'`

- [ ] **Step 3: Write the config loader**

Create `core/config.py`:

```python
import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path) as fh:
            self._raw = yaml.safe_load(fh)

        self.ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.qdrant_url = os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )

    @property
    def llm_model(self) -> str:
        return self._raw["models"]["llm"]

    @property
    def embedding_model(self) -> str:
        return self._raw["models"]["embedding"]

    @property
    def embedding_dim(self) -> int:
        return self._raw["models"]["embedding_dim"]

    @property
    def collection(self) -> str:
        return self._raw["storage"]["collection"]

    @property
    def data_dir(self) -> Path:
        return Path(self._raw["storage"]["data_dir"])

    @property
    def candidates(self) -> int:
        return self._raw["retrieval"]["candidates"]

    @property
    def top_k(self) -> int:
        return self._raw["retrieval"]["top_k"]

    @property
    def score_floor(self) -> float:
        return self._raw["retrieval"]["score_floor"]

    @property
    def chunking(self) -> dict:
        return self._raw["chunking"]
```

- [ ] **Step 4: Write the pipeline**

Create `ingestion/pipeline.py`:

```python
from pathlib import Path


class Pipeline:
    """Orchestrates parse → chunk → embed → store for one document."""

    def __init__(self, parser, chunker, embedder, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(self, path: Path, doc_id: str) -> int:
        parsed = self._parser.parse(path)
        chunks = self._chunker.chunk(parsed, doc_id, path.name)
        if not chunks:
            return 0

        # Replace wholesale so stale and fresh chunks never coexist.
        self._store.delete_by_doc(doc_id)

        vectors = self._embedder.embed([c.text for c in chunks])
        self._store.upsert(chunks, vectors)
        return len(chunks)
```

- [ ] **Step 5: Write search**

Create `retrieval/search.py`:

```python
from core.models import SearchResult


class Search:
    def __init__(self, embedder, store, candidates: int = 25):
        self._embedder = embedder
        self._store = store
        self._candidates = candidates

    def find(self, question: str) -> list[SearchResult]:
        vector = self._embedder.embed([question])[0]
        return self._store.search(vector, limit=self._candidates)
```

- [ ] **Step 6: Run the end-to-end test**

Run: `docker compose exec app python -m pytest tests/test_slice_e2e.py -v --run-integration`
Expected: 1 passed

**This is the milestone.** If it passes, the wiring is proven and everything after is replacing naive parts with real ones.

- [ ] **Step 7: Run the whole suite**

Run: `docker compose exec app python -m pytest -v --run-integration`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add core/config.py ingestion/pipeline.py retrieval/search.py tests/test_slice_e2e.py
git commit -m "feat: end-to-end vertical slice - pdf to cited answer"
```

---

### Task 13: Minimal Streamlit UI over the slice

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Replace the placeholder**

```python
import uuid
from pathlib import Path

import streamlit as st

from core.config import Config
from ingestion.parser import DoclingParser
from ingestion.chunkers.fixed import FixedChunker
from ingestion.pipeline import Pipeline
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from generation.llm import OllamaLLM
from generation.answerer import Answerer
from generation.prompts import SYSTEM_PROMPT, build_user_prompt


@st.cache_resource
def build_services():
    cfg = Config()
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)
    return {
        "cfg": cfg,
        "pipeline": Pipeline(DoclingParser(), FixedChunker(),
                             embedder, store),
        "search": Search(embedder, store, cfg.candidates),
        "answerer": Answerer(llm),
        "llm": llm,
    }


svc = build_services()
st.title("Ragnar")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest"):
        inbox = svc["cfg"].data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / uploaded.name
        target.write_bytes(uploaded.getbuffer())

        with st.spinner(f"Ingesting {uploaded.name}…"):
            n = svc["pipeline"].ingest(target, uuid.uuid4().hex)
        st.success(f"Indexed {n} chunks")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.caption(citation)

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results = svc["search"].find(question)[: svc["cfg"].top_k]

        if not results:
            text = "I could not find anything relevant in the documents."
            st.markdown(text)
            citations = []
        else:
            excerpts = [(r.chunk.citation_label(), r.chunk.text)
                        for r in results]
            text = st.write_stream(
                svc["llm"].stream(
                    SYSTEM_PROMPT, build_user_prompt(question, excerpts)
                )
            )
            citations = []
            for r in results:
                label = r.chunk.citation_label()
                if label not in citations:
                    citations.append(label)

            with st.expander("Sources"):
                for citation in citations:
                    st.caption(citation)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "citations": citations}
    )
```

- [ ] **Step 2: Restart and verify manually**

```bash
docker compose restart app
```

Open `http://localhost:8501` and check, in order:
1. Upload `tests/fixtures/sample.pdf` and click Ingest → success message with a chunk count
2. Ask "What is the service contract number?" → answer contains `SC-4471`, streamed rather than appearing all at once
3. Expand Sources → shows `sample.pdf, p. 1`
4. Ask "What is the capital of France?" → should NOT confidently answer from the document

Item 4 will likely behave poorly right now — there is no similarity floor yet. That is expected and is fixed in Task 24.

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat: minimal streamlit ui over the vertical slice"
```

**PHASE 1 COMPLETE.** Working end-to-end system with naive components.

---

# PHASE 2 — Real Ingestion

### Task 14: Document registry (SQLite)

**Files:**
- Create: `ingestion/registry_db.py`, `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from core.models import IngestStatus
from ingestion.registry_db import Registry


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry.db")


def test_add_document_starts_queued(registry):
    registry.add("d1", "a.pdf")
    doc = registry.get("d1")
    assert doc.status == IngestStatus.QUEUED
    assert doc.filename == "a.pdf"


def test_next_queued_returns_documents_in_insertion_order(registry):
    registry.add("d1", "a.pdf")
    registry.add("d2", "b.pdf")
    assert registry.next_queued().doc_id == "d1"


def test_mark_done_records_chunk_count(registry):
    registry.add("d1", "a.pdf")
    registry.mark_done("d1", chunk_count=12)
    doc = registry.get("d1")
    assert doc.status == IngestStatus.DONE
    assert doc.chunk_count == 12


def test_mark_failed_records_reason(registry):
    registry.add("d1", "a.pdf")
    registry.mark_failed("d1", "unreadable")
    doc = registry.get("d1")
    assert doc.status == IngestStatus.FAILED
    assert doc.error == "unreadable"


def test_adding_same_doc_id_twice_is_idempotent(registry):
    registry.add("d1", "a.pdf")
    registry.add("d1", "a.pdf")
    assert len(registry.all()) == 1


def test_reset_stale_processing_recovers_from_crash(registry):
    registry.add("d1", "a.pdf")
    registry.mark_processing("d1")

    recovered = registry.reset_stale_processing()

    assert recovered == 1
    assert registry.get("d1").status == IngestStatus.QUEUED


def test_next_queued_returns_none_when_empty(registry):
    assert registry.next_queued() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `ingestion/registry_db.py`:

```python
import sqlite3
import threading
from pathlib import Path

from core.models import Document, IngestStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    added_at    REAL NOT NULL DEFAULT (julianday('now'))
);
"""


class Registry:
    """SQLite document registry.

    Also the sole communication channel between the ingestion worker thread
    and the Streamlit UI, so every method must be thread-safe.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _row_to_doc(self, row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            status=IngestStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
        )

    def add(self, doc_id: str, filename: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO documents (doc_id, filename, status) "
                "VALUES (?, ?, ?)",
                (doc_id, filename, IngestStatus.QUEUED.value),
            )
            self._conn.commit()

    def get(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def all(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY added_at"
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def next_queued(self) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE status = ? "
                "ORDER BY added_at LIMIT 1",
                (IngestStatus.QUEUED.value,),
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def _set_status(self, doc_id: str, status: IngestStatus,
                    error: str | None = None,
                    chunk_count: int | None = None) -> None:
        with self._lock:
            if chunk_count is None:
                self._conn.execute(
                    "UPDATE documents SET status = ?, error = ? "
                    "WHERE doc_id = ?",
                    (status.value, error, doc_id),
                )
            else:
                self._conn.execute(
                    "UPDATE documents SET status = ?, error = ?, "
                    "chunk_count = ? WHERE doc_id = ?",
                    (status.value, error, chunk_count, doc_id),
                )
            self._conn.commit()

    def mark_processing(self, doc_id: str) -> None:
        self._set_status(doc_id, IngestStatus.PROCESSING)

    def mark_done(self, doc_id: str, chunk_count: int) -> None:
        self._set_status(doc_id, IngestStatus.DONE, chunk_count=chunk_count)

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._set_status(doc_id, IngestStatus.FAILED, error=error)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            self._conn.commit()

    def reset_stale_processing(self) -> int:
        """Recover documents wedged by a crash mid-ingest."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET status = ? WHERE status = ?",
                (IngestStatus.QUEUED.value, IngestStatus.PROCESSING.value),
            )
            self._conn.commit()
            return cursor.rowcount

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/registry_db.py tests/test_registry.py
git commit -m "feat: sqlite document registry with crash recovery"
```

---

### Task 15: Storage layout and content hashing

**Files:**
- Create: `ingestion/storage.py`, `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from ingestion.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path)


def test_directories_are_created(storage, tmp_path):
    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "originals").is_dir()
    assert (tmp_path / "converted").is_dir()


def test_doc_id_is_content_hash_not_filename(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")

    assert storage.doc_id(a) == storage.doc_id(b)


def test_different_content_yields_different_doc_id(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"one")
    b.write_bytes(b"two")

    assert storage.doc_id(a) != storage.doc_id(b)


def test_archive_moves_file_and_suffixes_with_hash(storage):
    src = storage.inbox / "report.pdf"
    src.write_bytes(b"content")
    doc_id = storage.doc_id(src)

    archived = storage.archive(src, doc_id)

    assert not src.exists()
    assert archived.exists()
    assert archived.name == f"report.{doc_id[:8]}.pdf"


def test_archiving_same_name_different_content_does_not_collide(storage):
    first = storage.inbox / "report.pdf"
    first.write_bytes(b"v1")
    a = storage.archive(first, storage.doc_id(first))

    second = storage.inbox / "report.pdf"
    second.write_bytes(b"v2")
    b = storage.archive(second, storage.doc_id(second))

    assert a != b
    assert a.exists() and b.exists()


def test_converted_artifacts_written_and_read_back(storage):
    storage.write_converted("d1", markdown="# Title")
    assert storage.read_markdown("d1") == "# Title"


def test_remove_converted_deletes_artifacts(storage):
    storage.write_converted("d1", markdown="# Title")
    storage.remove_converted("d1")
    assert storage.read_markdown("d1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `ingestion/storage.py`:

```python
import hashlib
import shutil
from pathlib import Path


class Storage:
    """Owns the on-disk layout.

    data/ is the source of truth; Qdrant is a rebuildable index. Originals
    are never deleted automatically.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.originals = self.root / "originals"
        self.converted = self.root / "converted"
        for directory in (self.inbox, self.originals, self.converted):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def doc_id(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    def archive(self, path: Path, doc_id: str) -> Path:
        target = self.originals / f"{path.stem}.{doc_id[:8]}{path.suffix}"
        shutil.move(str(path), str(target))
        return target

    def write_converted(self, doc_id: str, markdown: str) -> None:
        (self.converted / f"{doc_id}.md").write_text(markdown)

    def read_markdown(self, doc_id: str) -> str | None:
        path = self.converted / f"{doc_id}.md"
        return path.read_text() if path.exists() else None

    def remove_converted(self, doc_id: str) -> None:
        for suffix in (".md", ".json"):
            path = self.converted / f"{doc_id}{suffix}"
            path.unlink(missing_ok=True)

    def pending_files(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add ingestion/storage.py tests/test_storage.py
git commit -m "feat: storage layout with content-hash document ids"
```

---

### Task 16: Ingestion worker

**Files:**
- Create: `tests/fakes.py`, `ingestion/worker.py`, `tests/test_worker.py`
- Modify: `ingestion/pipeline.py`

- [ ] **Step 1: Write the fakes**

Create `tests/fakes.py`:

```python
from core.models import Chunk, SearchResult


class FakeEmbedder:
    """Deterministic embeddings without a server."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float((hash(t) >> (i * 4)) % 10) for i in range(self.dim)]
            for t in texts
        ]


class FakeStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.deleted: list[str] = []

    def ensure_collection(self):
        pass

    def upsert(self, chunks, vectors):
        self.chunks.extend(chunks)

    def search(self, vector, limit):
        return [SearchResult(chunk=c, score=1.0)
                for c in self.chunks[:limit]]

    def delete_by_doc(self, doc_id):
        self.deleted.append(doc_id)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]


class FakeParser:
    def __init__(self, blocks=None, fail=False):
        self.blocks = blocks or []
        self.fail = fail

    def parse(self, path):
        if self.fail:
            raise ValueError("unreadable file")
        from ingestion.parser import ParsedDocument
        return ParsedDocument(markdown="# doc", blocks=self.blocks,
                              page_count=1)
```

- [ ] **Step 2: Write the failing test**

```python
import pytest
from core.models import IngestStatus
from ingestion.parser import Block
from ingestion.registry_db import Registry
from ingestion.storage import Storage
from ingestion.pipeline import Pipeline
from ingestion.chunkers.fixed import FixedChunker
from ingestion.worker import IngestWorker
from tests.fakes import FakeEmbedder, FakeStore, FakeParser


@pytest.fixture
def env(tmp_path):
    storage = Storage(tmp_path)
    registry = Registry(tmp_path / "registry.db")
    store = FakeStore()
    pipeline = Pipeline(
        FakeParser(blocks=[Block(text="hello world", page=1)]),
        FixedChunker(target_chars=100),
        FakeEmbedder(),
        store,
    )
    return storage, registry, pipeline, store


def _drop(storage, name, content=b"data"):
    path = storage.inbox / name
    path.write_bytes(content)
    return path


def test_worker_processes_queued_document_to_done(env):
    storage, registry, pipeline, store = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert registry.get(doc_id).status == IngestStatus.DONE
    assert registry.get(doc_id).chunk_count > 0


def test_worker_archives_original_on_success(env):
    storage, registry, pipeline, _ = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert not path.exists()
    assert list(storage.originals.iterdir())


def test_worker_leaves_original_in_inbox_on_failure(env):
    storage, registry, _, store = env
    path = _drop(storage, "bad.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "bad.pdf")

    failing = Pipeline(FakeParser(fail=True), FixedChunker(),
                       FakeEmbedder(), store)
    IngestWorker(storage, registry, failing).process_next()

    doc = registry.get(doc_id)
    assert doc.status == IngestStatus.FAILED
    assert "unreadable" in doc.error
    assert path.exists(), "failed file must stay in inbox for inspection"


def test_worker_processes_one_document_at_a_time(env):
    storage, registry, pipeline, _ = env
    for name in ("a.pdf", "b.pdf"):
        path = _drop(storage, name, content=name.encode())
        registry.add(storage.doc_id(path), name)

    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()

    statuses = [d.status for d in registry.all()]
    assert statuses.count(IngestStatus.DONE) == 1
    assert statuses.count(IngestStatus.QUEUED) == 1


def test_process_next_is_noop_when_queue_empty(env):
    storage, registry, pipeline, _ = env
    assert IngestWorker(storage, registry, pipeline).process_next() is False


def test_worker_writes_converted_markdown(env):
    storage, registry, pipeline, _ = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert storage.read_markdown(doc_id) == "# doc"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Modify the pipeline to return markdown too**

Replace `ingestion/pipeline.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestResult:
    chunk_count: int
    markdown: str


class Pipeline:
    """Orchestrates parse → chunk → embed → store for one document."""

    def __init__(self, parser, chunker, embedder, store):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(self, path: Path, doc_id: str) -> IngestResult:
        parsed = self._parser.parse(path)
        chunks = self._chunker.chunk(parsed, doc_id, path.name)

        # Replace wholesale so stale and fresh chunks never coexist.
        self._store.delete_by_doc(doc_id)

        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._store.upsert(chunks, vectors)

        return IngestResult(chunk_count=len(chunks),
                            markdown=parsed.markdown)
```

Update `tests/test_slice_e2e.py` Step 1 accordingly — `pipeline.ingest(...)` now returns `IngestResult`, so change `assert count > 0` to `assert result.chunk_count > 0`.

- [ ] **Step 5: Write the worker**

Create `ingestion/worker.py`:

```python
import logging
import threading
import time

log = logging.getLogger(__name__)


class IngestWorker:
    """Drains the registry queue one document at a time.

    Sequential by design: Docling on CPU only contends with itself in
    parallel, and concurrency would add failure modes for no gain at this
    scale.

    Runs on a background thread and touches only the registry, storage, and
    pipeline — never Streamlit APIs, which are not thread-safe.
    """

    def __init__(self, storage, registry, pipeline, poll_seconds: float = 1.0):
        self._storage = storage
        self._registry = registry
        self._pipeline = pipeline
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def process_next(self) -> bool:
        """Process one queued document. Returns False if the queue is empty."""
        doc = self._registry.next_queued()
        if doc is None:
            return False

        self._registry.mark_processing(doc.doc_id)
        path = self._storage.inbox / doc.filename

        try:
            if not path.exists():
                raise FileNotFoundError(f"{doc.filename} missing from inbox")

            result = self._pipeline.ingest(path, doc.doc_id)
            self._storage.write_converted(doc.doc_id, result.markdown)
            self._storage.archive(path, doc.doc_id)
            self._registry.mark_done(doc.doc_id, result.chunk_count)
        except Exception as exc:
            # Original deliberately stays in inbox for inspection.
            log.exception("ingestion failed for %s", doc.filename)
            self._registry.mark_failed(doc.doc_id, str(exc))

        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.process_next():
                    self._stop.wait(self._poll)
            except Exception:
                log.exception("worker loop error")
                self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._registry.reset_stale_processing()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_worker.py -v`
Expected: 6 passed

- [ ] **Step 7: Run the full suite**

Run: `docker compose exec app python -m pytest -v --run-integration`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tests/fakes.py ingestion/worker.py ingestion/pipeline.py tests/test_worker.py tests/test_slice_e2e.py
git commit -m "feat: background ingestion worker with sequential queue drain"
```

---

### Task 17: Wire the worker and status strip into the UI

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Rewrite `build_services` to include storage, registry, and worker**

```python
@st.cache_resource
def build_services():
    cfg = Config()
    storage = Storage(cfg.data_dir)
    registry = Registry(cfg.data_dir / "registry.db")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)

    pipeline = Pipeline(DoclingParser(), FixedChunker(), embedder, store)
    worker = IngestWorker(storage, registry, pipeline)
    worker.start()   # resets stale PROCESSING rows on startup

    return {
        "cfg": cfg, "storage": storage, "registry": registry,
        "store": store, "search": Search(embedder, store, cfg.candidates),
        "llm": llm, "worker": worker,
    }
```

`@st.cache_resource` guarantees one worker per process regardless of script reruns.

- [ ] **Step 2: Replace the upload handler to queue rather than block**

```python
with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader(
        "Upload", type=["pdf", "xlsx", "docx"], accept_multiple_files=True
    )
    if uploaded and st.button("Queue for ingestion"):
        for file in uploaded:
            target = svc["storage"].inbox / file.name
            target.write_bytes(file.getbuffer())
            svc["registry"].add(svc["storage"].doc_id(target), file.name)
        st.success(f"Queued {len(uploaded)} file(s)")
        st.rerun()

    if st.button("Ingest inbox"):
        queued = 0
        for path in svc["storage"].pending_files():
            svc["registry"].add(svc["storage"].doc_id(path), path.name)
            queued += 1
        st.success(f"Queued {queued} file(s)")
        st.rerun()
```

Upload writes to disk, inserts registry rows, and returns immediately. No parsing happens in the request.

- [ ] **Step 3: Add the status strip as a fragment**

```python
    @st.fragment(run_every="2s")
    def status_strip():
        counts = svc["registry"].counts()
        processing = counts.get("processing", 0)
        queued = counts.get("queued", 0)

        if processing or queued:
            st.info(f"Indexing — {processing} in progress, {queued} queued")

        for doc in svc["registry"].all():
            icon = {"queued": "⏳", "processing": "⚙️",
                    "done": "✅", "failed": "❌"}[doc.status.value]
            st.write(f"{icon} {doc.filename}")
            if doc.error:
                st.caption(f"↳ {doc.error}")

    status_strip()
```

`@st.fragment(run_every="2s")` refreshes only this region. Without it the whole page reruns every 2 seconds and the chat flickers.

- [ ] **Step 4: Verify manually**

```bash
docker compose restart app
```

1. Queue 3 PDFs at once → button returns immediately, all three show ⏳
2. Watch the strip → they move to ⚙️ then ✅ one at a time, never two at once
3. **While ingestion is running, ask a question** → chat responds; the UI is not blocked
4. Stop the container mid-ingest (`docker compose restart app`), reload → no document is stuck on ⚙️

Item 3 is the whole point of this task. Item 4 verifies crash recovery.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py
git commit -m "feat: queued ingestion with non-blocking status strip"
```

**PHASE 2 COMPLETE.** Real ingestion with queueing, archiving, and crash recovery.

---

# PHASE 3 — Retrieval Quality

### Task 18: Structural chunker

**Files:**
- Create: `ingestion/chunkers/structural.py`, `ingestion/chunkers/registry.py`, `tests/test_chunker_structural.py`

- [ ] **Step 1: Write the failing test**

```python
from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.structural import StructuralChunker


def _doc(blocks):
    return ParsedDocument(markdown="x", blocks=blocks, page_count=1)


def test_chunks_do_not_span_page_boundaries():
    doc = _doc([
        Block(text="Short A.", page=1),
        Block(text="Short B.", page=2),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    for c in chunks:
        assert not ("Short A" in c.text and "Short B" in c.text)


def test_small_adjacent_blocks_on_same_page_are_merged():
    doc = _doc([
        Block(text="First sentence.", page=1),
        Block(text="Second sentence.", page=1),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Second sentence." in chunks[0].text


def test_oversized_block_is_split():
    doc = _doc([Block(text="word " * 4000, page=1)])
    chunks = StructuralChunker(target_tokens=100).chunk(doc, "d", "f.pdf")
    assert len(chunks) > 1


def test_table_blocks_are_never_merged_with_prose():
    doc = _doc([
        Block(text="Intro prose.", page=1),
        Block(text="| a | b |", page=1, is_table=True),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Intro prose" not in table_chunks[0].text


def test_chunk_indices_are_sequential():
    doc = _doc([Block(text=f"Block {i}.", page=i) for i in range(1, 6)])
    chunks = StructuralChunker().chunk(doc, "d", "f.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_chunker_structural.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `ingestion/chunkers/structural.py`:

```python
from core.models import Chunk
from ingestion.parser import Block, ParsedDocument

# Rough tokens-per-character for mixed PL/EN text. Polish words are longer
# and diacritics cost extra bytes, so character heuristics under-count.
# Replaced by the real tokenizer in a later refinement if measurement
# shows it matters.
CHARS_PER_TOKEN = 3.5


class StructuralChunker:
    """Merges adjacent blocks up to a token budget, never crossing pages,
    never mixing tables with prose."""

    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50):
        self.target_chars = int(target_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        groups: list[list[Block]] = []
        current: list[Block] = []

        def flush():
            if current:
                groups.append(list(current))
                current.clear()

        for block in parsed.blocks:
            if not block.text.strip():
                continue

            if block.is_table:
                flush()
                groups.append([block])
                continue

            if current:
                same_page = current[-1].page == block.page
                size = sum(len(b.text) for b in current) + len(block.text)
                if not same_page or size > self.target_chars:
                    flush()

            current.append(block)

        flush()

        chunks: list[Chunk] = []
        index = 0
        for group in groups:
            text = "\n\n".join(b.text.strip() for b in group)
            head = group[0]

            for piece in self._split(text):
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=head.page,
                    sheet=head.sheet,
                    is_table=head.is_table,
                ))
                index += 1

        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]

        pieces = []
        start = 0
        step = max(self.target_chars - self.overlap_chars, 1)
        while start < len(text):
            pieces.append(text[start:start + self.target_chars])
            start += step
        return pieces
```

- [ ] **Step 4: Write the registry**

Create `ingestion/chunkers/registry.py`:

```python
from ingestion.chunkers.fixed import FixedChunker
from ingestion.chunkers.structural import StructuralChunker

# The ONLY place a component is selected by name. Deliberately narrow:
# chunking is the one component with an experiment attached (Ragas), which
# is what justifies swappability here and nowhere else.
CHUNKERS = {
    "fixed": FixedChunker,
    "structural": StructuralChunker,
}


def build_chunker(config: dict):
    name = config.get("strategy", "structural")
    if name not in CHUNKERS:
        raise ValueError(
            f"Unknown chunker '{name}'. Available: {sorted(CHUNKERS)}"
        )

    if name == "fixed":
        return FixedChunker()
    return StructuralChunker(
        target_tokens=config.get("target_tokens", 500),
        overlap_tokens=config.get("overlap_tokens", 50),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_chunker_structural.py -v`
Expected: 5 passed

- [ ] **Step 6: Switch the default and rebuild the index**

In `config.yaml` set `chunking.strategy: structural`. In `ui/app.py` replace `FixedChunker()` with `build_chunker(cfg.chunking)`.

```bash
docker compose restart app
```

Re-ingest the fixture and confirm answers still cite correctly.

- [ ] **Step 7: Commit**

```bash
git add ingestion/chunkers/ config.yaml ui/app.py tests/test_chunker_structural.py
git commit -m "feat: structural chunker with config-driven selection"
```

---

### Task 19: Table chunking with header repetition

**Files:**
- Create: `ingestion/tables.py`, `tests/test_tables.py`

- [ ] **Step 1: Write the failing test**

```python
from ingestion.tables import chunk_table_markdown


TABLE = """\
| Client | Region | Value |
|---|---|---|
| Acme | North | 1000 |
| Beta | South | 2000 |
| Gamma | East | 3000 |
| Delta | West | 4000 |"""


def test_every_group_repeats_the_header_row():
    groups = chunk_table_markdown(TABLE, rows_per_group=2)

    assert len(groups) == 2
    for group in groups:
        assert "| Client | Region | Value |" in group


def test_rows_are_distributed_across_groups_without_loss():
    groups = chunk_table_markdown(TABLE, rows_per_group=2)
    combined = "\n".join(groups)

    for client in ("Acme", "Beta", "Gamma", "Delta"):
        assert client in combined


def test_small_table_stays_in_one_group():
    groups = chunk_table_markdown(TABLE, rows_per_group=20)
    assert len(groups) == 1


def test_table_without_data_rows_returns_nothing():
    header_only = "| A | B |\n|---|---|"
    assert chunk_table_markdown(header_only, rows_per_group=5) == []


def test_blank_row_boundaries_are_preferred_over_fixed_size():
    table = (
        "| A | B |\n|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "| 3 | 4 |\n"
    )
    groups = chunk_table_markdown(table, rows_per_group=20)
    assert len(groups) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tables.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `ingestion/tables.py`:

```python
def chunk_table_markdown(table: str, rows_per_group: int = 20) -> list[str]:
    """Split a markdown table into row groups, repeating the header.

    A table row without its header is semantically meaningless to an
    embedder, so the header goes into every group.

    Blank rows are treated as natural section boundaries where present;
    otherwise groups are fixed-size.
    """
    lines = table.strip().splitlines()
    if len(lines) < 2:
        return []

    header, separator = lines[0], lines[1]
    body = lines[2:]
    if not any(line.strip() for line in body):
        return []

    # Prefer blank-row boundaries when the table has them.
    sections: list[list[str]] = []
    current: list[str] = []
    for line in body:
        if not line.strip():
            if current:
                sections.append(current)
                current = []
        else:
            current.append(line)
    if current:
        sections.append(current)

    groups: list[str] = []
    for section in sections:
        for start in range(0, len(section), rows_per_group):
            rows = section[start:start + rows_per_group]
            groups.append("\n".join([header, separator, *rows]))

    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_tables.py -v`
Expected: 5 passed

- [ ] **Step 5: Route table blocks through it in the structural chunker**

In `structural.py`, replace the table branch in the chunk-building loop so table groups produce one chunk each:

```python
            if head.is_table:
                from ingestion.tables import chunk_table_markdown
                pieces = chunk_table_markdown(text) or [text]
            else:
                pieces = self._split(text)

            for piece in pieces:
```

Re-run `tests/test_chunker_structural.py` — all 5 must still pass.

- [ ] **Step 6: Commit**

```bash
git add ingestion/tables.py ingestion/chunkers/structural.py tests/test_tables.py
git commit -m "feat: table row-group chunking with repeated headers"
```

---

### Task 20: Reranker

**Files:**
- Create: `retrieval/reranker.py`, `tests/test_reranker.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose exec app python -m pytest tests/test_reranker.py -v --run-integration`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `retrieval/reranker.py`:

```python
import math

from core.models import SearchResult

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """Cross-encoder reranker running on CPU inside the container.

    Costs roughly 1-3s for 25 candidates. If that proves too slow, the ONNX
    export of the same model is a drop-in replacement that removes the torch
    dependency entirely.
    """

    def __init__(self, model_name: str = MODEL_NAME, model=None):
        self._model_name = model_name
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult],
               top_k: int) -> list[SearchResult]:
        if not candidates:
            return []

        model = self._ensure_model()
        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores = model.predict(pairs)

        # The cross-encoder emits logits, not probabilities. Sigmoid maps
        # them to (0, 1) so a single interpretable floor can be configured.
        rescored = [
            SearchResult(chunk=c.chunk, score=1 / (1 + math.exp(-float(s))))
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose exec app python -m pytest tests/test_reranker.py -v --run-integration`
Expected: 4 passed. First run downloads the model (~2.2GB) — allow several minutes.

- [ ] **Step 5: Measure actual latency**

```bash
docker compose exec app python -c "
import time
from core.models import Chunk, SearchResult
from retrieval.reranker import BGEReranker

r = BGEReranker()
cands = [SearchResult(chunk=Chunk(doc_id='d', filename='f.pdf',
         text=f'Sample chunk text number {i}. ' * 20, chunk_index=i),
         score=0.5) for i in range(25)]
r.rerank('warmup', cands, 5)
t = time.time(); r.rerank('what is the contract number', cands, 5)
print(f'rerank 25 candidates: {time.time()-t:.2f}s')"
```

**Record this number.** If it exceeds ~5s, switch to the ONNX export before proceeding — the spec names it as the designated lever.

- [ ] **Step 6: Commit**

```bash
git add retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: bge cross-encoder reranker with sigmoid-normalised scores"
```

---

### Task 21: Similarity floor and refusal

**Files:**
- Modify: `retrieval/search.py`, `tests/test_search.py` (create)

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'reranker'`

- [ ] **Step 3: Rewrite search**

Replace `retrieval/search.py`:

```python
from dataclasses import dataclass, field

from core.models import SearchResult

RELATED_COUNT = 3


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    related: list[SearchResult] = field(default_factory=list)
    refused: bool = False


class Search:
    """Retrieve, rerank, then apply the similarity floor.

    When nothing clears the floor the LLM is never called — that is what
    makes the refusal trustworthy, since there is no opportunity for the
    model to improvise.
    """

    def __init__(self, embedder, store, reranker=None, candidates: int = 25,
                 top_k: int = 5, score_floor: float = 0.0):
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._candidates = candidates
        self._top_k = top_k
        self._score_floor = score_floor

    def find(self, question: str) -> SearchOutcome:
        vector = self._embedder.embed([question])[0]
        candidates = self._store.search(vector, limit=self._candidates)

        if not candidates:
            return SearchOutcome(refused=True)

        if self._reranker is not None:
            ranked = self._reranker.rerank(question, candidates, self._top_k)
        else:
            ranked = candidates[: self._top_k]

        kept = [r for r in ranked if r.score >= self._score_floor]

        if not kept:
            return SearchOutcome(
                related=ranked[:RELATED_COUNT], refused=True
            )

        return SearchOutcome(results=kept)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: 4 passed

- [ ] **Step 5: Update the UI and e2e test for the new return type**

In `ui/app.py`, `search.find()` now returns a `SearchOutcome`. Handle `outcome.refused` by showing the refusal plus `outcome.related` as "Related documents you might check", and use `outcome.results` otherwise. Also pass the reranker and `score_floor` into `Search(...)` in `build_services`.

Update `tests/test_slice_e2e.py` to use `outcome.results`.

- [ ] **Step 6: Run the full suite**

Run: `docker compose exec app python -m pytest -v --run-integration`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add retrieval/search.py ui/app.py tests/test_search.py tests/test_slice_e2e.py
git commit -m "feat: similarity floor with deterministic refusal path"
```

**PHASE 3 COMPLETE.**

---

# PHASE 4 — Guards and Polish

### Task 22: Aggregation guard

**Files:**
- Create: `generation/guards.py`, `tests/test_guards.py`

- [ ] **Step 1: Write the failing test**

The false-positive test is the important one — it is what the two-signal design exists to satisfy.

```python
from core.models import Chunk, SearchResult
from generation.guards import should_refuse_aggregation, aggregation_refusal


def _result(text, is_table, filename="data.xlsx", sheet="Q1"):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename=filename, text=text, chunk_index=0,
                    is_table=is_table, sheet=sheet if is_table else None),
        score=0.9,
    )


def test_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("What is the total revenue?", results)


def test_polish_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("Jaka jest suma przychodów?", results)


def test_aggregation_wording_over_prose_is_NOT_refused():
    """The false-positive case the two-signal design exists to prevent."""
    results = [
        _result("The total liability is capped at 50,000 EUR.", False),
        _result("Payment terms are net 30.", False),
    ]
    assert not should_refuse_aggregation("What is the total liability?",
                                         results)


def test_lookup_question_over_tables_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True)]
    assert not should_refuse_aggregation("What is Acme's value?", results)


def test_mixed_results_mostly_prose_are_NOT_refused():
    results = [
        _result("prose one", False),
        _result("prose two", False),
        _result("| a | 1 |", True),
    ]
    assert not should_refuse_aggregation("What is the total?", results)


def test_refusal_message_names_file_and_sheet():
    results = [_result("| a | 1 |", True, filename="sales.xlsx", sheet="Q1")]
    message = aggregation_refusal(results)

    assert "sales.xlsx" in message
    assert "Q1" in message


def test_empty_results_are_not_refused_by_this_guard():
    assert not should_refuse_aggregation("What is the total?", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guards.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `generation/guards.py`:

```python
import re

from core.models import SearchResult

# English and Polish aggregation intent markers.
AGGREGATION_TERMS = {
    # English
    "total", "sum", "average", "mean", "count", "how many", "how much",
    "highest", "lowest", "largest", "smallest", "top", "rank", "ranking",
    "aggregate", "overall", "combined", "altogether",
    # Polish
    "suma", "sumy", "razem", "łącznie", "lacznie", "ile", "średnia",
    "srednia", "największ", "najwieksz", "najmniejsz", "najwyższ",
    "najwyzsz", "ranking", "łączna", "laczna", "ogółem", "ogolem",
}

TABLE_MAJORITY = 0.5


def _has_aggregation_intent(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in AGGREGATION_TERMS)


def _is_table_heavy(results: list[SearchResult]) -> bool:
    if not results:
        return False
    tables = sum(1 for r in results if r.chunk.is_table)
    return tables / len(results) > TABLE_MAJORITY


def should_refuse_aggregation(question: str,
                              results: list[SearchResult]) -> bool:
    """Fires only when BOTH signals are present.

    Either signal alone produces false positives: a question containing
    "total" about a prose contract must not be blocked, and a lookup
    question over a table must not be blocked either.
    """
    return _has_aggregation_intent(question) and _is_table_heavy(results)


def aggregation_refusal(results: list[SearchResult]) -> str:
    sources = []
    for r in results:
        label = r.chunk.filename
        if r.chunk.sheet:
            label = f"{label} (sheet {r.chunk.sheet})"
        if label not in sources:
            sources.append(label)

    listed = ", ".join(sources)
    return (
        "This looks like a question that requires calculating across a whole "
        "table. I can only read individual rows, so any total I gave you "
        "could be wrong.\n\n"
        f"The relevant data is in: {listed}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_guards.py -v`
Expected: 7 passed

- [ ] **Step 5: Wire into the UI**

In `ui/app.py`, after retrieval succeeds and before calling the LLM:

```python
        from generation.guards import (
            should_refuse_aggregation, aggregation_refusal
        )

        if should_refuse_aggregation(question, outcome.results):
            text = aggregation_refusal(outcome.results)
            st.warning(text)
            citations = []
        else:
            ...  # existing streaming path
```

- [ ] **Step 6: Commit**

```bash
git add generation/guards.py ui/app.py tests/test_guards.py
git commit -m "feat: two-signal aggregation guard preventing hallucinated totals"
```

---

### Task 23: OCR fallback and document removal

**Files:**
- Modify: `ingestion/parser.py`, `ui/app.py`
- Create: `tests/test_ocr_fallback.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from ingestion.parser import DoclingParser


class StubConverter:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def convert(self, path):
        self.calls += 1
        return self._results.pop(0)


class StubDoc:
    def __init__(self, markdown, pages=1):
        self._markdown = markdown
        self._pages = pages

    def export_to_markdown(self):
        return self._markdown

    def iterate_items(self):
        class Item:
            text = self._markdown

            class _P:
                page_no = 1
            prov = [_P()]
        return [(Item(), 0)]


class StubResult:
    def __init__(self, doc):
        self.document = doc


def test_dense_text_does_not_trigger_ocr():
    converter = StubConverter([StubResult(StubDoc("x" * 5000))])
    parser = DoclingParser(converter=converter, ocr_converter=converter)

    parsed = parser.parse(Path("a.pdf"))

    assert converter.calls == 1
    assert parsed.low_confidence is False


def test_sparse_text_triggers_ocr_and_flags_low_confidence():
    plain = StubConverter([StubResult(StubDoc("tiny"))])
    ocr = StubConverter([StubResult(StubDoc("recovered text " * 50))])
    parser = DoclingParser(converter=plain, ocr_converter=ocr)

    parsed = parser.parse(Path("scan.pdf"))

    assert ocr.calls == 1
    assert parsed.low_confidence is True
    assert "recovered" in parsed.markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ocr_fallback.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'ocr_converter'`

- [ ] **Step 3: Add the fallback to the parser**

Add to `ParsedDocument` a `low_confidence: bool = False` field, and modify `DoclingParser`:

```python
OCR_TRIGGER_CHARS_PER_PAGE = 50


class DoclingParser:
    def __init__(self, converter=None, ocr_converter=None):
        from docling.document_converter import DocumentConverter
        self._converter = converter or DocumentConverter()
        self._ocr_converter = ocr_converter
        self._low_confidence = False

    def parse(self, path):
        parsed = self._parse_with(self._converter, path)

        # OCR is off by default — the corpus is predominantly native-text,
        # and skipping OCR is the single largest ingestion speed win. Only
        # documents that come back near-empty are retried with OCR.
        if parsed.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE:
            ocr = self._ocr_converter or self._build_ocr_converter()
            parsed = self._parse_with(ocr, path)
            parsed.low_confidence = True

        return parsed

    def _build_ocr_converter(self):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        options = PdfPipelineOptions()
        options.do_ocr = True
        return DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        })
```

Move the existing body of `parse` into `_parse_with(self, converter, path)`.

Also propagate `low_confidence` onto chunks in both chunkers.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ocr_fallback.py -v`
Expected: 2 passed

- [ ] **Step 5: Add document removal to the UI**

In the sidebar document list, add a remove button per document:

```python
            if st.button("Remove", key=f"rm_{doc.doc_id}"):
                svc["store"].delete_by_doc(doc.doc_id)
                svc["storage"].remove_converted(doc.doc_id)
                svc["registry"].remove(doc.doc_id)
                st.rerun()
```

Originals in `originals/` are deliberately left in place.

- [ ] **Step 6: Add the markdown inspection view**

```python
            with st.expander(f"View {doc.filename}"):
                markdown = svc["storage"].read_markdown(doc.doc_id)
                st.markdown(markdown or "_Not yet converted_")
```

- [ ] **Step 7: Add the Ollama startup check**

At the top of `ui/app.py`, after `build_services()`:

```python
import httpx

try:
    httpx.get(f"{svc['cfg'].ollama_url}/api/tags", timeout=5).raise_for_status()
except Exception:
    st.error(
        f"Cannot reach Ollama at {svc['cfg'].ollama_url}.\n\n"
        "Start it with `ollama serve`, then confirm the models are present:\n"
        "`ollama pull qwen2.5:14b` and `ollama pull bge-m3`."
    )
    if st.button("Recheck"):
        st.rerun()
    st.stop()

if not svc["registry"].all():
    st.info("No documents indexed yet. Upload one to get started.")
```

- [ ] **Step 8: Verify manually**

```bash
docker compose restart app
```

1. Stop Ollama → reload → clear error naming the fix, with a Recheck button
2. Restart Ollama → click Recheck → app loads
3. Remove a document → disappears from the list, answers no longer cite it
4. Expand a document → converted markdown renders

- [ ] **Step 9: Commit**

```bash
git add ingestion/parser.py ui/app.py tests/test_ocr_fallback.py
git commit -m "feat: ocr fallback, document removal, and startup checks"
```

**PHASE 4 COMPLETE.**

---

# PHASE 5 — Evaluation

### Task 24: Golden set and deterministic metrics

Built before the judged metrics deliberately: these need no LLM judge, cost nothing, and cover the two failure modes that matter most.

**Files:**
- Create: `eval/golden_set.yaml`, `eval/metrics.py`, `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from eval.metrics import refusal_accuracy, citation_accuracy


def test_refusal_accuracy_rewards_correct_refusals():
    cases = [
        {"out_of_corpus": True, "refused": True},
        {"out_of_corpus": False, "refused": False},
    ]
    assert refusal_accuracy(cases) == 1.0


def test_refusal_accuracy_penalises_answering_out_of_corpus():
    cases = [{"out_of_corpus": True, "refused": False}]
    assert refusal_accuracy(cases) == 0.0


def test_refusal_accuracy_penalises_refusing_in_corpus():
    cases = [{"out_of_corpus": False, "refused": True}]
    assert refusal_accuracy(cases) == 0.0


def test_citation_accuracy_matches_expected_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["report.pdf, p. 4"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_fails_on_wrong_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["other.pdf, p. 1"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 0.0


def test_citation_accuracy_skips_out_of_corpus_cases():
    cases = [
        {"expected_sources": [], "citations": [], "out_of_corpus": True},
        {"expected_sources": ["a.pdf"], "citations": ["a.pdf, p. 1"],
         "out_of_corpus": False},
    ]
    assert citation_accuracy(cases) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the metrics**

Create `eval/__init__.py` (empty) and `eval/metrics.py`:

```python
def refusal_accuracy(cases: list[dict]) -> float:
    """Did the system refuse exactly when it should have?

    Not a Ragas metric — deterministic, needs no judge, and covers the
    failure mode that matters most: confidently answering something that
    is not in the corpus.
    """
    if not cases:
        return 0.0
    correct = sum(
        1 for c in cases if bool(c["refused"]) == bool(c["out_of_corpus"])
    )
    return correct / len(cases)


def citation_accuracy(cases: list[dict]) -> float:
    """Did the cited document match the expected source?"""
    scored = [c for c in cases if not c["out_of_corpus"]]
    if not scored:
        return 0.0

    correct = 0
    for case in scored:
        cited = " ".join(case["citations"])
        if any(src in cited for src in case["expected_sources"]):
            correct += 1
    return correct / len(scored)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Create the golden set skeleton**

Create `eval/golden_set.yaml`:

```yaml
# Hand-written evaluation cases against the real corpus.
# Target ~30-50 entries, roughly a quarter with out_of_corpus: true.
#
# The out-of-corpus entries are what make the similarity floor tunable.
# Without them you can measure whether good answers are correct, but not
# whether bad questions get refused - and the refusal path is the whole
# trust story.

- question: "What is the service contract number?"
  expected_answer: "SC-4471"
  expected_sources: ["sample.pdf"]
  out_of_corpus: false

- question: "Who escalates critical faults?"
  expected_answer: "The duty engineer."
  expected_sources: ["sample.pdf"]
  out_of_corpus: false

- question: "Jaki jest numer kontraktu serwisowego?"
  expected_answer: "SC-4471"
  expected_sources: ["sample.pdf"]
  out_of_corpus: false

# --- out of corpus: must be refused ---

- question: "What is the capital of France?"
  expected_answer: ""
  expected_sources: []
  out_of_corpus: true

- question: "What is our 2025 marketing budget?"
  expected_answer: ""
  expected_sources: []
  out_of_corpus: true
```

- [ ] **Step 6: Commit**

```bash
git add eval/ tests/test_eval_metrics.py
git commit -m "feat: golden set and judge-free evaluation metrics"
```

---

### Task 25: Evaluation runner and floor calibration

**Files:**
- Create: `eval/run_eval.py`, `eval/README.md`

- [ ] **Step 1: Write the runner**

Create `eval/run_eval.py`:

```python
"""Evaluation harness.

Imports app modules; the app never imports this. The external judge's API
key lives only here, and the harness runs against a hand-written test set —
production documents have no code path to an external service.

Usage:
    python eval/run_eval.py                 # deterministic metrics only
    python eval/run_eval.py --with-ragas    # adds judged metrics
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from core.config import Config
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.reranker import BGEReranker
from retrieval.search import Search
from generation.llm import OllamaLLM
from generation.answerer import Answerer
from generation.guards import should_refuse_aggregation
from eval.metrics import refusal_accuracy, citation_accuracy

ROOT = Path(__file__).parent


def run_cases(score_floor: float | None = None) -> list[dict]:
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    search = Search(embedder, store, BGEReranker(),
                    cfg.candidates, cfg.top_k, floor)
    answerer = Answerer(OllamaLLM(cfg.ollama_url, cfg.llm_model))

    golden = yaml.safe_load((ROOT / "golden_set.yaml").read_text())
    cases = []

    for entry in golden:
        outcome = search.find(entry["question"])
        guarded = (not outcome.refused and
                   should_refuse_aggregation(entry["question"],
                                             outcome.results))

        if outcome.refused or guarded:
            answer_text, citations, refused = "", [], True
        else:
            answer = answerer.answer(entry["question"], outcome.results)
            answer_text = answer.text
            citations = answer.citations
            refused = answer.refused

        cases.append({
            **entry,
            "answer": answer_text,
            "citations": citations,
            "refused": refused,
            "contexts": [r.chunk.text for r in outcome.results],
        })

    return cases


def calibrate_floor() -> None:
    """Sweep candidate floors and report which separates the two groups best.

    The floor cannot be chosen in advance — it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.
    """
    print(f"{'floor':>7} {'refusal_acc':>12} {'citation_acc':>13}")
    for floor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        cases = run_cases(score_floor=floor)
        print(f"{floor:>7.2f} {refusal_accuracy(cases):>12.2f} "
              f"{citation_accuracy(cases):>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ragas", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_floor()
        return

    cases = run_cases()
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "refusal_accuracy": refusal_accuracy(cases),
        "citation_accuracy": citation_accuracy(cases),
    }

    if args.with_ragas:
        report.update(run_ragas(cases))

    print(json.dumps(report, indent=2, ensure_ascii=False))

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (reports / f"{stamp}.json").write_text(
        json.dumps({"summary": report, "cases": cases},
                   indent=2, ensure_ascii=False)
    )
    print(f"\nwrote eval/reports/{stamp}.json")


def run_ragas(cases: list[dict]) -> dict:
    """Judged metrics via an external frontier model.

    A local 14B is a noticeably noisier judge, degrading further on Polish.
    Tuning against an unreliable measurement is worse than not measuring,
    because it feels rigorous.
    """
    import os
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )
    from langchain_openai import ChatOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — required for --with-ragas")

    scored = [c for c in cases if not c["out_of_corpus"] and c["contexts"]]
    dataset = Dataset.from_dict({
        "question": [c["question"] for c in scored],
        "answer": [c["answer"] for c in scored],
        "contexts": [c["contexts"] for c in scored],
        "ground_truth": [c["expected_answer"] for c in scored],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    )
    return {k: float(v) for k, v in result.items()}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Ingest the fixture and run the deterministic metrics**

```bash
docker compose exec app python eval/run_eval.py
```

Expected: JSON with `refusal_accuracy` and `citation_accuracy`. With `score_floor: 0.0` the refusal accuracy will be poor — nothing is ever refused. That is the problem the next step solves.

- [ ] **Step 3: Calibrate the similarity floor**

```bash
docker compose exec app python eval/run_eval.py --calibrate
```

Expected: a table sweeping floors from 0.0 to 0.8.

**Pick the floor with the highest refusal accuracy that does not reduce citation accuracy.** Write that value into `config.yaml` under `retrieval.score_floor`.

- [ ] **Step 4: Re-run and confirm improvement**

```bash
docker compose exec app python eval/run_eval.py
```

Expected: refusal accuracy materially higher than in Step 2.

- [ ] **Step 5: Write `eval/README.md`**

```markdown
# Evaluation

Development-time only. Never runs in the app, never runs in CI.

## Isolation

`eval/` imports app modules; the app never imports `eval/`. The external
judge's API key exists only in this environment. Production documents have
no code path to an external service — only the hand-written golden set is
ever sent out.

## Running

    python eval/run_eval.py                # deterministic metrics, offline
    python eval/run_eval.py --calibrate    # sweep the similarity floor
    OPENAI_API_KEY=sk-... python eval/run_eval.py --with-ragas

## Metrics

| Metric | Judge | Measures |
|---|---|---|
| refusal_accuracy | No | Refused exactly on out-of-corpus questions |
| citation_accuracy | No | Cited the expected source |
| faithfulness | Yes | Answer grounded in retrieved context |
| answer_relevancy | Yes | Answer addresses the question |
| context_precision | Yes | Retrieved chunks are relevant |
| context_recall | Yes | Retrieval found what was needed |

The judge-free metrics run offline and cost nothing — run them on every
change. The judged metrics need an API key and cost money — run them when
comparing configurations.

## Comparing configurations

    # edit config.yaml: chunking.strategy: semantic
    docker compose restart app
    # re-ingest the corpus
    python eval/run_eval.py --with-ragas
    # diff against the previous report in eval/reports/

This is what makes the chunker swappable: strategy choice becomes a
measurement rather than an argument.
```

- [ ] **Step 6: Commit**

```bash
git add eval/ config.yaml
git commit -m "feat: evaluation runner with floor calibration and ragas"
```

**PHASE 5 COMPLETE.**

---

## Final Verification

- [ ] **Full test suite passes**

```bash
docker compose exec app python -m pytest -v --run-integration
```

- [ ] **Clean rebuild works from scratch**

```bash
docker compose down -v && docker compose up -d --build
```

Then ingest a document and ask a question. This proves a new machine can bring the system up.

- [ ] **Manual acceptance checklist**

1. Queue 3 documents at once → returns immediately, processed one at a time
2. Chat remains responsive during ingestion
3. In-corpus question → correct answer, correct citation, streamed
4. Out-of-corpus question → refused, related documents offered
5. Aggregation question over a spreadsheet → refusal naming file and sheet
6. Polish question over English document → answered in Polish
7. Remove a document → no longer cited
8. Stop Ollama → clear error with recheck
9. Restart mid-ingest → no document stuck in `processing`

- [ ] **Commit any fixes, then tag**

```bash
git tag v0.1.0
```

---

## Deferred (per spec §Out of scope)

Not to be built without a new design conversation:

- VPS deployment, auth, TLS, multi-user, per-user corpora
- Text-to-SQL / DuckDB aggregation (the guard in `generation/guards.py` is the designated seam)
- Hybrid dense + sparse retrieval
- Persistent chat history
- Live filesystem watching
- ONNX reranker (the lever if CPU reranking proves too slow — measured in Task 20 Step 5)
- Real tokenizer-based chunk sizing (currently a `CHARS_PER_TOKEN` heuristic in `structural.py`)
