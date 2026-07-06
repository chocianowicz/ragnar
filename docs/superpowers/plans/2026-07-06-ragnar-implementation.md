# Ragnar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully offline macOS RAG chat app: documents (PDF/DOCX/XLSX/images/TXT/MD) are converted to markdown via Docling, chunked with provenance, embedded with BGE-M3, retrieved + reranked, and answered strictly from document content with citations — packaged as a `.dmg`.

**Architecture:** Single Python service. Ingestion pipeline (watchdog → Docling → provenance-aware chunking → Ollama embeddings → ChromaDB) feeds a retrieval chain (vector top-25 → `bge-reranker-v2-m3` → score floor → strict grounded prompt → Ollama LLM). FastAPI serves a plain HTML/JS chat UI, wrapped in a pywebview native window and frozen with PyInstaller.

**Tech Stack:** Python 3.12, FastAPI + uvicorn, Docling (+ docling-core HybridChunker), ChromaDB (embedded), sentence-transformers (CrossEncoder reranker), httpx (Ollama client), watchdog, pywebview, PyInstaller, pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-ragnar-design.md` — the authority on behavior. If plan and spec conflict, spec wins.

---

## File Structure

```
ragnar/
├── pyproject.toml
├── src/ragnar/
│   ├── __init__.py
│   ├── config.py          # data-dir paths + persisted user settings
│   ├── registry.py        # SQLite document registry (status, hashes, artifact paths)
│   ├── convert.py         # Docling conversion → .md + DoclingDocument JSON
│   ├── archive.py         # hash-suffixed archiving of originals
│   ├── chunker.py         # HybridChunker wrapper → chunks with page/sheet provenance
│   ├── ollama_client.py   # embed / chat / tags / pull against localhost:11434
│   ├── store.py           # ChromaDB wrapper + embed-model version guard
│   ├── reranker.py        # CrossEncoder wrapper, device detection
│   ├── retriever.py       # candidates → rerank → floor → passed/related
│   ├── llm.py             # strict grounded prompt construction
│   ├── chat.py            # ChatService: retrieve → generate | refuse, citations
│   ├── ingest.py          # IngestService: convert→chunk→embed→store→archive, remove
│   ├── watcher.py         # watchdog + per-file debounce + self-move suppression
│   ├── api.py             # FastAPI app: chat, documents, upload, status, setup
│   ├── main.py            # uvicorn thread + pywebview window
│   └── webui/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── tests/                 # mirrors src modules; fixtures in tests/fixtures/
├── build/
│   ├── spike/             # Phase 0 throwaway
│   ├── ragnar.spec        # PyInstaller spec
│   ├── fetch_models.py    # prefetch reranker weights + Docling artifacts + tokenizer
│   └── build.sh           # freeze + hdiutil .dmg
└── docs/
    ├── superpowers/...    # spec + this plan
    └── UAT.md             # manual test checklist
```

**Design rules for the executor:**
- Every service takes its dependencies via constructor (no module-level singletons) — this is what makes the test fakes work.
- Ollama and the reranker are NEVER called in unit tests; use the fakes given in each task. Tests needing real models are marked `@pytest.mark.ml` and excluded by default (`addopts = "-m 'not ml'"`).
- Commit after every green test, exactly as the steps say.

---

## Phase 0: Packaging spike (no TDD — throwaway code, the output is knowledge)

### Task 0: PyInstaller + torch/Docling/reranker spike

**Files:**
- Create: `build/spike/spike_main.py`, `build/spike/README.md`

- [ ] **Step 1: Create venv and install deps**

```bash
cd /Users/denis/Documents/AI/GitHub/ragnar
python3.12 -m venv .venv && source .venv/bin/activate
pip install docling sentence-transformers pyinstaller
```

- [ ] **Step 2: Write the spike script**

`build/spike/spike_main.py`:
```python
"""Throwaway packaging spike. Prints PASS/FAIL for the two risky deps."""
import sys, time
from pathlib import Path

def main():
    out = []
    t0 = time.time()
    try:
        from docling.document_converter import DocumentConverter
        pdf = Path(__file__).parent / "sample.pdf"
        result = DocumentConverter().convert(pdf)
        md = result.document.export_to_markdown()
        out.append(f"DOCLING: PASS ({len(md)} chars, {time.time()-t0:.1f}s)")
    except Exception as e:
        out.append(f"DOCLING: FAIL — {type(e).__name__}: {e}")
    t0 = time.time()
    try:
        from sentence_transformers import CrossEncoder
        m = CrossEncoder("BAAI/bge-reranker-v2-m3")
        scores = m.predict([("what is the capital of France?", "Paris is the capital of France."),
                            ("what is the capital of France?", "Bananas are yellow.")])
        assert scores[0] > scores[1]
        out.append(f"RERANKER: PASS device={m.model.device} ({time.time()-t0:.1f}s)")
    except Exception as e:
        out.append(f"RERANKER: FAIL — {type(e).__name__}: {e}")
    print("\n".join(out))

if __name__ == "__main__":
    main()
```
Put any small text-based PDF at `build/spike/sample.pdf` (e.g. print a one-page document to PDF).

- [ ] **Step 3: Run unfrozen first** — `python build/spike/spike_main.py` must print two PASS lines (models download on first run; this warms the HF cache). If this fails, fix the environment before freezing.

- [ ] **Step 4: Freeze**

```bash
pyinstaller --onedir --name ragnar-spike \
  --add-data "build/spike/sample.pdf:." \
  --collect-all docling --collect-all docling_core --collect-all docling_ibm_models \
  --collect-all easyocr --collect-all torch --collect-all transformers \
  --collect-all sentence_transformers \
  build/spike/spike_main.py
```
Expect a large `dist/ragnar-spike/` (several GB is normal for torch). Record the size.

- [ ] **Step 5: Test frozen, offline, cache-free** — the decisive step. To simulate a clean teammate Mac without a second machine: temporarily rename the HF cache and kill networking for the process:

```bash
mv ~/.cache/huggingface ~/.cache/huggingface.bak
./dist/ragnar-spike/ragnar-spike   # EXPECTED TO FAIL — proves models are NOT in the bundle yet
```
Then bundle the models explicitly: pre-download with `huggingface-cli download BAAI/bge-reranker-v2-m3` and `docling-tools models download`, add both directories via `--add-data`, point `CrossEncoder(<bundled path>)` and `DocumentConverter` (`artifacts_path=<bundled path>`) at them, re-freeze, and rerun with cache still renamed **and Wi-Fi off**. Success criteria (all four, from spec §5): both PASS lines; device reported (note MPS vs CPU); ran offline with no caches; bundle size recorded.
Restore the cache after: `mv ~/.cache/huggingface.bak ~/.cache/huggingface`.

- [ ] **Step 6: Record results and decide route** — write findings into `build/spike/README.md`: PASS/FAIL per dep, device, bundle size, exact flags that worked. If any dep failed after reasonable effort (timebox: half a day total), follow the spec's fallback chain (§5: ONNX reranker → embedded-runtime launcher → CPU) and note the chosen route — later packaging tasks (Task 16) must follow it.

- [ ] **Step 7: Commit** — `git add build/spike && git commit -m "spike: PyInstaller packaging results for docling + reranker"` (do NOT commit `dist/`, `build/spike/__pycache__`, or the venv; add a `.gitignore` first: `.venv/`, `dist/`, `*.spec` at repo root, `__pycache__/`, `*.pyc`).

**STOP after this task and report spike results to the human before continuing.**

---

## Phase 1: Scaffold + config

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/ragnar/__init__.py`, `tests/__init__.py`, `.gitignore`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "ragnar"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "httpx>=0.27",
    "watchdog>=5.0",
    "chromadb>=0.5",
    "docling>=2.15",
    "docling-core[chunking]>=2.15",
    "sentence-transformers>=3.3",
    "pywebview>=5.3",
    "python-multipart>=0.0.18",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pyinstaller>=6.11"]

[tool.pytest.ini_options]
addopts = "-m 'not ml'"
markers = ["ml: needs real ML models/Ollama, excluded by default"]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ragnar"]
```

- [ ] **Step 2: Install and smoke-test** — `pip install -e ".[dev]" && pytest` → "no tests ran" is success.
- [ ] **Step 3: Commit** — `git add -A && git commit -m "chore: project scaffold"`

### Task 2: Config module

**Files:** Create: `src/ragnar/config.py`, Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from ragnar.config import Config, Settings

def test_defaults_when_no_settings_file(tmp_path):
    cfg = Config(data_dir=tmp_path)
    assert cfg.settings.embed_model == "bge-m3"
    assert cfg.settings.onboarding_complete is False
    assert cfg.converted_dir == tmp_path / "converted"
    assert cfg.archive_dir == tmp_path / "archive" / "originals"

def test_save_and_reload_roundtrip(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.settings.watched_folder = str(tmp_path / "docs")
    cfg.settings.onboarding_complete = True
    cfg.save()
    cfg2 = Config(data_dir=tmp_path)
    assert cfg2.settings.watched_folder == str(tmp_path / "docs")
    assert cfg2.settings.onboarding_complete is True

def test_ensure_dirs_creates_tree(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.settings.watched_folder = str(tmp_path / "docs")
    cfg.ensure_dirs()
    assert cfg.converted_dir.is_dir() and cfg.archive_dir.is_dir()
    assert Path(cfg.settings.watched_folder).is_dir()
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_config.py -v` → ImportError.
- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

APP_NAME = "Ragnar"


def default_data_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME


@dataclass
class Settings:
    watched_folder: str = str(Path.home() / "Documents" / "RAG Chat" / "Documents")  # matches spec §5 default
    llm_model: str = "qwen2.5:14b"
    embed_model: str = "bge-m3"
    rerank_floor: float = 0.3
    onboarding_complete: bool = False


class Config:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or default_data_dir()
        self.settings_path = self.data_dir / "settings.json"
        self.converted_dir = self.data_dir / "converted"
        self.archive_dir = self.data_dir / "archive" / "originals"
        self.chroma_dir = self.data_dir / "chroma"
        self.registry_path = self.data_dir / "registry.sqlite3"
        self.settings = self._load()

    def _load(self) -> Settings:
        if self.settings_path.exists():
            return Settings(**json.loads(self.settings_path.read_text()))
        return Settings()

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(asdict(self.settings), indent=2))

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.converted_dir, self.archive_dir, self.chroma_dir,
                  Path(self.settings.watched_folder)):
            d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Verify pass** — `pytest tests/test_config.py -v` → 3 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat: config with persisted settings"`

---

## Phase 2: Registry

### Task 3: SQLite document registry

**Files:** Create: `src/ragnar/registry.py`, Test: `tests/test_registry.py`

Statuses: `processing | done | failed`. One row per source document.

- [ ] **Step 1: Write failing tests**

```python
from ragnar.registry import Registry

def make(tmp_path):
    return Registry(tmp_path / "reg.sqlite3")

def test_upsert_and_get(tmp_path):
    r = make(tmp_path)
    r.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h1", status="processing")
    d = r.get_by_source("/w/a.pdf")
    assert d.doc_id == "d1" and d.status == "processing"

def test_upsert_same_source_replaces(tmp_path):
    r = make(tmp_path)
    r.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h1", status="done")
    r.upsert(doc_id="d2", source_path="/w/a.pdf", content_hash="h2", status="processing")
    d = r.get_by_source("/w/a.pdf")
    assert d.doc_id == "d2" and d.content_hash == "h2"

def test_mark_done_stores_artifact_paths(tmp_path):
    r = make(tmp_path)
    r.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h1", status="processing")
    r.mark_done("d1", md_path="/c/d1.md", json_path="/c/d1.json", archive_path="/a/a-h1.pdf")
    d = r.get("d1")
    assert d.status == "done" and d.md_path == "/c/d1.md"

def test_mark_failed_stores_error(tmp_path):
    r = make(tmp_path)
    r.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h1", status="processing")
    r.mark_failed("d1", "conversion blew up")
    assert r.get("d1").status == "failed"
    assert "blew up" in r.get("d1").error

def test_list_all_and_delete(tmp_path):
    r = make(tmp_path)
    r.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h1", status="done")
    r.upsert(doc_id="d2", source_path="/w/b.pdf", content_hash="h2", status="done")
    assert len(r.list_all()) == 2
    r.delete("d1")
    assert [d.doc_id for d in r.list_all()] == ["d2"]
```

- [ ] **Step 2: Verify fail** — `pytest tests/test_registry.py -v`
- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import sqlite3, time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source_path TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  md_path TEXT, json_path TEXT, archive_path TEXT,
  ingested_at REAL
);
"""


@dataclass
class DocRecord:
    doc_id: str
    source_path: str
    content_hash: str
    status: str
    error: str | None
    md_path: str | None
    json_path: str | None
    archive_path: str | None
    ingested_at: float | None


class Registry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, doc_id: str, source_path: str, content_hash: str, status: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
            c.execute(
                "INSERT INTO documents (doc_id, source_path, content_hash, status) VALUES (?,?,?,?)",
                (doc_id, source_path, content_hash, status))

    def mark_done(self, doc_id: str, md_path: str, json_path: str, archive_path: str | None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET status='done', error=NULL, md_path=?, json_path=?, "
                "archive_path=?, ingested_at=? WHERE doc_id=?",
                (md_path, json_path, archive_path, time.time(), doc_id))

    def mark_failed(self, doc_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET status='failed', error=? WHERE doc_id=?", (error, doc_id))

    def _row_to_rec(self, row) -> DocRecord | None:
        return DocRecord(**dict(row)) if row else None

    def get(self, doc_id: str) -> DocRecord | None:
        with self._conn() as c:
            return self._row_to_rec(c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone())

    def get_by_source(self, source_path: str) -> DocRecord | None:
        with self._conn() as c:
            return self._row_to_rec(
                c.execute("SELECT * FROM documents WHERE source_path=?", (source_path,)).fetchone())

    def list_all(self) -> list[DocRecord]:
        with self._conn() as c:
            return [self._row_to_rec(r) for r in
                    c.execute("SELECT * FROM documents ORDER BY source_path").fetchall()]

    def delete(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def count_processing(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM documents WHERE status='processing'").fetchone()[0]
```

- [ ] **Step 4: Verify pass**, **Step 5: Commit** — `git commit -am "feat: sqlite document registry"`

---

## Phase 3: Conversion + archiving

### Task 4: Docling conversion wrapper

**Files:** Create: `src/ragnar/convert.py`, Test: `tests/test_convert.py`, Fixtures: `tests/fixtures/sample.md`

All formats (including `.txt`/`.md`) go through Docling so every document yields a DoclingDocument JSON with provenance. For `.md` sources the *source file's own text* is the canonical markdown (spec: "pass through unchanged"); for everything else the canonical markdown is Docling's export.

- [ ] **Step 1: Fixtures** — `tests/fixtures/sample.md`:

```markdown
# Vacation Policy

Employees receive 26 days of paid vacation per year.

## Carry-over

Up to 5 unused days may be carried into Q1 of the next year.
```

- [ ] **Step 2: Write failing tests**

```python
import json
from pathlib import Path
from ragnar.convert import Converter, ConversionError

FIXTURES = Path(__file__).parent / "fixtures"

def test_md_passthrough_keeps_source_text_verbatim(tmp_path):
    conv = Converter(converted_dir=tmp_path)
    res = conv.convert(FIXTURES / "sample.md", doc_id="d1")
    assert res.md_path.read_text() == (FIXTURES / "sample.md").read_text()
    assert json.loads(res.json_path.read_text())  # valid DoclingDocument JSON exists

def test_txt_becomes_markdown(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("The server room code is 4711.")
    conv = Converter(converted_dir=tmp_path / "out")
    res = conv.convert(src, doc_id="d2")
    assert "4711" in res.md_path.read_text()

def test_unsupported_extension_raises(tmp_path):
    src = tmp_path / "x.zip"; src.write_bytes(b"PK")
    conv = Converter(converted_dir=tmp_path / "out")
    import pytest
    with pytest.raises(ConversionError):
        conv.convert(src, doc_id="d3")
```

- [ ] **Step 3: Verify fail**, then implement:

```python
from __future__ import annotations
import shutil
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt", ".md"}


class ConversionError(Exception):
    pass


@dataclass
class ConversionResult:
    md_path: Path
    json_path: Path


class Converter:
    """Normalizes any supported file to (canonical .md, DoclingDocument .json)."""

    def __init__(self, converted_dir: Path, artifacts_path: str | None = None):
        self.converted_dir = converted_dir
        self.artifacts_path = artifacts_path  # bundled Docling models dir in frozen app
        self._converter = None  # lazy: Docling import is heavy

    def _docling(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            opts = PdfPipelineOptions(artifacts_path=self.artifacts_path, do_ocr=True)
            from docling.datamodel.base_models import InputFormat
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return self._converter

    def convert(self, source: Path, doc_id: str) -> ConversionResult:
        ext = source.suffix.lower()
        if ext not in SUPPORTED:
            raise ConversionError(f"unsupported file type: {ext}")
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.converted_dir / f"{doc_id}.md"
        json_path = self.converted_dir / f"{doc_id}.json"
        try:
            if ext == ".txt":
                # Docling has no txt input: wrap as markdown first, then parse that.
                tmp_md = self.converted_dir / f"{doc_id}.src.md"
                tmp_md.write_text(source.read_text(errors="replace"))
                doc = self._docling().convert(tmp_md).document
                tmp_md.replace(md_path)
            elif ext == ".md":
                doc = self._docling().convert(source).document
                shutil.copyfile(source, md_path)  # source text IS the canonical md
            else:
                doc = self._docling().convert(source).document
                md_path.write_text(doc.export_to_markdown())
            doc.save_as_json(json_path)
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(f"{type(e).__name__}: {e}") from e
        return ConversionResult(md_path=md_path, json_path=json_path)
```

- [ ] **Step 4: Verify pass** — first run downloads Docling models on the dev machine; that is fine (offline constraint applies to the shipped app, not dev).
- [ ] **Step 5: Add `@pytest.mark.ml` PDF/DOCX/XLSX structural tests** — create real fixtures (any one-page PDF; a DOCX saved from Pages/Word with a heading + paragraph; a small XLSX with a header row + 3 data rows) in `tests/fixtures/`, then:

```python
import pytest

@pytest.mark.ml
def test_xlsx_produces_table_markdown(tmp_path):
    conv = Converter(converted_dir=tmp_path)
    res = conv.convert(FIXTURES / "sample.xlsx", doc_id="dx")
    md = res.md_path.read_text()
    assert "|" in md  # table survived as a markdown table
```
Run once manually: `pytest tests/test_convert.py -m ml -v`.

- [ ] **Step 6: Commit** — `git commit -am "feat: docling conversion to md + provenance json"`

### Task 5: Archiver

**Files:** Create: `src/ragnar/archive.py`, Test: `tests/test_archive.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from ragnar.archive import Archiver, file_hash

def test_file_hash_stable(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("hello")
    assert file_hash(f) == file_hash(f)

def test_archive_moves_with_hash_suffix(tmp_path):
    watched, arch = tmp_path / "w", tmp_path / "arch"
    (watched / "sub").mkdir(parents=True)
    src = watched / "sub" / "a.pdf"; src.write_bytes(b"%PDF fake")
    moved = []
    a = Archiver(watched_folder=watched, archive_dir=arch, on_self_move=moved.append)
    h = file_hash(src)
    dest = a.archive(src, h)
    assert not src.exists()
    assert dest == arch / "sub" / f"a.{h[:12]}.pdf" and dest.exists()
    assert moved == [src]  # watcher suppression was notified BEFORE the move

def test_rearchiving_changed_file_does_not_collide(tmp_path):
    watched, arch = tmp_path / "w", tmp_path / "arch"; watched.mkdir()
    a = Archiver(watched_folder=watched, archive_dir=arch, on_self_move=lambda p: None)
    for content in (b"v1", b"v2"):
        src = watched / "a.pdf"; src.write_bytes(content)
        a.archive(src, file_hash(src))
    assert len(list(arch.glob("a.*.pdf"))) == 2
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
import hashlib, shutil
from pathlib import Path
from typing import Callable


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Archiver:
    """Moves ingested originals out of the watched folder, hash-suffixed.

    on_self_move is called with the source path BEFORE the move so the watcher
    can suppress the resulting deletion event (spec §1 Archiving).
    """

    def __init__(self, watched_folder: Path, archive_dir: Path,
                 on_self_move: Callable[[Path], None]):
        self.watched_folder = watched_folder
        self.archive_dir = archive_dir
        self.on_self_move = on_self_move

    def archive(self, source: Path, content_hash: str) -> Path:
        rel = source.relative_to(self.watched_folder)
        dest = self.archive_dir / rel.parent / f"{source.stem}.{content_hash[:12]}{source.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.on_self_move(source)
        shutil.move(str(source), str(dest))
        return dest
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: hash-suffixed archiver with watcher suppression hook"`

---

## Phase 4: Chunking with provenance

### Task 6: Chunker

**Files:** Create: `src/ragnar/chunker.py`, Test: `tests/test_chunker.py`

Uses docling-core's `HybridChunker`: hierarchy-aware (headings/sections/tables) then token-aware splitting/merging — this implements the spec's chunking intent, and its chunk metadata carries the provenance (page numbers) we need. Tests build a DoclingDocument programmatically (no ML involved).

- [ ] **Step 1: Write failing tests**

```python
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.base import BoundingBox
from ragnar.chunker import chunk_document

def build_doc() -> DoclingDocument:
    doc = DoclingDocument(name="policy")
    prov1 = ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=10), charspan=(0, 10))
    prov2 = ProvenanceItem(page_no=2, bbox=BoundingBox(l=0, t=0, r=100, b=10), charspan=(0, 10))
    doc.add_heading("Vacation Policy", prov=prov1)
    doc.add_text(label="text", text="Employees receive 26 days of paid vacation per year.", prov=prov1)
    doc.add_heading("Equipment", prov=prov2)
    doc.add_text(label="text", text="Laptops are replaced every 36 months.", prov=prov2)
    return doc

def test_chunks_carry_text_and_page_provenance():
    chunks = chunk_document(build_doc())
    assert len(chunks) >= 2
    vac = next(c for c in chunks if "26 days" in c.text)
    lap = next(c for c in chunks if "36 months" in c.text)
    assert vac.page == 1 and lap.page == 2

def test_chunk_indexes_are_sequential():
    chunks = chunk_document(build_doc())
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

def test_heading_context_is_included():
    chunks = chunk_document(build_doc())
    vac = next(c for c in chunks if "26 days" in c.text)
    assert "Vacation Policy" in vac.text  # contextualize() prefixes headings
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Chunk:
    text: str
    page: int | None
    chunk_index: int


@lru_cache(maxsize=1)
def _chunker(tokenizer_path: str | None, max_tokens: int):
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    if tokenizer_path:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        tok = HuggingFaceTokenizer.from_pretrained(tokenizer_path, max_tokens=max_tokens)
        return HybridChunker(tokenizer=tok, merge_peers=True)
    return HybridChunker(merge_peers=True)


def chunk_document(doc, tokenizer_path: str | None = None, max_tokens: int = 500) -> list[Chunk]:
    """Chunk a DoclingDocument into provenance-carrying units.

    tokenizer_path: local dir of the BGE-M3 tokenizer in the frozen app;
    None lets HybridChunker use its default (fine on the dev machine).
    """
    chunker = _chunker(tokenizer_path, max_tokens)
    out: list[Chunk] = []
    for i, ch in enumerate(chunker.chunk(doc)):
        page = None
        for item in ch.meta.doc_items:
            if item.prov:
                page = item.prov[0].page_no
                break
        out.append(Chunk(text=chunker.contextualize(ch), page=page, chunk_index=i))
    return out
```
Note: HybridChunker's default tokenizer downloads once on the dev machine. If the exact `add_heading`/`add_text`/`ProvenanceItem` signatures differ in the installed docling-core version, adapt the test builder to the installed API (check `python -c "import docling_core; print(docling_core.__version__)"` and the `DoclingDocument` docstrings) — the assertions must stay as written.

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: provenance-aware chunking via HybridChunker"`

- [ ] **Step 5 (ml-marked, depends on Task 4's XLSX fixture): verify table row-grouping + header repetition** — the spec requires that when a converted Excel table is split into multiple chunks, the header row is repeated into every group. Write:

```python
import pytest
from pathlib import Path
from ragnar.convert import Converter
from ragnar.chunker import chunk_document
from docling_core.types.doc import DoclingDocument

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.mark.ml
def test_xlsx_table_chunks_repeat_header_row(tmp_path):
    conv = Converter(converted_dir=tmp_path)
    res = conv.convert(FIXTURES / "sample.xlsx", doc_id="dx")  # needs >20 data rows to force >1 group
    doc = DoclingDocument.load_from_json(res.json_path)
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if "|" in c.text]
    assert len(table_chunks) >= 2
    header = table_chunks[0].text.splitlines()[0]
    assert all(header in c.text for c in table_chunks)  # header present in every group
```
If `HybridChunker`'s default table serialization does NOT repeat the header per chunk (check by running this test first — it may already pass, since HybridChunker's `TableSerializer` often includes headers per split), implement the fallback: post-process `chunk_document`'s output by detecting consecutive table-row chunks belonging to the same table (via `ch.meta.doc_items` pointing at the same `TableItem`) and prepending the first row's text to each subsequent chunk. Do NOT skip this test — it is the one place the plan verifies a spec-mandated behavior (spec §1: "column headers repeated per chunk for context").

- [ ] **Step 6: OCR low-confidence flagging** — spec §1 requires low-confidence OCR pages to be "ingested but flagged as lower-reliability sources." Docling's page prediction exposes per-page/per-cell confidence (check the installed version's `docling_core.types.doc.document.DoclingDocument` / `ConversionResult.confidence` — the exact attribute name varies by Docling release; inspect via `python -c "from docling.document_converter import DocumentConverter; r = DocumentConverter().convert('tests/fixtures/sample.pdf'); print(r.confidence)"`). Add a `low_confidence: bool` field to `Chunk`, populate it in `chunk_document` from the source page's confidence score against a threshold (e.g. mean OCR confidence < 0.7), thread it into `IngestService._chunk_embed_store`'s metadata (`"low_confidence": c.low_confidence`), and surface it in the UI document list (Task 15) as a small badge. Write one `@pytest.mark.ml` test against a real scanned-image fixture confirming `low_confidence` is `True` for it and `False` for a native-text PDF. Commit separately: `git commit -am "feat: OCR low-confidence flagging"`.
- [ ] **Step 7: Commit the table-header test** — `git commit -am "test: verify xlsx table header repetition across chunks"`

---

## Phase 5: Ollama client + vector store + reranker

### Task 7: Ollama client

**Files:** Create: `src/ragnar/ollama_client.py`, Test: `tests/test_ollama_client.py`

- [ ] **Step 1: Write failing tests** (httpx MockTransport — no real Ollama)

```python
import httpx, json
from ragnar.ollama_client import OllamaClient

def make_client(handler):
    return OllamaClient(base_url="http://testserver",
                        transport=httpx.MockTransport(handler))

def test_embed_batches_and_returns_vectors():
    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "bge-m3"
        return httpx.Response(200, json={"embeddings": [[0.1] * 4 for _ in body["input"]]})
    c = make_client(handler)
    vecs = c.embed("bge-m3", ["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 4

def test_chat_returns_message_content():
    def handler(req):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})
    assert make_client(handler).chat("qwen2.5:14b", [{"role": "user", "content": "x"}]) == "hi"

def test_is_up_false_when_connection_refused():
    def handler(req):
        raise httpx.ConnectError("refused")
    assert make_client(handler).is_up() is False

def test_has_model_checks_tags():
    def handler(req):
        return httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}]})
    c = make_client(handler)
    assert c.has_model("bge-m3") is True
    assert c.has_model("qwen2.5:14b") is False
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
import httpx


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=300.0)

    def is_up(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self, name: str) -> bool:
        try:
            models = self._client.get("/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return False
        names = {m["name"] for m in models}
        return name in names or f"{name}:latest" in names

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = self._client.post("/api/embed", json={"model": model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def chat(self, model: str, messages: list[dict]) -> str:
        r = self._client.post("/api/chat",
                              json={"model": model, "messages": messages, "stream": False})
        r.raise_for_status()
        return r.json()["message"]["content"]

    def pull(self, model: str) -> None:
        """Blocking pull; caller runs it in a background thread and polls has_model()."""
        with self._client.stream("POST", "/api/pull",
                                 json={"model": model}, timeout=None) as r:
            for _ in r.iter_lines():
                pass
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: ollama client (embed/chat/tags/pull)"`

### Task 8: Vector store

**Files:** Create: `src/ragnar/store.py`, Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests** (real ChromaDB on tmp dir; embeddings are plain lists — no models)

```python
from ragnar.store import VectorStore, Candidate

E1, E2, E3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]

def test_add_query_roundtrip(tmp_path):
    s = VectorStore(tmp_path, embed_model_id="bge-m3")
    s.add_chunks("d1", texts=["alpha", "beta"], embeddings=[E1, E2],
                 metadatas=[{"source": "a.pdf", "page": 1, "chunk_index": 0},
                            {"source": "a.pdf", "page": 2, "chunk_index": 1}])
    got = s.query(E3, n=1)
    assert got[0].text == "alpha" and got[0].metadata["page"] == 1

def test_delete_document_removes_all_its_chunks(tmp_path):
    s = VectorStore(tmp_path, embed_model_id="bge-m3")
    s.add_chunks("d1", ["alpha"], [E1], [{"source": "a.pdf", "page": 1, "chunk_index": 0}])
    s.add_chunks("d2", ["beta"], [E2], [{"source": "b.pdf", "page": 1, "chunk_index": 0}])
    s.delete_document("d1")
    assert [c.text for c in s.query(E1, n=5)] == ["beta"]

def test_embed_model_mismatch_detected(tmp_path):
    VectorStore(tmp_path, embed_model_id="bge-m3")
    s2 = VectorStore(tmp_path, embed_model_id="other-model")
    assert s2.needs_reembed is True

def test_reset_clears_and_updates_model(tmp_path):
    s = VectorStore(tmp_path, embed_model_id="bge-m3")
    s.add_chunks("d1", ["alpha"], [E1], [{"source": "a.pdf", "page": 1, "chunk_index": 0}])
    s2 = VectorStore(tmp_path, embed_model_id="new-model")
    s2.reset()
    assert s2.needs_reembed is False and s2.query(E1, n=5) == []
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import chromadb

COLLECTION = "chunks"


@dataclass
class Candidate:
    text: str
    metadata: dict
    doc_id: str


class VectorStore:
    def __init__(self, path: Path, embed_model_id: str):
        self.embed_model_id = embed_model_id
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(
            COLLECTION, metadata={"embed_model": embed_model_id})
        stored = (self.col.metadata or {}).get("embed_model")
        self.needs_reembed = stored != embed_model_id

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION)
        self.col = self.client.get_or_create_collection(
            COLLECTION, metadata={"embed_model": self.embed_model_id})
        self.needs_reembed = False

    def add_chunks(self, doc_id: str, texts: list[str],
                   embeddings: list[list[float]], metadatas: list[dict]) -> None:
        self.col.add(ids=[f"{doc_id}:{m['chunk_index']}" for m in metadatas],
                     documents=texts, embeddings=embeddings,
                     metadatas=[{**m, "doc_id": doc_id} for m in metadatas])

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def query(self, embedding: list[float], n: int = 25) -> list[Candidate]:
        res = self.col.query(query_embeddings=[embedding], n_results=n)
        out = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            out.append(Candidate(text=text, metadata=meta, doc_id=meta["doc_id"]))
        return out
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: chroma vector store with embed-model guard"`

### Task 9: Reranker

**Files:** Create: `src/ragnar/reranker.py`, Test: `tests/test_reranker.py`

- [ ] **Step 1: Write failing tests** (fake CrossEncoder injected — no model download)

```python
from ragnar.reranker import Reranker
from ragnar.store import Candidate

class FakeCrossEncoder:
    def predict(self, pairs):
        # score = crude keyword overlap, deterministic
        return [sum(w in text.lower() for w in q.lower().split()) for q, text in pairs]

def cands(*texts):
    return [Candidate(text=t, metadata={"chunk_index": i}, doc_id="d") for i, t in enumerate(texts)]

def test_rerank_orders_by_score_and_truncates():
    r = Reranker(model=FakeCrossEncoder())
    got = r.rerank("vacation days policy",
                   cands("bananas are yellow", "vacation policy: 26 days", "the days are long"),
                   top_k=2)
    assert got[0].candidate.text == "vacation policy: 26 days"
    assert len(got) == 2 and got[0].score >= got[1].score

def test_rerank_empty_input():
    assert Reranker(model=FakeCrossEncoder()).rerank("q", [], top_k=5) == []
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
from dataclasses import dataclass
from ragnar.store import Candidate


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float


class Reranker:
    def __init__(self, model=None, model_path: str = "BAAI/bge-reranker-v2-m3"):
        # model injected in tests; real CrossEncoder loaded lazily otherwise
        self._model = model
        self._model_path = model_path

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_path)
        return self._model

    def rerank(self, query: str, candidates: list[Candidate], top_k: int = 5) -> list[ScoredCandidate]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.text) for c in candidates])
        scored = [ScoredCandidate(c, float(s)) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]
```
Add one `@pytest.mark.ml` test loading the real model and asserting an on-topic pair outscores an off-topic one, and note the score range observed (needed to calibrate `rerank_floor`).

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: cross-encoder reranker wrapper"`

---

## Phase 6: Retrieval + chat

### Task 10: Retriever

**Files:** Create: `src/ragnar/retriever.py`, Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

```python
from ragnar.retriever import Retriever, RetrievalResult
from ragnar.store import Candidate
from ragnar.reranker import ScoredCandidate

class FakeStore:
    def __init__(self, cands): self.cands = cands
    def query(self, emb, n=25): return self.cands[:n]

class FakeEmbedder:
    def embed_query(self, text): return [0.0]

class FakeReranker:
    def __init__(self, scores): self.scores = scores
    def rerank(self, q, cands, top_k):
        s = [ScoredCandidate(c, self.scores[i]) for i, c in enumerate(cands)]
        s.sort(key=lambda x: x.score, reverse=True)
        return s[:top_k]

def cand(t): return Candidate(text=t, metadata={}, doc_id="d")

def test_passed_contains_only_above_floor():
    r = Retriever(FakeStore([cand("a"), cand("b"), cand("c")]), FakeEmbedder(),
                  FakeReranker([0.9, 0.5, 0.1]), floor=0.4, top_k=3)
    res = r.retrieve("q")
    assert [sc.candidate.text for sc in res.passed] == ["a", "b"]

def test_all_below_floor_yields_refusal_with_related():
    r = Retriever(FakeStore([cand("a"), cand("b")]), FakeEmbedder(),
                  FakeReranker([0.2, 0.1]), floor=0.4, top_k=3)
    res = r.retrieve("q")
    assert res.passed == [] and [sc.candidate.text for sc in res.related] == ["a", "b"]

def test_empty_store():
    res = Retriever(FakeStore([]), FakeEmbedder(), FakeReranker([]), floor=0.4).retrieve("q")
    assert res.passed == [] and res.related == []
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from ragnar.reranker import ScoredCandidate


@dataclass
class RetrievalResult:
    passed: list[ScoredCandidate] = field(default_factory=list)
    related: list[ScoredCandidate] = field(default_factory=list)


class Retriever:
    def __init__(self, store, embedder, reranker,
                 floor: float = 0.3, n_candidates: int = 25, top_k: int = 5):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.floor, self.n_candidates, self.top_k = floor, n_candidates, top_k

    def retrieve(self, query: str) -> RetrievalResult:
        emb = self.embedder.embed_query(query)
        candidates = self.store.query(emb, n=self.n_candidates)
        ranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        passed = [sc for sc in ranked if sc.score >= self.floor]
        if passed:
            return RetrievalResult(passed=passed, related=ranked)
        return RetrievalResult(passed=[], related=ranked[:3])
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: retriever with rerank floor and refusal path"`

### Task 11: Prompt + ChatService

**Files:** Create: `src/ragnar/llm.py`, `src/ragnar/chat.py`, Test: `tests/test_chat.py`

- [ ] **Step 1: Write failing tests**

```python
from ragnar.chat import ChatService
from ragnar.retriever import RetrievalResult
from ragnar.reranker import ScoredCandidate
from ragnar.store import Candidate

def sc(text, source="a.pdf", page=1, score=0.9):
    return ScoredCandidate(Candidate(text=text,
        metadata={"source": source, "page": page, "chunk_index": 0}, doc_id="d1"), score)

class FakeRetriever:
    def __init__(self, result): self.result = result
    def retrieve(self, q): return self.result

class FakeLLM:
    def __init__(self, reply): self.reply, self.last_messages = reply, None
    def chat(self, model, messages):
        self.last_messages = messages
        return self.reply

def test_grounded_answer_includes_citations_from_metadata():
    svc = ChatService(FakeRetriever(RetrievalResult(passed=[sc("26 days of vacation")])),
                      FakeLLM("You get 26 days [1]."), llm_model="m")
    resp = svc.answer("how many vacation days?")
    assert resp.refused is False
    assert resp.citations == [{"source": "a.pdf", "page": 1}]
    assert "26 days" in resp.answer

def test_chunks_and_strict_instructions_reach_the_llm():
    llm = FakeLLM("ok")
    svc = ChatService(FakeRetriever(RetrievalResult(passed=[sc("SECRET-CHUNK-TEXT")])),
                      llm, llm_model="m")
    svc.answer("q?")
    prompt_blob = " ".join(m["content"] for m in llm.last_messages)
    assert "SECRET-CHUNK-TEXT" in prompt_blob
    assert "NO_ANSWER" in prompt_blob  # refusal instruction present

def test_no_passed_chunks_refuses_without_calling_llm():
    llm = FakeLLM("should never be returned")
    svc = ChatService(FakeRetriever(RetrievalResult(passed=[], related=[sc("close but no", score=0.2)])),
                      llm, llm_model="m")
    resp = svc.answer("q?")
    assert resp.refused is True and llm.last_messages is None
    assert resp.related == [{"source": "a.pdf", "page": 1}]

def test_llm_declares_no_answer_becomes_refusal():
    svc = ChatService(FakeRetriever(RetrievalResult(passed=[sc("irrelevant")])),
                      FakeLLM("NO_ANSWER"), llm_model="m")
    assert svc.answer("q?").refused is True

def test_duplicate_citations_are_deduped():
    svc = ChatService(FakeRetriever(RetrievalResult(passed=[sc("a"), sc("b")])),
                      FakeLLM("x"), llm_model="m")
    assert svc.answer("q?").citations == [{"source": "a.pdf", "page": 1}]
```

- [ ] **Step 2: Verify fail**, implement `llm.py`:

```python
from __future__ import annotations
from ragnar.reranker import ScoredCandidate

SYSTEM_PROMPT = """You are a document assistant. You answer questions using ONLY the \
document excerpts provided in the user message. Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge, never guess.
2. Cite the excerpt number for each claim, like [1] or [2].
3. If the excerpts do not contain the information needed to answer, reply with \
exactly: NO_ANSWER
4. Answer in the same language as the question."""


def build_messages(question: str, chunks: list[ScoredCandidate]) -> list[dict]:
    lines = []
    for i, sc in enumerate(chunks, start=1):
        m = sc.candidate.metadata
        loc = f" (page {m['page']})" if m.get("page") else ""
        lines.append(f"[{i}] {m.get('source', 'unknown')}{loc}:\n{sc.candidate.text}")
    user = "Document excerpts:\n\n" + "\n\n".join(lines) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
```

and `chat.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from ragnar.llm import build_messages

REFUSAL_TEXT = ("I don't have information about that in the ingested documents. "
                "These documents look closest to your question:")


@dataclass
class ChatResponse:
    answer: str
    refused: bool
    citations: list[dict] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)


def _refs(scored) -> list[dict]:
    seen, out = set(), []
    for sc in scored:
        m = sc.candidate.metadata
        key = (m.get("source"), m.get("page"))
        if key not in seen:
            seen.add(key)
            out.append({"source": m.get("source"), "page": m.get("page")})
    return out


class ChatService:
    def __init__(self, retriever, llm, llm_model: str):
        self.retriever, self.llm, self.llm_model = retriever, llm, llm_model

    def answer(self, question: str) -> ChatResponse:
        r = self.retriever.retrieve(question)
        if not r.passed:
            return ChatResponse(answer=REFUSAL_TEXT, refused=True, related=_refs(r.related))
        text = self.llm.chat(self.llm_model, build_messages(question, r.passed))
        if text.strip() == "NO_ANSWER":
            return ChatResponse(answer=REFUSAL_TEXT, refused=True, related=_refs(r.passed))
        return ChatResponse(answer=text, refused=False, citations=_refs(r.passed))
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: strict grounded chat service with refusal + citations"`

---

## Phase 7: Ingestion orchestration + watcher

### Task 12: IngestService

**Files:** Create: `src/ragnar/ingest.py`, Test: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests** (fake converter/embedder; real registry/store/archiver on tmp dirs)

```python
import json
from pathlib import Path
from ragnar.ingest import IngestService
from ragnar.registry import Registry
from ragnar.store import VectorStore
from ragnar.archive import Archiver
from ragnar.convert import ConversionError, ConversionResult
from ragnar.chunker import Chunk

class FakeConverter:
    def __init__(self, converted_dir, fail=False):
        self.converted_dir, self.fail = Path(converted_dir), fail
    def convert(self, source, doc_id):
        if self.fail:
            raise ConversionError("boom")
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        md = self.converted_dir / f"{doc_id}.md"; md.write_text(source.read_text())
        js = self.converted_dir / f"{doc_id}.json"; js.write_text("{}")
        return ConversionResult(md_path=md, json_path=js)

class FakeEmbedder:
    def embed_texts(self, texts): return [[float(len(t)), 0.0] for t in texts]
    def embed_query(self, text): return [float(len(text)), 0.0]

def fake_chunker(json_path):
    return [Chunk(text="chunk-1", page=1, chunk_index=0)]

def make_service(tmp_path, fail=False):
    watched = tmp_path / "w"; watched.mkdir(exist_ok=True)
    reg = Registry(tmp_path / "reg.sqlite3")
    store = VectorStore(tmp_path / "chroma", embed_model_id="bge-m3")
    arch = Archiver(watched, tmp_path / "arch", on_self_move=lambda p: None)
    svc = IngestService(registry=reg, store=store, archiver=arch,
                        converter=FakeConverter(tmp_path / "conv", fail=fail),
                        embedder=FakeEmbedder(), chunk_fn=fake_chunker)
    return svc, watched, reg, store

def test_successful_ingest_registers_stores_archives(tmp_path):
    svc, watched, reg, store = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("# Hello")
    svc.ingest_file(f)
    rec = reg.list_all()[0]
    assert rec.status == "done" and not f.exists() and Path(rec.archive_path).exists()
    assert store.query([7.0, 0.0], n=1)[0].metadata["source"] == "a.md"

def test_failed_conversion_marks_failed_and_leaves_file(tmp_path):
    svc, watched, reg, _ = make_service(tmp_path, fail=True)
    f = watched / "bad.md"; f.write_text("x")
    svc.ingest_file(f)
    assert reg.list_all()[0].status == "failed" and f.exists()

def test_reingest_changed_file_replaces_chunks(tmp_path):
    svc, watched, reg, store = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("v1"); svc.ingest_file(f)
    old_id = reg.list_all()[0].doc_id
    f.write_text("v2-longer"); svc.ingest_file(f)
    recs = reg.list_all()
    assert len(recs) == 1 and recs[0].doc_id != old_id
    assert all(c.doc_id != old_id for c in store.query([2.0, 0.0], n=10))

def test_unchanged_file_is_skipped(tmp_path):
    svc, watched, reg, _ = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("same")
    svc.ingest_file(f)
    rec1 = reg.list_all()[0]
    # simulate the same content reappearing at the same path
    Path(rec1.archive_path).rename(f)
    svc.ingest_file(f)
    assert reg.list_all()[0].doc_id == rec1.doc_id  # untouched

def test_remove_document_cleans_everything(tmp_path):
    svc, watched, reg, store = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("# Hello"); svc.ingest_file(f)
    rec = reg.list_all()[0]
    svc.remove_document(rec.doc_id)
    assert reg.list_all() == []
    assert not Path(rec.archive_path).exists() and not Path(rec.md_path).exists()
    assert store.query([7.0, 0.0], n=5) == []

def test_reingest_deletes_stale_converted_artifacts(tmp_path):
    svc, watched, reg, _ = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("v1"); svc.ingest_file(f)
    old = reg.list_all()[0]
    old_md, old_json = Path(old.md_path), Path(old.json_path)
    assert old_md.exists() and old_json.exists()
    Path(old.archive_path).rename(f)  # bring the (changed) file back to re-trigger ingest
    f.write_text("v2-longer")
    svc.ingest_file(f)
    assert not old_md.exists() and not old_json.exists()  # stale artifacts cleaned up

def test_chunk_with_no_page_stores_zero_not_none(tmp_path):
    # Chroma metadata rejects None values; page=None must be coerced to 0.
    def chunker_no_page(json_path):
        return [Chunk(text="chunk-1", page=None, chunk_index=0)]
    watched = tmp_path / "w"; watched.mkdir()
    reg = Registry(tmp_path / "reg.sqlite3")
    store = VectorStore(tmp_path / "chroma", embed_model_id="bge-m3")
    arch = Archiver(watched, tmp_path / "arch", on_self_move=lambda p: None)
    svc = IngestService(registry=reg, store=store, archiver=arch,
                        converter=FakeConverter(tmp_path / "conv"),
                        embedder=FakeEmbedder(), chunk_fn=chunker_no_page)
    f = watched / "a.md"; f.write_text("no page info")
    svc.ingest_file(f)  # must not raise
    assert store.query([12.0, 0.0], n=1)[0].metadata["page"] == 0
```

- [ ] **Step 1b: Write failing tests for re-embed and startup scan**

```python
def test_reembed_all_restores_chunks_from_json_without_reconverting(tmp_path):
    svc, watched, reg, store = make_service(tmp_path)
    f = watched / "a.md"; f.write_text("# Hello"); svc.ingest_file(f)
    store.reset()  # simulate an embed-model switch wiping the collection
    assert store.query([7.0, 0.0], n=5) == []
    svc.reembed_all()
    assert store.query([7.0, 0.0], n=1)[0].metadata["source"] == "a.md"

def test_scan_existing_ingests_files_present_at_startup(tmp_path):
    svc, watched, reg, store = make_service(tmp_path)
    (watched / "a.md").write_text("hello")
    (watched / "b.md").write_text("world")
    svc.scan_existing()
    assert {r.source_path.split("/")[-1] for r in reg.list_all()} == {"a.md", "b.md"}
    assert all(r.status == "done" for r in reg.list_all())
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
import uuid
from pathlib import Path
from ragnar.archive import file_hash
from ragnar.convert import ConversionError


def load_docling_json(json_path: Path):
    from docling_core.types.doc import DoclingDocument
    return DoclingDocument.load_from_json(json_path)


def default_chunk_fn(json_path: Path):
    from ragnar.chunker import chunk_document
    return chunk_document(load_docling_json(json_path))


class IngestService:
    def __init__(self, registry, store, archiver, converter, embedder,
                 chunk_fn=default_chunk_fn):
        self.registry, self.store, self.archiver = registry, store, archiver
        self.converter, self.embedder, self.chunk_fn = converter, embedder, chunk_fn

    def ingest_file(self, path: Path) -> None:
        h = file_hash(path)
        existing = self.registry.get_by_source(str(path))
        if existing and existing.content_hash == h and existing.status == "done":
            return  # unchanged file re-appeared; nothing to do
        if existing:
            self.store.delete_document(existing.doc_id)
            for p in (existing.md_path, existing.json_path):  # avoid orphaned converted/ files
                if p and Path(p).exists():
                    Path(p).unlink()
        doc_id = uuid.uuid4().hex[:12]
        self.registry.upsert(doc_id=doc_id, source_path=str(path), content_hash=h,
                             status="processing")
        try:
            conv = self.converter.convert(path, doc_id)
        except ConversionError as e:
            self.registry.mark_failed(doc_id, str(e))
            return
        self._chunk_embed_store(doc_id, conv.json_path, path.name)
        archive_path = self.archiver.archive(path, h)
        self.registry.mark_done(doc_id, md_path=str(conv.md_path),
                                json_path=str(conv.json_path),
                                archive_path=str(archive_path))

    def _chunk_embed_store(self, doc_id: str, json_path: Path, source_name: str) -> None:
        import time
        chunks = self.chunk_fn(json_path)
        if not chunks:
            return
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        now = time.time()
        # Chroma rejects None metadata values; page=None (no provenance) becomes 0.
        self.store.add_chunks(
            doc_id, texts=[c.text for c in chunks], embeddings=embeddings,
            metadatas=[{"source": source_name, "page": c.page if c.page is not None else 0,
                        "chunk_index": c.chunk_index, "ingested_at": now} for c in chunks])

    def reembed_all(self) -> None:
        """Re-chunk + re-embed every 'done' document from its stored conversion JSON,
        without re-converting or touching the watched folder. Used when the embedding
        model changes and the vector store has just been reset (spec §1: 'triggers a
        full re-embed of the corpus')."""
        for rec in self.registry.list_all():
            if rec.status == "done" and rec.json_path and Path(rec.json_path).exists():
                source_name = Path(rec.source_path).name
                self._chunk_embed_store(rec.doc_id, Path(rec.json_path), source_name)

    def remove_document(self, doc_id: str) -> None:
        rec = self.registry.get(doc_id)
        if rec is None:
            return
        self.store.delete_document(doc_id)
        for p in (rec.md_path, rec.json_path, rec.archive_path):
            if p and Path(p).exists():
                Path(p).unlink()
        self.registry.delete(doc_id)

    def handle_deleted(self, path: Path) -> None:
        """Watched-folder deletion of a not-yet-archived file."""
        rec = self.registry.get_by_source(str(path))
        if rec and rec.status != "done":
            self.store.delete_document(rec.doc_id)
            self.registry.delete(rec.doc_id)

    def scan_existing(self) -> None:
        """Startup catch-up: (re)ingest every file currently in the watched folder.
        Covers files dropped while the app was closed, and rows stuck in 'processing'
        after a crash (ingest_file re-attempts anything not status=='done')."""
        for p in sorted(Path(self.archiver.watched_folder).rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                self.ingest_file(p)
```

Also create an `Embedder` adapter in `ingest.py` (used by main wiring, trivially thin, no test needed beyond integration):

```python
class OllamaEmbedder:
    def __init__(self, client, model: str):
        self.client, self.model = client, model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(self.model, texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(self.model, [text])[0]
```

- [ ] **Step 3: Verify pass**, **Step 4: Commit** — `git commit -am "feat: ingest orchestration (convert-chunk-embed-store-archive)"`

### Task 13: Watcher

**Files:** Create: `src/ragnar/watcher.py`, Test: `tests/test_watcher.py`

- [ ] **Step 1: Write failing tests** (short debounce, real tmp dirs, thread-based)

```python
import time
from pathlib import Path
from ragnar.watcher import FolderWatcher

class Recorder:
    def __init__(self):
        self.ingested, self.deleted = [], []
    def ingest(self, p): self.ingested.append(Path(p).name)
    def delete(self, p): self.deleted.append(Path(p).name)

def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond(): return True
        time.sleep(0.05)
    return False

def test_new_file_triggers_ingest_after_quiet_period(tmp_path):
    rec = Recorder()
    w = FolderWatcher(tmp_path, on_file_ready=rec.ingest, on_file_deleted=rec.delete,
                      debounce_seconds=0.2)
    w.start()
    try:
        (tmp_path / "a.md").write_text("hello")
        assert wait_for(lambda: rec.ingested == ["a.md"])
    finally:
        w.stop()

def test_rapid_writes_are_debounced_to_one_ingest(tmp_path):
    rec = Recorder()
    w = FolderWatcher(tmp_path, on_file_ready=rec.ingest, on_file_deleted=rec.delete,
                      debounce_seconds=0.3)
    w.start()
    try:
        f = tmp_path / "a.md"
        for i in range(5):
            f.write_text(f"v{i}"); time.sleep(0.05)
        assert wait_for(lambda: len(rec.ingested) == 1)
        time.sleep(0.5)
        assert rec.ingested == ["a.md"]
    finally:
        w.stop()

def test_suppressed_path_does_not_fire_deletion(tmp_path):
    rec = Recorder()
    w = FolderWatcher(tmp_path, on_file_ready=rec.ingest, on_file_deleted=rec.delete,
                      debounce_seconds=0.2)
    f = tmp_path / "a.md"; f.write_text("x")
    w.start()
    try:
        assert wait_for(lambda: rec.ingested == ["a.md"])
        w.suppress(f)          # what the Archiver's on_self_move calls
        f.unlink()             # simulates the archive move
        time.sleep(0.5)
        assert rec.deleted == []
        # and suppression is one-shot: a REAL later deletion still fires
        f.write_text("y")
        assert wait_for(lambda: len(rec.ingested) == 2)
        f.unlink()
        assert wait_for(lambda: rec.deleted == ["a.md"])
    finally:
        w.stop()

def test_ready_files_are_processed_one_at_a_time_not_concurrently(tmp_path):
    active = []
    max_concurrent = []
    def slow_ingest(p):
        active.append(1)
        max_concurrent.append(len(active))
        time.sleep(0.15)
        active.pop()
    w = FolderWatcher(tmp_path, on_file_ready=slow_ingest, on_file_deleted=lambda p: None,
                      debounce_seconds=0.1)
    w.start()
    try:
        for name in ("a.md", "b.md", "c.md"):
            (tmp_path / name).write_text(name)
        assert wait_for(lambda: len(max_concurrent) == 3, timeout=5.0)
        assert max(max_concurrent) == 1  # never more than one in flight
    finally:
        w.stop()

def test_hidden_files_ignored(tmp_path):
    rec = Recorder()
    w = FolderWatcher(tmp_path, on_file_ready=rec.ingest, on_file_deleted=rec.delete,
                      debounce_seconds=0.2)
    w.start()
    try:
        (tmp_path / ".DS_Store").write_text("junk")
        time.sleep(0.5)
        assert rec.ingested == []
    finally:
        w.stop()
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
import queue
import threading
from pathlib import Path
from typing import Callable
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class FolderWatcher(FileSystemEventHandler):
    """Watches a folder; fires on_file_ready after a per-file quiet period.

    Ready paths are handed to a single worker thread, so ingestion of a bulk
    drop runs one file at a time (spec assumes sequential processing; running
    Docling/embedding calls concurrently across per-file Timer threads would
    race on shared resources like the lazy-initialized Docling converter).

    suppress(path) marks the next deletion event for that path as self-inflicted
    (the archiver moving the file out) so it is swallowed once.
    """

    def __init__(self, folder: Path, on_file_ready: Callable, on_file_deleted: Callable,
                 debounce_seconds: float = 1.5):
        self.folder = Path(folder)
        self.on_file_ready, self.on_file_deleted = on_file_ready, on_file_deleted
        self.debounce = debounce_seconds
        self._timers: dict[str, threading.Timer] = {}
        self._suppressed: set[str] = set()
        self._lock = threading.Lock()
        self._observer = Observer()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._drain_queue, daemon=True)

    def start(self) -> None:
        self._worker.start()
        self._observer.schedule(self, str(self.folder), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()
        self._observer.stop()
        self._observer.join(timeout=5)
        self._queue.put(None)  # sentinel: stop the worker
        self._worker.join(timeout=5)

    def suppress(self, path: Path) -> None:
        with self._lock:
            self._suppressed.add(str(path))
            t = self._timers.pop(str(path), None)
        if t:
            t.cancel()

    @staticmethod
    def _relevant(path_str: str) -> bool:
        return not Path(path_str).name.startswith(".")

    def _schedule(self, path_str: str) -> None:
        if not self._relevant(path_str):
            return
        with self._lock:
            old = self._timers.pop(path_str, None)
            if old:
                old.cancel()
            t = threading.Timer(self.debounce, self._fire, args=(path_str,))
            self._timers[path_str] = t
        t.start()

    def _fire(self, path_str: str) -> None:
        with self._lock:
            self._timers.pop(path_str, None)
            if path_str in self._suppressed:
                self._suppressed.discard(path_str)
                return
        self._queue.put(path_str)  # hand off to the single worker; don't run inline

    def _drain_queue(self) -> None:
        while True:
            path_str = self._queue.get()
            if path_str is None:  # stop sentinel
                return
            if Path(path_str).exists():
                self.on_file_ready(Path(path_str))

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule(event.dest_path)

    def on_deleted(self, event):
        if event.is_directory or not self._relevant(event.src_path):
            return
        with self._lock:
            t = self._timers.pop(event.src_path, None)
            if event.src_path in self._suppressed:
                self._suppressed.discard(event.src_path)
                if t:
                    t.cancel()
                return
        if t:
            t.cancel()
        self.on_file_deleted(Path(event.src_path))
```

- [ ] **Step 3: Verify pass** (these are timing tests; if flaky on CI, raise `wait_for` timeout — do not add sleeps to production code).
- [ ] **Step 4: Commit** — `git commit -am "feat: debounced folder watcher with self-move suppression"`

---

## Phase 8: API + Web UI

### Task 14: FastAPI app

**Files:** Create: `src/ragnar/api.py`, Test: `tests/test_api.py`

Endpoints: `GET /` (index.html) · `POST /api/chat` · `GET /api/documents` · `DELETE /api/documents/{id}` · `GET /api/documents/{id}/markdown` · `POST /api/upload` · `GET /api/status` · `POST /api/setup/pull` + `GET /api/setup/pull/{model}` · `POST /api/setup/complete`.

- [ ] **Step 1: Write failing tests** (FastAPI TestClient; fakes injected through `create_app`)

```python
from pathlib import Path
from fastapi.testclient import TestClient
from ragnar.api import create_app
from ragnar.chat import ChatResponse
from ragnar.config import Config

class FakeChat:
    def answer(self, q):
        return ChatResponse(answer=f"echo:{q}", refused=False,
                            citations=[{"source": "a.pdf", "page": 1}])

class FakeIngest:
    def __init__(self): self.removed = []
    def remove_document(self, doc_id): self.removed.append(doc_id)

class FakeOllama:
    def __init__(self, up=True): self.up = up
    def is_up(self): return self.up
    def has_model(self, m): return True
    def pull(self, m): pass

def make_client(tmp_path, registry=None):
    from ragnar.registry import Registry
    cfg = Config(data_dir=tmp_path)
    cfg.settings.watched_folder = str(tmp_path / "watched")
    cfg.ensure_dirs()
    reg = registry or Registry(cfg.registry_path)
    app = create_app(config=cfg, registry=reg, chat=FakeChat(),
                     ingest=FakeIngest(), ollama=FakeOllama())
    return TestClient(app), cfg, reg

def test_chat_endpoint(tmp_path):
    client, cfg, reg = make_client(tmp_path)
    reg.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h", status="done")  # non-empty corpus
    r = client.post("/api/chat", json={"question": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo:hi" and body["citations"][0]["source"] == "a.pdf"

def test_upload_saves_into_watched_folder(tmp_path):
    client, cfg, _ = make_client(tmp_path)
    r = client.post("/api/upload", files={"file": ("x.md", b"# hi", "text/markdown")})
    assert r.status_code == 200
    assert (Path(cfg.settings.watched_folder) / "x.md").read_bytes() == b"# hi"

def test_upload_rejects_unsupported_type(tmp_path):
    client, *_ = make_client(tmp_path)
    r = client.post("/api/upload", files={"file": ("x.exe", b"MZ", "application/x-dos")})
    assert r.status_code == 422

def test_documents_list_and_markdown_view(tmp_path):
    client, cfg, reg = make_client(tmp_path)
    md = cfg.converted_dir / "d1.md"; md.write_text("# converted")
    reg.upsert(doc_id="d1", source_path="/w/a.pdf", content_hash="h", status="processing")
    reg.mark_done("d1", md_path=str(md), json_path="j", archive_path=None)
    docs = client.get("/api/documents").json()
    assert docs[0]["doc_id"] == "d1" and docs[0]["status"] == "done"
    assert client.get("/api/documents/d1/markdown").text == "# converted"

def test_delete_document(tmp_path):
    client, *_ = make_client(tmp_path)
    assert client.delete("/api/documents/d1").status_code == 200

def test_chat_with_empty_corpus_says_so_instead_of_generic_refusal(tmp_path):
    client, cfg, reg = make_client(tmp_path)
    r = client.post("/api/chat", json={"question": "anything"}).json()
    assert r["refused"] is True
    assert "no documents" in r["answer"].lower()

def test_status_reports_setup_state(tmp_path):
    client, *_ = make_client(tmp_path)
    s = client.get("/api/status").json()
    assert s["ollama_up"] is True and s["onboarding_complete"] is False

def test_setup_complete_persists(tmp_path):
    client, cfg, _ = make_client(tmp_path)
    r = client.post("/api/setup/complete",
                    json={"watched_folder": str(tmp_path / "docs2"), "llm_model": "llama3.1:8b"})
    assert r.status_code == 200
    assert Config(data_dir=tmp_path).settings.onboarding_complete is True
```

- [ ] **Step 2: Verify fail**, implement:

```python
from __future__ import annotations
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from ragnar.convert import SUPPORTED

WEBUI_DIR = Path(__file__).parent / "webui"


class ChatIn(BaseModel):
    question: str


class SetupIn(BaseModel):
    watched_folder: str | None = None
    llm_model: str | None = None


def create_app(config, registry, chat, ingest, ollama) -> FastAPI:
    app = FastAPI()
    pulls: dict[str, str] = {}  # model -> "pulling" | "done" | "error: ..."

    @app.get("/")
    def index():
        return FileResponse(WEBUI_DIR / "index.html")

    @app.get("/static/{name}")
    def static(name: str):
        f = WEBUI_DIR / name
        if not f.is_file() or "/" in name:
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/chat")
    def chat_endpoint(body: ChatIn):
        if not registry.list_all():
            return {"answer": "No documents have been ingested yet — add one to get started.",
                    "refused": True, "citations": [], "related": []}
        r = chat.answer(body.question)
        return {"answer": r.answer, "refused": r.refused,
                "citations": r.citations, "related": r.related}

    @app.get("/api/documents")
    def documents():
        return [{"doc_id": d.doc_id, "source_path": d.source_path, "status": d.status,
                 "error": d.error, "ingested_at": d.ingested_at}
                for d in registry.list_all()]

    @app.get("/api/documents/{doc_id}/markdown", response_class=PlainTextResponse)
    def doc_markdown(doc_id: str):
        rec = registry.get(doc_id)
        if not rec or not rec.md_path or not Path(rec.md_path).exists():
            raise HTTPException(404)
        return Path(rec.md_path).read_text()

    @app.delete("/api/documents/{doc_id}")
    def doc_delete(doc_id: str):
        ingest.remove_document(doc_id)
        return {"ok": True}

    @app.post("/api/upload")
    def upload(file: UploadFile):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED:
            raise HTTPException(422, f"unsupported file type {ext}")
        dest = Path(config.settings.watched_folder) / Path(file.filename).name
        dest.write_bytes(file.file.read())  # watcher picks it up from here
        return {"ok": True, "filename": dest.name}

    @app.get("/api/status")
    def status():
        return {"ollama_up": ollama.is_up(),
                "onboarding_complete": config.settings.onboarding_complete,
                "llm_model": config.settings.llm_model,
                "llm_model_ready": ollama.has_model(config.settings.llm_model),
                "embed_model_ready": ollama.has_model(config.settings.embed_model),
                "ingesting": registry.count_processing()}

    @app.post("/api/setup/pull")
    def setup_pull(body: dict):
        model = body["model"]
        pulls[model] = "pulling"

        def run():
            try:
                ollama.pull(model)
                pulls[model] = "done"
            except Exception as e:
                pulls[model] = f"error: {e}"
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    @app.get("/api/setup/pull/{model:path}")
    def pull_status(model: str):
        return {"status": pulls.get(model, "unknown")}

    @app.post("/api/setup/complete")
    def setup_complete(body: SetupIn):
        if body.watched_folder:
            config.settings.watched_folder = body.watched_folder
        if body.llm_model:
            config.settings.llm_model = body.llm_model
        config.settings.onboarding_complete = True
        config.save()
        config.ensure_dirs()
        return {"ok": True}

    return app
```

- [ ] **Step 3: Verify pass** (index test will 404 until Task 15 creates webui files — write `test_index` in Task 15 instead).
- [ ] **Step 4: Commit** — `git commit -am "feat: fastapi app with chat/documents/upload/status/setup"`

### Task 15: Web UI

**Files:** Create: `src/ragnar/webui/index.html`, `src/ragnar/webui/app.js`, `src/ragnar/webui/style.css`; Test: add `test_index_served` to `tests/test_api.py`

No unit tests for JS (manual UAT covers it). Single-page app with three views driven by `/api/status`: **wizard** (Ollama check → model pull → folder → first doc), **chat**, **documents panel**.

- [ ] **Step 1: Add the serving test**

```python
def test_index_served(tmp_path):
    client, *_ = make_client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200 and "Ragnar" in r.text
```

- [ ] **Step 2: index.html**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ragnar</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<div id="wizard" class="view hidden">
  <h1>Welcome to Ragnar</h1>
  <div id="wiz-ollama" class="step hidden">
    <h2>1 · Ollama</h2>
    <p>Ragnar needs <a href="https://ollama.com/download" target="_blank">Ollama</a> running locally. Install it, then click Recheck.</p>
    <button onclick="recheck()">Recheck</button>
  </div>
  <div id="wiz-models" class="step hidden">
    <h2>2 · Models</h2>
    <label>Chat model:
      <select id="llm-select">
        <option value="qwen2.5:14b">Qwen2.5 14B — best quality, needs ≥16 GB RAM</option>
        <option value="llama3.1:8b">Llama 3.1 8B — lighter, needs ≥8 GB RAM</option>
      </select></label>
    <button onclick="pullModels()">Download models</button>
    <p id="pull-progress"></p>
  </div>
  <div id="wiz-folder" class="step hidden">
    <h2>3 · Watched folder</h2>
    <p>Files dropped into this folder are ingested automatically.</p>
    <input id="folder-input" size="50">
    <button onclick="completeSetup()">Finish setup</button>
  </div>
</div>

<div id="main" class="view hidden">
  <aside id="docs-panel">
    <h2>Documents <span id="ingesting-badge" class="hidden"></span></h2>
    <input type="file" id="file-input" multiple>
    <ul id="doc-list"></ul>
    <pre id="md-view" class="hidden"></pre>
  </aside>
  <section id="chat-panel">
    <div id="messages"></div>
    <p id="first-doc-hint" class="hidden">Add your first document on the left to start asking questions.</p>
    <form id="chat-form">
      <input id="question" placeholder="Ask about your documents…" autocomplete="off">
      <button>Send</button>
    </form>
  </section>
</div>
<div id="ollama-warning" class="banner hidden">⚠ Ollama is not running — chat is unavailable.</div>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: app.js** — complete behavior:

```javascript
const $ = (id) => document.getElementById(id);
const show = (id) => $(id).classList.remove("hidden");
const hide = (id) => $(id).classList.add("hidden");

let status = null;
let selectedLlmModel = null;  // set by pullModels(), sent by completeSetup()

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}

async function refreshStatus() {
  status = await api("/api/status");
  route();
}

function route() {
  ["wizard", "main"].forEach(hide);
  ["wiz-ollama", "wiz-models", "wiz-folder"].forEach(hide);
  $("ollama-warning").classList.toggle("hidden", status.ollama_up);
  if (!status.onboarding_complete) {
    show("wizard");
    if (!status.ollama_up) show("wiz-ollama");
    else if (!status.llm_model_ready || !status.embed_model_ready) show("wiz-models");
    else show("wiz-folder");
  } else {
    show("main");
    refreshDocs();
  }
}

async function recheck() { await refreshStatus(); }

async function pullModels() {
  const llm = $("llm-select").value;
  for (const m of [llm, "bge-m3"]) await api("/api/setup/pull", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({model: m})});
  $("pull-progress").textContent = "Downloading… this can take several minutes.";
  const timer = setInterval(async () => {
    const a = await api(`/api/setup/pull/${llm}`), b = await api("/api/setup/pull/bge-m3");
    $("pull-progress").textContent = `chat model: ${a.status} · embeddings: ${b.status}`;
    if (a.status === "done" && b.status === "done") {
      clearInterval(timer);
      // NOTE: do not call /api/setup/complete here — that flag means "onboarding
      // finished", and setting it now would skip the folder step (step 3) entirely.
      // Just refresh status; route() advances to wiz-folder once models are ready.
      selectedLlmModel = llm;
      await refreshStatus();
    }
  }, 2000);
}

async function completeSetup() {
  await api("/api/setup/complete", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({watched_folder: $("folder-input").value || null,
                          llm_model: selectedLlmModel})});
  await refreshStatus();
}

async function refreshDocs() {
  const docs = await api("/api/documents");
  $("first-doc-hint").classList.toggle("hidden", docs.length > 0);
  const badge = $("ingesting-badge");
  badge.classList.toggle("hidden", status.ingesting === 0);
  badge.textContent = status.ingesting ? `ingesting ${status.ingesting}…` : "";
  $("doc-list").innerHTML = "";
  for (const d of docs) {
    const li = document.createElement("li");
    const name = d.source_path.split("/").pop();
    li.innerHTML = `<span class="doc-name" title="${d.error ?? ""}">${name}
        <em class="st-${d.status}">${d.status}</em></span>`;
    const view = document.createElement("button");
    view.textContent = "view";
    view.onclick = async () => { $("md-view").textContent =
        await api(`/api/documents/${d.doc_id}/markdown`); show("md-view"); };
    const rm = document.createElement("button");
    rm.textContent = "remove";
    rm.onclick = async () => { await api(`/api/documents/${d.doc_id}`,
        {method: "DELETE"}); refreshDocs(); };
    li.append(view, rm);
    $("doc-list").append(li);
  }
}

$("file-input").addEventListener("change", async (e) => {
  for (const f of e.target.files) {
    const fd = new FormData(); fd.append("file", f);
    await api("/api/upload", {method: "POST", body: fd});
  }
  e.target.value = "";
  setTimeout(refreshDocs, 500);
});

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  addMsg("user", q);
  const thinking = addMsg("assistant", "…");
  try {
    const r = await api("/api/chat", {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: q})});
    let html = escapeHtml(r.answer);
    const refs = r.refused ? r.related : r.citations;
    if (refs?.length) {
      html += `<details><summary>Sources</summary><ul>` + refs.map(c =>
        `<li>${escapeHtml(c.source)}${c.page ? ` — page ${c.page}` : ""}</li>`).join("") +
        `</ul></details>`;
    }
    thinking.innerHTML = html;
    thinking.classList.toggle("refused", r.refused);
  } catch (err) {
    thinking.textContent = `Error: ${err.message}`;
  }
});

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  $("messages").append(div);
  div.scrollIntoView();
  return div;
}

function escapeHtml(s) {
  const d = document.createElement("div"); d.textContent = s; return d.innerHTML;
}

refreshStatus();
setInterval(async () => { if (status?.onboarding_complete) {
  status = await api("/api/status"); refreshDocs(); } }, 5000);
```

- [ ] **Step 4: style.css** — keep it minimal (system font, flex layout: `#main{display:flex;height:100vh}`, `#docs-panel{width:320px;overflow:auto;border-right:1px solid #ddd;padding:12px}`, `#chat-panel{flex:1;display:flex;flex-direction:column}`, `#messages{flex:1;overflow:auto;padding:16px}`, `.msg.user{text-align:right;background:#e8f0fe}`, `.msg{margin:8px;padding:10px;border-radius:8px;background:#f5f5f5}`, `.refused{background:#fff3e0}`, `.hidden{display:none}`, `.banner{position:fixed;bottom:0;width:100%;background:#b00020;color:#fff;padding:8px;text-align:center}`, `.st-failed{color:#b00020}`, `.st-processing{color:#e65100}`). Executor: write a complete file following these rules.
- [ ] **Step 5: Verify** — `pytest tests/test_api.py -v` all pass; then manual smoke: run Task 16's dev entry (or a 5-line uvicorn script) and click through in a browser.
- [ ] **Step 6: Commit** — `git commit -am "feat: web ui (wizard, chat, documents panel)"`

---

## Phase 9: Wiring + native window

### Task 16: main.py — composition root + pywebview

**Files:** Create: `src/ragnar/main.py`, Test: `tests/test_main.py` (wiring only)

- [ ] **Step 1: Write failing test**

```python
from ragnar.main import build_services
from ragnar.config import Config

def test_build_services_wires_everything(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.settings.watched_folder = str(tmp_path / "w")
    svc = build_services(cfg)
    assert svc.app is not None and svc.watcher is not None
    assert svc.ingest.registry is svc.registry

def test_embed_model_change_triggers_reembed_not_silent_data_loss(tmp_path, monkeypatch):
    # Regression test for the bug where switching embed models wiped the vector
    # store and left it empty forever (registry rows stayed 'done' with nothing
    # to re-trigger ingestion). build_services must call ingest.reembed_all().
    calls = []
    monkeypatch.setattr("ragnar.ingest.IngestService.reembed_all", lambda self: calls.append(1))
    cfg = Config(data_dir=tmp_path)
    cfg.settings.watched_folder = str(tmp_path / "w")
    cfg.settings.embed_model = "bge-m3"
    build_services(cfg)  # first run: store is fresh, embed_model matches -> no reembed needed
    assert calls == []
    cfg2 = Config(data_dir=tmp_path)
    cfg2.settings.embed_model = "new-embed-model"  # simulate a model switch on next launch
    build_services(cfg2)
    assert calls == [1]  # reembed_all was invoked exactly once
```

- [ ] **Step 2: Implement**

```python
from __future__ import annotations
import threading
from dataclasses import dataclass
from pathlib import Path
from ragnar.api import create_app
from ragnar.archive import Archiver
from ragnar.chat import ChatService
from ragnar.config import Config
from ragnar.convert import Converter
from ragnar.ingest import IngestService, OllamaEmbedder
from ragnar.ollama_client import OllamaClient
from ragnar.registry import Registry
from ragnar.reranker import Reranker
from ragnar.retriever import Retriever
from ragnar.store import VectorStore
from ragnar.watcher import FolderWatcher

HOST, PORT = "127.0.0.1", 8756


@dataclass
class Services:
    app: object
    watcher: FolderWatcher
    ingest: IngestService
    registry: Registry


def build_services(cfg: Config, bundled_models_dir: Path | None = None) -> Services:
    cfg.ensure_dirs()
    registry = Registry(cfg.registry_path)
    ollama = OllamaClient()
    embedder = OllamaEmbedder(ollama, cfg.settings.embed_model)
    store = VectorStore(cfg.chroma_dir, embed_model_id=cfg.settings.embed_model)
    needs_reembed = store.needs_reembed
    if needs_reembed:
        store.reset()  # wipes vectors; re-embed of the existing corpus happens below,
                       # once `ingest` exists, via reembed_all() (spec §1: embed-model
                       # mismatch "triggers a full re-embed of the corpus")
    reranker = Reranker(model_path=str(bundled_models_dir / "reranker")
                        if bundled_models_dir else "BAAI/bge-reranker-v2-m3")
    retriever = Retriever(store, embedder, reranker, floor=cfg.settings.rerank_floor)
    chat = ChatService(retriever, ollama, llm_model=cfg.settings.llm_model)
    converter = Converter(cfg.converted_dir,
                          artifacts_path=str(bundled_models_dir / "docling")
                          if bundled_models_dir else None)
    watcher_holder: list[FolderWatcher] = []
    archiver = Archiver(Path(cfg.settings.watched_folder), cfg.archive_dir,
                        on_self_move=lambda p: watcher_holder[0].suppress(p)
                        if watcher_holder else None)
    ingest = IngestService(registry=registry, store=store, archiver=archiver,
                           converter=converter, embedder=embedder)
    if needs_reembed:
        ingest.reembed_all()
    watcher = FolderWatcher(Path(cfg.settings.watched_folder),
                            on_file_ready=ingest.ingest_file,
                            on_file_deleted=ingest.handle_deleted)
    watcher_holder.append(watcher)
    app = create_app(config=cfg, registry=registry, chat=chat, ingest=ingest, ollama=ollama)
    return Services(app=app, watcher=watcher, ingest=ingest, registry=registry)


def run() -> None:
    import uvicorn, webview
    cfg = Config()
    svc = build_services(cfg, bundled_models_dir=_bundled_models_dir())
    svc.ingest.scan_existing()  # catch up on files dropped / crashes while the app was closed
    svc.watcher.start()
    server = uvicorn.Server(uvicorn.Config(svc.app, host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    webview.create_window("Ragnar", f"http://{HOST}:{PORT}", width=1200, height=800)
    webview.start()
    svc.watcher.stop()


def _bundled_models_dir() -> Path | None:
    import sys
    if getattr(sys, "frozen", False):  # running inside PyInstaller bundle
        return Path(sys._MEIPASS) / "models"
    return None


if __name__ == "__main__":
    run()
```
Note: `build_services` must not import docling/torch at module level (they're lazy inside Converter/Reranker) so this test stays fast. If the wiring test is slow, that's a regression — check for eager imports.

- [ ] **Step 3: Verify pass**; also manual dev run: `python -m ragnar.main` opens a native window (requires Ollama running for full function).
- [ ] **Step 4: Commit** — `git commit -am "feat: composition root + pywebview entry"`

---

## Phase 10: Packaging

### Task 17: Model prefetch + PyInstaller build + dmg

**Files:** Create: `build/fetch_models.py`, `build/ragnar.spec`, `build/build.sh`, `docs/INSTALL.md`

**Follow the route chosen in Task 0.** The steps below assume PyInstaller worked; if the spike selected a fallback, adapt per the spec's fallback chain and record the deviation in the plan.

- [ ] **Step 1: fetch_models.py** — downloads into `build/models/` (build-time only, dev machine):

```python
"""Build-time model prefetch. Run once before build.sh."""
from pathlib import Path
from huggingface_hub import snapshot_download

OUT = Path(__file__).parent / "models"
snapshot_download("BAAI/bge-reranker-v2-m3", local_dir=OUT / "reranker")
snapshot_download("BAAI/bge-m3", local_dir=OUT / "tokenizer",
                  allow_patterns=["tokenizer*", "*.json"])  # tokenizer only, not weights
import subprocess
subprocess.run(["docling-tools", "models", "download", "-o", str(OUT / "docling")], check=True)
```

- [ ] **Step 2: ragnar.spec** — start from `pyi-makespec --windowed --name Ragnar src/ragnar/main.py`, then add (using the exact flags proven in Task 0):
  - `datas`: `src/ragnar/webui → ragnar/webui`, `build/models → models`
  - the `--collect-all` equivalents (`collect_all("docling")`, etc.) from the spike
  - `info_plist={"NSHighResolutionCapable": True}`
- [ ] **Step 3: build.sh**

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python build/fetch_models.py
pyinstaller build/ragnar.spec --noconfirm
mkdir -p dist/dmg && cp -R "dist/Ragnar.app" dist/dmg/
ln -sf /Applications dist/dmg/Applications
hdiutil create -volname Ragnar -srcfolder dist/dmg -ov -format UDZO dist/Ragnar.dmg
echo "dmg: $(du -h dist/Ragnar.dmg | cut -f1)"
```

- [ ] **Step 4: docs/INSTALL.md** — teammate instructions: install Ollama; open dmg, drag to Applications; first launch is **right-click → Open → Open** (unsigned app, Gatekeeper); wizard walks through the rest.
- [ ] **Step 5: Verify** — build, then repeat the Task 0 clean-machine protocol **on the built Ragnar.app**: HF cache renamed, Wi-Fi off, launch, complete the wizard against a running Ollama (Ollama model pulls DO need network — turn Wi-Fi on only for the pull step, off again for chat), ingest one PDF, ask one question that's answered, one that must be refused.
- [ ] **Step 6: Commit** — `git commit -am "feat: packaging (model prefetch, pyinstaller spec, dmg)"`

---

## Phase 11: Integration + UAT

### Task 18: End-to-end integration test

**Files:** Test: `tests/test_integration.py`

Deterministic E2E without ML: real registry/store/archiver/watcher/api; fake embedder (keyword-vector), fake reranker-model (keyword overlap — reuse `FakeCrossEncoder`), fake LLM; converter faked for speed. Verifies the spec's core promise end-to-end: right chunk retrieved and cited; out-of-corpus question refused with related docs.

- [ ] **Step 1: Write the test**

```python
import time
from pathlib import Path
from fastapi.testclient import TestClient
# tests/ has __init__.py (Task 1), so fakes defined in other test modules are
# importable directly — no need to redefine or copy-paste them here.
from tests.test_ingest import FakeConverter
from tests.test_reranker import FakeCrossEncoder
from tests.test_chat import FakeLLM

VOCAB = ["vacation", "laptop", "server", "days", "code"]

class KeywordEmbedder:
    def _vec(self, text):
        t = text.lower()
        return [float(t.count(w)) for w in VOCAB]
    def embed_texts(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)

def test_full_pipeline_answer_and_refusal(tmp_path):
    # Assemble like main.build_services, but substitute: FakeConverter (imported above)
    # for Docling, KeywordEmbedder (defined above) for Ollama embeddings,
    # Reranker(model=FakeCrossEncoder()) (imported above) for the real cross-encoder,
    # and FakeLLM("Employees get 26 days [1].") (imported above) in place of Ollama chat.
    ...
    # 1. drop a file into the watched folder
    watched = Path(cfg.settings.watched_folder)
    (watched / "policy.md").write_text("# Vacation\nEmployees receive 26 vacation days.")
    watcher.start()
    try:
        deadline = time.time() + 10
        while registry.count_processing() >= 0 and not registry.list_all():
            assert time.time() < deadline, "ingestion never happened"
            time.sleep(0.1)
        while registry.list_all()[0].status == "processing":
            time.sleep(0.1)
        assert registry.list_all()[0].status == "done"
        assert not (watched / "policy.md").exists()  # archived
        client = TestClient(app)
        # 2. in-corpus question → answered with citation
        r = client.post("/api/chat", json={"question": "how many vacation days?"}).json()
        assert r["refused"] is False and r["citations"][0]["source"] == "policy.md"
        # 3. out-of-corpus question → refusal with related docs
        r = client.post("/api/chat", json={"question": "what is the wifi password?"}).json()
        assert r["refused"] is True
    finally:
        watcher.stop()
```
The executor fills in the wiring block from Task 16's `build_services`, substituting the fakes; keep floor at a value where keyword overlap separates the two questions (e.g. floor=1.0 with FakeCrossEncoder integer scores).

- [ ] **Step 2: Verify pass**, **Step 3: Commit** — `git commit -am "test: end-to-end pipeline (ingest→retrieve→answer/refuse)"`

### Task 19: Calibrate rerank floor + UAT checklist

**Files:** Create: `docs/UAT.md`; Modify: `src/ragnar/config.py` (default `rerank_floor` if calibration says so)

- [ ] **Step 1: Calibrate floor with the real reranker** — write a scratch script (not committed) that loads the real CrossEncoder, scores ~10 relevant and ~10 irrelevant (question, chunk) pairs from your own sample docs, prints score distributions. Pick a floor that cleanly separates them (bge-reranker-v2-m3 raw scores are logits, typically ≈ −10…+10; sigmoid them or floor on raw — decide from the data and document the choice in `config.py` as a comment). Update the `Settings.rerank_floor` default.
- [ ] **Step 2: Write docs/UAT.md** — manual checklist mirroring spec §Testing:

```markdown
# Ragnar UAT checklist
Setup: fresh data dir (`rm -rf ~/Library/Application\ Support/Ragnar`), Ollama running.
- [ ] Wizard: Ollama down → step 1 shows guidance; start Ollama → Recheck advances
- [ ] Wizard: model pull shows progress and completes
- [ ] Wizard: folder default accepted / custom path works
- [ ] Upload a PDF via UI → status processing → done; original vanishes from watched folder into archive
- [ ] Drop DOCX + XLSX into watched folder → both ingest
- [ ] Drop a scanned (image-only) PDF → ingests via OCR
- [ ] Ask question answered by a doc → grounded answer + correct source/page citation
- [ ] Ask question NOT in any doc → refusal + related documents listed
- [ ] Ask in Polish about a Polish doc → answered in Polish
- [ ] "view" shows converted markdown; tables from XLSX render as markdown tables
- [ ] "remove" deletes doc; asking about its content now refuses
- [ ] Replace a file with changed content at same name → answers reflect new content
- [ ] Quit and relaunch → corpus intact, no wizard
- [ ] Kill Ollama mid-session → banner appears; chat errors gracefully
```

- [ ] **Step 3: Execute the checklist against the dev build** (`python -m ragnar.main`), fix what fails, then against the frozen `.app` (Task 17 Step 5 covers the offline part).
- [ ] **Step 4: Commit** — `git commit -am "docs: UAT checklist + calibrated rerank floor"`

---

## Execution notes

- **Order is strict through Phase 7** (each phase builds on the previous). Phases 8–9 (API/UI/wiring) could interleave, but do them in order anyway — simpler.
- **Task 0 gates everything**: report spike results to the human before Task 1.
- **LLM model benchmarking** (Qwen2.5 14B vs Llama 3.1 8B, spec §3) happens informally during Task 19 UAT on the dev Mac; both stay offered in the wizard regardless.
- Deviations from this plan (API mismatches in docling-core, chromadb, etc.) are expected in the small — keep the *assertions and behaviors* fixed, adapt the incidental API calls, and note deviations in commit messages.
