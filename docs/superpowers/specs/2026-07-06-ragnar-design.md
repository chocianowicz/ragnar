# Ragnar — Local Document RAG Chat — Design

## Purpose

A fully offline, local RAG (retrieval-augmented generation) chat tool for a small team. Users ingest their own documents (PDF, Word, Excel, scanned images, text) and ask questions in a chat UI. The system answers **only** from ingested documents, citing sources, and explicitly refuses (with related-document suggestions) when the answer isn't in the corpus.

Distributed as a native macOS `.dmg` app. Each teammate installs their own copy and runs an independent instance against their own local document corpus — no shared server, no network exposure required.

## Constraints

- **Fully offline**: no external API calls. Local LLM, local embeddings, local reranker, local vector store — all via Ollama / local Python libraries.
- **Hardware**: Apple Silicon Macs.
- **Users**: small team, each running their own instance sequentially (no concurrency requirements).
- **Distribution**: unsigned `.dmg` (no Apple Developer account) — users bypass Gatekeeper via right-click → Open, documented in a short setup note.

## Architecture

```
[Watched folder] ─┐
[UI upload]────────┼─→ [Docling: convert to .md] → [archive/originals]
                    │            │
                    │            ↓
                    └─→ [Chunk + embed .md] → [ChromaDB]
                                                    ↑
[Chat UI] → [FastAPI /chat] → [Retriever + Reranker] ─┘
                    ↓
            [Ollama LLM] → answer + citations (or refusal + related docs)
```

Packaged as a native app: `pywebview` wraps the FastAPI backend + HTML/JS chat UI in a native window; `PyInstaller` bundles the Python runtime and dependencies into a `.app`, wrapped into a `.dmg`.

## Components

### 1. Ingestion pipeline

Every supported file is first normalized to markdown, then only the markdown is chunked/embedded. This replaces bespoke per-format parsing logic with one conversion step and a single, uniform chunking strategy.

- **Watcher**: background process (`watchdog`) monitors a configured folder for new/changed/deleted files.
- **Upload**: FastAPI endpoint saves uploaded files into the same watched folder, so both paths converge on one code path.
- **Conversion (Docling)**: PDF, DOCX, XLSX, and scanned images are converted via **Docling** (layout-aware, preserves table structure as markdown tables, includes built-in OCR for scanned pages/images). User-authored TXT/MD files in the watched folder pass through unchanged. Two artifacts are produced per source document, both written to the app's internal data directory (`data/converted/`, **not** inside the watched folder, so generated files can never be re-ingested as new documents):
  - a `.md` file — the canonical human-readable representation, shown in the UI when a user inspects a source;
  - the Docling document JSON — retains per-element provenance (page numbers, bounding boxes, sheet names) that markdown flattening loses.
- **Archiving**: once conversion succeeds, the original raw file is moved into `archive/originals/` (outside the watched folder, mirroring the source folder structure), suffixed with its content hash to avoid collisions when a changed file is later re-dropped at the same path. The pipeline suppresses watcher events for moves it initiated itself, so archiving never triggers the deletion path. Originals are kept for reference/re-conversion, never deleted automatically.
- **Document removal**: since originals leave the watched folder after ingestion, users remove a document via a per-document "remove" action in the UI's document list, which deletes its chunks, converted artifacts, and archived original. Deleting a still-unprocessed file from the watched folder also works (normal deletion path).
- **Chunking**: performed from the Docling document (not the flat `.md`) so each chunk carries provenance — split on heading/section boundaries first, then within a section by paragraph, sized using the embedding model's own tokenizer (~500 tokens, ~50 overlap). Tables (from converted Excel sheets) are chunked by row-group: rows grouped on blank-row boundaries where present, else fixed-size groups (e.g. 20 rows), with the header row repeated into every group's chunk. Goal: each chunk is one coherent semantic unit (section, paragraph, table fragment), not an arbitrary token window.
- **Embedding model**: **BGE-M3** via Ollama (`ollama pull bge-m3`) — multilingual, handles long context (up to 8192 tokens), good fit if documents mix languages (e.g. Polish/English). The model name + version is stored as ChromaDB collection metadata; if it ever changes, the app detects the mismatch and triggers a full re-embed of the corpus rather than mixing incompatible vectors.
- **Versioning**: file identity = path + content hash. When a changed file appears at a known path, prior chunks for that document are deleted and it's re-converted + re-ingested (the previous archived original stays, distinguished by hash suffix).
- **Metadata per chunk**: source filename, page/sheet number (taken from the Docling document's per-element provenance during chunking), chunk index, ingestion timestamp — used later for citations.
- **Failure handling**: files that fail conversion are marked failed with a reason and skipped (original stays in the watched folder, not archived); the rest of the batch continues. Low-confidence OCR pages are converted but flagged as lower-reliability sources.
- **Debounce**: file events are debounced per-file (e.g. 1-2s of quiet time after the last write) before triggering conversion, so bulk drops or in-progress saves don't cause redundant or partial-read processing.
- **Chat availability during ingestion**: chat always answers from whatever is currently indexed; the UI shows a subtle "still ingesting N files" indicator rather than blocking chat.

### 2. Storage

- **ChromaDB**, embedded/on-disk, one instance per user's app install. Stores chunk text, embeddings, and metadata.

### 3. Retrieval + generation

- Query embedded with BGE-M3 (same model as ingestion, for consistency).
- **Retrieval**: top ~25 candidates from ChromaDB via vector similarity.
- **Reranking**: candidates reranked by `bge-reranker-v2-m3` (cross-encoder, run via `sentence-transformers`/`FlagEmbedding` locally on Apple Silicon MPS) — narrows to top 5 most relevant chunks. Reranker weights are bundled inside the app (not fetched from the HuggingFace Hub at runtime) to keep first-run fully offline; cached under the app's own data directory.
- **Similarity floor**: a concrete cutoff on the reranker's relevance score (exact value determined empirically during build/testing) — if no candidate clears it, treat as "no relevant documents found": skip the LLM call, return a deterministic refusal + the closest N chunks as "related documents you might check."
- **Prompt construction**: strict system prompt instructing the LLM to answer only from the provided chunks, cite source file + page/sheet per claim, and avoid answering when context is insufficient.
- **LLM**: local instruction-tuned model via Ollama (e.g. Qwen2.5 14B or Llama 3.1 8B — exact choice benchmarked once built, depends on available RAM).
- **Citations**: derived from the metadata of chunks actually used (not parsed from LLM prose) — more reliable than trusting the model's self-reported citations.

### 4. Web UI (native app window)

- FastAPI serves a single-page chat interface (plain HTML/JS/CSS, no build step).
- Chat view with message history; each answer shows expandable source citations (filename + page/sheet).
- Upload widget (drag-and-drop or file picker) posting to the ingestion endpoint, with ingestion status (processing/done/failed) shown per file.
- Document list panel: all ingested documents with status, a view of the converted markdown, and a per-document "remove" action (deletes chunks, converted artifacts, and archived original).
- Server binds to `localhost` only — no network exposure, no auth needed, since each user runs their own independent instance.

### 5. Native app packaging + onboarding

- `pywebview` window wraps the existing web UI (same code, no separate native UI to build).
- `PyInstaller` bundles Python backend + dependencies (including `torch`/`transformers` for the reranker and Docling, the bundled reranker model weights, and **all** Docling model artifacts — layout, TableFormer, and OCR models, which Docling otherwise downloads from HuggingFace on first run; prefetched at build time via `docling-tools models download` and pointed at via `artifacts_path` in the app data dir) into a `.app`; wrapped into a `.dmg` for distribution.
- Ollama is **not** bundled (separate installer, large binary) — the app detects and guides instead.
- **Packaging risk flag**: reliably bundling `torch`'s native libraries and MPS backend support via PyInstaller on Apple Silicon is not guaranteed to work cleanly (known issues: missing hidden imports, binary bloat, silent fallback to CPU). This risk now applies to **two** heavy ML dependencies (the reranker and Docling), increasing both bundle size and surface area for bundling issues. This should be validated with a small spike before committing further implementation time; if bundling proves unreliable, the documented fallback is running both on CPU (slower, but functionally correct).

**First-run onboarding wizard** (shown once, or whenever setup is incomplete):
1. **Ollama check** — pings `localhost:11434`; if unavailable, shows install instructions + download link with a "Recheck" button.
2. **Model selection** — presents vetted model choices (LLM + BGE-M3 embedding model) with size/RAM guidance; triggers `ollama pull` with progress indicator.
3. **Watched folder** — file-picker to choose a folder; defaults to `~/Documents/RAG Chat/Documents` if skipped.
4. **First document prompt** — nudges the user to upload/drop their first file before landing on the empty chat screen.

After first run, the app launches directly into the chat screen unless a check (Ollama/model) fails again.

## Error handling summary

| Scenario | Behavior |
|---|---|
| Ollama not running/installed | Clear in-app message + guidance; startup check + "Recheck" |
| Corrupt/unparseable file | Marked failed with reason, skipped, rest of batch continues |
| Low-confidence OCR | Ingested but flagged as lower-reliability source |
| Empty corpus | Chat clearly states no documents are ingested yet |
| No relevant chunks found | Deterministic refusal + closest N chunks as related documents |
| Unsigned app / Gatekeeper | Documented right-click → Open workaround for users |
| torch/MPS bundling fails in PyInstaller | Falls back to CPU-only reranking (slower, functionally correct) |

## Testing approach

- Unit tests per source format (PDF/DOCX/XLSX/scanned-image fixtures → converted output, via Docling). Assert structural properties (headings present, table row counts, key strings, page provenance populated) rather than exact markdown output, so tests survive Docling upgrades.
- Unit tests for chunking (heading/section boundaries, paragraph overlap, table row-grouping, provenance carried onto each chunk) operating on Docling-document fixtures.
- Test that archiving behaves correctly: original moved to `archive/originals/` (hash-suffixed) only on conversion success; left in place on failure; the archive move does not trigger the watcher's deletion path.
- Test document removal via the UI action: chunks, converted artifacts, and archived original all cleaned up.
- Integration test for ingest → retrieve → rerank round-trip against a small fixture corpus with known answers, verifying both correct retrieval and correct refusal on out-of-corpus questions.
- Manual UAT checklist for onboarding wizard and chat UI once built (Ollama detection, model pull, folder picker, upload, chat, citations, refusal case).

## Explicitly out of scope (v1)

- Multi-user / shared instance support (each teammate runs independently).
- Code signing / notarization.
- Authentication (not needed — localhost-only, single user per instance).
- Concurrent request handling / job queues (sequential small-team usage only).
