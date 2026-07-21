# Ragnar — Dockerized Local RAG — Design

**Status:** Active. Supersedes the deployment model in `2026-07-06-ragnar-design.md` (native macOS `.dmg`), which is retained for reference. The retrieval and citation thinking from that spec largely carries forward; the packaging, storage, and UI decisions do not.

## Purpose

An offline RAG (retrieval-augmented generation) system over a company's own PDF and Excel documents. Users ask questions in a chat UI and receive answers drawn **only** from ingested documents, with citations to source file and page. When the answer is not in the corpus, the system refuses explicitly rather than improvising.

Delivered as a Docker Compose stack running locally on macOS, with deliberate portability to a VPS at a later stage.

## Constraints

- **Offline runtime**: no external API calls during ingestion, retrieval, or generation. Local LLM, embeddings, reranker, and vector store.
- **Development-time exception**: the evaluation harness may call an external LLM judge, operating only on a hand-written test set. Production documents have no code path to any external service (see §7).
- **Hardware**: Apple Silicon Mac, 24–32GB unified memory.
- **Scale**: ~200 documents, predominantly native-text PDF with some Excel. Mixed Polish/English.
- **Users**: single user, localhost. Multi-user, auth, and TLS are explicitly out of scope for v1.

## Deployment model

Ollama runs **natively on the macOS host**; everything else runs in Docker.

Docker Desktop on macOS is a Linux VM with no access to Metal, so anything inside a container is CPU-only. Running Ollama on the host preserves GPU acceleration for the two workloads that need it — generation and embedding. Docling and the reranker run CPU-bound inside the container, which is acceptable: Docling is batch work, and the reranker costs 1–3s per query.

```
┌─ macOS host ─────────────────────────────────┐
│  Ollama (native, Metal GPU) :11434           │
│    ├─ qwen2.5:14b     ← generation           │
│    └─ bge-m3          ← embeddings           │
│         ▲                                     │
│  ┌──────┼─ Docker Desktop ─────────────────┐ │
│  │      │ host.docker.internal              │ │
│  │  ┌───┴───────────┐   ┌────────────────┐ │ │
│  │  │ app           │──▶│ qdrant         │ │ │
│  │  │ Streamlit     │   │ (official img) │ │ │
│  │  │ Docling  (CPU)│   │ named volume   │ │ │
│  │  │ reranker (CPU)│   └────────────────┘ │ │
│  │  └───────────────┘                      │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

**Portability:** `OLLAMA_BASE_URL` is configuration, never hardcoded. Moving to a VPS changes that one variable. Nothing else in the stack is macOS-specific.

## Model selection

| Role | Model | Rationale |
|---|---|---|
| Generation | `qwen2.5:14b` (Ollama) | Strongest general option for mixed PL/EN. Comfortable in 24–32GB alongside the container stack. |
| Embedding | `bge-m3` (Ollama) | Strong multilingual and cross-lingual retrieval — a Polish question can legitimately retrieve English chunks. |
| Reranking | `bge-reranker-v2-m3` (CPU, in-container) | Cross-encoder; multilingual, matched to the embedder. |

`llama3.1:8b` is the documented fallback if the system later moves to weaker hardware. Polish-specialised models (Bielik) were considered and rejected: stronger on Polish, weaker on English, which is the wrong trade for a mixed corpus.

## Project structure

```
core/
  models.py          # Document, Chunk, SearchResult, IngestStatus
  interfaces.py      # Protocols: DocumentParser, Chunker, Embedder,
                     #            VectorStore, Reranker
  config.py          # config.yaml + env, validated at startup

ingestion/
  parser.py          # DoclingParser
  chunkers/
    structural.py    # default
    semantic.py      # alternative, for A/B testing
    registry.py      # name → Chunker  ← the only selection point
  tables.py          # row-group chunking + header repetition
  pipeline.py        # parse → chunk → embed → store
  worker.py          # background queue drain

retrieval/
  embedder.py        # OllamaEmbedder (bge-m3)
  store.py           # QdrantStore
  reranker.py        # BGEReranker
  search.py          # retrieve → rerank → threshold

generation/
  guards.py          # aggregation detection + refusal
  prompts.py         # system prompt, PL/EN aware
  answerer.py        # prompt → Ollama → answer + citations

ui/app.py            # Streamlit

eval/                # imports app core; app never imports eval
  golden_set.yaml
  run_eval.py        # Ragas + external judge; holds the API key

tests/
```

### Modularity decision

Components sit behind `Protocol` interfaces with **one implementation each**, except chunking, which is config-selectable via `chunkers/registry.py`.

The rationale is narrow and deliberate. Chunking strategy is the highest-leverage tunable in a RAG system and the one whose winner cannot be predicted in advance — and because this project has a golden set and a Ragas harness, swapping chunkers becomes a measurable experiment rather than speculation. That argument does not extend to components with no experiment attached, so there is no general factory, no config-driven parser selection, and no alternative embedder. Interfaces exist for testability (faking the embedder in tests) and to keep a future swap to a one-file change.

## Components

### 1. Storage layout

```
data/                   # bind-mounted from the host
  inbox/                # drop zone; Streamlit uploads land here
  originals/            # moved here post-ingestion, hash-suffixed
  converted/            # <doc_id>.md + <doc_id>.json
  registry.db           # SQLite: filename, hash, status, error, chunk count
qdrant_storage/         # named Docker volume
```

**`data/` is the source of truth; Qdrant is a rebuildable index.** If the vector store is lost or the embedding model changes, re-embed from `converted/` without re-parsing, or re-parse from `originals/`. Backup means backing up `data/` and ignoring the vector store.

- **`inbox/` is a staging area, not a library.** Files leave it after successful ingestion, so an empty inbox means "everything is indexed." Failed files remain with their error recorded.
- **`originals/` is hash-suffixed** — `report.pdf` → `report.a3f9c2.pdf` — so a changed version of the same filename does not collide and versions stay distinguishable. Originals are never auto-deleted.
- **`converted/` holds two artifacts.** The `.md` is human-readable and shown in the UI. The Docling JSON is what the chunker reads; it carries per-element provenance (page numbers, sheet names) that flattening to markdown destroys.
- **`registry.db`** tracks documents including failed ones, which have no chunks and so do not exist in Qdrant at all. It also serves the UI's document list without querying the vector store, and is the sole communication channel between the ingestion worker and the UI.
- **Document ID = content hash.** Dropping the same file twice is a no-op dedupe. Re-ingesting a changed file at a known path yields a new ID, deletes the old chunks, and retains both originals.

Qdrant uses a named volume rather than a bind mount: it performs many small writes and uses mmap, and Docker Desktop's macOS bind mounts are slower and have had mmap edge cases. Named volumes live in the Linux VM's own filesystem. Losing browsability from Finder is acceptable for something defined as a rebuildable cache.

### 2. Ingestion pipeline

**Queueing.** Upload writes files to `inbox/`, inserts `queued` rows into the registry, and returns immediately. A background worker — a thread inside the Streamlit process, held as a singleton via `@st.cache_resource` — drains the queue **one document at a time**. Sequential by design: Docling on CPU would only contend with itself in parallel, and concurrency would add failure modes and buy nothing at this scale.

```
upload (instant) → inbox/ + registry: queued
                            ↓
       worker:  queued → processing → done | failed
                            ↓
       UI polls registry every ~2s → status strip
```

The worker touches only SQLite and the pipeline, never Streamlit APIs, which are not thread-safe. `@st.fragment(run_every="2s")` refreshes only the status strip, so the chat does not flicker.

**Crash recovery:** on startup, any row stuck in `processing` resets to `queued`. Without this, a container killed mid-ingest leaves documents permanently wedged.

**Parse (Docling).** OCR is **off by default** — the corpus is predominantly native-text, and this is the single largest speed win. OCR is enabled per-document only when text extraction returns near-empty (under ~50 characters per page), catching occasional scans without paying the cost on the rest. Documents taking the OCR path carry a `low_confidence` flag through to their citations.

**Chunk.** Two implementations behind the `Chunker` Protocol, selected in `config.yaml`:

- `structural` (default) — split on heading/section boundaries, then by paragraph within a section, sized with **BGE-M3's own tokenizer** at ~500 tokens with ~50 overlap.
- `semantic` — embedding-similarity boundary detection, present for A/B testing.

Tokenizer-based sizing rather than character counts matters specifically for this corpus: Polish words are longer and diacritics cost extra bytes, so character heuristics systematically misjudge chunk size on mixed PL/EN text.

Tables bypass both chunkers and route through `tables.py`: row groups split on blank-row boundaries where present, else fixed groups of ~20 rows, **with the header row repeated into every group**. A table row without its header is semantically meaningless to an embedder.

**Chunk metadata** — the basis for all citations:

```python
doc_id: str          # content hash
filename: str        # original name, for display
page: int | None     # from Docling provenance
sheet: str | None    # for Excel
chunk_index: int
is_table: bool       # drives the aggregation guard
low_confidence: bool # OCR-derived
```

**Embed.** BGE-M3 via Ollama on the host GPU, batched. The Qdrant collection records the embedding model name and version; a mismatch at startup triggers a warning and an offered re-embed rather than silently mixing incompatible vectors.

**Store.** Qdrant upsert with provenance as payload. Re-ingestion deletes by `doc_id` first, so stale and fresh chunks never coexist.

### 3. Retrieval

```
question → embed (bge-m3, host GPU)
         → Qdrant top-25 by cosine
         → rerank (bge-reranker-v2-m3, CPU) → top-5
         → threshold check
         → aggregation guard
```

25 candidates in, 5 out. The cross-encoder reads query and chunk together — far more accurate than vector similarity, too slow to run corpus-wide. If CPU reranking proves too slow, the ONNX export of the same model is a drop-in speedup that removes the torch dependency; deferred rather than done upfront.

**Similarity floor.** The reranker emits logits, not probabilities; they require a sigmoid to become interpretable, and the useful cutoff is corpus-dependent and must be determined empirically. It is calibrated against the golden set — specifically its out-of-corpus entries (§7).

When nothing clears the floor, the system **skips the LLM call entirely** and returns a deterministic refusal plus the closest chunks framed as "related documents you might check." Not invoking the model is what makes the refusal trustworthy.

### 4. Excel and the aggregation guard

RAG handles spreadsheets well for lookup and poorly for computation:

| Question type | Example | Supported |
|---|---|---|
| Lookup | "What's the contract value for client X?" | Yes |
| Definition | "What does the 'status' column mean?" | Yes |
| Aggregation | "Total revenue across all regions?" | **No** |
| Filtering/ranking | "Which clients billed over 100k?" | **No** |

The failure mode being defended against is specific: on an aggregation question the model does not decline — it sees a handful of rows and produces a confident, plausible, wrong total. In a business context that is worse than a refusal.

**The guard fires on two signals together:** aggregation intent in the question (sum, total, average, count, ranking; suma, razem, ile, średnia, największy) **and** retrieved chunks predominantly `is_table`. Either signal alone produces false positives — a question containing "total" about a prose contract must not be blocked. When it fires, the response names the file and sheet so the user can compute it themselves.

This is the designated seam for a future DuckDB/text-to-SQL tool: same detection, different branch. Deferred until real usage shows whether such questions are common.

### 5. Generation

Strict system prompt: answer only from provided chunks, state when they are insufficient. One clause is specific to this corpus — **answer in the language of the question, regardless of the source language.** BGE-M3 retrieves cross-lingually, so Polish questions will legitimately surface English chunks; Qwen2.5 handles translation-in-place, but only when instructed.

**Citations derive from the metadata of chunks actually retrieved, never parsed from model prose.** Deduplicated by (filename, page). Chunks flagged `low_confidence` are visibly marked.

**Responses stream.** 14B on Metal runs ~15–25 tok/s, so a 300-token answer takes 15–20 seconds. Streamed via `st.write_stream` this reads as responsive; as a blocking spinner it reads as broken.

### 6. Streamlit UI

**Sidebar** — upload widget, "Ingest inbox" button for files dropped via Finder, corpus stats, and the document list with per-document status and failure reasons. Clicking a document displays its converted `.md`, which Streamlit renders natively.

**Main** — chat via `st.chat_message` / `st.chat_input`, streamed answers, citations in an expander showing filename, page or sheet, and chunk text.

**Removal** deletes chunks from Qdrant by `doc_id`, converted artifacts, and the registry row — but leaves the original in `originals/`.

Chat history lives in `st.session_state` and is lost on browser refresh; persistence is deferred.

### 7. Evaluation harness

**Isolation.** `eval/` imports app core modules; nothing in the app imports `eval/`. The external judge's API key exists only in the eval environment. This is enforced structurally, not by convention.

**Golden set** — ~30–50 entries in `eval/golden_set.yaml`, each with question, expected answer, expected source document(s), and an out-of-corpus flag. Roughly a quarter should be out-of-corpus: questions that sound like they belong but whose answers are in no document. Without these, the refusal path — the entire trust story — is unmeasured.

**Metrics, in two tiers:**

| Metric | Judge required | Measures |
|---|---|---|
| `faithfulness` | Yes | Answer grounded in retrieved context |
| `answer_relevancy` | Yes | Answer addresses the question |
| `context_precision` | Yes | Retrieved chunks are relevant |
| `context_recall` | Yes | Retrieval found what was needed |
| `answer_correctness` | Yes | Match against reference answer |
| **refusal accuracy** | No | Refused on out-of-corpus questions |
| **citation correctness** | No | Cited the expected document |

The final two are custom rather than Ragas, deterministic, and free to run. They cover the two failure modes that matter most: confidently answering what it shouldn't, and citing the wrong source. They are built first — they catch real regressions before the judged metrics are calibrated.

**Judge caveat, recorded deliberately:** judged metrics use an external frontier model because a local 14B is a noticeably noisier judge, degrading further on Polish. Tuning the system against an unreliable measurement is worse than not measuring, because it feels rigorous.

**Output** is a timestamped report with per-question scores and aggregates, committed so runs can be diffed. The workflow this enables:

```
edit config.yaml (chunker: semantic) → re-ingest → run eval → diff against last report
```

## Error handling

| Scenario | Behaviour |
|---|---|
| Ollama unreachable | Startup check; message naming the fix (`ollama serve`, `ollama pull …`) with a recheck button |
| Embedding model mismatch | Warn at startup, offer full re-embed; never mix vector spaces |
| Document fails to parse | Marked `failed` with reason, left in `inbox/`, batch continues |
| Low-confidence OCR | Ingested, flagged, marked visibly in citations |
| Empty corpus | Chat states plainly that nothing is indexed |
| Nothing clears similarity floor | Deterministic refusal + related documents; no LLM call |
| Aggregation question over tables | Refusal naming file and sheet |
| Container killed mid-ingest | `processing` rows reset to `queued` on startup |
| Qdrant volume lost | Re-embed from `converted/`; no re-parse needed |

## Testing approach

- **Parsing**: per-format fixtures (native PDF, scanned PDF, XLSX) asserting structural properties — headings present, table row counts, page provenance populated — rather than exact markdown, so tests survive Docling upgrades.
- **Chunking**: operating on Docling-document fixtures — heading/section boundaries, paragraph overlap, table row-grouping, header repetition, provenance carried onto every chunk.
- **Guard**: aggregation intent detection across PL and EN, including the false-positive case (prose document, question containing "total") which must *not* fire.
- **Registry/worker**: queue transitions, crash recovery reset, failed documents left in `inbox/`.
- **Retrieval round-trip**: small fixture corpus with known answers, verifying correct retrieval, correct citation, and correct refusal on out-of-corpus questions.
- **Fakes over mocks**: the `Embedder` and `VectorStore` Protocols exist partly so unit tests can substitute deterministic implementations without Ollama or Qdrant running.
- **Evaluation** is not a test suite and does not run in CI — it requires the real corpus and an API key.

## Out of scope (v1)

- VPS deployment, auth, TLS, multi-user, per-user corpora — deferred wholesale.
- Text-to-SQL / DuckDB aggregation over spreadsheets (seam designed, implementation deferred).
- Hybrid dense + sparse retrieval (BGE-M3 produces sparse vectors, Qdrant supports them; deferred until Ragas shows dense retrieval is the bottleneck).
- Persistent chat history.
- Live filesystem watching — replaced by explicit ingest actions.
- ONNX reranker (documented as the speedup lever if CPU reranking proves too slow).
