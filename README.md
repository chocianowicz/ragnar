<div align="center">

# RAGnar

**A local RAG chat tool for a small team.**

It answers from your documents and nothing else — with citations, and a refusal
when the answer isn't in there.

</div>

---

## The name

**RAGnar** — RAG + Ragnar

| | |
|---|---|
| **RAG** | retrieval-augmented generation — the answer is built from retrieved text, not from what a model happens to remember |
| **nar** | it reads like a Norse name, and it stuck |

*Only what the documents say.* That is the entire product. A question goes in, and
what comes back is either something the corpus actually supports — with the file and
page to check it against — or an admission that it isn't there.

---

## What it does

You drag a PDF into the sidebar and ask a question. In between, RAGnar:

1. **Parses the file** with Docling, keeping page and sheet provenance. OCR stays off
   by default; if a document averages fewer than 50 characters per page, the whole
   file is re-parsed with OCR and every chunk from it is marked low-confidence — a
   scanned document still works, without making every native-text PDF pay for it.
2. **Chunks it structurally** — grouped by page and size, never split across a page
   boundary. Tables are chunked by row group with the header row repeated in each
   one, so no chunk is a headerless fragment nobody can read.
3. **Embeds and stores** — BGE-M3 vectors in Qdrant, each paired with a lexical
   (sparse) vector of the same chunk.
4. **Retrieves and narrows** — dense and lexical search run together and are fused by
   reciprocal rank; 25 candidates, reranked by a cross-encoder down to 5, then
   measured against a similarity floor.
5. **Answers, or refuses** — the model sees only the retrieved excerpts, and the
   citations are assembled from those chunks rather than from anything the model wrote.

Every model in that chain runs on your own machine.

---

## What it looks like

A browser tab at `localhost:8501`. One chat, three collapsed panels, nothing else:

```
┌───────────────────┬────────────────────────────────────────────────┐
│  RAGnar           │                                                │
│  local RAG chat   │   what is the notice period?                   │
│                   │                                                │
│  ▸ Settings       │   Three months, counted from the end of the    │
│  ▸ Documents      │   calendar month in which notice is given.     │
│  ▸ Chats          │                                                │
│                   │   ▾ Sources                                    │
│                   │       contract.pdf, p. 12                      │
│                   │       contract.pdf, p. 13                      │
│                   │                                                │
│                   │   [ Ask about your documents             ]     │
└───────────────────┴────────────────────────────────────────────────┘
```

**Documents** takes the upload and shows what has been ingested. **Chats** keeps past
conversations. **Settings** holds the two choices worth making per question — which
model answers, and how sure it has to be — with everything else behind an advanced
toggle.

Ingestion runs in the background with an estimated time remaining, so a slow scan of a
200-page PDF doesn't block the chat you're already having.

---

## Folders

A flat list of documents works at ten and stops working at fifty. Folders group them —
one per client, project or topic — and answer the question a large corpus raises: *only
look in here.*

```
☑ Select all

▾ ☐ Acme Corp                                  3/12
    ☑ ✅ nda.pdf              [ Acme Corp ▾ ]   ✕
    ☑ ✅ sow.pdf              [ Acme Corp ▾ ]   ✕
    ☐ ✅ invoice.xlsx         [ Acme Corp ▾ ]   ✕
▸ ☑ Northwind Ltd                             12/12
▸ ☑ Unfiled                                   32/32
```

The checkbox you already know, at three levels: **Select all**, then a folder, then a
document. Ticking a folder ticks everything in it, and only ticked documents are
searched. Streamlit has no half-filled checkbox, so a partly-ticked folder reads as
unchecked and the `3/12` count carries the real state.

A document is filed with the picker on its own row. The tick and the picker do
different jobs on purpose: the tick means *search this*, and giving it a second meaning
would make filing a document silently narrow the next answer.

**A chat remembers what it was scoped to.** Ask about Acme with only Acme ticked, come
back tomorrow, and reopening that conversation restores that scope. Folders are stored
by identity rather than by name, so renaming one changes nothing, and a document added
to Acme after the chat was saved comes back ticked with the rest.

**Nothing here can lose a document.** Deleting a folder returns its documents to
Unfiled; the ✕ on a row is still the only thing that removes one. `Unfiled` is not a
folder but the absence of one, which is why the name is reserved.

Re-uploading a corrected file keeps its folder. A document's id is a hash of its
contents, so fixing a typo produces a new, unrelated document — it inherits the folder
of the file it replaces rather than quietly landing in Unfiled.

---

## Why refusing is the feature

Most document chatbots will answer anything. Ask about a contract clause that isn't in
the corpus and you get a fluent paragraph assembled from the model's general knowledge —
indistinguishable, in tone, from the answers that are actually grounded.

RAGnar decides whether it *can* answer before the model is involved:

```
question
   └─▶ embed ─▶ retrieve 25 ─▶ rerank ─▶ top 5
                                          │
        best score below 0.55? ──yes──▶ refuse, and list related documents
                                          │
     aggregation words + mostly ──yes──▶ refuse, and name the file and sheet
     table chunks?                        │
                                          no
                                          ▼
                            answer, cited from the chunks actually used
```

Three things follow:

- **The refusal is deterministic.** Below the floor the LLM is never called at all, so
  there is no prompt to talk it out of and no temperature setting that makes it guess.
- **Citations cannot be invented.** They are built from the metadata of retrieved
  chunks, never parsed out of the model's prose — the model cannot cite a document that
  was not retrieved.
- **The one question RAG structurally cannot answer is refused by name.** "What's the
  total?" against a spreadsheet asks for arithmetic across a whole table when retrieval
  only ever sees a handful of rows. Two signals have to agree — aggregation wording
  *and* a majority of table chunks — before it fires, so ordinary questions about
  tables still get answered.

What you get instead of a wrong number:

> This looks like a question that requires calculating across a whole table. I can only
> read individual rows, so any total I gave you could be wrong.
>
> The relevant data is in: budget.xlsx (sheet 2025)

---

## Privacy

Not a policy — a property of how it is built.

- The LLM runs on your machine, through Ollama.
- Embeddings and reranking run on your machine.
- The vector store is a container on your machine.
- Documents are written to `./data` and never leave it.

Two containers are defined, `app` and `qdrant`, and neither is given a key to anything
external — `docker compose config` is the whole story. There is no account, no API key,
and no per-token bill.

The one network access in the whole project is HuggingFace downloading the reranker
weights, once. `HF_TOKEN` is optional and only raises the rate limit while that happens.

---

## Where things land

Plain files on disk, next to the code:

```
data/
├── inbox/            ← uploads land here first
├── originals/        ← the file as you gave it
├── converted/        ← Docling output, cached by content hash
├── registry.db       ← what has been ingested, and how it went
├── chats.db          ← past conversations
└── settings.json     ← what this instance has tuned, layered over config.yaml

qdrant_storage        ← a Docker volume holding the vectors
```

Converted documents are cached by content hash, so re-uploading the same file costs
nothing and a chunking change can be replayed without re-parsing. The vectors are the
one thing that would have to be rebuilt from scratch — see Known gaps.

---

## Install

Ollama has to run natively on the host rather than in a container, because Docker on
macOS cannot reach the GPU:

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3

cp .env.example .env      # optional: HF_TOKEN silences a rate-limit warning

docker compose up -d --build
```

Open `http://localhost:8501`, upload a document, ask a question.

Both containers use `restart: unless-stopped`. Docker Desktop under memory pressure will
SIGKILL the app, and without that line it stays dead until you notice — but a deliberate
`docker compose stop` is still respected.

---

## Settings

Two controls are in the panel itself, because they are the two things worth deciding
per question:

| Setting | Default | What it changes |
|---|---|---|
| Answer model | `qwen2.5:14b` | Which local model writes the answer |
| Strictness | Balanced (`0.55`) | How sure the app must be before it answers at all |

Everything else is a deployment choice — set once, then left — and lives behind
**Show advanced settings**:

| Setting | Default | What it changes |
|---|---|---|
| Re-rank results | on | Off is faster, less precise, and **disables the relevance floor** |
| Candidates considered | 25 | Passages fetched before re-ranking picks the best few. Only shown when re-ranking is on: without it the top few are kept as the search ranked them and the pool is never used |
| Temperature | 0 | Higher wanders further from the excerpts |
| Rephrase the question | on | Searches several rewordings *beside* your own wording, and searches again if the first pass is thin. After a refusal, also tries the broader group the subject belongs to, if the chat or the documents state the link; that answer is marked indirect. One or two extra model calls, one more on a refusal |
| Remember context | on | Resolves what a follow-up refers to before searching, and lets the answer see the recent chat. Off, each question stands alone and the chat cannot supply a link for an indirect answer. One model call per follow-up |
| Check the draft answer | off | Drafts, judges whether the excerpts support it, searches again if not. Two model calls, and no golden-set evidence yet that it helps |
| Chunk size | 500 tokens | Target size per chunk, 50-token overlap |
| Table rows per chunk | 20 | Rows per table chunk, header repeated in each |

Your question is always searched as you asked it. The rephrasing stage adds alternative
phrasings alongside it and fuses the results, never substituting for it — the lexical
half of hybrid search matches on your own identifiers, so replacing your wording could
only lose documents.

Strictness is the one to understand before touching. Its three stops are **provisional,
not calibrated**: on a real corpus, out-of-corpus questions scored 0.50–0.503 and
relevant ones 0.578 and up, so Balanced at 0.55 sits in that gap with margin either
side. That is five data points, not a golden set. Lenient turns refusals into confident
guesses.

---

## What's in an answer

- **The answer**, streamed, written only from the retrieved excerpts and in the language
  of the question — a Polish question against an English contract is answered in Polish.
- **Sources**, an expander listing every chunk that fed the answer:
  `contract.pdf, p. 12` · `budget.xlsx, sheet 2025` · `notes.docx`
- **Or a refusal.** "I could not find an answer to this in the indexed documents,"
  followed — when retrieval did find related material — by the documents that came
  closest: *"The closest passages were in contract.pdf, p. 12 and policy.pdf, p. 4,
  but none matched closely enough to answer from."* That distinguishes "the corpus is
  silent on this" from "the corpus covers this area but not your question", and it
  names documents only, never quoting them. A question with no bearing on the corpus
  at all gets the bare sentence, so the refusal never invents a connection.

---

## Supported formats

`.pdf` · `.xlsx` · `.docx`

---

## Configuration

`config.yaml` holds the pipeline defaults:

| Key | Default | Notes |
|---|---|---|
| `models.llm` | `qwen2.5:14b` | Answer generation, via Ollama |
| `models.embedding` | `bge-m3` | 1024-dimensional vectors |
| `models.reranker` | `BAAI/bge-reranker-v2-m3` | Cross-encoder, downloaded once |
| `models.keep_alive` | `10m` | How long Ollama holds a model after the last call |
| `chunking.target_tokens` | `500` | 50-token overlap |
| `chunking.table_rows_per_group` | `20` | Header repeated per group |
| `retrieval.hybrid` | `true` | Pair the dense vector with a lexical one |
| `retrieval.candidates` | `25` | Fetched before reranking |
| `retrieval.top_k` | `5` | Kept after reranking |
| `retrieval.score_floor` | `0.55` | Below this, refuse |
| `agentic.max_hops` | `2` | Extra searches when a first pass comes back thin |
| `agentic.variants` | `3` | Alternative phrasings searched beside the original |

Which of the extra stages run is *not* configured here — those are per-instance choices
made in Settings and stored in `data/settings.json`, next to the instance's data rather
than in the repo.

Environment (`.env`):

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | Yes | `http://host.docker.internal:11434` — Ollama on the host |
| `QDRANT_URL` | Yes | `http://qdrant:6333` — the sibling container |
| `HF_TOKEN` | No | Raises the HuggingFace rate limit while the reranker downloads |

---

## Known gaps

Tracked rather than glossed over:

- **The similarity floor is hand-tuned**, not calibrated — see `eval/README.md`. It
  needs a much larger golden set before the number deserves trust.
- **No recovery path if the vector store is lost.** Re-embedding from the converted
  document cache is designed for and not implemented.

The test suite runs against real Ollama, Qdrant and Docling rather than mocks, which is
why it is slow and why it catches integration breakage that mocks would hide.

---

## Requirements

**Memory.** The answer model is the whole budget: `qwen2.5:14b` is about 14 GB
resident once loaded, and the app keeps it loaded for `models.keep_alive` after
the last question so a pause does not cost a cold reload. On a 32 GB machine
that also runs Docker Desktop and Qdrant, that leaves little spare — running a
long batch (the eval harness asks ~50 questions back to back) wants a quiet
machine. Lower `keep_alive`, or choose a smaller answer model, if memory is
tight.

Docker Desktop, and [Ollama](https://ollama.com/) running natively on the host with
`qwen2.5:14b` and `bge-m3` pulled. Apple Silicon is the tested configuration; the 14B
model wants real memory, and the reranker adds a one-time download on first run.

`eval/` is deliberately isolated from the running app and never imported by it — the
evaluation harness cannot change the behaviour it is measuring.
