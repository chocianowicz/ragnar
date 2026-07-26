# RAGnar

A fully offline RAG (retrieval-augmented generation) chat tool for a small team.
Upload your own PDF and Excel documents, ask questions in a chat interface, and get answers drawn **only** from what's actually in those documents — with citations, and an honest refusal when the answer isn't there.

## Why

Generic chatbots either don't know your documents or hallucinate answers that sound right but aren't.
Ragnar is scoped narrowly on purpose: it never answers from outside knowledge, it always cites its source (file + page/sheet), and when nothing in the corpus is relevant, it says so instead of guessing.

Everything runs locally — local LLM, local embeddings, local reranker, local vector store.
No document ever leaves the machine it's running on.

## How it works

```
Upload PDF/Excel ──▶ Docling (parse + OCR fallback) ──▶ chunk (structural + table-aware)
                                                              │
                                                              ▼
                                              embed (BGE-M3) ──▶ Qdrant
                                                                   │
Chat question ──▶ embed ──▶ retrieve ──▶ rerank ──▶ similarity floor
                                                          │
                                              clears floor? ──no──▶ refuse + related docs
                                                          │
                                                         yes
                                                          │
                                              aggregation guard (tables only)
                                                          │
                                              fires? ──yes──▶ refuse, name the file/sheet
                                                          │
                                                          no
                                                          ▼
                                          Qwen2.5 (local, via Ollama) ──▶ streamed answer + citations
```

- **Parsing** — [Docling](https://github.com/docling-project/docling) converts PDF/Excel into structured text with page/sheet provenance. OCR only kicks in when a page's extracted text looks too sparse to be a real digital page, so scanned documents still work without slowing down the common case.
- **Chunking** — text is grouped by heading/section, never split across pages, with page provenance carried onto every chunk. Tables get their own row-group chunking with the header row repeated in every group, so a chunk is never a headerless fragment.
- **Retrieval** — [BGE-M3](https://huggingface.co/BAAI/bge-m3) embeddings in [Qdrant](https://qdrant.tech/), narrowed by a `bge-reranker-v2-m3` cross-encoder, then filtered by a similarity floor. Below the floor, the LLM is never even called — the refusal is deterministic, not the model's judgment call.
- **Aggregation guard** — a question like "what's the total revenue?" against a spreadsheet is a request RAG structurally can't satisfy honestly (it sees a handful of rows, not the whole table). A two-signal guard (aggregation wording *and* table-heavy results) catches this and refuses with a pointer to the source, instead of quietly making up a plausible-sounding number.
- **Generation** — [Qwen2.5](https://ollama.com/library/qwen2.5) via [Ollama](https://ollama.com/), answering only from retrieved excerpts, in the language of the question regardless of the source document's language. Citations are built from the chunks actually used — never parsed from the model's own text.

## Stack

Python · Streamlit · Docling · Qdrant · Ollama (Qwen2.5 + BGE-M3) · `sentence-transformers` (reranker) · Docker Compose

## Running it

Requires [Ollama](https://ollama.com/) running natively on the host (for GPU acceleration — Docker on macOS can't reach the GPU) and Docker Desktop.

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3

cp .env.example .env   # optional: add HF_TOKEN to silence a HuggingFace rate-limit warning

docker compose up -d --build
```

Open `http://localhost:8501`, upload a document, ask a question.

## Status

Working end-to-end: upload → background ingestion → chat with streamed, cited answers. 83 tests passing against real Ollama/Qdrant/Docling — no mocks in the integration path.

Known gaps, tracked as follow-up work rather than silently glossed over:
- Excel parsing is designed for (citations carry a sheet field) but not yet exercised against a real spreadsheet — no `.xlsx` test fixture exists yet.
- The similarity floor is hand-tuned from limited real usage, not a properly calibrated golden set (see `eval/README.md`).
- No recovery path yet if the vector store is lost — re-embedding from the converted document cache is designed for but not implemented.

## Project layout

```
core/         shared data models, config loading
ingestion/    parsing, chunking, background worker, on-disk storage
retrieval/    embedding, vector store, reranking, similarity floor
generation/   LLM client, prompts, citation logic, aggregation guard
ui/           Streamlit chat app
eval/         evaluation harness — isolated from the running app, never imported by it
tests/        83 tests, mostly integration-style against the real stack
```

Design docs and the original implementation plan live under `docs/superpowers/`.
