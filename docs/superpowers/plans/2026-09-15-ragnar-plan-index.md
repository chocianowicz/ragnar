# RAGnar Improvement Plan — Index

The improvement plan (https://claude.ai/artifact/LEStegxK82amXjsQe5c5Ed) spans six
independent subsystems. Per the writing-plans scope rule, each becomes its own
implementation plan that produces working, testable software on its own. Execute
them in this order; each phase's "entry" line says what it depends on.

| # | Plan file | Scope | Entry condition |
|---|---|---|---|
| 0 | `2026-09-15-phase-0-foundations.md` | keep_alive + warm-up; table-summary identifier fix; aggregation-guard fix; injection scan + banner; dead code; README corrections; golden-set `turns:` schema | Stack running; 253 unit tests green |
| 1 | `2026-09-15-phase-1-structure.md` | Extract `answer_job` → `generation/answering.py`; promote test doubles to `tests/fakes.py`; dedupe `_pool`/`_widen`; `Chunk.key()`; `dataclasses.asdict` traces; publish at citation build; `RetrievalRequest` dataclass; split `ui/app.py`; fragment polling; `migrate.py` tests | Phase 0 merged |
| 2 | `phase-2-settings.md` *(to write)* | Decide rewrite/self-correct on golden-set evidence; collapse to "Thorough search"; user panel + admin toggle; chunking to admin, off the rerun path; Strictness stops; `data/settings.json` layering; 👍/👎 golden-set promotion; draft-and-review queue | Phase 1 merged; golden set has ≥ 30 human-verified cases |
| 3 | `phase-3-speed.md` *(to write)* | Native host-side reranker service on MPS with in-container fallback; helper model for follow-up/agentic calls; adaptive candidate count | Phase 1 merged; golden set exists (to prove scores unchanged) |
| 4 | `phase-4-accuracy.md` *(to write)* | `rebuild-index` command; heading-aware chunking; cross-page prose with page ranges; OCR flag on citations; document versions; two-level delete; floor calibration → Strictness stops | Phases 1–3 merged; one re-ingest scheduled |

Cross-cutting items from §9 (injection) are split: retrieval-time scan and banner in
Phase 0 (no re-ingest); trust levels, quarantine, history hygiene and helper-prompt
exclusion in Phase 2 alongside the settings work; ingest-time hidden-text heuristics
in Phase 4 with the re-ingest.

Conventions shared by every plan:

- Tests run inside the container: `docker compose exec -T app python -m pytest …`.
  The stack must be up (`docker compose up -d`). Integration tests additionally need
  Ollama on the host (`ollama serve`) and are selected with `--run-integration`.
- Branch from `fix/main-p0-correctness`. One branch per phase, named `phase-N-<name>`.
- Commit after every task with the message given in the plan, ending with the
  attribution trailer the executing harness specifies.
- Nothing is pushed by a plan. Pushing and merging are the human's decisions.
