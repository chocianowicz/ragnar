# Evaluation

Development-time only. Never runs in the app, never runs in CI automatically.

## Isolation

`eval/` imports app modules; the app never imports `eval/`. The Ragas
judge is configured only here, so production documents have no code path
to an external service through the app itself. The eval harness itself
does send contexts to the judge, which is why private golden cases are
filtered out before judging (see below).

## Running

    docker compose exec app python eval/run_eval.py             # deterministic metrics
    docker compose exec app python eval/run_eval.py --calibrate # sweep the similarity floor

Every run streams each finished case to `eval/reports/answers-<stamp>.jsonl`
(flushed and fsync'd). If a run crashes or freezes, continue it:

    python eval/run_eval.py --resume eval/reports/answers-<stamp>.jsonl

A resume refuses a sink written under a different config (model,
collection, floor, stages), so two runs can't be mixed in one file.

### Calibrating the similarity floor

    python eval/run_eval.py --calibrate --retrieval-only --multi-query --floors 0.01:0.99:0.01

Run it after a re-ingest, since reranker scores depend on the chunks, and
after the golden check (below) is reviewed. Retrieval runs once, and every
floor is derived from each case's recorded best score. The table shows
`missed` (out-of-corpus questions answered: the worse error) and `false`
(answerable questions refused) side by side, along with a 95% range at the
best floor. Pick the lowest floor on a flat stretch of low `missed`, not a
lone peak.

`--multi-query` matches the UI's "Rephrase the question" setting, which
decides refusal on the multi-query pool. Multi-hop and self-correct only
run once something has cleared the floor, so they can't change the
decision, and `--calibrate` rejects them. Leave `--multi-query` off if the
UI setting is off. The sweep writes a `calibrate-<stamp>.jsonl` sink and
resumes with `--resume`, like a normal run.

### Checking the golden set

    python eval/check_golden.py [--resume eval/reports/golden-check-<stamp>.jsonl]

An LLM audit of every entry against evidence from the live index. It
flags wrong or partial expected answers, questions with more than one
answer, questions that don't say which document they mean ("the book"),
and out-of-corpus probes the library can answer after all. Private
entries are judged by the local answer model only. It writes a report
and never edits the YAML.

### Judged metrics (Ragas)

    python eval/run_ragas.py --answers eval/reports/answers-<stamp>.jsonl
    python eval/run_ragas.py --answers ... --resume eval/reports/ragas-<stamp>.jsonl
    python eval/run_ragas.py --summarize eval/reports/ragas-<stamp>.jsonl

This scores a finished answers file and never re-runs the pipeline. The
judge is `glm-5.3:cloud` via the local Ollama; `RAGAS_JUDGE_*` in
`.env.example` overrides it. Cases are judged in batches of 5 (`--batch-size`),
and each batch is appended to `ragas-<stamp>.jsonl` before the next starts.
A hung judge call times out after 180s, and a case the judge fails on is
recorded as `null` instead of stopping the run. `--noise` adds
`noise_sensitivity`, which roughly doubles judge calls, so use it on
finalist runs only.

**Private documents never reach the judge.** A golden entry marked
`private: true` is skipped, and so is any case whose retrieved contexts came
from a private entry's document. The skip counts go into the ragas file's
provenance line.

### Multi-hop (HybridQA)

    python eval/ingest_hybridqa.py                  # builds the 'hybridqa_structural' collection
    python eval/run_eval.py --collection hybridqa_structural --golden eval/golden_hybridqa_draft.yaml
    python eval/run_eval.py --collection hybridqa_structural --golden eval/golden_hybridqa_draft.yaml --multi-hop
    python eval/run_eval.py --collection hybridqa_structural --golden eval/golden_hybridqa_draft.yaml --multi-hop --multi-query

HybridQA is a public Wikipedia benchmark where each answer needs a table
row plus a linked passage: 125 cases, 89 of them multi-hop and 35
out-of-corpus probes. Ingest uses production chunking from `config.yaml`.
The stage flags route through `AgenticSearch`, wired the way the UI wires
it. `golden_aggregation.yaml` runs on the same collection as a negative
control for the aggregation guard.

### Comparing two runs

    python eval/stats.py eval/reports/answers-A.jsonl eval/reports/answers-B.jsonl
    python eval/stats.py eval/reports/ragas-A.jsonl eval/reports/ragas-B.jsonl

This prints B minus A for each metric, pairing the runs case by case, with a
95% bootstrap range. A difference whose range crosses 0 is marked "not
significant". Every metric in a report carries the same kind of range.

## Metrics

| Metric | Judge | Measures |
|---|---|---|
| refusal_accuracy | No | Refused exactly when it should have |
| missed_refusal_rate | No | Out-of-corpus questions answered anyway (hallucination risk) |
| false_refusal_rate | No | Answerable questions declined (over-caution) |
| citation_accuracy | No | Cited an expected source |
| citation_precision | No | Share of cited labels that are expected sources |
| multi_hop_retrieval_recall | No | Multi-hop: every expected source was retrieved |
| multi_hop_citation_accuracy | No | Multi-hop: every expected source was cited |
| aggregation_guard | No | Guard caught / false positives on `aggregation:` cases |
| faithfulness, no_invented_numbers | Yes | Answer claims supported by the contexts |
| answer_relevancy, answer_correctness, answer_similarity | Yes | Answer against the question and the reference |
| context_precision, context_recall, context_entity_recall | Yes | Retrieval quality |
| same_language | Yes | Answer in the question's language (PL/EN) |
| rewrite_preserves_intent | Yes | Follow-up rewrite asks what the user meant |
| context_recall on refused cases | Yes | Evidence was there and the model still declined |
| noise_sensitivity (`--noise`) | Yes | Irrelevant chunks pulling the answer wrong |

How to read them together: high context_recall with low answer_correctness
means the evidence arrives and generation doesn't use it. Low context_recall
points at retrieval. A multi_hop_retrieval_recall well above
multi_hop_citation_accuracy means the second hop was found and the answer
ignored it.

## Current state

`golden_set.yaml` holds 222 hand-reviewed cases over 17 PDFs, ported from
branch `main_alex`. 51 are out-of-corpus (8 of them near-miss probes about
books not in the library), 7 are follow-up chains, and 41 are marked
`private`. Every source must be indexed first, and `run_eval.py` refuses
to run if one is missing. The old 5-case fixture set this section
described is gone.

The set was audited with `check_golden.py` on 2026-09-24, and 41 of 215
entries were flagged:
- 28 questions didn't say which document they meant ("the book", "this
  e-book"). They now name the title.
- 8 out-of-corpus probes turned out to be answerable from copyright pages.
  They became in-corpus cases and were replaced with near-miss probes.
- 2 expected answers were incomplete and were corrected.
- One private case was dropped.

`retrieval.score_floor` is 0.49, calibrated on 2026-09-25 against the
222-case set with `--calibrate --retrieval-only --multi-query --floors
0.01:0.99:0.01`. No floor gives a flat stretch of low `missed`: it falls
steadily while `false` rises. 0.49 is the lowest floor of the 0.49-0.54
plateau (missed 0.314, false 0.146, refusal accuracy 0.815, inside the 95%
range of the best floor, 0.833 at 0.12-0.13). The UI's Strict and Lenient
stops are the lowest floors of the neighbouring plateaus, 0.64 and 0.18.

The scores are probabilities. Until 2026-09-25 the reranker applied a
sigmoid on top of CrossEncoder.predict's own, which squeezed every score
into 0.50-0.73; floors from before then (0.55, 0.62) are on that scale,
where x maps to ln(x / (1 - x)) here. Rankings, and so refusals at an
equivalent floor, did not change.

## Comparing configurations

    # edit config.yaml: chunking.strategy: fixed (or back to structural)
    docker compose restart app
    # re-ingest the corpus via the UI or a script
    docker compose exec app python eval/run_eval.py
    # diff against the previous report in eval/reports/
