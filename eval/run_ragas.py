"""Ragas judged-metric harness.

Eval-only; the app never imports this. Scores the answers an
eval/run_eval.py run already produced (its answers-*.jsonl sink), so the
judge sees exactly what the pipeline answered and nothing is re-generated.

Crash-safe: cases are judged in small batches and every batch's scores are
appended and fsync'd to ragas-<stamp>.jsonl before the next starts. After a
crash or a freeze, rerun with --resume on that file; only the batch in
flight is lost. The summary is always computed from the JSONL, so a partial
file can be summarised too.

Private documents never reach the judge: a case is skipped when its entry
is marked `private: true`, or when any retrieved context came from a
document some private entry cites — a public question can still retrieve a
private chunk.

Usage:
    python eval/run_ragas.py --answers eval/reports/answers-<stamp>.jsonl
    python eval/run_ragas.py --answers ... --noise      # + noise_sensitivity
    python eval/run_ragas.py --answers ... --resume eval/reports/ragas-<stamp>.jsonl
    python eval/run_ragas.py --summarize eval/reports/ragas-<stamp>.jsonl
"""
import argparse
import json
import math
import os
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The legacy metric singletons are deprecated in favour of
# ragas.metrics.collections, but only they implement the SingleTurnMetric
# interface evaluate() drives.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (AspectCritic, NoiseSensitivity, answer_correctness,
                           answer_relevancy, answer_similarity,
                           context_entity_recall, context_precision,
                           context_recall, faithfulness)
from ragas.run_config import RunConfig

from core.config import Config
from eval.metrics import _cites, _should_refuse
from eval.run_eval import append_line
from eval.stats import column_mean, metric_ci, read_jsonl, row_key

ROOT = Path(__file__).parent
DEFAULT_JUDGE = "glm-5.3:cloud"


def build_judge(cfg: Config) -> tuple[ChatOpenAI, str]:
    """OpenAI-compatible judge; defaults to the local Ollama endpoint.

    A ':cloud' model served through local Ollama still runs remotely, so
    the private-document filter below applies either way.
    """
    model = os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE).strip()
    judge = ChatOpenAI(
        model=model,
        base_url=os.environ.get("RAGAS_JUDGE_BASE_URL",
                                f"{cfg.ollama_url}/v1").strip(),
        api_key=os.environ.get("RAGAS_JUDGE_API_KEY", "ollama").strip(),
        temperature=0,
        # Per request. Without it a stalled connection hung a run for hours:
        # RunConfig's timeout did not reach it.
        timeout=120,
    )
    return judge, model


def build_embeddings(cfg: Config) -> OpenAIEmbeddings:
    """The app's own embedding model, local, via Ollama's /v1 endpoint."""
    return OpenAIEmbeddings(
        model=cfg.embedding_model, base_url=f"{cfg.ollama_url}/v1",
        api_key="ollama",
        # Ollama takes text, not the pre-tokenised ids this would send.
        check_embedding_ctx_length=False,
        timeout=120,
    )


def private_docs(cases: list[dict]) -> set[str]:
    return {src for c in cases if c.get("private")
            for src in c.get("expected_sources", [])}


def skip_reason(case: dict, private: set[str]) -> str | None:
    """Why a case must not go to the judge, or None if it may."""
    if case.get("private"):
        return "private"
    if any(_cites(doc, case.get("context_sources", [])) for doc in private):
        return "private_context"
    if _should_refuse(case):
        return "should_refuse"   # covered by refusal metrics in run_eval
    if not case["contexts"]:
        return "no_contexts"
    return None


# Metrics per task kind. An answered case gets the full set; a case the
# app refused despite being answerable gets context_recall alone, which
# says whether the evidence was there and the model still declined.
same_language = AspectCritic(
    name="same_language",
    definition="Is the response written in the same language as the "
               "user_input? Answer Yes if the languages match, No otherwise.",
)
no_invented_numbers = AspectCritic(
    name="no_invented_numbers",
    definition="Does the response state any number, amount, date, or value "
               "that cannot be found in the retrieved contexts? Answer No if "
               "a figure appears that is absent from the contexts, Yes "
               "otherwise.",
)
rewrite_preserves_intent = AspectCritic(
    name="rewrite_preserves_intent",
    definition="The user_input is a conversation followed by the user's last "
               "message. The response is that last message rewritten to "
               "stand alone. Answer Yes if the rewrite asks exactly what the "
               "last message asks, read in the context of the conversation; "
               "No if it changes, drops or adds anything.",
)
# Judge runs at temperature 0, so the default of three generations would be
# three identical calls.
answer_relevancy.strictness = 1

METRICS = {
    "answered": [faithfulness, answer_relevancy, answer_correctness,
                 answer_similarity, context_precision, context_recall,
                 context_entity_recall, no_invented_numbers, same_language],
    "refused": [context_recall],
    "rewrite": [rewrite_preserves_intent],
}


def tasks(cases: list[dict], noise: bool = False) -> tuple[list, Counter]:
    """(key, kind, sample) for every judgeable case, plus skip counts."""
    private = private_docs(cases)
    out, skipped = [], Counter()
    for case in cases:
        reason = skip_reason(case, private)
        if reason:
            skipped[reason] += 1
            continue
        key = row_key(case)
        # Relevancy is judged against what was actually searched.
        asked = case.get("resolved_question") or case["question"]
        if case["refused"] or not case["answer"]:
            out.append((key, "refused", SingleTurnSample(
                user_input=asked, reference=case["expected_answer"],
                retrieved_contexts=case["contexts"])))
        else:
            out.append((key, "answered", SingleTurnSample(
                user_input=asked, response=case["answer"],
                reference=case["expected_answer"],
                retrieved_contexts=case["contexts"])))
            if noise:
                out.append((key, "noise", SingleTurnSample(
                    user_input=asked, response=case["answer"],
                    reference=case["expected_answer"],
                    retrieved_contexts=case["contexts"])))
        if case.get("resolved_question"):
            convo = "\n".join(f"{t['role']}: {t['content']}"
                              for t in case.get("history", []))
            out.append((key, "rewrite", SingleTurnSample(
                user_input=f"{convo}\nuser (last message): {case['question']}",
                response=case["resolved_question"])))
    return out, skipped


def score(pending: list, out: Path, judge, embeddings,
          batch_size: int) -> None:
    """Judge pending tasks batch by batch, appending each batch to out."""
    # 600s: noise_sensitivity makes many judge calls per case and a reasoning
    # judge timed 40% of them out at 180s. Requests time out separately.
    # Parallel judge calls. ollama.com caps concurrent requests per account
    # and answers the excess with 429 "waiting for a concurrent request
    # slot", which surfaced here as timeouts; set RAGAS_JUDGE_WORKERS to fit.
    workers = int(os.environ.get("RAGAS_JUDGE_WORKERS", "4"))
    run_config = RunConfig(max_workers=workers, timeout=600, max_retries=3)
    metrics = {**METRICS, "noise": [NoiseSensitivity(mode="relevant")]}
    by_kind: dict[str, list] = {}
    for key, kind, sample in pending:
        by_kind.setdefault(kind, []).append((key, sample))

    done = 0
    with open(out, "a", encoding="utf-8") as handle:
        for kind, items in by_kind.items():
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                result = evaluate(
                    EvaluationDataset([s for _, s in batch]),
                    metrics=metrics[kind], llm=judge, embeddings=embeddings,
                    run_config=run_config,
                    # One failed judge call becomes NaN for that case, not
                    # a dead run.
                    raise_exceptions=False, show_progress=False,
                )
                for (key, _), scores in zip(batch, result.scores):
                    append_line(handle, {"key": key, "kind": kind, **{
                        m: (None if v is None or math.isnan(v) else float(v))
                        for m, v in scores.items()}})
                done += len(batch)
                print(f"[{done}/{len(pending)}] {kind}", file=sys.stderr)


def summarize(path: Path) -> dict:
    """Means with 95% ranges, per kind and metric, from a ragas JSONL."""
    prov, rows = read_jsonl(path)
    summary = {"ragas_file": str(path), "provenance": prov, "kinds": {}}
    for kind in sorted({r["kind"] for r in rows}):
        kind_rows = [r for r in rows if r["kind"] == kind]
        columns = sorted({k for r in kind_rows for k in r}
                         - {"key", "kind"})
        summary["kinds"][kind] = {
            c: {**(metric_ci(column_mean(c), kind_rows) or {}),
                "failed": sum(1 for r in kind_rows if r.get(c) is None)}
            for c in columns
        }
    out = path.with_suffix(".summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["kinds"], indent=2), f"\nwrote {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path,
                        help="answers-*.jsonl written by eval/run_eval.py")
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue an interrupted ragas-*.jsonl")
    parser.add_argument("--summarize", type=Path, default=None,
                        help="only summarise an existing ragas-*.jsonl")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--noise", action="store_true",
                        help="add noise_sensitivity; roughly doubles judge "
                             "calls, so meant for finalist runs only")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.answers:
        parser.error("--answers is required (run eval/run_eval.py first)")

    cfg = Config()
    answers_prov, cases = read_jsonl(args.answers)
    todo, skipped = tasks(cases, noise=args.noise)
    print(f"{len(todo)} judge tasks; skipped {dict(skipped)}",
          file=sys.stderr)

    judge, model = build_judge(cfg)
    out = args.resume or (ROOT / "reports" /
                          f"ragas-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    old_prov, rows = read_jsonl(out)
    if old_prov is None:
        with open(out, "a", encoding="utf-8") as handle:
            append_line(handle, {"_provenance": {
                "answers": str(args.answers), "judge": model,
                "answers_provenance": answers_prov,
                "skipped": dict(skipped),
                "started": datetime.now().isoformat(timespec="seconds"),
            }})
    elif old_prov["answers"] != str(args.answers):
        raise SystemExit(f"{out} scores {old_prov['answers']}, "
                         f"not {args.answers}")
    done = {(r["key"], r["kind"]) for r in rows}
    pending = [t for t in todo if (t[0], t[1]) not in done]
    print(f"scores -> {out} ({len(done)} done, {len(pending)} to go)",
          file=sys.stderr)

    score(pending, out, judge, build_embeddings(cfg), args.batch_size)
    summarize(out)


if __name__ == "__main__":
    main()
