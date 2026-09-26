"""Evaluation harness.

Imports app modules; the app never imports this. The external judge's API
key lives only here, and the harness runs against a hand-written test set —
production documents have no code path to an external service.

Usage:
    python eval/run_eval.py                  # deterministic metrics only
    python eval/run_eval.py --resume eval/reports/answers-<stamp>.jsonl
    python eval/run_eval.py --collection hybridqa_structural --multi-hop \\
        --golden eval/golden_hybridqa_draft.yaml
    python eval/run_eval.py --calibrate       # sweep the similarity floor
    python eval/run_eval.py --golden mine.yaml
    python eval/run_eval.py --calibrate --floors 0.30:0.70:0.01
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow `python eval/run_eval.py` to find the app packages even though
# running a script (rather than `-m`) puts this file's directory, not the
# repo root, at the front of sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from core.config import Config
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.reranker import BGEReranker
from retrieval.search import Search
from retrieval.agentic import AgenticSearch
from generation.llm import OllamaLLM
from generation import followup, injection
from generation.answerer import (Answerer, AnswerMode, classify,
                                 citation_labels)
from eval.metrics import (refusal_accuracy, citation_accuracy, normalise,
                          _should_refuse, aggregation_guard, METRICS,
                          missed_refusal_rate, false_refusal_rate)
from eval.stats import metric_ci, read_jsonl

ROOT = Path(__file__).parent


def private_companion(path: Path) -> Path:
    """golden_set.yaml -> golden_set.private.yaml, beside it."""
    return path.with_name(f"{path.stem}.private{path.suffix}")


def load_golden(path: Path) -> list[dict]:
    """A golden set, plus its gitignored private companion if present.

    Cases about personal documents cannot live in a tracked file, so they
    sit in <name>.private.yaml. Without that file a run covers the public
    cases only, which is what anyone cloning the repo gets.
    """
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    companion = private_companion(path)
    if companion.exists():
        entries += yaml.safe_load(companion.read_text(encoding="utf-8")) or []
    if not entries:
        raise SystemExit(f"{path} is empty — nothing to evaluate.")
    return entries


def case_flagged(results) -> bool:
    """Whether any retrieved passage carries instruction-shaped text.

    Recorded per case because an adversarial golden entry asserts two
    things — the visible fact was answered, and the planted instruction
    was flagged — and `citations` holds label strings, which cannot say.
    """
    return any(injection.flag(r.chunk.text) for r in results)


def turns_of(entry: dict) -> list[str]:
    """The questions an entry asks, in order.

    A plain entry has one `question:`. A follow-up chain has `turns:`, and
    only the last turn is scored — the earlier ones exist to give it
    something to refer back to, exactly as a user would.
    """
    has_q, has_t = "question" in entry, "turns" in entry
    if has_q == has_t:
        raise SystemExit(
            f"golden entry must have either question: or turns:, got "
            f"{sorted(entry)}"
        )
    return [entry["question"]] if has_q else list(entry["turns"])


def check_corpus(store: QdrantStore, golden: list[dict]) -> None:
    """Refuse to score against a corpus that cannot contain the answers.

    The harness searches whatever is in the live collection. If the
    expected documents were never ingested, every in-corpus case refuses,
    every out-of-corpus case refuses too, and the run reports a plausible
    refusal_accuracy instead of an error — a number that looks like a
    measurement and means nothing. Cheap to check, and the failure mode it
    prevents is silent.
    """
    # Normalised on both sides: the indexed filename is decomposed on
    # macOS and the golden set's is precomposed, so a plain set difference
    # reports a document that is sitting right there as missing.
    expected = {
        normalise(src) for entry in golden
        if not entry.get("out_of_corpus")
        for src in entry.get("expected_sources", [])
    }
    if not expected:
        return

    # One cheap unfiltered probe; the payload carries the filename.
    indexed = {
        normalise(r.chunk.filename)
        for r in store.search([0.0] * store.dim, limit=10_000)
    }
    missing = sorted(expected - indexed)
    if missing:
        raise SystemExit(
            "These golden-set sources are not in the indexed corpus:\n  "
            + "\n  ".join(missing)
            + f"\n\nIngest them into collection '{store.collection}' first, "
              "or the scores below measure nothing."
        )


def build_report(cases: list[dict], golden_path, retrieval_only: bool,
                 prov: dict | None = None) -> dict:
    """The summary written to stdout and to eval/reports/.

    `retrieval_only` is recorded rather than implied: both metrics are
    derived from retrieval, so a run without the answer model produces
    real numbers — and someone reading the report later has to be able to
    tell that the answers themselves were never generated.

    Every metric carries a 95% bootstrap range (eval/stats.py). Multi-hop
    metrics are left out when the set has no multi-hop case.
    """
    has_multihop = any(c.get("multihop") for c in cases)
    latencies = sorted(c["latency_s"] for c in cases if "latency_s" in c)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "golden_set": str(golden_path or "eval/golden_set.yaml"),
        "retrieval_only": bool(retrieval_only),
        "provenance": prov,
        **{name: metric_ci(fn, cases) for name, fn in METRICS.items()
           if has_multihop or not name.startswith("multi_hop")},
        "aggregation_guard": aggregation_guard(cases),
    }
    if latencies:
        report["latency_s"] = {
            "p50": latencies[len(latencies) // 2],
            "p90": latencies[min(int(len(latencies) * 0.9),
                                  len(latencies) - 1)],
            "max": latencies[-1],
        }
    return report


def case_key(entry: dict) -> str:
    """Identity of a golden entry across a crash and a resume.

    The question alone is not unique: two books can both be asked who they
    are dedicated to. Question plus expected sources is.
    """
    return json.dumps([turns_of(entry)[-1],
                       sorted(entry.get("expected_sources", []))],
                      ensure_ascii=False)


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT.parent,
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(cfg: Config, golden_path: Path, collection: str,
               stages: dict, retrieval_only: bool, score_floor: float) -> dict:
    """What produced a run. `config` must match for a resume to append."""
    return {
        "git": git_revision(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "golden_set": str(golden_path),
            "collection": collection,
            "llm_model": cfg.llm_model,
            "embedding_model": cfg.embedding_model,
            "reranker_model": cfg.reranker_model,
            "candidates": cfg.candidates,
            "top_k": cfg.top_k,
            "score_floor": score_floor,
            "stages": stages,
            "retrieval_only": bool(retrieval_only),
        },
    }


def append_line(handle, row: dict) -> None:
    """One JSON line, on disk before the next case starts."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_cases(score_floor: float | None = None,
              golden_path: Path | None = None,
              verify_corpus: bool = True,
              retrieval_only: bool = False,
              collection: str | None = None,
              stages: dict | None = None,
              sink: Path | None = None) -> list[dict]:
    """Run the golden set through retrieval + answering.

    stages switches on the AgenticSearch stages the UI offers
    (multi_query, multi_hop, self_correct); with none on, this is the plain
    Search path.

    sink, when given, gets one JSON line per finished case. If it already
    holds cases from an interrupted run of the same config, those are kept
    and skipped, so a crash costs only the case in flight.
    """
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor
    collection = collection or cfg.collection
    stages = {k: v for k, v in (stages or {}).items() if v}
    golden_path = golden_path or (ROOT / "golden_set.yaml")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, collection, cfg.embedding_dim)
    search = Search(embedder, store,
                    BGEReranker(cfg.reranker_model,
                                max_length=cfg.reranker_max_length),
                    cfg.candidates, cfg.top_k, floor)
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)
    answerer = Answerer(llm)
    # Wired as ui/services.py does, so stages measure what the UI runs.
    agentic = AgenticSearch(
        search, llm,
        max_hops=int(cfg.agentic.get("max_hops", 2)),
        variants=int(cfg.agentic.get("variants", 3)),
    ) if stages else None

    golden = load_golden(golden_path)
    if verify_corpus:
        check_corpus(store, golden)

    prov = provenance(cfg, golden_path, collection, stages,
                      retrieval_only, floor)
    cases: list[dict] = []
    handle = None
    if sink:
        old_prov, cases = read_jsonl(sink)
        if old_prov and old_prov["config"] != prov["config"]:
            raise SystemExit(
                f"{sink} was written by a different config; resuming would "
                f"mix two runs.\n  was: {old_prov['config']}\n  now: "
                f"{prov['config']}")
        sink.parent.mkdir(parents=True, exist_ok=True)
        handle = open(sink, "a", encoding="utf-8")
        if old_prov is None:
            append_line(handle, {"_provenance": prov})
        if cases:
            print(f"resuming {sink}: {len(cases)} of {len(golden)} done",
                  file=sys.stderr)
    done = {case_key(c) for c in cases}

    try:
        for n, entry in enumerate(golden, 1):
            if case_key(entry) in done:
                continue
            cases.append(run_one(entry, search, agentic, stages, llm,
                                 answerer, retrieval_only))
            if handle:
                append_line(handle, cases[-1])
            print(f"[{n}/{len(golden)}] {cases[-1]['latency_s']:.1f}s "
                  f"{cases[-1]['question'][:60]}", file=sys.stderr)
    finally:
        if handle:
            handle.close()

    return cases


def run_one(entry, search, agentic, stages, llm, answerer,
            retrieval_only) -> dict:
    started = time.perf_counter()
    history: list[dict] = []
    outcome, mode = None, None
    answer_text, citations, refused, resolved = "", [], True, None

    for question in turns_of(entry):
        # The history the last turn was asked with; the follow-up judge
        # needs it to tell whether the rewrite kept the meaning.
        prior = list(history)
        search_question, was_resolved = question, False
        if history:
            search_question, was_resolved = followup.resolve(
                llm, question, history)
        if agentic:
            outcome, _ = agentic.find(search_question, **stages)
        else:
            outcome = search.find(search_question)
        mode = classify(question, outcome.refused, outcome.results)

        if mode is not AnswerMode.ANSWER:
            answer_text, citations, refused = "", [], True
        elif retrieval_only:
            # Citations come from the retrieved chunks, never from the
            # model, so they are already known. Skipping generation
            # costs the answer text and nothing either metric reads.
            answer_text = ""
            citations = citation_labels(outcome.results)
            refused = False
        else:
            answer = answerer.answer(question, outcome.results,
                                     history=history)
            answer_text, citations, refused = (
                answer.text, answer.citations, answer.refused)
        resolved = search_question if was_resolved else None
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer_text}]

    return {
        **entry,
        "question": turns_of(entry)[-1],
        "resolved_question": resolved,
        "history": prior,
        "answer": answer_text,
        "citations": citations,
        "refused": refused,
        "mode": mode.value,
        "contexts": [r.chunk.text for r in outcome.results],
        "context_sources": [r.chunk.citation_label()
                            for r in outcome.results],
        "flagged": case_flagged(outcome.results),
        # What the floor is compared against. Recorded so a
        # calibration sweep can be derived from this one run.
        "best_score": outcome.trace.best_score,
        "latency_s": round(time.perf_counter() - started, 2),
    }


def parse_floors(spec: str) -> list[float]:
    """"start:stop:step" -> the floors to sweep, stop inclusive."""
    try:
        start, stop, step = (float(p) for p in spec.split(":"))
    except ValueError:
        raise SystemExit(
            f"--floors expects start:stop:step, got {spec!r}"
        )
    if step <= 0 or stop < start:
        raise SystemExit(f"--floors range is empty: {spec!r}")
    floors, value = [], start
    while value <= stop + 1e-9:
        floors.append(round(value, 4))
        value += step
    return floors


DEFAULT_FLOORS = "0.05:0.95:0.05"


def at_floor(cases: list[dict], floor: float) -> list[dict]:
    """The cases as they would have come out under `floor`.

    Derived rather than re-run. Retrieval and re-ranking do not depend on
    the floor — it only decides which already-scored passages survive — so
    a case refuses exactly when its best score falls below it.
    """
    return [{**c, "refused": c.get("best_score") is None
             or c["best_score"] < floor} for c in cases]


def sweep(cases: list[dict], floors: list[float]):
    """(floor, refusal_accuracy, missed, false, wrong questions) per floor.

    missed = out-of-corpus answered anyway (the hallucination risk), false =
    answerable questions refused. refusal_accuracy weighs them equally;
    choosing a floor should not.
    """
    rows = []
    for floor in floors:
        scored = at_floor(cases, floor)
        wrong = [c["question"] for c in scored
                 if bool(c["refused"]) != _should_refuse(c)]
        rows.append((floor, refusal_accuracy(scored),
                     missed_refusal_rate(scored), false_refusal_rate(scored),
                     wrong))
    return rows


def calibrate_floor(floors: list[float],
                    golden_path: Path | None = None,
                    retrieval_only: bool = False,
                    collection: str | None = None,
                    stages: dict | None = None,
                    sink: Path | None = None) -> None:
    """Sweep candidate floors and report the trade-off at each.

    The floor cannot be chosen in advance - it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.
    """
    cases = run_cases(golden_path=golden_path, retrieval_only=retrieval_only,
                      collection=collection, stages=stages, sink=sink)

    print(f"{'floor':>6} {'accuracy':>9} {'missed':>7} {'false':>6}  wrong")
    rows = sweep(cases, floors)
    for floor, accuracy, missed, false, wrong in rows:
        summary = "; ".join(q[:38] for q in wrong[:3])
        if len(wrong) > 3:
            summary += f" (+{len(wrong) - 3} more)"
        print(f"{floor:>6.2f} {accuracy:>9.3f} {missed:>7.3f} {false:>6.3f}"
              f"  {summary}")

    best = max(rows, key=lambda row: row[1])
    ci = metric_ci(lambda cs: refusal_accuracy(at_floor(cs, best[0])), cases)
    print(f"\nBest separation at floor {best[0]:.2f}: refusal accuracy "
          f"{best[1]:.3f} (95% range {ci['lo']:.3f}-{ci['hi']:.3f}), "
          f"missed {best[2]:.3f}, false {best[3]:.3f}, "
          f"{len(best[4])} wrong of {len(cases)}.")
    print("missed = out-of-corpus answered (worse); false = answerable "
          "refused. Prefer the lowest floor on a flat stretch of low "
          "`missed`, not a lone peak.")
    print("Citation accuracy is unaffected by the floor; it was "
          f"{citation_accuracy(cases):.2f} on this run.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="skip answer generation. Both metrics come from retrieval, so "
             "this measures the same things without loading the answer "
             "model — ~14 GB on the reference machine.",
    )
    parser.add_argument(
        "--golden", type=Path, default=None,
        help="golden set to run (default: eval/golden_set.yaml). Keeps a "
             "real-corpus set separate from the fixture one.",
    )
    parser.add_argument(
        "--floors", default=DEFAULT_FLOORS,
        help=f"calibration sweep as start:stop:step (default {DEFAULT_FLOORS})",
    )
    parser.add_argument(
        "--collection", default=None,
        help="Qdrant collection to search (default: config.yaml's). "
             "The HybridQA benchmark lives in 'hybridqa_structural'.",
    )
    for stage in ("multi-hop", "multi-query", "self-correct"):
        parser.add_argument(f"--{stage}", action="store_true",
                            help=f"turn on the {stage} AgenticSearch stage")
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="continue an interrupted run from its answers-*.jsonl sink",
    )
    args = parser.parse_args()

    reports = ROOT / "reports"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stages = {"multi_hop": args.multi_hop, "multi_query": args.multi_query,
              "self_correct": args.self_correct}

    if args.calibrate:
        # Multi-hop and self-correct only run after something has cleared
        # the floor, so they cannot move the refuse/answer decision; they
        # would only make the recorded best score a later pass's. The UI's
        # "Rephrase the question" gate is multi-query alone.
        if args.multi_hop or args.self_correct:
            parser.error("--calibrate takes --multi-query only: multi-hop "
                         "and self-correct run after the floor decision")
        sink = args.resume or reports / f"calibrate-{stamp}.jsonl"
        print(f"cases -> {sink}", file=sys.stderr)
        calibrate_floor(parse_floors(args.floors), golden_path=args.golden,
                        retrieval_only=args.retrieval_only,
                        collection=args.collection, stages=stages, sink=sink)
        return

    # Every case lands here as it finishes, so a crash or a Ctrl-C loses
    # only the case in flight. Rerun with --resume on this path.
    sink = args.resume or reports / f"answers-{stamp}.jsonl"
    print(f"answers -> {sink}", file=sys.stderr)
    cases = run_cases(golden_path=args.golden,
                      retrieval_only=args.retrieval_only,
                      collection=args.collection, stages=stages, sink=sink)
    prov, _ = read_jsonl(sink)
    report = build_report(cases, args.golden, args.retrieval_only, prov)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = sink.with_name(sink.stem.replace("answers-", "") + ".json")
    out.write_text(
        json.dumps({"summary": report, "cases": cases},
                   indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
