"""Evaluation harness.

Imports app modules; the app never imports this. The external judge's API
key lives only here, and the harness runs against a hand-written test set —
production documents have no code path to an external service.

Usage:
    python eval/run_eval.py                  # deterministic metrics only
    python eval/run_eval.py --calibrate       # sweep the similarity floor
    python eval/run_eval.py --golden mine.yaml
    python eval/run_eval.py --calibrate --floors 0.45:0.70:0.01
"""
import argparse
import json
import sys
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
from generation.llm import OllamaLLM
from generation import followup
from generation.answerer import Answerer, AnswerMode, classify
from eval.metrics import refusal_accuracy, citation_accuracy

ROOT = Path(__file__).parent


def load_golden(path: Path) -> list[dict]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not entries:
        raise SystemExit(f"{path} is empty — nothing to evaluate.")
    return entries


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
    expected = {
        src for entry in golden
        if not entry.get("out_of_corpus")
        for src in entry.get("expected_sources", [])
    }
    if not expected:
        return

    # One cheap unfiltered probe; the payload carries the filename.
    indexed = {
        r.chunk.filename
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


def run_cases(score_floor: float | None = None,
              golden_path: Path | None = None,
              verify_corpus: bool = True) -> list[dict]:
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    search = Search(embedder, store,
                    BGEReranker(cfg.reranker_model,
                                max_length=cfg.reranker_max_length),
                    cfg.candidates, cfg.top_k, floor)
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)
    answerer = Answerer(llm)

    golden = load_golden(golden_path or (ROOT / "golden_set.yaml"))
    if verify_corpus:
        check_corpus(store, golden)
    cases = []

    for entry in golden:
        history: list[dict] = []
        outcome = None
        answer_text, citations, refused, resolved = "", [], True, None

        for question in turns_of(entry):
            search_question, was_resolved = question, False
            if history:
                search_question, was_resolved = followup.resolve(
                    llm, question, history)
            outcome = search.find(search_question)
            mode = classify(question, outcome.refused, outcome.results)

            if mode is not AnswerMode.ANSWER:
                answer_text, citations, refused = "", [], True
            else:
                answer = answerer.answer(question, outcome.results,
                                         history=history)
                answer_text, citations, refused = (
                    answer.text, answer.citations, answer.refused)
            resolved = search_question if was_resolved else None
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer_text}]

        cases.append({
            **entry,
            "question": turns_of(entry)[-1],
            "resolved_question": resolved,
            "answer": answer_text,
            "citations": citations,
            "refused": refused,
            "contexts": [r.chunk.text for r in outcome.results],
        })

    return cases


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


DEFAULT_FLOORS = "0.40:0.80:0.05"


def calibrate_floor(floors: list[float],
                    golden_path: Path | None = None) -> None:
    """Sweep candidate floors and report which separates the two groups best.

    The floor cannot be chosen in advance - it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.

    The sweep used to step by 0.1, which could not produce the 0.55 the
    config actually ships - the grid has to be fine enough to contain the
    answer it is meant to find.
    """
    print(f"{'floor':>7} {'refusal_acc':>12} {'citation_acc':>13}  misses")
    for floor in floors:
        cases = run_cases(score_floor=floor, golden_path=golden_path,
                          verify_corpus=floor == floors[0])
        # Name what each floor gets wrong. A pair of aggregate numbers says
        # a floor is worse without saying which question it broke, which is
        # the thing you need in order to judge whether the trade is right.
        missed = [
            c["question"] for c in cases
            if bool(c["refused"]) != bool(c["out_of_corpus"])
        ]
        summary = "; ".join(q[:40] for q in missed[:3])
        if len(missed) > 3:
            summary += f" (+{len(missed) - 3} more)"
        print(f"{floor:>7.2f} {refusal_accuracy(cases):>12.2f} "
              f"{citation_accuracy(cases):>13.2f}  {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--golden", type=Path, default=None,
        help="golden set to run (default: eval/golden_set.yaml). Keeps a "
             "real-corpus set separate from the fixture one.",
    )
    parser.add_argument(
        "--floors", default=DEFAULT_FLOORS,
        help=f"calibration sweep as start:stop:step (default {DEFAULT_FLOORS})",
    )
    args = parser.parse_args()

    if args.calibrate:
        calibrate_floor(parse_floors(args.floors), golden_path=args.golden)
        return

    cases = run_cases(golden_path=args.golden)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "golden_set": str(args.golden or "eval/golden_set.yaml"),
        "refusal_accuracy": refusal_accuracy(cases),
        "citation_accuracy": citation_accuracy(cases),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (reports / f"{stamp}.json").write_text(
        json.dumps({"summary": report, "cases": cases},
                   indent=2, ensure_ascii=False)
    )
    print(f"\nwrote eval/reports/{stamp}.json")


if __name__ == "__main__":
    main()
