"""Evaluation harness.

Imports app modules; the app never imports this. The external judge's API
key lives only here, and the harness runs against a hand-written test set —
production documents have no code path to an external service.

Usage:
    python eval/run_eval.py                 # deterministic metrics only
    python eval/run_eval.py --calibrate      # sweep the similarity floor
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
from generation.answerer import Answerer
from generation.guards import should_refuse_aggregation
from eval.metrics import refusal_accuracy, citation_accuracy

ROOT = Path(__file__).parent


def run_cases(score_floor: float | None = None) -> list[dict]:
    cfg = Config()
    floor = cfg.score_floor if score_floor is None else score_floor

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    search = Search(embedder, store, BGEReranker(cfg.reranker_model),
                    cfg.candidates, cfg.top_k, floor)
    answerer = Answerer(OllamaLLM(cfg.ollama_url, cfg.llm_model))

    golden = yaml.safe_load((ROOT / "golden_set.yaml").read_text())
    cases = []

    for entry in golden:
        outcome = search.find(entry["question"])
        guarded = (not outcome.refused and
                   should_refuse_aggregation(entry["question"],
                                             outcome.results))

        if outcome.refused or guarded:
            answer_text, citations, refused = "", [], True
        else:
            answer = answerer.answer(entry["question"], outcome.results)
            answer_text = answer.text
            citations = answer.citations
            refused = answer.refused

        cases.append({
            **entry,
            "answer": answer_text,
            "citations": citations,
            "refused": refused,
            "contexts": [r.chunk.text for r in outcome.results],
        })

    return cases


def calibrate_floor() -> None:
    """Sweep candidate floors and report which separates the two groups best.

    The floor cannot be chosen in advance - it depends on the corpus. This
    is what the out-of-corpus golden entries exist for.
    """
    print(f"{'floor':>7} {'refusal_acc':>12} {'citation_acc':>13}")
    for floor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        cases = run_cases(score_floor=floor)
        print(f"{floor:>7.2f} {refusal_accuracy(cases):>12.2f} "
              f"{citation_accuracy(cases):>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_floor()
        return

    cases = run_cases()
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
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
