"""LLM audit of a golden set before it is trusted for evaluation.

Every entry is checked against evidence pulled from the live index:

- in-corpus: the best excerpts from its expected source document (searched
  with question + expected answer). The judge says whether the excerpts
  support the expected answer, whether the question has one clear answer,
  and whether it can be understood without knowing which document it is
  about — "What is the ISBN of the book?" cannot, in a library of eighteen.
- out-of-corpus: the best excerpts from the whole library. The judge says
  whether any of them answers the question after all; if one does, the
  probe is not out-of-corpus and would score a correct answer as a miss.

Private entries (`private: true`) and any case whose evidence came from a
private document are judged by the local answer model only, never by the
cloud judge. The golden set is never edited: the output is a report to
review by hand. Crash-safe like the other harnesses: one fsync'd line per
entry, --resume to continue.

Usage:
    python eval/check_golden.py
    python eval/check_golden.py --resume eval/reports/golden-check-<stamp>.jsonl
    python eval/check_golden.py --golden eval/golden_hybridqa_draft.yaml \\
        --collection hybridqa_structural
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Config
from generation.llm import OllamaLLM
from retrieval.embedder import OllamaEmbedder
from retrieval.reranker import BGEReranker
from retrieval.search import Search
from retrieval.store import QdrantStore
from eval.metrics import _cites, normalise
from eval.run_eval import append_line, case_key, load_golden, turns_of
from eval.run_ragas import DEFAULT_JUDGE
from eval.stats import read_jsonl

ROOT = Path(__file__).parent
EVIDENCE = 8

IN_CORPUS = """You audit a test set for a search-and-answer system over a \
library of about twenty documents (books, manuals, personal paperwork; \
English and Polish). Each test case is a question, the answer the system \
is expected to give, and excerpts from the document the answer is supposed \
to come from.

Decide one verdict:
- "ok": the excerpts clearly support the expected answer, and the question \
has one clear answer.
- "wrong_answer": the excerpts contradict the expected answer.
- "partial_answer": the expected answer leaves out or distorts part of what \
the excerpts say is the answer.
- "not_in_excerpts": the excerpts neither support nor contradict it.
- "ambiguous": the document supports more than one different answer.
- "not_standalone": a user searching the whole library could not tell which \
document or thing the question is about (e.g. "the book", "this e-book", \
"the author" with no title). For a conversation, judge the last message \
read after the earlier ones.

Reply with JSON only:
{"verdict": "...", "corrected_answer": "<better expected answer, or null>", \
"quote": "<verbatim excerpt text that decides it, or empty>", \
"note": "<one sentence>"}"""

OUT_OF_CORPUS = """You audit a test set for a search-and-answer system over \
a library of about twenty documents. This question is meant to be \
UNANSWERABLE from the library, so the system should refuse it. Below are \
the library excerpts that match it best.

Decide one verdict:
- "ok": no excerpt answers the question.
- "answerable": an excerpt answers it (fully or substantially), so it is not \
a valid unanswerable probe.

Reply with JSON only:
{"verdict": "...", "corrected_answer": "<the answer the excerpt gives, or \
null>", "quote": "<verbatim excerpt text, or empty>", \
"note": "<one sentence>"}"""


def parse_verdict(text: str) -> dict:
    """The judge's JSON, tolerating code fences and prose around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def doc_ids_by_filename(store: QdrantStore) -> dict[str, set[str]]:
    """normalised filename -> doc_ids, read off the index itself."""
    ids: dict[str, set[str]] = {}
    for r in store.search([0.0] * store.dim, limit=100_000):
        ids.setdefault(normalise(r.chunk.filename), set()).add(r.chunk.doc_id)
    return ids


def excerpts(results) -> str:
    return "\n\n".join(f"[{i}] ({r.chunk.citation_label()})\n{r.chunk.text}"
                       for i, r in enumerate(results, 1))


def check(entry: dict, search: Search, ids: dict, private: set[str],
          cloud: OllamaLLM, local: OllamaLLM) -> dict:
    turns = turns_of(entry)
    conversation = "\n".join(f"user: {t}" for t in turns)
    if entry.get("out_of_corpus"):
        outcome = search.find(turns[-1], score_floor=0.0)
        system = OUT_OF_CORPUS
        prompt = f"Question:\n{conversation}\n\nExcerpts:\n"
    else:
        scope = sorted({d for src in entry["expected_sources"]
                        for d in ids.get(normalise(src), ())})
        if not scope:
            return {"verdict": "source_not_indexed",
                    "note": f"{entry['expected_sources']} not in the index"}
        outcome = search.find(f"{turns[-1]} {entry['expected_answer']}",
                              doc_ids=scope, score_floor=0.0)
        system = IN_CORPUS
        prompt = (f"Question:\n{conversation}\n\n"
                  f"Expected answer: {entry['expected_answer']}\n\n"
                  f"Excerpts from {', '.join(entry['expected_sources'])}:\n")

    sources = [r.chunk.citation_label() for r in outcome.results]
    is_private = bool(entry.get("private")) or any(
        _cites(doc, sources) for doc in private)
    judge = local if is_private else cloud
    verdict = parse_verdict(judge.generate(
        system, prompt + excerpts(outcome.results)))
    return {**verdict, "verdict": verdict.get("verdict", "unparsed"),
            "judge": judge.model, "evidence_sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path,
                        default=ROOT / "golden_set.yaml")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = Config()
    store = QdrantStore(cfg.qdrant_url, args.collection or cfg.collection,
                        cfg.embedding_dim)
    search = Search(OllamaEmbedder(cfg.ollama_url, cfg.embedding_model),
                    store,
                    BGEReranker(cfg.reranker_model,
                                max_length=cfg.reranker_max_length),
                    candidates=40, top_k=EVIDENCE, score_floor=0.0)
    cloud = OllamaLLM(cfg.ollama_url,
                      os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE))
    local = OllamaLLM(cfg.ollama_url, cfg.llm_model)

    golden = load_golden(args.golden)
    private = {src for e in golden if e.get("private")
               for src in e.get("expected_sources", [])}
    ids = doc_ids_by_filename(store)

    out = args.resume or (ROOT / "reports" /
                          f"golden-check-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    out.parent.mkdir(exist_ok=True)
    _, rows = read_jsonl(out)
    # A judge error is retried on resume; everything else is final.
    done = {r["key"] for r in rows if r["verdict"] != "error"}
    print(f"-> {out} ({len(done)} of {len(golden)} done)", file=sys.stderr)

    with open(out, "a", encoding="utf-8") as handle:
        for n, entry in enumerate(golden, 1):
            key = case_key(entry)
            if key in done:
                continue
            try:
                result = check(entry, search, ids, private, cloud, local)
            except Exception as exc:    # one bad call must not end the run
                result = {"verdict": "error", "note": repr(exc)[:300]}
            rows.append({"key": key, "question": turns_of(entry)[-1],
                         "expected_answer": entry.get("expected_answer"),
                         "expected_sources": entry.get("expected_sources"),
                         "out_of_corpus": bool(entry.get("out_of_corpus")),
                         **result})
            append_line(handle, rows[-1])
            print(f"[{n}/{len(golden)}] {result['verdict']:<18} "
                  f"{turns_of(entry)[-1][:60]}", file=sys.stderr)

    latest = list({r["key"]: r for r in rows}.values())
    print("\n" + json.dumps(Counter(r["verdict"] for r in latest)))
    for r in latest:
        if r["verdict"] != "ok":
            print(f"\n[{r['verdict']}] {r['question']}\n"
                  f"  expected: {r['expected_answer']}\n"
                  f"  suggested: {r.get('corrected_answer')}\n"
                  f"  note: {r.get('note')}")


if __name__ == "__main__":
    main()
