"""Bootstrap error bars for eval metrics and run-to-run differences.

With ~150 cases a 3-5 point swing between two runs can be pure sampling
noise. Every number the harness reports gets a 95% range from resampling
its cases, and two runs are compared case by case (paired), so the range
covers the difference actually being claimed.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

RESAMPLES = 2000
SEED = 7


def _is_score(v) -> bool:
    # Ragas writes NaN for a case the judge failed on; that case has no
    # score, not a score of zero.
    return v is not None and not math.isnan(float(v))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def metric_ci(metric, cases: list, resamples: int = RESAMPLES,
              seed: int = SEED) -> dict | None:
    """{"value", "lo", "hi", "n"}: metric(cases) and its 95% range.

    metric is any function of a case list, so the deterministic metrics in
    eval/metrics.py get error bars without being rewritten per case.
    """
    if not cases:
        return None
    rng = random.Random(seed)
    draws = (metric(rng.choices(cases, k=len(cases)))
             for _ in range(resamples))
    stats = sorted(v for v in draws if not math.isnan(v))
    if not stats:
        return None
    return {"value": round(metric(cases), 4),
            "lo": round(stats[int(0.025 * len(stats))], 4),
            "hi": round(stats[min(int(0.975 * len(stats)), len(stats) - 1)], 4),
            "n": len(cases)}


def read_jsonl(path: Path) -> tuple[dict | None, list[dict]]:
    """(provenance, rows) from an answers or ragas JSONL.

    Tolerates a torn last line: a crash mid-write leaves half a JSON
    object at the end, and that case simply runs again on resume.
    """
    prov, rows = None, []
    if not path.exists():
        return prov, rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_provenance" in row:
            prov = row["_provenance"]
        else:
            rows.append(row)
    return prov, rows


def row_key(row: dict) -> str:
    """Case identity shared by answers rows and ragas rows.

    Same shape as run_eval.case_key: the question alone is not unique.
    """
    return row.get("key") or json.dumps(
        [row["question"], sorted(row.get("expected_sources", []))],
        ensure_ascii=False)


def column_mean(column: str):
    """A metric over ragas rows: the mean of one judged column."""
    def metric(rows):
        values = [float(r[column]) for r in rows if _is_score(r.get(column))]
        return _mean(values) if values else math.nan
    return metric


def paired_diff(metric, a: dict, b: dict, **kw) -> dict | None:
    """metric(B) - metric(A), resampling the cases both runs have.

    a and b map case key -> row. Resampling keys (not the two runs
    independently) keeps each case paired with itself, which is what makes
    a small real difference visible. "significant" is False when the 95%
    range crosses zero.
    """
    keys = sorted(a.keys() & b.keys())

    def diff(ks):
        return metric([b[k] for k in ks]) - metric([a[k] for k in ks])

    ci = metric_ci(diff, keys, **kw)
    if ci is None or math.isnan(ci["value"]):
        return None
    return {**ci, "significant": not ci["lo"] <= 0 <= ci["hi"]}


def compare(path_a: Path, path_b: Path) -> dict:
    """Every metric two runs share, as B - A with a 95% range."""
    from eval.metrics import METRICS

    _, rows_a = read_jsonl(path_a)
    _, rows_b = read_jsonl(path_b)
    # A follow-up has an "answered" and a "rewrite" ragas row under one
    # case key; kind keeps them from overwriting each other.
    a = {(row_key(r), r.get("kind")): r for r in rows_a}
    b = {(row_key(r), r.get("kind")): r for r in rows_b}
    if rows_a and "refused" in rows_a[0]:
        metrics = METRICS
    else:
        columns = {k for r in rows_a for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        metrics = {c: column_mean(c) for c in sorted(columns)}
    return {name: paired_diff(fn, a, b) for name, fn in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two runs (answers-*.jsonl or ragas-*.jsonl) "
                    "case by case: B minus A, with a 95% bootstrap range.")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args()

    print(f"{'metric':32} {'B-A':>7} {'95% range':>18}  n")
    for name, d in compare(args.run_a, args.run_b).items():
        if d is None:
            continue
        verdict = "" if d["significant"] else "  (not significant)"
        print(f"{name:32} {d['value']:>+7.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  {d['n']}{verdict}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
