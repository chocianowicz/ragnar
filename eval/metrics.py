import re


def _should_refuse(case: dict) -> bool:
    """Whether this case is supposed to end in a refusal.

    Usually that is exactly "the answer is not in the corpus". But an
    aggregation question is in-corpus and must still be refused — the
    guard exists so the model never totals rows it has only partly seen —
    so a case may say so explicitly with expect_refusal.
    """
    if "expect_refusal" in case:
        return bool(case["expect_refusal"])
    return bool(case["out_of_corpus"])


def refusal_accuracy(cases: list[dict]) -> float:
    """Did the system refuse exactly when it should have?

    Not a Ragas metric — deterministic, needs no judge, and covers the
    failure mode that matters most: confidently answering something that
    is not in the corpus.
    """
    if not cases:
        return 0.0
    correct = sum(1 for c in cases
                  if bool(c["refused"]) == _should_refuse(c))
    return correct / len(cases)


def citation_accuracy(cases: list[dict]) -> float:
    """Did the cited document match the expected source?"""
    scored = [c for c in cases if not c["out_of_corpus"]]
    if not scored:
        return 0.0

    correct = 0
    for case in scored:
        cited = " ".join(case["citations"])
        if any(re.search(rf"\b{re.escape(src)}\b", cited)
               for src in case["expected_sources"]):
            correct += 1
    return correct / len(scored)
