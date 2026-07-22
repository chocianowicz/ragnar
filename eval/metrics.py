def refusal_accuracy(cases: list[dict]) -> float:
    """Did the system refuse exactly when it should have?

    Not a Ragas metric — deterministic, needs no judge, and covers the
    failure mode that matters most: confidently answering something that
    is not in the corpus.
    """
    if not cases:
        return 0.0
    correct = sum(
        1 for c in cases if bool(c["refused"]) == bool(c["out_of_corpus"])
    )
    return correct / len(cases)


def citation_accuracy(cases: list[dict]) -> float:
    """Did the cited document match the expected source?"""
    scored = [c for c in cases if not c["out_of_corpus"]]
    if not scored:
        return 0.0

    correct = 0
    for case in scored:
        cited = " ".join(case["citations"])
        if any(src in cited for src in case["expected_sources"]):
            correct += 1
    return correct / len(scored)
