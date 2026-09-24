import re
import unicodedata


def normalise(name: str) -> str:
    """A filename in a form two sources can be compared in.

    macOS stores filenames decomposed — "Me" plus a combining acute —
    while anything typed, pasted or written in a config file is
    precomposed. They render identically and compare unequal, which turns
    a correct citation into a scored miss.
    """
    return unicodedata.normalize("NFC", name)


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


def _cites(source: str, labels: list[str]) -> bool:
    """Whether any label names `source`, as a whole word run.

    Labels carry a page suffix ("a.pdf, p. 3") and HybridQA sub-pages
    extend a title ("... Olympics – Men's super-G"), so equality is too
    strict; a word boundary still keeps "a.pdf" from matching "aa.pdf".
    """
    joined = normalise(" ".join(labels))
    return re.search(rf"\b{re.escape(normalise(source))}\b", joined) is not None


def citation_accuracy(cases: list[dict]) -> float:
    """Did the cited document match the expected source?"""
    # Cases that are supposed to end in a refusal name no sources, so
    # scoring them here counted a correct refusal as a citation miss.
    scored = [c for c in cases if not _should_refuse(c)]
    if not scored:
        return 0.0

    correct = sum(1 for case in scored
                  if any(_cites(src, case["citations"])
                         for src in case["expected_sources"]))
    return correct / len(scored)


def missed_refusal_rate(cases: list[dict]) -> float:
    """Of the cases that should refuse, the share that answered anyway.

    The worse half of refusal_accuracy: each one is an answer to a
    question the corpus cannot support.
    """
    scored = [c for c in cases if _should_refuse(c)]
    if not scored:
        return 0.0
    return sum(1 for c in scored if not c["refused"]) / len(scored)


def false_refusal_rate(cases: list[dict]) -> float:
    """Of the answerable cases, the share that were declined."""
    scored = [c for c in cases if not _should_refuse(c)]
    if not scored:
        return 0.0
    return sum(1 for c in scored if c["refused"]) / len(scored)


def citation_precision(cases: list[dict]) -> float:
    """Of the labels an answer cites, the share naming an expected source.

    citation_accuracy only asks whether the right source is somewhere in
    the list, which citing everything retrieved passes for free.
    """
    scored = [c for c in cases if not _should_refuse(c) and c["citations"]]
    if not scored:
        return 0.0
    per_case = [
        sum(1 for label in c["citations"]
            if any(_cites(src, [label]) for src in c["expected_sources"]))
        / len(c["citations"])
        for c in scored
    ]
    return sum(per_case) / len(per_case)


def _multihop(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c.get("multihop") and not _should_refuse(c)]


def multi_hop_citation_accuracy(cases: list[dict]) -> float:
    """Over multi-hop cases: did the answer cite EVERY expected source?"""
    scored = _multihop(cases)
    if not scored:
        return 0.0
    return sum(1 for c in scored
               if all(_cites(src, c["citations"])
                      for src in c["expected_sources"])) / len(scored)


def multi_hop_retrieval_recall(cases: list[dict]) -> float:
    """Over multi-hop cases: did retrieval surface EVERY expected source?

    Read beside multi_hop_citation_accuracy: a gap between the two means
    the second hop was found and the answer did not use it.
    """
    scored = _multihop(cases)
    if not scored:
        return 0.0
    return sum(1 for c in scored
               if all(_cites(src, c.get("context_sources", []))
                      for src in c["expected_sources"])) / len(scored)


def aggregation_guard(cases: list[dict]) -> dict | None:
    """How the aggregation guard did on cases labelled `aggregation:`.

    caught: real aggregations the guard refused. false_positive: single
    cell lookups it refused anyway. None when no case is labelled.
    """
    labelled = [c for c in cases if "aggregation" in c]
    if not labelled:
        return None

    def fired(c):
        return c.get("mode") == "aggregation_refused"

    def truthy(v):
        return str(v).lower() == "true"

    real = [c for c in labelled if truthy(c["aggregation"])]
    lookup = [c for c in labelled if not truthy(c["aggregation"])]
    return {
        "caught": f"{sum(map(fired, real))}/{len(real)}",
        "false_positive": f"{sum(map(fired, lookup))}/{len(lookup)}",
    }


# Every per-run deterministic metric, by report name.
METRICS = {
    "refusal_accuracy": refusal_accuracy,
    "missed_refusal_rate": missed_refusal_rate,
    "false_refusal_rate": false_refusal_rate,
    "citation_accuracy": citation_accuracy,
    "citation_precision": citation_precision,
    "multi_hop_citation_accuracy": multi_hop_citation_accuracy,
    "multi_hop_retrieval_recall": multi_hop_retrieval_recall,
}
