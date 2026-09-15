import re

from core.models import SearchResult

# Aggregation intent markers that are whole words in their own right, so
# both edges can be anchored without discarding real matches.
AGGREGATION_TERMS = {
    # English
    "total", "sum", "average", "mean", "count", "how many", "how much",
    "highest", "lowest", "largest", "smallest", "top", "rank", "ranking",
    "aggregate", "overall", "combined", "altogether",
    # Polish - invariant forms
    "razem", "ile", "ogółem", "ogolem", "ranking",
    # Polish - "suma" declined. Enumerated rather than stemmed because the
    # stem "sum" also begins "sumienie" (conscience), an ordinary word in
    # exactly the contract-and-policy prose this corpus is made of.
    "suma", "sumy", "sumę", "sume", "sumie", "sumą", "sum",
}

# Polish inflects, so its aggregation markers are stems, not words:
# "największa", "największą", "największych" all share "najwięks" and none
# of them end there. Anchoring the right edge with \b - as every term used
# to be - made these unmatchable, since the character after the stem is
# always an inflection vowel rather than the word boundary \b demands.
# Matched at word start only.
#
# Over-firing here is cheap: should_refuse_aggregation needs table-heavy
# results as well, so a stem that catches an unrelated word on its own
# refuses nothing.
AGGREGATION_STEMS = {
    "najwięks", "najwieks",       # największa, największą, ...
    "najmniejsz",                 # najmniejsza, najmniejszy, ...
    "najwyżs", "najwyzs",         # najwyższa, najwyższe, ...
    "najniżs", "najnizs",         # najniższa, najniższe, ...
    "sumaryczn",                  # sumaryczna, sumaryczny (aggregate adj.)
    "łączn", "laczn",             # łącznie, łączna, łączną, ...
    "średni", "sredni",           # średnia, średnią, średniej, ...
    "liczb",                      # liczba, liczbę (count)
}

TABLE_MAJORITY = 0.5


def _has_aggregation_intent(question: str) -> bool:
    lowered = question.lower()
    for term in AGGREGATION_TERMS:
        if " " in term:
            if term in lowered:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", lowered):
            return True
    return any(
        re.search(rf"\b{re.escape(stem)}", lowered)
        for stem in AGGREGATION_STEMS
    )


def _is_table_heavy(results: list[SearchResult]) -> bool:
    if not results:
        return False
    tables = sum(1 for r in results if r.chunk.is_table)
    return tables / len(results) > TABLE_MAJORITY


def should_refuse_aggregation(question: str,
                               results: list[SearchResult]) -> bool:
    """Fires only when BOTH signals are present.

    Either signal alone produces false positives: a question containing
    "total" about a prose contract must not be blocked, and a lookup
    question over a table must not be blocked either.

    Defers entirely when a precomputed aggregate summary was retrieved —
    the LLM can read a ready, deterministically-correct total from it, so
    there is nothing to refuse.
    """
    if any(r.chunk.is_summary for r in results):
        return False
    return _has_aggregation_intent(question) and _is_table_heavy(results)


def aggregation_refusal(results: list[SearchResult]) -> str:
    sources = []
    for r in results:
        label = r.chunk.filename
        if r.chunk.sheet:
            label = f"{label} (sheet {r.chunk.sheet})"
        if label not in sources:
            sources.append(label)

    listed = ", ".join(sources)
    return (
        "This looks like a question that requires calculating across a whole "
        "table. I can only read individual rows, so any total I gave you "
        "could be wrong.\n\n"
        f"The relevant data is in: {listed}"
    )
