import re

from core.models import SearchResult
from ingestion.table_summary import columns_of

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


def _summary_answers(question: str, results: list[SearchResult]) -> bool:
    """Whether a retrieved summary covers a column the question names.

    A precomputed total is only a reason not to refuse when it is a total
    of the thing being asked about. Any summary at all used to lift the
    guard — including one for an unrelated column that happened to embed
    near the question.
    """
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3}
    for result in results:
        if not result.chunk.is_summary:
            continue
        for column in columns_of(result.chunk.text):
            if any(w in words for w in re.findall(r"\w+", column.lower())):
                return True
    return False


def should_refuse_aggregation(question: str,
                               results: list[SearchResult]) -> bool:
    """Fires only when BOTH signals are present.

    Either signal alone produces false positives: a question containing
    "total" about a prose contract must not be blocked, and a lookup
    question over a table must not be blocked either.

    Defers when a precomputed aggregate summary of the column the question
    asks about was retrieved — the LLM can read a ready,
    deterministically-correct total from it, so there is nothing to refuse.
    A summary of some other column is not a reason to answer.
    """
    if _summary_answers(question, results):
        return False
    return _has_aggregation_intent(question) and _is_table_heavy(results)


def aggregation_refusal(results: list[SearchResult], lang: str = "en") -> str:
    """The refusal, in the language of the question (generation/language)."""
    from generation import language

    sources = []
    for r in results:
        label = r.chunk.filename
        if r.chunk.sheet:
            label = language.text(lang, "sheet", filename=label,
                                  sheet=r.chunk.sheet)
        if label not in sources:
            sources.append(label)

    return language.text(lang, "aggregation", listed=", ".join(sources))
