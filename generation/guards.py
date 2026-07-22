import re

from core.models import SearchResult

# English and Polish aggregation intent markers.
AGGREGATION_TERMS = {
    # English
    "total", "sum", "average", "mean", "count", "how many", "how much",
    "highest", "lowest", "largest", "smallest", "top", "rank", "ranking",
    "aggregate", "overall", "combined", "altogether",
    # Polish
    "suma", "sumy", "razem", "łącznie", "lacznie", "ile", "średnia",
    "srednia", "największ", "najwieksz", "najmniejsz", "najwyższ",
    "najwyzsz", "ranking", "łączna", "laczna", "ogółem", "ogolem",
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
    return False


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
    """
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
