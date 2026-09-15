import pytest

from core.models import Chunk, SearchResult
from generation.guards import should_refuse_aggregation, aggregation_refusal


def _result(text, is_table, filename="data.xlsx", sheet="Q1"):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename=filename, text=text, chunk_index=0,
                    is_table=is_table, sheet=sheet if is_table else None),
        score=0.9,
    )


def test_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("What is the total revenue?", results)


def test_polish_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("Jaka jest suma przychodów?", results)


def test_aggregation_wording_over_prose_is_NOT_refused():
    """The false-positive case the two-signal design exists to prevent."""
    results = [
        _result("The total liability is capped at 50,000 EUR.", False),
        _result("Payment terms are net 30.", False),
    ]
    assert not should_refuse_aggregation("What is the total liability?",
                                         results)


def test_lookup_question_over_tables_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True)]
    assert not should_refuse_aggregation("What is Acme's value?", results)


def test_mixed_results_mostly_prose_are_NOT_refused():
    results = [
        _result("prose one", False),
        _result("prose two", False),
        _result("| a | 1 |", True),
    ]
    assert not should_refuse_aggregation("What is the total?", results)


def test_refusal_message_names_file_and_sheet():
    results = [_result("| a | 1 |", True, filename="sales.xlsx", sheet="Q1")]
    message = aggregation_refusal(results)

    assert "sales.xlsx" in message
    assert "Q1" in message


def test_empty_results_are_not_refused_by_this_guard():
    assert not should_refuse_aggregation("What is the total?", [])


def test_word_containing_aggregation_substring_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True), _result("| Beta | 2000 |", True)]
    assert not should_refuse_aggregation("Is the laptop covered under warranty?", results)
    assert not should_refuse_aggregation("What is the discount policy?", results)


def test_polish_word_containing_aggregation_substring_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True)]
    assert not should_refuse_aggregation("Gdzie mogę kupić bilet?", results)


# Polish inflects its aggregation markers, so the terms for them are stems.
# Anchoring the right edge with \b made every one of these unmatchable -
# the character after the stem is an inflection vowel, never a boundary -
# so these questions reached the model with table rows and an invitation to
# do arithmetic. One case per stem family.
@pytest.mark.parametrize("question", [
    "Jaka jest największa kwota w tabeli?",     # najwięks- superlative
    "Która pozycja jest najmniejsza?",          # najmniejsz-
    "Jaka jest najwyższa stawka?",              # najwyżs-
    "Jaka jest najniższa cena?",                # najniżs-
    "Podaj sumę wszystkich pozycji",            # suma, declined
    "Ile wynosi łączna kwota?",                 # łączn-
    "Jaka jest średnia cena za sztukę?",        # średni-
])
def test_inflected_polish_aggregation_questions_are_refused(question):
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation(question, results)


@pytest.mark.parametrize("question", [
    # "sumienie" (conscience) starts with the same letters as "suma" - the
    # reason suma is enumerated by form rather than matched as a stem.
    "Czy umowa obejmuje klauzulę sumienia?",
    "Jaki jest okres wypowiedzenia?",
    "Kto podpisał umowę?",
    "Jakie są warunki płatności?",
])
def test_ordinary_polish_questions_over_tables_are_NOT_refused(question):
    results = [_result("| Acme | 1000 |", True), _result("| Beta | 2 |", True)]
    assert not should_refuse_aggregation(question, results)


def _summary(text=("Aggregate column summary (Q1 sheet): 2 rows total. "
                   "Revenue — total (sum): 6000; average (mean): 3000; "
                   "minimum: 1000; maximum: 5000; count: 2")):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.xlsx", text=text, chunk_index=0,
                    is_summary=True, sheet="Q1"),
        score=0.9,
    )


def test_aggregation_defers_to_a_summary_of_the_column_asked_about():
    # aggregation intent + table results, and a summary of exactly the
    # column in the question -> the model can read the total; do NOT refuse
    results = [_summary(), _result("| a | 1 |", True)]
    assert not should_refuse_aggregation("What is the total revenue?", results)


def test_aggregation_still_refuses_when_the_summary_is_about_something_else():
    """The failure this fixes: any retrieved summary used to switch the
    guard off, including one summarising an unrelated column."""
    results = [_summary("Aggregate column summary: 1809 rows total. "
                        "Column A BMg — total (sum): 12.3; average (mean): "
                        "0.01; minimum: 0; maximum: 1.5; count: 1804"),
               _result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("What is the total revenue?", results)
