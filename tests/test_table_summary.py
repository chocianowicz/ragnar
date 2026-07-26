import pandas as pd
from ingestion.table_summary import summarize_table


def test_summarizes_numeric_columns_deterministically():
    df = pd.DataFrame({"Name": ["A", "B", "C"], "Salary": ["100", "200", "300"]})
    s = summarize_table(df, sheet="Pay")

    assert s is not None
    assert "Salary" in s
    assert "600" in s          # sum
    assert "200" in s          # mean
    assert "Pay sheet" in s
    assert "3 rows total" in s


def test_non_numeric_columns_are_ignored():
    df = pd.DataFrame({"Name": ["A", "B", "C"], "City": ["London", "Paris", "Rome"]})
    assert summarize_table(df) is None


def test_id_like_column_is_not_treated_as_numeric():
    df = pd.DataFrame({"ID": ["E001", "E002", "E003"]})
    assert summarize_table(df) is None


def test_empty_table_returns_none():
    assert summarize_table(pd.DataFrame()) is None
    assert summarize_table(None) is None


def test_mostly_numeric_column_with_one_blank_still_summarised():
    df = pd.DataFrame({"Value": ["10", "20", "30", "40", ""]})
    s = summarize_table(df)
    assert s is not None
    assert "100" in s   # sum of the four numbers
    assert "count: 4" in s


def test_duplicate_column_names_are_skipped_not_crashed():
    df = pd.DataFrame([[100, "Acme", 200], [300, "Beta", 400]],
                       columns=["Value", "Name", "Value"])
    s = summarize_table(df)  # must not raise
    # The unambiguous "Name" column isn't numeric, so nothing to summarize.
    assert s is None


def test_duplicate_columns_dont_block_summarising_the_unique_ones():
    df = pd.DataFrame(
        [[100, 1, 100], [300, 2, 300]],
        columns=["Value", "Quantity", "Value"],  # Value duplicated
    )
    s = summarize_table(df)
    assert s is not None
    assert "Quantity" in s   # the one unambiguous numeric column
    assert "Value" not in s  # ambiguous duplicate, correctly skipped
