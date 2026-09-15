import re

import pandas as pd

# A column counts as numeric (and thus aggregatable) when at least this
# fraction of its values parse as numbers. Keeps "ID" columns like "E001"
# out while catching "Salary" columns stored as strings.
NUMERIC_THRESHOLD = 0.8

# Header words that mark a column as an identifier rather than a
# quantity. Summing an identifier column is never meaningful, and on the
# real corpus it produced "CN code — total (sum): 129445222522".
_IDENTIFIER_HEADER = re.compile(
    r"\b(id|code|no\.?|nr|number|ref|reference|sku|cn|isbn|ean|key)\b", re.I)

# Without a header hint, a column is treated as an identifier when every
# value is integral and nearly all are distinct — but only with enough
# rows to be sure. Three distinct integers is just a small table.
_IDENTIFIER_MIN_ROWS = 5
_IDENTIFIER_UNIQUE_SHARE = 0.9


def _looks_like_identifier(name, numeric: pd.Series) -> bool:
    if _IDENTIFIER_HEADER.search(str(name)):
        return True
    values = numeric.dropna()
    if len(values) < _IDENTIFIER_MIN_ROWS:
        return False
    integral = bool((values == values.round()).all())
    distinct_share = values.nunique() / len(values)
    return integral and distinct_share >= _IDENTIFIER_UNIQUE_SHARE


def _fmt(value) -> str:
    f = float(value)
    return str(int(f)) if f == int(f) else f"{f:.2f}"


def summarize_table(df: pd.DataFrame, sheet: str | None = None) -> str | None:
    """Natural-language aggregate summary of a table's numeric columns.

    Returns None if the table has no numeric column worth summarising.

    Computed deterministically in pandas, then stored as its own chunk, so
    aggregate answers ("total salary", "how many rows") come from a ready
    value rather than asking the LLM to add up rows it may only partially
    see. The wording deliberately includes total/sum/average/minimum/
    maximum/count so it embeds near aggregation-style questions.
    """
    if df is None or len(df) == 0:
        return None

    # Duplicate column names (blank/merged headers are common in real
    # spreadsheets) make df[col] return a DataFrame instead of a Series,
    # which pd.to_numeric rejects outright - and it's ambiguous which
    # same-named column would even be meant. Skip them rather than crash.
    dupe_names = set(df.columns[df.columns.duplicated(keep=False)])
    unique_columns = [c for c in df.columns if c not in dupe_names]

    parts: list[str] = []
    for col in unique_columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        n = int(numeric.notna().sum())
        if n == 0 or n < NUMERIC_THRESHOLD * len(df):
            continue
        if _looks_like_identifier(col, numeric):
            continue
        parts.append(
            f"{col} — total (sum): {_fmt(numeric.sum())}; "
            f"average (mean): {_fmt(numeric.mean())}; "
            f"minimum: {_fmt(numeric.min())}; maximum: {_fmt(numeric.max())}; "
            f"count: {n}"
        )

    if not parts:
        return None

    where = f" ({sheet} sheet)" if sheet else ""
    head = f"Aggregate column summary{where}: {len(df)} rows total."
    return head + " " + " ".join(parts)


# Column names as they appear in a summary this module wrote. Kept here,
# next to the format it parses, so the two cannot drift apart unnoticed.
_COLUMN_IN_SUMMARY = re.compile(r"([^;:.]+?) — total \(sum\)")


def columns_of(summary_text: str) -> list[str]:
    """The columns a summary covers, parsed back out of its text.

    A summary is stored as a chunk with no structured metadata, so the
    aggregation guard has to recover the column names from the prose to
    decide whether the summary answers the question being asked.
    """
    names = []
    for raw in _COLUMN_IN_SUMMARY.findall(summary_text):
        # The match runs back to the previous separator, which may leave
        # a trailing count ("count: 3 Revenue") in front of the name.
        name = re.sub(r"^\s*\d+\s*", "", raw).strip()
        if name:
            names.append(name)
    return names
