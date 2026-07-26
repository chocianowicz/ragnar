import pandas as pd

# A column counts as numeric (and thus aggregatable) when at least this
# fraction of its values parse as numbers. Keeps "ID" columns like "E001"
# out while catching "Salary" columns stored as strings.
NUMERIC_THRESHOLD = 0.8


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
