from ingestion.tables import chunk_table_markdown


TABLE = """\
| Client | Region | Value |
|---|---|---|
| Acme | North | 1000 |
| Beta | South | 2000 |
| Gamma | East | 3000 |
| Delta | West | 4000 |"""


def test_every_group_repeats_the_header_row():
    groups = chunk_table_markdown(TABLE, rows_per_group=2)

    assert len(groups) == 2
    for group in groups:
        assert "| Client | Region | Value |" in group


def test_rows_are_distributed_across_groups_without_loss():
    groups = chunk_table_markdown(TABLE, rows_per_group=2)
    combined = "\n".join(groups)

    for client in ("Acme", "Beta", "Gamma", "Delta"):
        assert client in combined


def test_small_table_stays_in_one_group():
    groups = chunk_table_markdown(TABLE, rows_per_group=20)
    assert len(groups) == 1


def test_table_without_data_rows_returns_nothing():
    header_only = "| A | B |\n|---|---|"
    assert chunk_table_markdown(header_only, rows_per_group=5) == []


def test_blank_row_boundaries_are_preferred_over_fixed_size():
    table = (
        "| A | B |\n|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "| 3 | 4 |\n"
    )
    groups = chunk_table_markdown(table, rows_per_group=20)
    assert len(groups) == 2


WIDE = "| CN code | Description |\n|---|---|\n" + "\n".join(
    f"| {31020000 + i} | {'Ammonium nitrate and related compounds, ' * 6}|"
    for i in range(20)
)


def test_wide_table_is_bounded_by_size_not_just_row_count():
    """Twenty rows of sentence-length cells came to ~14,000 characters on
    the real corpus, against a ~1,750-character prose target."""
    unbounded = chunk_table_markdown(WIDE, rows_per_group=20)
    bounded = chunk_table_markdown(WIDE, rows_per_group=20, max_chars=1750)

    assert len(unbounded) == 1
    assert len(max(unbounded, key=len)) > 4000
    assert len(bounded) > 1
    assert all(len(g) <= 1750 for g in bounded)


def test_size_bound_still_repeats_the_header():
    groups = chunk_table_markdown(WIDE, rows_per_group=20, max_chars=1750)

    assert all("| CN code | Description |" in g for g in groups)


def test_no_rows_are_lost_to_the_size_bound():
    groups = chunk_table_markdown(WIDE, rows_per_group=20, max_chars=1750)
    combined = "\n".join(groups)

    for i in range(20):
        assert str(31020000 + i) in combined


def test_a_single_row_wider_than_the_budget_is_kept_whole():
    """Splitting inside a row would produce the headerless fragment this
    function exists to prevent."""
    huge = "| A | B |\n|---|---|\n| 1 | " + "x" * 5000 + " |"

    groups = chunk_table_markdown(huge, rows_per_group=20, max_chars=100)

    assert len(groups) == 1
    assert "| A | B |" in groups[0]
    assert "x" * 5000 in groups[0]


def test_row_count_still_bounds_a_narrow_table():
    groups = chunk_table_markdown(TABLE, rows_per_group=2, max_chars=100_000)
    assert len(groups) == 2


def test_max_chars_none_preserves_the_old_behaviour():
    assert (chunk_table_markdown(TABLE, rows_per_group=2)
            == chunk_table_markdown(TABLE, rows_per_group=2, max_chars=None))
