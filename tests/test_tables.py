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
