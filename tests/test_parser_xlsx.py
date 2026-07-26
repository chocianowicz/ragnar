import pytest
from pathlib import Path
from ingestion.parser import DoclingParser

FIXTURE = Path("tests/fixtures/sample.xlsx")


@pytest.mark.integration
def test_xlsx_tables_become_blocks_with_content():
    parsed = DoclingParser().parse(FIXTURE)

    table_blocks = [b for b in parsed.blocks if b.is_table]
    assert table_blocks, "Excel tables must produce blocks (were being dropped)"

    combined = "\n".join(b.text for b in table_blocks)
    assert "Acme" in combined
    assert "Delta" in combined


@pytest.mark.integration
def test_xlsx_blocks_carry_sheet_names():
    parsed = DoclingParser().parse(FIXTURE)

    sheets = {b.sheet for b in parsed.blocks if b.is_table}
    assert "Q1 Sales" in sheets
    assert "Q2 Sales" in sheets


@pytest.mark.integration
def test_xlsx_table_chunks_are_searchable_and_cite_the_sheet():
    from ingestion.chunkers.structural import StructuralChunker

    parsed = DoclingParser().parse(FIXTURE)
    chunks = StructuralChunker().chunk(parsed, "d1", "sample.xlsx")

    assert chunks, "Excel must produce chunks — otherwise nothing is indexed"
    # The sheet name should reach the citation label, not a synthetic page.
    labels = {c.citation_label() for c in chunks}
    assert any("sheet Q1 Sales" in lbl for lbl in labels)


@pytest.mark.integration
def test_xlsx_emits_precomputed_aggregate_summary_block():
    parsed = DoclingParser().parse(FIXTURE)
    summaries = [b for b in parsed.blocks if b.is_summary]
    assert summaries, "each numeric table should get an aggregate summary block"
    # Q1 sheet values 1000+2000+3000 = 6000 must be precomputed, not left to the LLM
    combined = " ".join(b.text for b in summaries)
    assert "6000" in combined
    assert not any(b.is_table for b in summaries)  # summary reads as prose
