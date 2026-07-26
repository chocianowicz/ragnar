from core.models import Chunk


def test_chunk_citation_label_prefers_page():
    c = Chunk(doc_id="a1", filename="report.pdf", text="x",
              chunk_index=0, page=4)
    assert c.citation_label() == "report.pdf, p. 4"


def test_chunk_citation_label_uses_sheet_for_spreadsheets():
    c = Chunk(doc_id="a1", filename="sales.xlsx", text="x",
              chunk_index=0, sheet="Q1")
    assert c.citation_label() == "sales.xlsx, sheet Q1"


def test_chunk_citation_label_falls_back_to_filename():
    c = Chunk(doc_id="a1", filename="notes.md", text="x", chunk_index=0)
    assert c.citation_label() == "notes.md"
