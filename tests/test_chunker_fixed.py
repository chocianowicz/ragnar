from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.fixed import FixedChunker


def _doc():
    return ParsedDocument(
        markdown="ignored",
        blocks=[
            Block(text="alpha " * 100, page=1),
            Block(text="beta " * 100, page=2),
        ],
        page_count=2,
    )


def test_chunker_carries_provenance_onto_every_chunk():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")

    assert len(chunks) > 2
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.filename == "sample.pdf" for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)


def test_chunker_indexes_chunks_sequentially():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_never_splits_across_pages():
    chunks = FixedChunker(target_chars=200).chunk(_doc(), "d1", "sample.pdf")
    for c in chunks:
        assert "alpha" not in c.text or "beta" not in c.text
