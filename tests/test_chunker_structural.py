import pytest

from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.fixed import FixedChunker
from ingestion.chunkers.structural import StructuralChunker
from ingestion.chunkers.registry import build_chunker


def _doc(blocks):
    return ParsedDocument(markdown="x", blocks=blocks, page_count=1)


def test_chunks_do_not_span_page_boundaries():
    doc = _doc([
        Block(text="Short A.", page=1),
        Block(text="Short B.", page=2),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    for c in chunks:
        assert not ("Short A" in c.text and "Short B" in c.text)


def test_small_adjacent_blocks_on_same_page_are_merged():
    doc = _doc([
        Block(text="First sentence.", page=1),
        Block(text="Second sentence.", page=1),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Second sentence." in chunks[0].text


def test_oversized_block_is_split():
    doc = _doc([Block(text="word " * 4000, page=1)])
    chunks = StructuralChunker(target_tokens=100).chunk(doc, "d", "f.pdf")
    assert len(chunks) > 1


def test_table_blocks_are_never_merged_with_prose():
    doc = _doc([
        Block(text="Intro prose.", page=1),
        Block(text="| a | b |", page=1, is_table=True),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Intro prose" not in table_chunks[0].text


def test_chunk_indices_are_sequential():
    doc = _doc([Block(text=f"Block {i}.", page=i) for i in range(1, 6)])
    chunks = StructuralChunker().chunk(doc, "d", "f.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_build_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker({"strategy": "bogus"})


def test_build_chunker_structural_applies_config():
    chunker = build_chunker({"strategy": "structural", "target_tokens": 300})
    assert isinstance(chunker, StructuralChunker)
    assert chunker.target_chars == int(300 * 3.5)


def test_build_chunker_fixed_returns_fixed_chunker():
    chunker = build_chunker({"strategy": "fixed"})
    assert isinstance(chunker, FixedChunker)
