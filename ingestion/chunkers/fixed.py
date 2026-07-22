from core.models import Chunk
from ingestion.parser import ParsedDocument


class FixedChunker:
    """Naive character-window chunker used only for the Phase 1 slice.

    Chunks never span blocks, so page provenance stays unambiguous.
    """

    def __init__(self, target_chars: int = 2000, overlap_chars: int = 200):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for block in parsed.blocks:
            text = block.text.strip()
            start = 0
            while start < len(text):
                piece = text[start:start + self.target_chars]
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=block.page,
                    sheet=block.sheet,
                    is_table=block.is_table,
                    low_confidence=parsed.low_confidence,
                ))
                index += 1
                step = self.target_chars - self.overlap_chars
                start += max(step, 1)

        return chunks
