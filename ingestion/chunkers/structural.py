from core.models import Chunk
from ingestion.parser import Block, ParsedDocument

# Rough tokens-per-character for mixed PL/EN text. Polish words are longer
# and diacritics cost extra bytes, so character heuristics under-count.
CHARS_PER_TOKEN = 3.5


class StructuralChunker:
    """Merges adjacent blocks up to a token budget, never crossing pages,
    never mixing tables with prose."""

    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50):
        self.target_chars = int(target_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        groups: list[list[Block]] = []
        current: list[Block] = []

        def flush():
            if current:
                groups.append(list(current))
                current.clear()

        for block in parsed.blocks:
            if not block.text.strip():
                continue

            if block.is_table:
                flush()
                groups.append([block])
                continue

            if current:
                same_page = current[-1].page == block.page
                size = sum(len(b.text) for b in current) + len(block.text)
                if not same_page or size > self.target_chars:
                    flush()

            current.append(block)

        flush()

        chunks: list[Chunk] = []
        index = 0
        for group in groups:
            text = "\n\n".join(b.text.strip() for b in group)
            head = group[0]

            for piece in self._split(text):
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=head.page,
                    sheet=head.sheet,
                    is_table=head.is_table,
                ))
                index += 1

        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]

        pieces = []
        start = 0
        step = max(self.target_chars - self.overlap_chars, 1)
        while start < len(text):
            pieces.append(text[start:start + self.target_chars])
            start += step
        return pieces
