from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


def _default_converter() -> DocumentConverter:
    # OCR is off by default — the corpus is predominantly native-text PDFs,
    # and running OCR unconditionally is both slow and (per exploration)
    # triggers model downloads even when nothing needs OCR'ing. OCR is
    # enabled explicitly elsewhere only when text extraction comes back
    # near-empty.
    options = PdfPipelineOptions()
    options.do_ocr = False
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


@dataclass
class Block:
    text: str
    page: int | None = None
    is_table: bool = False
    sheet: str | None = None


@dataclass
class ParsedDocument:
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0

    @property
    def chars_per_page(self) -> float:
        if self.page_count == 0:
            return 0.0
        return len(self.markdown) / self.page_count


class DoclingParser:
    """Converts a source file into markdown plus provenance-carrying blocks.

    Blocks are what the chunker consumes; the markdown is for human display
    only. Flattening to markdown loses page numbers, so the two are kept
    separate deliberately.
    """

    def __init__(self, converter: DocumentConverter | None = None):
        self._converter = converter or _default_converter()

    def parse(self, path: Path) -> ParsedDocument:
        doc = self._converter.convert(str(path)).document

        blocks: list[Block] = []
        pages: set[int] = set()

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov:
                page = getattr(prov[0], "page_no", None)
            if page is not None:
                pages.add(page)

            blocks.append(Block(
                text=text,
                page=page,
                is_table=type(item).__name__.lower().startswith("table"),
            ))

        return ParsedDocument(
            markdown=doc.export_to_markdown(),
            blocks=blocks,
            page_count=len(pages) or 1,
        )
