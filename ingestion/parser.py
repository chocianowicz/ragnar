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


def _ocr_converter() -> DocumentConverter:
    # Used as a fallback when the OCR-disabled default converter comes back
    # with near-empty text — typically scanned/image-only PDFs.
    options = PdfPipelineOptions()
    options.do_ocr = True
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


# Below this density (chars of extracted text per page), we suspect the
# extraction missed content (e.g. a scanned page) and re-parse with OCR.
OCR_TRIGGER_CHARS_PER_PAGE = 50


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
    low_confidence: bool = False

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

    def __init__(self, converter: DocumentConverter | None = None,
                 ocr_converter: DocumentConverter | None = None):
        self._converter = converter or _default_converter()
        self._ocr_converter = ocr_converter

    def parse(self, path: Path) -> ParsedDocument:
        parsed = self._parse_with(self._converter, path)

        if parsed.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE:
            ocr_converter = self._ocr_converter or _ocr_converter()
            parsed = self._parse_with(ocr_converter, path)
            parsed.low_confidence = True

        return parsed

    def _parse_with(self, converter: DocumentConverter, path: Path) -> ParsedDocument:
        doc = converter.convert(str(path)).document

        blocks: list[Block] = []
        pages: set[int] = set()

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov and isinstance(prov, (list, tuple)):
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
