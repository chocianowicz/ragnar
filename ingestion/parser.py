import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    OcrMode, PdfPipelineOptions, RapidOcrOptions)
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


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:          # no torch, or a build without MPS
        return False


def _ocr_converter() -> DocumentConverter:
    # Used as a fallback when the OCR-disabled default converter comes back
    # with near-empty text — typically scanned/image-only PDFs.
    options = PdfPipelineOptions()
    options.do_ocr = True
    # RapidOCR finds text regions by itself and, on a full-page scan with no
    # detectable ones, comes back with nothing at all — a one-page scanned
    # PDF measured 0 characters on this machine until this was set, 1448
    # after. It is the mode, not a boolean: FULL_PAGE is what
    # force_full_page_ocr meant before Docling deprecated that field, and
    # DEFAULT is the region-detection behaviour that failed.
    options.ocr_options.mode = OcrMode.FULL_PAGE
    if _mps_available():
        # RapidOCR's torch backend can run on the Mac GPU, but it defaults
        # to CPU. Same text, about 2x faster: 15 scanned pages took 21.6s
        # on MPS and 42.5s on CPU. Only switched on where MPS exists, so
        # the container keeps the setup it already had.
        options.ocr_options = RapidOcrOptions(
            backend="torch", mode=OcrMode.FULL_PAGE,
            rapidocr_params={"EngineConfig.torch.use_mps": True})
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


# Below this density (chars of extracted text per page), we suspect the
# extraction missed content (e.g. a scanned page) and re-parse with OCR.
OCR_TRIGGER_CHARS_PER_PAGE = 50


def _sheet_name(item, doc) -> str | None:
    """Sheet title for a table that came from a spreadsheet, else None.

    In Excel, each table's parent is a group labelled 'sheet' whose name is
    the sheet title. PDFs have no such group, so this returns None and the
    table is located by page number instead.
    """
    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    try:
        group = parent.resolve(doc)
    except Exception:
        return None
    if "sheet" in str(getattr(group, "label", "")).lower():
        return getattr(group, "name", None)
    return None


def _looks_like_a_table(item, doc) -> bool:
    """Filters out Docling's occasional misclassification of repetitive or
    fixed-position text as a table.

    A genuine table (PDF or spreadsheet) has at least two columns; a
    single-column match is far more likely to be misread prose than real
    tabular data, and even if it were a genuine single-column table,
    treating it as prose (chunked by the token budget, like any other
    text) is harmless. Doesn't catch every misclassification — a fake
    table can occasionally get split into 2+ columns too — but it's a
    cheap, safe filter for the common case with no real downside.
    """
    try:
        df = item.export_to_dataframe(doc)
    except Exception:
        return True  # can't verify — trust Docling's own classification
    return df.shape[1] >= 2


def _table_summary(item, doc, sheet: str | None) -> str | None:
    """Deterministic aggregate summary for a table item, or None.

    A summary is a nice-to-have on top of the table's own (already-indexed)
    content, never load-bearing for it - any failure here (malformed data,
    an unexpected pandas edge case) must degrade to "no summary" rather than
    take down parsing of the whole document, so the whole thing is one
    try/except rather than two.
    """
    from ingestion.table_summary import summarize_table
    try:
        df = item.export_to_dataframe(doc)
        return summarize_table(df, sheet=sheet)
    except Exception:
        return None


@dataclass
class Block:
    text: str
    page: int | None = None
    is_table: bool = False
    sheet: str | None = None
    is_summary: bool = False


@dataclass
class ParsedDocument:
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0
    low_confidence: bool = False

    @property
    def chars_per_page(self) -> float:
        # Extracted text only. The markdown also carries a "<!-- image -->"
        # placeholder per picture, so a scan with no text at all still
        # scored above the OCR trigger and was never OCR'd.
        if self.page_count == 0:
            return 0.0
        return sum(len(b.text) for b in self.blocks) / self.page_count


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
            if self._ocr_converter is None:
                self._ocr_converter = _ocr_converter()
            parsed = self._parse_with(self._ocr_converter, path)
            parsed.low_confidence = True

        return parsed

    def _parse_with(self, converter: DocumentConverter, path: Path) -> ParsedDocument:
        doc = converter.convert(str(path)).document

        blocks: list[Block] = []
        pages: set[int] = set()

        for item, _level in doc.iterate_items():
            raw_is_table = type(item).__name__.lower().startswith("table")

            if raw_is_table:
                # Table items carry no `.text` — their content lives in a
                # structured grid that must be exported explicitly. Without
                # this, every table (Excel sheets, PDF tables) is silently
                # dropped and never indexed.
                try:
                    text = item.export_to_markdown(doc)
                except Exception:
                    text = ""
                sheet = _sheet_name(item, doc)
                is_table = _looks_like_a_table(item, doc)
            else:
                text = getattr(item, "text", "") or ""
                sheet = None
                is_table = False

            if not text.strip():
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov and isinstance(prov, (list, tuple)):
                page = getattr(prov[0], "page_no", None)

            # For spreadsheet tables the "page number" is just the sheet
            # ordinal — the sheet name is the meaningful locator, so drop the
            # synthetic page and let citations read "file.xlsx, sheet Q1".
            if sheet is not None:
                page = None
            elif page is not None:
                pages.add(page)

            blocks.append(Block(
                text=text,
                page=page,
                is_table=is_table,
                sheet=sheet,
            ))

            # For each table, also emit a precomputed aggregate summary as
            # its own block. Retrieval can then surface a ready "total = X"
            # fact for aggregation questions instead of asking the LLM to add
            # up rows it may only partially see. is_table=False so it reads as
            # a prose fact, not a table fragment.
            if is_table:
                summary = _table_summary(item, doc, sheet)
                if summary:
                    blocks.append(Block(
                        text=summary, page=page, sheet=sheet,
                        is_table=False, is_summary=True,
                    ))

        return ParsedDocument(
            markdown=doc.export_to_markdown(),
            blocks=blocks,
            # Docling's own count. Counting only pages that yielded text
            # made a 300-page scan with one stray letter a "1-page" document
            # of healthy density.
            page_count=len(doc.pages) or len(pages) or 1,
        )


# host_server.py on the Mac, where Docling's layout and table models run on
# the GPU. Docker on macOS cannot reach Metal, so in the container they run
# on CPU (~30 min for a 300-page book). Unset keeps parsing in-process.
PARSER_URL = os.environ.get("PARSER_URL", "").rstrip("/")
# A whole book can take minutes even on the GPU.
PARSE_TIMEOUT = 1800.0

log = logging.getLogger(__name__)


def to_dict(parsed: ParsedDocument) -> dict:
    return asdict(parsed)


def from_dict(data: dict) -> ParsedDocument:
    """Rebuild what to_dict() serialized, refusing anything that isn't that.

    Missing or misnamed keys raise rather than defaulting: a remote response
    is a contract, and a half-built ParsedDocument would index a document
    with no text and report success. RemoteParser catches the exceptions and
    parses locally instead.
    """
    markdown = data["markdown"]
    if not isinstance(markdown, str):
        raise TypeError(f"markdown must be a string, got {type(markdown).__name__}")
    blocks = data["blocks"]
    if not isinstance(blocks, list):
        raise TypeError(f"blocks must be a list, got {type(blocks).__name__}")
    return ParsedDocument(
        markdown=markdown,
        blocks=[Block(**b) for b in blocks],
        page_count=data["page_count"],
        low_confidence=data["low_confidence"],
    )


class RemoteParser:
    """Parses through host_server.py and falls back to local Docling.

    The host runs the same DoclingParser code, so only the device changes.
    The file is sent as bytes rather than as a path, so it works whatever
    the container and the host call the directory.
    """

    def __init__(self, url: str = PARSER_URL, fallback=None,
                 client: httpx.Client | None = None):
        self._url = url.rstrip("/")
        self._fallback = fallback
        self._client = client or httpx.Client()

    def _fallback_parser(self):
        """The local parser, built on first use.

        Built lazily so constructing a RemoteParser costs nothing when the
        host is doing the work, and kept after the first failure so a host
        that is down does not rebuild (and reload) Docling per document.
        """
        if self._fallback is None:
            self._fallback = DoclingParser()
        return self._fallback

    def parse(self, path: Path) -> ParsedDocument:
        if not self._url:
            # No url configured. Returning early is not just an optimisation:
            # httpx resolves "" + "/parse" as a *relative* URL, so without
            # this an unset PARSER_URL would still build and send a request
            # that could only fail.
            return self._fallback_parser().parse(path)
        try:
            response = self._client.post(
                f"{self._url}/parse", content=path.read_bytes(),
                headers={"X-Filename": path.name}, timeout=PARSE_TIMEOUT)
            response.raise_for_status()
            return from_dict(response.json())
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            log.warning("host parser at %s failed on %s (%s); parsing "
                        "in-process", self._url, path.name, exc)
            return self._fallback_parser().parse(path)


def build_parser():
    """RemoteParser when PARSER_URL is set, otherwise local Docling."""
    return RemoteParser() if PARSER_URL else DoclingParser()
