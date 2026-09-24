import pytest
from pathlib import Path
from docling.datamodel.pipeline_options import OcrMode
from ingestion.parser import DoclingParser

FIXTURE = Path("tests/fixtures/sample.pdf")


@pytest.mark.integration
def test_parser_extracts_text_and_pages():
    parsed = DoclingParser().parse(FIXTURE)

    assert "SC-4471" in parsed.markdown
    assert "Escalation" in parsed.markdown

    pages = {b.page for b in parsed.blocks if b.page is not None}
    assert pages == {1, 2}


@pytest.mark.integration
def test_parser_reports_text_density_for_ocr_decision():
    parsed = DoclingParser().parse(FIXTURE)
    # native-text PDF — well above the OCR trigger threshold
    assert parsed.chars_per_page > 50


def test_default_converter_has_ocr_disabled():
    from ingestion.parser import _default_converter
    from docling.datamodel.base_models import InputFormat

    converter = _default_converter()
    pdf_options = converter.format_to_options[InputFormat.PDF]
    assert pdf_options.pipeline_options.do_ocr is False


def test_ocr_converter_ocrs_every_page():
    """RapidOCR segments the page itself and comes back with nothing when it
    finds no regions — a one-page scanned PDF measured 0 characters on this
    machine until the mode was set to FULL_PAGE, 1,448 after. This converter
    only runs on documents that already looked empty, so the tradeoff (OCR
    over the whole page rather than over detected regions) is one it should
    take, and Docling defaulting to region detection is what made it silent.

    Asserted as the mode rather than force_full_page_ocr, which Docling has
    deprecated in favour of it."""
    from ingestion.parser import _ocr_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode

    converter = _ocr_converter()
    pdf_options = converter.format_to_options[InputFormat.PDF]
    assert pdf_options.pipeline_options.do_ocr is True
    assert pdf_options.pipeline_options.ocr_options.mode is OcrMode.FULL_PAGE


class FakeTableItem:
    def __init__(self, df):
        self._df = df

    def export_to_dataframe(self, doc):
        return self._df


def test_single_column_table_is_not_treated_as_a_genuine_table():
    import pandas as pd
    from ingestion.parser import _looks_like_a_table

    single_col = pd.DataFrame({"only": ["a", "b", "c"]})
    assert _looks_like_a_table(FakeTableItem(single_col), doc=None) is False


def test_multi_column_table_is_treated_as_a_genuine_table():
    import pandas as pd
    from ingestion.parser import _looks_like_a_table

    two_col = pd.DataFrame({"Name": ["a"], "Value": [1]})
    assert _looks_like_a_table(FakeTableItem(two_col), doc=None) is True


def test_export_failure_defaults_to_trusting_doclings_classification():
    from ingestion.parser import _looks_like_a_table

    class Broken:
        def export_to_dataframe(self, doc):
            raise RuntimeError("boom")

    assert _looks_like_a_table(Broken(), doc=None) is True


def test_ocr_fallback_reads_the_whole_page():
    """A scanned page is one bitmap the layout model may not box; OCR of
    only detected regions returned nothing for a 1-page scan."""
    from docling.datamodel.base_models import InputFormat
    from ingestion.parser import _ocr_converter

    options = _ocr_converter().format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_ocr and options.ocr_options.mode == OcrMode.FULL_PAGE


def test_ocr_fallback_uses_the_mac_gpu_only_where_there_is_one(monkeypatch):
    from docling.datamodel.base_models import InputFormat
    from ingestion import parser

    def ocr(mps):
        monkeypatch.setattr(parser, "_mps_available", lambda: mps)
        return (parser._ocr_converter().format_to_options[InputFormat.PDF]
                .pipeline_options.ocr_options)

    on_mac = ocr(True)
    assert on_mac.backend == "torch" and on_mac.mode == OcrMode.FULL_PAGE
    assert on_mac.rapidocr_params == {"EngineConfig.torch.use_mps": True}
    elsewhere = ocr(False)
    assert elsewhere.mode == OcrMode.FULL_PAGE
    assert "EngineConfig.torch.use_mps" not in (
        getattr(elsewhere, "rapidocr_params", None) or {})


def test_image_placeholders_do_not_count_as_text():
    """A scan's markdown is "<!-- image -->" per picture and nothing else;
    counting that as text kept it above the trigger, so it was never OCR'd."""
    from ingestion.parser import OCR_TRIGGER_CHARS_PER_PAGE, ParsedDocument

    scan = ParsedDocument(markdown="<!-- image -->\n\n" * 9, blocks=[],
                          page_count=1)
    assert scan.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE


def test_a_long_scan_is_not_one_dense_page():
    """Two stray letters over 300 scanned pages is not 2 chars on 1 page."""
    from ingestion.parser import Block, OCR_TRIGGER_CHARS_PER_PAGE, ParsedDocument

    scan = ParsedDocument(markdown="I\n\nI", page_count=300,
                          blocks=[Block(text="I", page=1)] * 2)
    assert scan.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE
