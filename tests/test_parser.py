import pytest
from pathlib import Path
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
