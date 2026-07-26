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
