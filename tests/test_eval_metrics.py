from eval.metrics import refusal_accuracy, citation_accuracy


def test_refusal_accuracy_rewards_correct_refusals():
    cases = [
        {"out_of_corpus": True, "refused": True},
        {"out_of_corpus": False, "refused": False},
    ]
    assert refusal_accuracy(cases) == 1.0


def test_refusal_accuracy_penalises_answering_out_of_corpus():
    cases = [{"out_of_corpus": True, "refused": False}]
    assert refusal_accuracy(cases) == 0.0


def test_refusal_accuracy_penalises_refusing_in_corpus():
    cases = [{"out_of_corpus": False, "refused": True}]
    assert refusal_accuracy(cases) == 0.0


def test_citation_accuracy_matches_expected_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["report.pdf, p. 4"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_fails_on_wrong_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["other.pdf, p. 1"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 0.0


def test_citation_accuracy_skips_out_of_corpus_cases():
    cases = [
        {"expected_sources": [], "citations": [], "out_of_corpus": True},
        {"expected_sources": ["a.pdf"], "citations": ["a.pdf, p. 1"],
         "out_of_corpus": False},
    ]
    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_rejects_substring_filename_match():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["quarterly_report.pdf, p. 4"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 0.0


def test_expect_refusal_overrides_out_of_corpus():
    """An aggregation question is in-corpus but must still be refused: the
    guard exists so the model never adds up rows it only partly sees.
    Without this the golden set would score a correct refusal as a miss."""
    cases = [{"out_of_corpus": False, "expect_refusal": True, "refused": True}]
    assert refusal_accuracy(cases) == 1.0


def test_expect_refusal_defaults_to_out_of_corpus():
    cases = [{"out_of_corpus": True, "refused": True},
             {"out_of_corpus": False, "refused": False}]
    assert refusal_accuracy(cases) == 1.0


def test_citation_accuracy_matches_across_unicode_normalisation():
    """macOS stores filenames decomposed ("Me" + combining acute); anything
    typed or copied is precomposed. Same name on screen, different string,
    so a correct citation scored as a miss."""
    cases = [{"out_of_corpus": False,
              "citations": ["NDC 3.0 México_spanish.pdf, p. 22"],
              "expected_sources": ["NDC 3.0 México_spanish.pdf"]}]

    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_ignores_cases_that_should_refuse():
    """An aggregation case expects a refusal, so it names no sources.
    Scoring it for citations counted a correct refusal as a citation miss."""
    cases = [
        {"out_of_corpus": False, "expect_refusal": True,
         "citations": [], "expected_sources": []},
        {"out_of_corpus": False, "citations": ["a.pdf, p. 1"],
         "expected_sources": ["a.pdf"]},
    ]

    assert citation_accuracy(cases) == 1.0
