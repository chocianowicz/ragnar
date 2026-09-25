"""Refusals in the language of the question.

They are fixed text written by code, so they used to be English whatever
the question was asked in. Detection is deterministic: no model call.
"""
import pytest

from core.models import Chunk, SearchResult
from generation import language
from generation.answerer import no_results_message
from generation.guards import aggregation_refusal


@pytest.mark.parametrize("question", [
    "Jaki jest cel klimatyczny UE na 2040 rok?",
    "jaki jest cel emisyjny niemiec",          # no diacritics, lower case
    "Co to jest CBAM?",
    "Ile wynosi benchmark dla cementu?",
    "Wyjaśnij mechanizm CBAM.",                # diacritics alone
])
def test_polish_questions_are_recognised(question):
    assert language.of(question) == "pl"


@pytest.mark.parametrize("question", [
    "What is the net zero goal of Poland?",
    "What does the CBAM say about CO emissions?",
    "How many mi of pipeline are covered?",
    "Show the Polska 2040 plan",               # a name is not a language
])
def test_english_questions_stay_english(question):
    assert language.of(question) == "en"


@pytest.mark.parametrize("question", [
    "Wie hoch ist das Klimaziel Frankreichs?",
    "Was ist das Rezept für Pierogi?",         # "was" is German too
    "Wat is het klimaatdoel van de EU?",       # "is" is Dutch too
    "¿Cuál es el objetivo climático de México?",
    "Quel est l'objectif climatique de la France ?",
])
def test_other_languages_are_told_apart_from_english(question):
    assert language.of(question) == language.OTHER


def _near(*files):
    return [SearchResult(chunk=Chunk(doc_id=f, filename=f, text="t",
                                     chunk_index=0, page=1), score=0.6)
            for f in files]


def test_a_polish_refusal_is_polish_throughout():
    text = no_results_message(_near("a.pdf", "b.pdf"), model_declined=True,
                              lang="pl")

    assert text.startswith("Nie udało się znaleźć odpowiedzi")
    assert "a.pdf, p. 1 i b.pdf, p. 1" in text
    assert "could not" not in text


def test_english_is_unchanged_and_the_fallback():
    english = no_results_message(_near("a.pdf"), lang="en")
    assert english == no_results_message(_near("a.pdf"))
    assert no_results_message([], lang="xx") == language.text("en",
                                                              "no_answer")


def test_the_aggregation_refusal_follows_the_question():
    results = [SearchResult(chunk=Chunk(doc_id="d", filename="t.xlsx",
                                        text="t", chunk_index=0, sheet="Q2"),
                            score=0.6)]
    assert "arkusz Q2" in aggregation_refusal(results, "pl")
    assert "sheet Q2" in aggregation_refusal(results)


class _LLM:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def generate(self, system, user, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return self.reply


def test_a_translation_that_keeps_every_name_is_used():
    llm = _LLM("Keine Antwort gefunden. Am nächsten: a.pdf, p. 1.")
    out = language.translate(llm, "Wie hoch?", "No answer. Closest: a.pdf, p. 1.",
                             ["a.pdf, p. 1"])
    assert out.startswith("Keine Antwort")


def test_a_translation_that_loses_a_name_is_discarded():
    """It might name a different document; English is safer."""
    llm = _LLM("Keine Antwort gefunden. Am nächsten: b.pdf, S. 1.")
    message = "No answer. Closest: a.pdf, p. 1."
    assert language.translate(llm, "Wie hoch?", message,
                              ["a.pdf, p. 1"]) == message


def test_a_failed_translation_falls_back_to_english():
    message = "No answer."
    assert language.translate(_LLM(fail=True), "q", message, []) == message


def test_english_and_polish_refusals_never_call_the_model():
    from generation.answering import Settings, _refuse

    class Job:
        chunks, status = [], ""

        def append(self, piece):
            self.chunks.append(piece)

    settings = Settings(model="m", temperature=0, floor=0.5, candidates=25,
                        use_reranker=True, follow_up=False, multi_query=False,
                        multi_hop=False, self_correct=False)
    llm = _LLM("unused")
    for lang in ("en", "pl"):
        _refuse(Job(), "q", "msg", lang, llm=llm, settings=settings)
    assert llm.calls == 0
