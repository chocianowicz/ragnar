"""The refusal message names what the documents *do* cover.

"I could not find anything relevant" is false whenever retrieval found
coherent passages that simply were not about the thing asked. On the real
corpus, "What is Poland's net zero goal?" returns EU-wide climate-neutrality
material at 0.068 — relevant to the topic, silent on the country. Telling
the user only "nothing relevant" sends them looking for a bug.

Naming the closest documents is safe in a way that quoting them is not: a
filename cannot be mistaken for an answer, and nothing here calls the model,
so the refusal stays deterministic.
"""
from core.models import Chunk, SearchResult
from generation.answerer import NO_RESULTS_MESSAGE, no_results_message


def _result(filename, page, score=0.08):
    return SearchResult(
        chunk=Chunk(doc_id=filename, filename=filename, text="t",
                    chunk_index=0, page=page),
        score=score,
    )


def test_no_candidates_at_all_keeps_the_bare_message():
    """Nothing was retrieved, so there is nothing honest to point at."""
    assert no_results_message([]) == NO_RESULTS_MESSAGE


def test_near_misses_are_named():
    message = no_results_message([_result("EU NDC.pdf", 46)])

    assert "EU NDC.pdf, p. 46" in message
    assert message != NO_RESULTS_MESSAGE


def test_the_message_says_the_passages_were_not_close_enough():
    """The distinction the user needs: the documents were not silent, they
    were off-target. Without it the refusal reads as a retrieval failure."""
    message = no_results_message([_result("EU NDC.pdf", 46)])

    assert "closely enough" in message


def test_a_declined_answer_says_they_did_not_contain_it_instead():
    """These passages *did* clear the floor — the model read them and found
    no answer. Saying they "did not match closely enough" would be a lie
    about which stage refused."""
    message = no_results_message([_result("EU NDC.pdf", 46)],
                                 model_declined=True)

    assert "closely enough" not in message
    assert "do not answer it" in message


def test_documents_are_deduplicated_keeping_the_best_page():
    """Five chunks from one PDF is one document to go and read, and the
    page worth naming is the closest-scoring one."""
    message = no_results_message([
        _result("EU NDC.pdf", 46, score=0.08),
        _result("EU NDC.pdf", 24, score=0.06),
        _result("EU NDC.pdf", 33, score=0.05),
    ])

    assert message.count("EU NDC.pdf") == 1
    assert "p. 46" in message
    assert "p. 24" not in message


def test_at_most_three_documents_are_named():
    """A refusal is not a search-results page."""
    message = no_results_message(
        [_result(f"doc{i}.pdf", 1, score=0.30 - i / 100) for i in range(6)])

    assert message.count(".pdf") == 3
    assert "doc0.pdf" in message and "doc2.pdf" in message
    assert "doc3.pdf" not in message


def test_the_message_never_quotes_the_passages():
    """Naming a document is a pointer; quoting it is an answer smuggled
    past the floor."""
    hit = SearchResult(
        chunk=Chunk(doc_id="d", filename="f.pdf",
                    text="The EU will be climate-neutral by 2050.",
                    chunk_index=0, page=1),
        score=0.52)

    assert "climate-neutral" not in no_results_message([hit])


def test_a_chunk_without_a_page_is_still_named():
    hit = SearchResult(
        chunk=Chunk(doc_id="d", filename="book.pdf", text="t", chunk_index=0),
        score=0.52)

    assert "book.pdf" in no_results_message([hit])


def test_the_declined_answer_the_user_sees_names_the_documents():
    """The wiring, not just the helper: a model that declines must produce
    the informative refusal, or the helper is dead code."""
    from generation.answerer import Answerer
    from generation.prompts import NO_ANSWER

    class _LLM:
        def generate(self, system, user, *, model=None, temperature=None,
                     history=None):
            return NO_ANSWER

    answer = Answerer(_LLM()).answer("q", [_result("EU NDC.pdf", 46)])

    assert answer.refused
    assert answer.citations == []
    assert "EU NDC.pdf, p. 46" in answer.text
    assert "do not answer it" in answer.text


def test_a_question_the_corpus_knows_nothing_about_names_nothing():
    """Measured on the real corpus, questions with no bearing on it at all
    ("What is the capital of Mongolia?", "How do I bake sourdough?") peg
    their whole candidate list below 0.001 relevance, while genuine near
    misses reach 0.012-0.31.

    Naming documents from the pegged case would manufacture a connection
    the reranker explicitly did not find — worse than saying nothing,
    because the user goes and reads them."""
    pegged = [_result(f"doc{i}.pdf", 1, score=0.0002) for i in range(3)]

    assert no_results_message(pegged) == NO_RESULTS_MESSAGE


def test_only_the_candidates_above_the_noise_are_named():
    """One real near miss alongside pegged noise: name the one, not the
    three, or the message pads a true answer with two false ones."""
    message = no_results_message([
        _result("real.pdf", 46, score=0.068),
        _result("noise1.pdf", 1, score=0.00016),
        _result("noise2.pdf", 2, score=0.00004),
    ])

    assert "real.pdf, p. 46" in message
    assert "noise1.pdf" not in message and "noise2.pdf" not in message
