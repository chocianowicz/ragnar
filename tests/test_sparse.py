import pytest

from retrieval.sparse import encode, tokenize, _term_id, K1


def test_numeric_codes_survive_as_single_tokens():
    """The whole reason this exists: a dense vector ranked the chunk
    holding CN code 31022100 at 291 for a question naming it."""
    assert "31022100" in tokenize("CN code 31022100 ammonium sulphate")


def test_tokenizer_keeps_accented_words_whole():
    assert tokenize("Transformaciones para la Vida") == [
        "transformaciones", "para", "la", "vida"]
    assert tokenize("najwyższa łączna kwota") == [
        "najwyższa", "łączna", "kwota"]


def test_tokenizer_lowercases():
    assert tokenize("Cement CLINKERS") == ["cement", "clinkers"]


def test_term_ids_are_stable_across_processes():
    """Python's hash() is salted per process; using it would silently
    break every stored vector on restart."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "from retrieval.sparse import _term_id; print(_term_id('31022100'))"],
        capture_output=True, text=True, check=True,
    )
    assert int(out.stdout.strip()) == _term_id("31022100")


def test_encode_returns_sorted_unique_indices():
    indices, values = encode("alpha beta alpha gamma")

    assert len(indices) == 3
    assert indices == sorted(indices)
    assert len(values) == len(indices)


def test_repeated_terms_saturate_rather_than_accumulate():
    """Otherwise a chunk that repeats a common word outranks one that
    actually answers the question."""
    once = dict(zip(*encode("alpha")))
    thrice = dict(zip(*encode("alpha alpha alpha")))
    term = _term_id("alpha")

    assert thrice[term] > once[term]
    assert thrice[term] < 3 * once[term]
    assert thrice[term] < K1 + 1          # bounded above by the asymptote


def test_encode_handles_text_with_no_tokens():
    assert encode("!!! ???") == ([], [])
    assert encode("") == ([], [])


def test_matching_token_produces_a_shared_index():
    """A query and the chunk that answers it have to land on the same id."""
    q_idx, _ = encode("What is the benchmark for CN code 31022100?")
    c_idx, _ = encode("| 31022100 | Ammonium sulphate | 0.022 |")

    assert set(q_idx) & set(c_idx)
    assert _term_id("31022100") in set(q_idx) & set(c_idx)
