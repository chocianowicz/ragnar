"""Lexical (sparse) encoding, to sit alongside the dense embedding.

Dense vectors are very good at meaning and very bad at identifiers. On the
real corpus, the chunk holding the benchmark row for CN code 31022100
ranked 291st for a question naming that code, because "31022100" carries
almost no semantic signal and the surrounding rows all embed to nearly the
same place. A lexical match finds it immediately — the token is either in
the chunk or it isn't.

No model is involved. This is term frequency over a Unicode word
tokenizer, and Qdrant applies the IDF half at query time (the sparse index
is created with Modifier.IDF), which is also what keeps the statistics
correct as documents are added and removed: the alternative is maintaining
corpus-wide document frequencies here and rewriting them on every ingest.
"""
from __future__ import annotations

import re
import zlib

# Unicode-aware: keeps Polish, Spanish and French words whole, and keeps
# runs of digits whole so "31022100" is one token rather than eight.
_TOKEN = re.compile(r"\w+", re.UNICODE)

# BM25's term-frequency saturation. The second and third occurrence of a
# term should count for less than the first, or a chunk that repeats a
# common word outranks one that actually answers the question.
K1 = 1.2

# Length normalisation (BM25's `b`) is deliberately skipped. It exists to
# stop long documents winning by sheer surface area, and it needs a
# corpus-wide average length to be meaningful. Chunks here are already
# bounded to a target size by the chunker, so the spread it corrects for
# is mostly absent, and tracking that average across ingests would be real
# bookkeeping for little gain.


def tokenize(text: str) -> list[str]:
    """Lowercased word and number tokens.

    Chinese and Japanese are not word-segmented by this, so lexical search
    contributes little for those documents — the dense half still does its
    job, which is the point of running both.
    """
    return [m.group().lower() for m in _TOKEN.finditer(text)]


def _term_id(token: str) -> int:
    """Stable 32-bit id for a token.

    Deliberately not Python's hash(): that is salted per process, so the
    same token would map to different ids across restarts and every stored
    vector would stop matching the queries that should find it.
    """
    return zlib.crc32(token.encode("utf-8"))


def encode(text: str) -> tuple[list[int], list[float]]:
    """(indices, values) for one text, ready for a Qdrant SparseVector.

    Returns empty lists for text with no tokens; callers should treat that
    as "no lexical signal" rather than an error — a chunk of punctuation is
    unusual but not invalid.
    """
    counts: dict[int, int] = {}
    for token in tokenize(text):
        counts[_term_id(token)] = counts.get(_term_id(token), 0) + 1

    if not counts:
        return [], []

    indices = sorted(counts)
    values = [
        (counts[i] * (K1 + 1)) / (counts[i] + K1)
        for i in indices
    ]
    return indices, values
