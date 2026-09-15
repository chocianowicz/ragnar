"""Tests for the dense -> hybrid migration.

It rewrites the index, so the one thing that must never happen is a
partial copy being mistaken for a complete one. Until now it had been
verified by running it once, successfully, which is not the same thing.
"""
import os
import uuid

import pytest

from core.config import Config
from core.models import Chunk
from retrieval.migrate import migrate, _dense_of
from retrieval.store import QdrantStore, DENSE


def _chunk(doc_id, text, index=0):
    return Chunk(doc_id=doc_id, filename="f.pdf", text=text, chunk_index=index)


def _cfg(collection):
    """A Config pointed at a throwaway collection with 8-dim vectors."""
    cfg = Config()
    cfg._raw["storage"]["collection"] = collection
    cfg._raw["models"]["embedding_dim"] = 8
    return cfg


@pytest.fixture
def dense_collection():
    """A pre-hybrid collection with a few points, dropped afterwards."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(os.environ["QDRANT_URL"], name, dim=8, hybrid=False)
    store.ensure_collection()
    store.upsert([_chunk("d1", "ammonium sulphate 31022100", 0),
                  _chunk("d1", "notice period three months", 1)],
                 [[0.1] * 8, [0.2] * 8])
    yield name, store
    for n in (name, f"{name}_hybrid"):
        try:
            QdrantStore(os.environ["QDRANT_URL"], n, dim=8).drop_collection()
        except Exception:
            pass


@pytest.mark.integration
def test_dry_run_changes_nothing(dense_collection, capsys):
    name, store = dense_collection

    migrate(_cfg(name), apply=False)

    assert "Would build" in capsys.readouterr().out
    assert not store.is_hybrid


@pytest.mark.integration
def test_apply_builds_a_hybrid_copy_and_leaves_the_source_alone(dense_collection):
    name, store = dense_collection

    migrate(_cfg(name), apply=True)

    target = QdrantStore(os.environ["QDRANT_URL"], f"{name}_hybrid", dim=8)
    assert target.is_hybrid
    assert target._client.get_collection(f"{name}_hybrid").points_count == 2
    # the source is untouched and still serving
    assert not store.is_hybrid
    assert store._client.get_collection(name).points_count == 2


@pytest.mark.integration
def test_the_lexical_half_is_built_from_the_stored_text(dense_collection):
    """The whole point: no re-parse, no re-embed — the text is already in
    the payload, so an exact identifier is findable afterwards."""
    name, _ = dense_collection
    migrate(_cfg(name), apply=True)

    target = QdrantStore(os.environ["QDRANT_URL"], f"{name}_hybrid", dim=8)
    hits = target.search([0.15] * 8, limit=2, text="31022100")

    assert "31022100" in hits[0].chunk.text


@pytest.mark.integration
def test_migrating_an_already_hybrid_collection_is_a_no_op(dense_collection,
                                                          capsys):
    name, _ = dense_collection
    migrate(_cfg(name), apply=True)
    capsys.readouterr()

    migrate(_cfg(f"{name}_hybrid"), apply=True)

    assert "Nothing to do" in capsys.readouterr().out


@pytest.mark.integration
def test_refuses_to_overwrite_an_existing_target(dense_collection):
    """A second run must not silently rebuild over a previous attempt."""
    name, _ = dense_collection
    migrate(_cfg(name), apply=True)

    with pytest.raises(SystemExit, match="already exists"):
        migrate(_cfg(name), apply=True)


def test_dense_of_accepts_both_stored_shapes():
    """Points may carry a bare vector or a named one."""
    class P:
        def __init__(self, v):
            self.vector = v

    assert _dense_of(P([0.1, 0.2])) == [0.1, 0.2]
    assert _dense_of(P({DENSE: [0.3], "sparse": None})) == [0.3]
