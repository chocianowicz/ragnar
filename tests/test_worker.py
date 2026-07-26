import pytest
from core.models import IngestStatus
from ingestion.parser import Block
from ingestion.registry_db import Registry
from ingestion.storage import Storage
from ingestion.pipeline import Pipeline
from ingestion.chunkers.fixed import FixedChunker
from ingestion.worker import IngestWorker
from tests.fakes import FakeEmbedder, FakeStore, FakeParser


@pytest.fixture
def env(tmp_path):
    storage = Storage(tmp_path)
    registry = Registry(tmp_path / "registry.db")
    store = FakeStore()
    pipeline = Pipeline(
        FakeParser(blocks=[Block(text="hello world", page=1)]),
        FixedChunker(target_chars=100),
        FakeEmbedder(),
        store,
    )
    return storage, registry, pipeline, store


def _drop(storage, name, content=b"data"):
    path = storage.inbox / name
    path.write_bytes(content)
    return path


def test_worker_processes_queued_document_to_done(env):
    storage, registry, pipeline, store = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert registry.get(doc_id).status == IngestStatus.DONE
    assert registry.get(doc_id).chunk_count > 0


def test_worker_archives_original_on_success(env):
    storage, registry, pipeline, _ = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert not path.exists()
    assert list(storage.originals.iterdir())


def test_worker_leaves_original_in_inbox_on_failure(env):
    storage, registry, _, store = env
    path = _drop(storage, "bad.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "bad.pdf")

    failing = Pipeline(FakeParser(fail=True), FixedChunker(),
                       FakeEmbedder(), store)
    IngestWorker(storage, registry, failing).process_next()

    doc = registry.get(doc_id)
    assert doc.status == IngestStatus.FAILED
    assert "unreadable" in doc.error
    assert path.exists(), "failed file must stay in inbox for inspection"


def test_worker_processes_one_document_at_a_time(env):
    storage, registry, pipeline, _ = env
    for name in ("a.pdf", "b.pdf"):
        path = _drop(storage, name, content=name.encode())
        registry.add(storage.doc_id(path), name)

    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()

    statuses = [d.status for d in registry.all()]
    assert statuses.count(IngestStatus.DONE) == 1
    assert statuses.count(IngestStatus.QUEUED) == 1


def test_process_next_is_noop_when_queue_empty(env):
    storage, registry, pipeline, _ = env
    assert IngestWorker(storage, registry, pipeline).process_next() is False


def test_worker_writes_converted_markdown(env):
    storage, registry, pipeline, _ = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert storage.read_markdown(doc_id) == "# doc"
