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


class _EchoParser:
    """Returns the file's own bytes as its single block, so a test can tell
    which file on disk a given doc_id was ingested from."""

    def parse(self, path):
        from ingestion.parser import ParsedDocument
        text = path.read_bytes().decode()
        return ParsedDocument(markdown=text,
                              blocks=[Block(text=text, page=1)],
                              page_count=1)


def _drop(storage, name, content=b"data"):
    """Place a file in the inbox the way an upload does: hashed first, then
    written under its own doc_id."""
    doc_id = storage.doc_id_for_bytes(content)
    path = storage.inbox_path(doc_id, name)
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


def test_two_documents_sharing_a_filename_are_indexed_separately(env):
    """The inbox used to be keyed by filename while the registry keyed on
    content, so the second upload of a shared name replaced the first's
    bytes — and the first doc_id was then ingested from the wrong file,
    marked done, and cited under a hash describing different content."""
    storage, registry, _, store = env
    # A parser that echoes the bytes it was handed, so the assertion can
    # tell which file each doc_id was actually ingested from — the whole
    # point of the bug. FakeParser returns fixed blocks and cannot.
    pipeline = Pipeline(_EchoParser(), FixedChunker(target_chars=100),
                        FakeEmbedder(), store)

    ids = []
    for content in (b"contract A", b"contract B"):
        path = _drop(storage, "umowa.pdf", content=content)
        doc_id = storage.doc_id(path)
        registry.add(doc_id, "umowa.pdf")
        ids.append(doc_id)

    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()
    worker.process_next()

    assert [registry.get(i).status for i in ids] == [IngestStatus.DONE] * 2
    indexed = {
        doc_id: " ".join(c.text for c in store.chunks if c.doc_id == doc_id)
        for doc_id in ids
    }
    # Each id holds its own file's content and none of the other's. Before
    # the fix, both ids resolved to the same inbox path and ids[0] indexed
    # "contract B".
    assert "contract A" in indexed[ids[0]] and "contract B" not in indexed[ids[0]]
    assert "contract B" in indexed[ids[1]] and "contract A" not in indexed[ids[1]]


def test_chunks_carry_the_display_filename_not_the_inbox_path(env):
    """The inbox names files by doc_id; citations must not."""
    storage, registry, pipeline, store = env
    path = _drop(storage, "umowa.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "umowa.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert store.chunks
    assert all(c.filename == "umowa.pdf" for c in store.chunks)


def test_document_queued_under_a_bare_filename_still_ingests(env):
    """Upgrade in place: anything already queued before the inbox was keyed
    by doc_id is sitting under its bare name."""
    storage, registry, pipeline, _ = env
    legacy = storage.inbox / "old.pdf"
    legacy.write_bytes(b"queued before the change")
    doc_id = storage.doc_id(legacy)
    registry.add(doc_id, "old.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert registry.get(doc_id).status == IngestStatus.DONE


def test_worker_writes_converted_markdown(env):
    storage, registry, pipeline, _ = env
    path = _drop(storage, "a.pdf")
    doc_id = storage.doc_id(path)
    registry.add(doc_id, "a.pdf")

    IngestWorker(storage, registry, pipeline).process_next()

    assert storage.read_markdown(doc_id) == "# doc"
