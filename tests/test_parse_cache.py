import json

import pytest

from core.models import IngestStatus
from ingestion.chunkers.fixed import FixedChunker
from ingestion.parser import Block, ParsedDocument
from ingestion.pipeline import Pipeline
from ingestion.registry_db import Registry
from ingestion.storage import Storage
from ingestion.worker import IngestWorker
from tests.fakes import FakeEmbedder, FakeStore


def test_a_cached_parse_round_trips_every_block_field(tmp_path):
    """Block fields carry page numbers and table/summary flags into
    citations, so losing one on the way through the cache would silently
    change what an answer says it was based on."""
    storage = Storage(tmp_path)
    parsed = ParsedDocument(
        markdown="# doc",
        blocks=[
            Block(text="prose", page=1),
            Block(text="| a | b |", page=2, is_table=True),
            Block(text="total = 3", page=2, is_summary=True),
            Block(text="| x | y |", page=None, is_table=True, sheet="Q1"),
        ],
        page_count=2,
        low_confidence=True,
    )
    storage.write_parsed("abc", parsed)

    cached = storage.read_parsed("abc")

    assert cached.blocks == parsed.blocks
    assert cached.low_confidence is True
    # Markdown is deliberately not duplicated: it is already in the .md file
    # that write_converted wrote under the same doc_id.
    assert cached.markdown == ""


def test_no_cache_yet_returns_none(tmp_path):
    assert Storage(tmp_path).read_parsed("missing") is None


def test_an_unreadable_cache_file_is_a_miss_not_a_crash(tmp_path):
    """A truncated write from an interrupted ingest must not make the
    document permanently un-ingestable."""
    storage = Storage(tmp_path)
    (storage.converted / "abc.blocks.json").write_text("{not json")

    assert storage.read_parsed("abc") is None


def test_an_older_cache_format_is_a_miss_not_a_crash(tmp_path):
    """The shape has changed across versions. An old file must be ignored and
    re-parsed, never trusted into an index that then cites nothing."""
    storage = Storage(tmp_path)
    (storage.converted / "abc.blocks.json").write_text(
        json.dumps({"blocks": [{"text": "t"}]}))          # no low_confidence

    assert storage.read_parsed("abc") is None


def test_an_unknown_block_field_is_a_miss_not_a_crash(tmp_path):
    storage = Storage(tmp_path)
    (storage.converted / "abc.blocks.json").write_text(json.dumps({
        "low_confidence": False,
        "blocks": [{"text": "t", "removed_field": 1}],
    }))

    assert storage.read_parsed("abc") is None


def test_removing_a_document_removes_its_cached_parse(tmp_path):
    storage = Storage(tmp_path)
    storage.write_converted("abc", "# doc")
    storage.write_parsed("abc", ParsedDocument(markdown="# doc",
                                               page_count=1))

    storage.remove_converted("abc")

    assert storage.read_markdown("abc") is None
    assert storage.read_parsed("abc") is None

# ── Through the worker: a miss costs time, a wrong hit costs correctness ─────

class CountingParser:
    """Parses for real, but counts and can change its answer, so a test can
    tell a fresh parse from a cached one."""

    def __init__(self, text="first parse"):
        self.calls = 0
        self.text = text

    def parse(self, path):
        self.calls += 1
        return ParsedDocument(markdown=f"# {self.text}",
                              blocks=[Block(text=self.text, page=1)],
                              page_count=1)


@pytest.fixture
def env(tmp_path):
    storage = Storage(tmp_path)
    registry = Registry(tmp_path / "registry.db")
    store = FakeStore()
    parser = CountingParser()
    pipeline = Pipeline(parser, FixedChunker(target_chars=100),
                        FakeEmbedder(), store)
    return storage, registry, pipeline, store, parser


def _queue(storage, registry, name="a.pdf", content=b"data"):
    doc_id = storage.doc_id_for_bytes(content)
    storage.inbox_path(doc_id, name).write_bytes(content)
    registry.add(doc_id, name)
    return doc_id


def test_the_first_ingest_writes_a_cache(env):
    storage, registry, pipeline, _, _ = env
    doc_id = _queue(storage, registry)

    IngestWorker(storage, registry, pipeline).process_next()

    assert storage.read_parsed(doc_id) is not None


def test_re_ingesting_the_same_content_does_not_parse_again(env):
    """The point of the cache: a chunking change replays without Docling."""
    storage, registry, pipeline, _, parser = env
    doc_id = _queue(storage, registry)
    worker = IngestWorker(storage, registry, pipeline)

    worker.process_next()
    assert parser.calls == 1

    # Re-queue the same content, which is what a re-chunk does.
    storage.restore_to_inbox("a.pdf", doc_id)
    registry.requeue(doc_id)
    worker.process_next()

    assert parser.calls == 1, "the cached parse was not used"


def test_a_cache_hit_reindexes_from_the_cached_blocks(env):
    """A cache hit must not be mistaken for 'nothing to do': the chunks are
    rebuilt and the index replaced, which is the whole reason to re-ingest."""
    storage, registry, pipeline, store, parser = env
    doc_id = _queue(storage, registry)
    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()
    first = [c.text for c in store.chunks]

    # The parser's answer has changed: if the cache is used, the indexed text
    # must still be the cached one.
    parser.text = "second parse"
    storage.restore_to_inbox("a.pdf", doc_id)
    registry.requeue(doc_id)
    worker.process_next()

    assert [c.text for c in store.chunks] == first
    assert "first parse" in store.chunks[0].text


def test_a_cache_hit_does_not_blank_the_converted_markdown(env):
    """read_parsed returns empty markdown on purpose, so writing its result
    back through write_converted would erase the human-readable file."""
    storage, registry, pipeline, _, _ = env
    doc_id = _queue(storage, registry)
    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()

    storage.restore_to_inbox("a.pdf", doc_id)
    registry.requeue(doc_id)
    worker.process_next()

    assert storage.read_markdown(doc_id) == "# first parse"


def test_different_content_is_parsed_separately(env):
    """The cache is keyed by content hash, so two files are two parses."""
    storage, registry, pipeline, _, parser = env
    worker = IngestWorker(storage, registry, pipeline)

    for content in (b"one", b"two"):
        _queue(storage, registry, content=content)
        worker.process_next()

    assert parser.calls == 2


def test_a_corrupt_cache_falls_back_to_parsing(env):
    """A truncated file must cost a re-parse, not a failed document."""
    storage, registry, pipeline, _, parser = env
    doc_id = _queue(storage, registry)
    worker = IngestWorker(storage, registry, pipeline)
    worker.process_next()

    (storage.converted / f"{doc_id}.blocks.json").write_text("{broken")
    storage.restore_to_inbox("a.pdf", doc_id)
    registry.requeue(doc_id)
    worker.process_next()

    assert parser.calls == 2
    assert registry.get(doc_id).status == IngestStatus.DONE


def test_ingest_reports_the_parse_it_used(env):
    """The caller writes the cache from ingest()'s result, so the parsed
    document has to come back out."""
    storage, registry, pipeline, _, _ = env
    doc_id = _queue(storage, registry)
    path = storage.inbox_path(doc_id, "a.pdf")

    result = pipeline.ingest(path, doc_id, filename="a.pdf")

    assert result.parsed is not None
    assert result.parsed.blocks[0].text == "first parse"


def test_ingest_skips_the_parser_when_handed_a_parse(env):
    storage, registry, pipeline, _, parser = env
    doc_id = _queue(storage, registry)
    path = storage.inbox_path(doc_id, "a.pdf")
    given = ParsedDocument(markdown="# given",
                           blocks=[Block(text="given", page=1)],
                           page_count=1)

    result = pipeline.ingest(path, doc_id, filename="a.pdf", parsed=given)

    assert parser.calls == 0
    assert result.parsed is given

