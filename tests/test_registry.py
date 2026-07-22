import pytest
from core.models import IngestStatus
from ingestion.registry_db import Registry


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry.db")


def test_add_document_starts_queued(registry):
    registry.add("d1", "a.pdf")
    doc = registry.get("d1")
    assert doc.status == IngestStatus.QUEUED
    assert doc.filename == "a.pdf"


def test_next_queued_returns_documents_in_insertion_order(registry):
    registry.add("d1", "a.pdf")
    registry.add("d2", "b.pdf")
    assert registry.next_queued().doc_id == "d1"


def test_mark_done_records_chunk_count(registry):
    registry.add("d1", "a.pdf")
    registry.mark_done("d1", chunk_count=12)
    doc = registry.get("d1")
    assert doc.status == IngestStatus.DONE
    assert doc.chunk_count == 12


def test_mark_failed_records_reason(registry):
    registry.add("d1", "a.pdf")
    registry.mark_failed("d1", "unreadable")
    doc = registry.get("d1")
    assert doc.status == IngestStatus.FAILED
    assert doc.error == "unreadable"


def test_adding_same_doc_id_twice_is_idempotent(registry):
    registry.add("d1", "a.pdf")
    registry.add("d1", "a.pdf")
    assert len(registry.all()) == 1


def test_reset_stale_processing_recovers_from_crash(registry):
    registry.add("d1", "a.pdf")
    registry.mark_processing("d1")

    recovered = registry.reset_stale_processing()

    assert recovered == 1
    assert registry.get("d1").status == IngestStatus.QUEUED


def test_next_queued_returns_none_when_empty(registry):
    assert registry.next_queued() is None
