import pytest
from core.models import IngestStatus
from ingestion.registry_db import Registry, estimate_eta


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


def test_requeue_puts_a_done_document_back_to_queued(registry):
    registry.add("d1", "a.pdf")
    registry.mark_done("d1", chunk_count=5)

    registry.requeue("d1")

    assert registry.get("d1").status == IngestStatus.QUEUED


def test_requeue_puts_a_failed_document_back_to_queued(registry):
    registry.add("d1", "a.pdf")
    registry.mark_failed("d1", "unreadable")

    registry.requeue("d1")

    doc = registry.get("d1")
    assert doc.status == IngestStatus.QUEUED
    assert doc.error is None


# --- ETA estimation ----------------------------------------------------------

def test_estimate_eta_none_when_nothing_remaining():
    assert estimate_eta(done=[(1000, 5.0)], remaining_bytes=0,
                        remaining_count=0, current_elapsed=0.0) is None


def test_estimate_eta_none_before_any_document_is_timed():
    # Two queued docs but no completed timings yet -> not estimable.
    assert estimate_eta(done=[], remaining_bytes=2000,
                        remaining_count=2, current_elapsed=0.0) is None


def test_estimate_eta_uses_bytes_per_second_throughput():
    # 1000 bytes took 10s -> 100 B/s. 3000 bytes remain -> 30s.
    eta = estimate_eta(done=[(1000, 10.0)], remaining_bytes=3000,
                       remaining_count=2, current_elapsed=0.0)
    assert eta == pytest.approx(30.0)


def test_estimate_eta_falls_back_to_per_document_average_without_sizes():
    # No byte sizes known; two docs averaged 6s each, three remain -> 18s.
    eta = estimate_eta(done=[(0, 4.0), (0, 8.0)], remaining_bytes=0,
                       remaining_count=3, current_elapsed=0.0)
    assert eta == pytest.approx(18.0)


def test_estimate_eta_discounts_time_already_spent_and_floors_at_zero():
    eta = estimate_eta(done=[(1000, 10.0)], remaining_bytes=1000,
                       remaining_count=1, current_elapsed=4.0)
    assert eta == pytest.approx(6.0)

    # Already over the estimate -> clamps to 0, never negative.
    over = estimate_eta(done=[(1000, 10.0)], remaining_bytes=1000,
                        remaining_count=1, current_elapsed=99.0)
    assert over == 0.0


def test_ingest_eta_reports_counts_and_survives_untimed_history(registry):
    registry.add("d1", "a.pdf", size_bytes=100)
    registry.add("d2", "b.pdf", size_bytes=100)
    registry.mark_done("d1", chunk_count=3)  # done but ~0s duration

    processing, queued, eta = registry.ingest_eta()

    assert (processing, queued) == (0, 1)
    # A ~0s duration yields no usable rate -> None rather than a bogus 0.
    assert eta is None


def test_registry_migrates_a_pre_timing_database(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    # Simulate a database created before bytes/started_at/finished_at existed.
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, filename TEXT NOT "
        "NULL, status TEXT NOT NULL, error TEXT, chunk_count INTEGER NOT NULL "
        "DEFAULT 0, added_at REAL NOT NULL DEFAULT (julianday('now')))"
    )
    conn.execute(
        "INSERT INTO documents (doc_id, filename, status) VALUES "
        "('old1', 'legacy.pdf', 'done')"
    )
    conn.commit()
    conn.close()

    reg = Registry(path)  # must not raise; adds the missing columns
    assert reg.get("old1").status == IngestStatus.DONE
    # New columns are usable on the migrated row.
    reg.mark_processing("old1")
    assert reg.get("old1").status == IngestStatus.PROCESSING
    processing, queued, eta = reg.ingest_eta()
    assert processing == 1
