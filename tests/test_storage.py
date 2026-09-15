import pytest
from ingestion.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path)


def test_directories_are_created(storage, tmp_path):
    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "originals").is_dir()
    assert (tmp_path / "converted").is_dir()


def test_doc_id_is_content_hash_not_filename(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")

    assert storage.doc_id(a) == storage.doc_id(b)


def test_different_content_yields_different_doc_id(storage):
    a = storage.inbox / "a.pdf"
    b = storage.inbox / "b.pdf"
    a.write_bytes(b"one")
    b.write_bytes(b"two")

    assert storage.doc_id(a) != storage.doc_id(b)


def test_archive_moves_file_and_suffixes_with_hash(storage):
    src = storage.inbox / "report.pdf"
    src.write_bytes(b"content")
    doc_id = storage.doc_id(src)

    archived = storage.archive(src, doc_id)

    assert not src.exists()
    assert archived.exists()
    assert archived.name == f"report.{doc_id[:8]}.pdf"


def test_archiving_same_name_different_content_does_not_collide(storage):
    first = storage.inbox / "report.pdf"
    first.write_bytes(b"v1")
    a = storage.archive(first, storage.doc_id(first))

    second = storage.inbox / "report.pdf"
    second.write_bytes(b"v2")
    b = storage.archive(second, storage.doc_id(second))

    assert a != b
    assert a.exists() and b.exists()


def test_converted_artifacts_written_and_read_back(storage):
    storage.write_converted("d1", markdown="# Title")
    assert storage.read_markdown("d1") == "# Title"


def test_remove_converted_deletes_artifacts(storage):
    storage.write_converted("d1", markdown="# Title")
    storage.remove_converted("d1")
    assert storage.read_markdown("d1") is None


def test_restore_to_inbox_round_trips_an_archived_original(storage):
    src = storage.inbox / "report.pdf"
    src.write_bytes(b"the original content")
    doc_id = storage.doc_id(src)
    storage.archive(src, doc_id)
    assert not (storage.inbox / "report.pdf").exists()

    ok = storage.restore_to_inbox("report.pdf", doc_id)

    assert ok is True
    restored = storage.inbox_path(doc_id, "report.pdf")
    assert restored.exists()
    assert restored.read_bytes() == b"the original content"


def test_restore_to_inbox_returns_false_when_original_missing(storage):
    assert storage.restore_to_inbox("gone.pdf", "deadbeef" * 8) is False


def test_inbox_path_separates_same_named_files_by_content(storage):
    """The keyspace bug: the registry keys on content, the inbox used to
    key on filename, so two different `umowa.pdf` uploads shared one path
    and the second silently replaced the first's bytes."""
    id_a = storage.doc_id_for_bytes(b"contract A")
    id_b = storage.doc_id_for_bytes(b"contract B")

    path_a = storage.inbox_path(id_a, "umowa.pdf")
    path_b = storage.inbox_path(id_b, "umowa.pdf")

    assert path_a != path_b
    assert path_a.suffix == ".pdf" and path_b.suffix == ".pdf"


def test_inbox_path_is_stable_for_the_same_document(storage):
    doc_id = storage.doc_id_for_bytes(b"contract A")
    assert (storage.inbox_path(doc_id, "umowa.pdf")
            == storage.inbox_path(doc_id, "umowa.pdf"))


def test_doc_id_for_bytes_matches_doc_id_on_disk(storage, tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(b"identical content")
    assert storage.doc_id_for_bytes(b"identical content") == storage.doc_id(path)


def test_restore_to_inbox_does_not_collide_on_shared_filename(storage):
    """Re-chunking two same-named documents restored both to one path."""
    ids = []
    for content in (b"contract A", b"contract B"):
        doc_id = storage.doc_id_for_bytes(content)
        src = storage.inbox_path(doc_id, "umowa.pdf")
        src.write_bytes(content)
        storage.archive(src, doc_id, filename="umowa.pdf")
        ids.append(doc_id)

    assert storage.restore_to_inbox("umowa.pdf", ids[0]) is True
    assert storage.restore_to_inbox("umowa.pdf", ids[1]) is True

    assert storage.inbox_path(ids[0], "umowa.pdf").read_bytes() == b"contract A"
    assert storage.inbox_path(ids[1], "umowa.pdf").read_bytes() == b"contract B"
