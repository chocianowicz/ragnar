import os
from pathlib import Path

import pytest

from ui import sources


@pytest.fixture
def static_dir(tmp_path, monkeypatch):
    target = tmp_path / "static"
    monkeypatch.setattr(sources, "STATIC", target)
    return target


def _original(tmp_path, name="umowa.pdf", content=b"%PDF-1.4 fake"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_publish_returns_a_servable_url(tmp_path, static_dir):
    url = sources.publish(_original(tmp_path), "abc123def456ghi789", "umowa.pdf")

    assert url.startswith("/app/static/")
    assert url.endswith(".pdf")          # so the server sends application/pdf
    assert (static_dir / Path(url).name).exists()


def test_publish_uses_a_hard_link_not_a_copy(tmp_path, static_dir):
    """A 20MB PDF should cost a directory entry, not 20MB."""
    original = _original(tmp_path)
    url = sources.publish(original, "abc123def456ghi789", "umowa.pdf")

    published = static_dir / Path(url).name
    assert os.stat(published).st_ino == os.stat(original).st_ino


def test_two_documents_sharing_a_filename_get_distinct_urls(tmp_path, static_dir):
    a = sources.publish(_original(tmp_path, "a.pdf", b"A"), "aaa111", "umowa.pdf")
    b = sources.publish(_original(tmp_path, "b.pdf", b"B"), "bbb222", "umowa.pdf")

    assert a != b


def test_publish_is_idempotent(tmp_path, static_dir):
    original = _original(tmp_path)
    first = sources.publish(original, "abc123", "umowa.pdf")
    second = sources.publish(original, "abc123", "umowa.pdf")

    assert first == second


def test_publish_returns_none_when_the_original_is_gone(tmp_path, static_dir):
    assert sources.publish(tmp_path / "missing.pdf", "abc123", "gone.pdf") is None


def test_unpublish_removes_the_served_file(tmp_path, static_dir):
    """Deleting a document must stop serving it, or the delete button lies."""
    url = sources.publish(_original(tmp_path), "abc123", "umowa.pdf")
    published = static_dir / Path(url).name
    assert published.exists()

    sources.unpublish("abc123", "umowa.pdf")

    assert not published.exists()


def test_unpublish_is_safe_when_nothing_was_published(static_dir):
    sources.unpublish("never", "seen.pdf")


@pytest.mark.parametrize("name,expected", [
    ("a.pdf", True), ("a.PDF", True),
    ("a.xlsx", False), ("a.docx", False), ("a", False),
])
def test_viewable_only_claims_types_a_browser_renders(name, expected):
    assert sources.viewable(name) is expected
