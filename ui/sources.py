"""Making an original file reachable by URL, so a citation can open it.

Streamlit serves files from a `static` folder beside the entrypoint, and
only from there. Originals live in data/originals, so this publishes a
hard link into ui/static — a link, not a copy, so a 20MB PDF costs a
directory entry rather than 20MB. Both paths sit on the same filesystem
(the repo is one bind mount), and there is a copy fallback for the case
where they do not.

Why bother, rather than showing the converted text: the converted text is
what the page number indexes, but it is not what the reader wants to see
when checking a quote in a contract. A browser opening the real PDF at
`#page=4` puts them in front of the actual document, formatting intact.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# Types a browser will render rather than download. Anything else is still
# published — the browser downloads it, which is the correct outcome for a
# spreadsheet — but only these get a "view" affordance.
VIEWABLE = {".pdf"}


def _published_name(doc_id: str, filename: str) -> str:
    """Named by doc_id, not by the uploaded filename.

    Two documents can share a name, and the URL has to distinguish them for
    the same reason the inbox does. The extension is kept so the server
    sends a usable content type.
    """
    return f"{doc_id[:16]}{Path(filename).suffix.lower()}"


def publish(original: Path, doc_id: str, filename: str) -> str | None:
    """Make `original` reachable, returning its URL path or None.

    Idempotent: an already-published file is left alone. Returns None if
    the original is missing or cannot be published, so the caller can fall
    back to offering a download instead of rendering a dead link.
    """
    if not original.exists():
        return None

    STATIC.mkdir(parents=True, exist_ok=True)
    target = STATIC / _published_name(doc_id, filename)

    if not target.exists():
        try:
            os.link(original, target)
        except OSError:
            # Different filesystem, or a platform without hard links.
            try:
                shutil.copy2(original, target)
            except OSError as exc:
                log.warning("could not publish %s: %s", filename, exc)
                return None

    return f"/app/static/{target.name}"


def unpublish(doc_id: str, filename: str) -> None:
    """Drop the published link when its document is removed.

    Without this, deleting a document from the app would leave the file
    still downloadable by anyone who knows the URL — the opposite of what
    the delete button promises.
    """
    (STATIC / _published_name(doc_id, filename)).unlink(missing_ok=True)


def viewable(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIEWABLE
