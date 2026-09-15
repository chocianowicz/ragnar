"""The standalone document page a citation opens in a new tab."""
import streamlit as st

from ui import sources


def render_document_page(svc, doc_id: str, page: str | None) -> None:
    """A standalone reader for one document, opened from a citation.

    This is its own page rather than an expanding panel so a citation can
    be opened in a new tab and kept beside the conversation — checking a
    source should not cost you your place in the chat.

    The converted text is shown rather than the original file because that
    is what the citation refers to: page and sheet provenance is recorded
    during conversion. The original is offered as a download, since the
    app runs in a container with no access to the browser's file handlers
    and cannot serve the raw bytes as a URL without static file serving
    turned on.
    """
    doc = svc["registry"].get(doc_id)
    if doc is None:
        st.error("That document is no longer indexed.")
        return

    st.title(doc.filename)
    if page:
        st.caption(f"Cited from page {page} — use your browser's find "
                   f"(⌘F / Ctrl-F) to jump to the passage.")

    original = svc["storage"].archived_path(doc.filename, doc.doc_id)
    published = sources.publish(original, doc.doc_id, doc.filename)
    if published and sources.viewable(doc.filename):
        anchor = f"{published}#page={page}" if page else published
        st.link_button("Open the original file ↗", anchor)
    elif original.exists():
        st.download_button("⬇ Download the original file",
                           data=original.read_bytes(),
                           file_name=doc.filename, key="reader_download")
    else:
        st.caption("Original file not found — showing the converted text only.")

    st.divider()
    st.markdown(svc["storage"].read_markdown(doc_id) or "_Not yet converted_")
