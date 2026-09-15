import uuid
from urllib.parse import quote

import httpx
import streamlit as st

from generation.guards import aggregation_refusal
from generation.answerer import (
    AnswerMode, classify, NO_RESULTS_MESSAGE, citation_labels,
    build_citations,
)
from history.chat_store import chat_title
from ui.services import build_services
from ui import sources
from ui.panels import settings, documents, chats

st.set_page_config(page_title="RAGnar", page_icon="📚")

svc = build_services()

try:
    httpx.get(f"{svc['cfg'].ollama_url}/api/tags", timeout=5).raise_for_status()
except Exception:
    st.error(
        f"Cannot reach Ollama at {svc['cfg'].ollama_url}.\n\n"
        "Start it with `ollama serve`, then confirm the models are present:\n"
        "`ollama pull qwen2.5:14b` and `ollama pull bge-m3`."
    )
    if st.button("Recheck"):
        st.rerun()
    st.stop()


def render_document_page(doc_id: str, page: str | None) -> None:
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


if "doc" in st.query_params:
    render_document_page(st.query_params["doc"],
                         st.query_params.get("page"))
    st.stop()

SIDEBAR_HEADER_CSS = """
<style>
[class*="st-key-remove_container_"] button:hover,
[class*="st-key-del_chat_container_"] button:hover {
    background-color: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
}
/* Make the Settings/Documents/Chats expander labels read like st.header */
div[data-testid="stExpander"] summary p {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
</style>
"""

with st.sidebar:
    st.title("RAGnar - local RAG chat app")
    st.markdown(SIDEBAR_HEADER_CSS, unsafe_allow_html=True)

    query = settings.render(svc)
    documents.render(svc)
    chats.render(svc)

all_docs = svc["registry"].all()
if not all_docs:
    st.info("No documents indexed yet. Upload one to get started.")

selected_doc_ids = [
    d.doc_id for d in all_docs
    if st.session_state.get(f"sel_{d.doc_id}", True)
]
# None = unfiltered search (identical to today's behavior) whenever
# everything happens to be selected; only pass an explicit filter — which
# may be an empty list, correctly refusing — for a genuine subset.
doc_ids_filter = (
    None if set(selected_doc_ids) == {d.doc_id for d in all_docs}
    else selected_doc_ids
)


def render_sources(citations: list) -> None:
    """Show each cited passage, with a way into the document it came from.

    Citations saved before this existed are plain strings; render those as
    the captions they used to be rather than dropping older conversations.
    """
    for citation in citations:
        if not isinstance(citation, dict):
            st.caption(str(citation))
            continue

        label = citation.get("label", "source")
        score = citation.get("score")
        heading = f"{label}" + (f"  ·  {score:.2f}" if score is not None else "")
        with st.expander(heading):
            # The passage the model was actually given, not a fresh lookup:
            # this is what the answer was built from.
            st.markdown(
                f"> {citation.get('text', '').strip()[:1500]}"
                .replace("\n", "\n> ")
            )
            doc_id = citation.get("doc_id")
            if doc_id:
                # Links, not buttons: st.link_button opens a new tab, so
                # checking a source does not cost you your place in the
                # chat.
                filename = citation.get("filename", "")
                page = citation.get("page")
                url = f"?doc={quote(doc_id)}"
                if page:
                    url += f"&page={quote(str(page))}"

                published = sources.publish(
                    svc["storage"].archived_path(filename, doc_id),
                    doc_id, filename,
                ) if filename else None

                cols = st.columns(2)
                with cols[0]:
                    if published and sources.viewable(filename):
                        # #page= is understood by the PDF viewers built into
                        # every current browser, so this lands on the cited
                        # page of the real document rather than near it.
                        anchor = f"{published}#page={page}" if page else published
                        st.link_button("Open the original ↗", anchor,
                                       help="The real file, at the cited page")
                    elif published:
                        st.link_button("Open the original ↗", published,
                                       help="Downloads in a new tab")
                with cols[1]:
                    st.link_button("Converted text ↗", url,
                                   help="What the page number indexes")


def render_trace(trace) -> None:
    """How the answer was found — shown because a local question takes tens
    of seconds, and because a refusal is only actionable if you can see
    whether nothing matched or everything scored just under the floor."""
    if trace is None:
        return
    with st.expander("How this answer was found"):
        bits = [
            f"**{trace.get('candidates', 0)}** passages retrieved"
            + (" by meaning and wording" if trace.get("hybrid") else ""),
        ]
        if trace.get("reranked"):
            bits.append(f"**{trace['reranked']}** re-checked by the ranker")
        bits.append(
            f"**{trace.get('kept', 0)}** cleared the relevance floor "
            f"({trace.get('floor', 0):.2f})"
        )
        if trace.get("best_score") is not None:
            bits.append(f"best score **{trace['best_score']:.3f}**")
        st.markdown(" · ".join(bits))
        seconds = trace.get("seconds") or {}
        if seconds:
            st.caption("  ".join(f"{k} {v:.1f}s" for k, v in seconds.items()))


if "messages" not in st.session_state:
    st.session_state.messages = []
# None until the current conversation has been saved for the first time.
st.session_state.setdefault("current_chat_id", None)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            st.caption("Sources")
            render_sources(message["citations"])
        render_trace(message.get("trace"))

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Retrieval alone runs to tens of seconds locally. Narrate it, so
        # the wait is legible instead of a bare spinner.
        with st.status("Searching your documents", expanded=True) as status:
            outcome = svc["search"].find(
                question, doc_ids=doc_ids_filter,
                score_floor=query["floor"], candidates=query["candidates"],
                use_reranker=query["use_reranker"],
                on_step=lambda label: status.update(label=label),
            )
            status.update(label="Writing the answer", state="complete",
                          expanded=False)

        mode = classify(question, outcome.refused, outcome.results)
        trace = {
            "candidates": outcome.trace.candidates,
            "reranked": outcome.trace.reranked,
            "kept": outcome.trace.kept,
            "floor": outcome.trace.floor,
            "best_score": outcome.trace.best_score,
            "hybrid": outcome.trace.hybrid,
            "seconds": outcome.trace.seconds,
        }

        if mode is AnswerMode.NO_RESULTS:
            text = NO_RESULTS_MESSAGE
            st.markdown(text)
            citations = []

            related_labels = citation_labels(outcome.related)
            if related_labels:
                with st.expander("Related documents you might check"):
                    for label in related_labels:
                        st.caption(label)
        elif mode is AnswerMode.AGGREGATION_REFUSED:
            text = aggregation_refusal(outcome.results)
            st.warning(text)
            citations = []
        else:
            citations = build_citations(outcome.results)
            text = st.write_stream(svc["answerer"].stream(
                question, outcome.results,
                model=query["model"], temperature=query["temperature"],
            ))
            st.caption("Sources")
            render_sources(citations)

        render_trace(trace)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "citations": citations,
         "trace": trace}
    )

    # Persist the conversation. Mint an id on first save so a chat only
    # appears in the list once it actually has content. Rerun so the newly
    # saved/updated chat shows in the Chats panel immediately (the sidebar
    # renders above this handler, so it hasn't seen the save yet this run).
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = uuid.uuid4().hex
    svc["chats"].save(
        st.session_state.current_chat_id,
        chat_title(st.session_state.messages),
        st.session_state.messages,
    )
    st.rerun()
