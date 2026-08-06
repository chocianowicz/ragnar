import uuid

import httpx
import streamlit as st

from generation.guards import aggregation_refusal
from generation.answerer import (
    AnswerMode, classify, NO_RESULTS_MESSAGE, citation_labels,
)
from history.chat_store import chat_title
from ui.services import build_services
from ui.panels import settings, documents, chats

st.set_page_config(page_title="RAGnar", page_icon="📚")

svc = build_services()

try:
    httpx.get(f"{svc['cfg'].ollama_url}/api/tags",
              timeout=5).raise_for_status()
except Exception:
    st.error(
        f"Cannot reach Ollama at {svc['cfg'].ollama_url}.\n\n"
        "Start it with `ollama serve`, then confirm the models are present:\n"
        "`ollama pull qwen2.5:14b` and `ollama pull bge-m3`."
    )
    if st.button("Recheck"):
        st.rerun()
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

if "messages" not in st.session_state:
    st.session_state.messages = []
# None until the current conversation has been saved for the first time.
st.session_state.setdefault("current_chat_id", None)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.caption(citation)

if question := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        outcome = svc["search"].find(
            question, doc_ids=doc_ids_filter,
            score_floor=query["floor"], use_reranker=query["use_reranker"],
        )
        mode = classify(question, outcome.refused, outcome.results)

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
            citations = citation_labels(outcome.results)
            # Build conversation history from all messages before the
            # current question (which was just appended). Strip the
            # app-only 'citations' key — the LLM doesn't need it.
            previous = st.session_state.messages[:-1]
            history = [
                {k: v for k, v in m.items() if k != "citations"}
                for m in previous
            ]
            text = st.write_stream(svc["answerer"].stream(
                question, outcome.results,
                model=query["model"], temperature=query["temperature"],
                history=history,
            ))
            with st.expander("Sources"):
                for citation in citations:
                    st.caption(citation)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "citations": citations}
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
