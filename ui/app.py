import time

import httpx
import streamlit as st

from generation.answering import Settings, answer as run_answer
from history.chat_store import chat_title
from ui.services import build_services
from ui.jobs import JobRegistry, new_chat_id
from ui.chat import render_transcript
from ui.reader import render_document_page
from ui import sources
from ui.panels import settings, documents, chats

st.set_page_config(page_title="RAGnar", page_icon="📚")

svc = build_services()


@st.cache_resource
def job_registry() -> JobRegistry:
    """Shared across sessions on purpose: an answer started in one tab has
    to be visible to the tab that reloads, and to a second tab opened while
    it runs."""
    return JobRegistry()


jobs = job_registry()


def commit(job) -> None:
    """Move a finished job into its conversation and persist it.

    Goes through the store rather than session state, because the job's
    chat may not be the one on screen — you can ask a question, wander off
    to another conversation, and the answer still has to land in the chat
    it belongs to.
    """
    saved = svc["chats"].get(job.chat_id)
    messages = list(saved.messages) if saved else []
    messages.append(
        {"role": "assistant",
         "content": job.text or (f"Something went wrong: {job.error}"
                                 if job.error else ""),
         "citations": job.citations, "trace": job.trace}
    )
    svc["chats"].save(job.chat_id, chat_title(messages), messages)
    if st.session_state.current_chat_id == job.chat_id:
        st.session_state.messages = messages


def reap_finished_jobs() -> None:
    """Land any answer that finished while its chat was not on screen.

    Without this, walking away from a question means the answer completes
    on its thread and is never written anywhere — the work is done and
    silently thrown away, which is the failure this whole mechanism exists
    to prevent.
    """
    for job in jobs.finished():
        commit(job)
        jobs.pop(job.chat_id)

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


if "doc" in st.query_params:
    render_document_page(svc, st.query_params["doc"],
                         st.query_params.get("page"))
    st.stop()

APP_CSS = """
<style>
[class*="st-key-remove_container_"] button:hover,
[class*="st-key-del_chat_container_"] button:hover {
    background-color: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
}
/* Make the Settings/Documents/Chats expander labels read like st.header.
   Scoped to the sidebar: the chat also uses expanders now, for citations
   and the retrieval trace, and those are asides rather than headings. */
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}

/* The retrieval trace is supporting detail, not part of the answer.
   Smaller and dimmed so the eye separates the two without having to read
   them. Opacity rather than a fixed grey, so it holds in both themes. */
[class*="st-key-trace_"] {
    opacity: 0.7;
}
[class*="st-key-trace_"] p,
[class*="st-key-trace_"] li,
[class*="st-key-trace_"] pre,
[class*="st-key-trace_"] code,
[class*="st-key-trace_"] div[data-testid="stMarkdownContainer"] {
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}
/* st.caption is already small; keep it a touch smaller still so the two
   levels inside the trace stay distinguishable. */
[class*="st-key-trace_"] div[data-testid="stCaptionContainer"] p {
    font-size: 0.74rem !important;
}
/* The trace's own summary line sits between the two: clearly a label,
   clearly not a heading. */
[class*="st-key-tracewrap_"] div[data-testid="stExpander"] summary p {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    opacity: 0.8;
}
</style>
"""

if "messages" not in st.session_state:
    st.session_state.messages = []
# None until the current conversation has been saved for the first time.
st.session_state.setdefault("current_chat_id", None)

# Land any answer that finished while this chat was not on screen —
# before the transcript is drawn, so it appears in this pass rather than
# sitting invisible until the next thing the user clicks.
reap_finished_jobs()

with st.sidebar:
    st.title("RAGnar - local RAG chat app")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    query = settings.render(svc)
    documents.render(svc)
    chats.render(svc, jobs)

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


render_transcript(st.session_state.messages)

if question := st.chat_input("Ask about your documents",
                             disabled=jobs.running(
                                 st.session_state.current_chat_id or "")):
    # Mint the id before starting, so the job is filed under the
    # conversation it belongs to even on its first question.
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = new_chat_id()

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    # Persist the question before the answer exists, so the conversation
    # shows up in the Chats panel straight away. Otherwise walking away
    # from a running question leaves it findable by nothing.
    svc["chats"].save(st.session_state.current_chat_id,
                      chat_title(st.session_state.messages),
                      st.session_state.messages)
    settings = Settings(
        model=query["model"], temperature=query["temperature"],
        floor=query["floor"], candidates=query["candidates"],
        use_reranker=query["use_reranker"], follow_up=query["follow_up"],
        rewrite=query["rewrite"], multi_query=query["multi_query"],
        multi_hop=query["multi_hop"], self_correct=query["self_correct"])

    jobs.start(
        st.session_state.current_chat_id,
        lambda job: run_answer(
            job, question, history=history, doc_ids=doc_ids_filter,
            settings=settings, search=svc["search"],
            answerer=svc["answerer"], agentic=svc["agentic"], llm=svc["llm"],
            publish=lambda doc_id, filename: sources.publish(
                svc["storage"].archived_path(filename, doc_id),
                doc_id, filename),
        ),
    )
    st.rerun()


active = jobs.get(st.session_state.current_chat_id or "")
if active is not None:
    with st.chat_message("assistant"):
        partial = active.text
        if partial:
            st.markdown(partial)
        related = active.trace.get("related") or []
        if related and not active.citations:
            with st.expander("Related documents you might check"):
                for label in related:
                    st.caption(label)
        if active.done:
            commit(active)
            jobs.pop(active.chat_id)
            st.rerun()
        else:
            st.caption(f"{active.status}…  ·  {active.elapsed():.0f}s")

if jobs.any_running():
    # Redraw while anything is still working — including a question left
    # running in another conversation, so its indicator stays honest and
    # its answer is filed as soon as it lands. A rerun costs a repaint;
    # the answer is on a thread that does not notice.
    time.sleep(1.0)
    st.rerun()

