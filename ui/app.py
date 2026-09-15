import time
from urllib.parse import quote

import httpx
import streamlit as st

from generation.guards import aggregation_refusal
from generation.answerer import (
    AnswerMode, classify, NO_RESULTS_MESSAGE, citation_labels,
    build_citations, declined,
)
from generation import followup
from generation.prompts import NO_ANSWER
from history.chat_store import chat_title
from ui.services import build_services
from ui.jobs import JobRegistry, new_chat_id
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


def render_trace(trace, key: str) -> None:
    """How the answer was found — shown because a local question takes tens
    of seconds, and because a refusal is only actionable if you can see
    whether nothing matched or everything scored just under the floor.

    Wrapped in keyed containers so CSS can render the whole thing smaller
    and dimmed: it is supporting detail, and should not compete with the
    answer for attention.
    """
    if trace is None:
        return
    with st.container(key=f"tracewrap_{key}"), \
            st.expander("How this answer was found"), \
            st.container(key=f"trace_{key}"):
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

        if trace.get("resolved_question"):
            st.caption("Searched for the follow-up as:")
            # st.text: this is model output and must not be able to inject
            # markup into the page.
            st.text(trace["resolved_question"])

        agentic = trace.get("agentic")
        if agentic:
            st.divider()
            st.markdown(
                f"**{agentic.get('llm_calls', 0)}** extra model call(s) "
                f"before the answer · pool widened to "
                f"**{agentic.get('pool_size', 0)}** passages"
            )
            if agentic.get("rewritten_query"):
                # st.text, not markdown: this is model output and must not
                # be able to inject formatting or markup into the page.
                st.caption("Searched instead for:")
                st.text(agentic["rewritten_query"])
            queries = agentic.get("queries") or []
            if len(queries) > 1:
                st.caption(f"{len(queries)} phrasings searched:")
                for q in queries:
                    st.text(f"• {q}")
            if agentic.get("hops"):
                st.caption(f"Followed up {agentic['hops']} time(s) "
                           f"after finding gaps.")
            if agentic.get("self_corrected"):
                st.caption("Draft answer judged incomplete; searched again.")
            for note in agentic.get("notes") or []:
                st.caption(f"⚠ {note}")


for turn, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            st.caption("Sources")
            render_sources(message["citations"])
        else:
            # No answer came out of the passages, so no sources — but the
            # near misses are still worth offering as somewhere to look.
            related = (message.get("trace") or {}).get("related") or []
            if related:
                with st.expander("Related documents you might check"):
                    for label in related:
                        st.caption(label)
        render_trace(message.get("trace"), key=f"h{turn}")

def answer_job(job, question, history, doc_ids_filter, query):
    """The whole answer, start to finish, on a background thread.

    Reports progress into the job and appends generated text as it
    arrives. Touches no Streamlit API: everything here can outlive the
    script run that started it, and st.* is only safe on the script's own
    thread.
    """
    search_question, resolved = question, False
    if history and query["follow_up"]:
        job.status = "Working out what the question refers to"
        search_question, resolved = followup.resolve(
            svc["llm"], question, history, model=query["model"])

    step = lambda label: setattr(job, "status", label)
    extras = any((query["rewrite"], query["multi_query"],
                  query["multi_hop"], query["self_correct"]))
    if extras:
        outcome, agentic = svc["agentic"].find(
            search_question, doc_ids=doc_ids_filter,
            score_floor=query["floor"], candidates=query["candidates"],
            use_reranker=query["use_reranker"],
            rewrite=query["rewrite"], multi_query=query["multi_query"],
            multi_hop=query["multi_hop"], self_correct=query["self_correct"],
            on_step=step,
        )
    else:
        outcome = svc["search"].find(
            search_question, doc_ids=doc_ids_filter,
            score_floor=query["floor"], candidates=query["candidates"],
            use_reranker=query["use_reranker"], on_step=step)
        agentic = None

    mode = classify(question, outcome.refused, outcome.results)
    job.mode = mode.name
    job.trace = {
        "candidates": outcome.trace.candidates,
        "reranked": outcome.trace.reranked,
        "kept": outcome.trace.kept,
        "floor": outcome.trace.floor,
        "best_score": outcome.trace.best_score,
        "hybrid": outcome.trace.hybrid,
        "seconds": outcome.trace.seconds,
        "resolved_question": search_question if resolved else None,
    }
    if agentic is not None:
        job.trace["agentic"] = {
            "rewritten_query": agentic.rewritten_query,
            "queries": agentic.queries,
            "hops": agentic.hops,
            "self_corrected": agentic.self_corrected,
            "pool_size": agentic.pool_size,
            "llm_calls": agentic.llm_calls,
            "notes": agentic.notes,
        }

    if mode is AnswerMode.NO_RESULTS:
        job.append(NO_RESULTS_MESSAGE)
        job.trace["related"] = citation_labels(outcome.related)
        return
    if mode is AnswerMode.AGGREGATION_REFUSED:
        job.append(aggregation_refusal(outcome.results))
        return

    job.citations = build_citations(outcome.results)
    job.status = "Writing the answer"

    # Hold the opening back until it is clear whether this is an answer or
    # the model declining, so the sentinel never appears on screen. It is
    # the first thing emitted when it is emitted at all, so a short buffer
    # settles it.
    buffer, deciding = "", True
    for piece in svc["answerer"].stream(
            question, outcome.results, model=query["model"],
            temperature=query["temperature"], history=history):
        if deciding:
            buffer += piece
            if declined(buffer):
                break
            if len(buffer.strip()) < len(NO_ANSWER):
                continue          # still could go either way
            deciding = False
            job.append(buffer)
            continue
        job.append(piece)

    if declined(buffer):
        # The passages looked relevant but did not answer. Not an answer,
        # so no sources: they did not produce this.
        job.chunks.clear()
        job.citations = []
        job.mode = AnswerMode.NO_RESULTS.name
        job.append(NO_RESULTS_MESSAGE)
        job.trace["model_declined"] = True
        job.trace["related"] = citation_labels(outcome.results)
    elif deciding:
        job.append(buffer)        # stream ended inside the buffer


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
    jobs.start(
        st.session_state.current_chat_id, question,
        lambda job: answer_job(job, question, history, doc_ids_filter, query),
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
        else:  # noqa: RET505
            st.caption(f"{active.status}…  ·  {active.elapsed():.0f}s")

if jobs.any_running():
    # Redraw while anything is still working — including a question left
    # running in another conversation, so its indicator stays honest and
    # its answer is filed as soon as it lands. A rerun costs a repaint;
    # the answer is on a thread that does not notice.
    time.sleep(1.0)
    st.rerun()

