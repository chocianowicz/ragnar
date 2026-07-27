import uuid

import httpx
import streamlit as st

from core.config import Config
from history.chat_store import ChatStore, chat_title
from ingestion.parser import DoclingParser
from ingestion.chunkers.registry import build_chunker
from ingestion.chunkers.structural import StructuralChunker
from ingestion.pipeline import Pipeline
from ingestion.storage import Storage
from ingestion.registry_db import Registry
from ingestion.worker import IngestWorker
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from retrieval.reranker import BGEReranker
from generation.llm import OllamaLLM
from generation.guards import aggregation_refusal
from generation.answerer import (
    Answerer, AnswerMode, classify, NO_RESULTS_MESSAGE, citation_labels,
)


@st.cache_resource
def build_services():
    cfg = Config()
    storage = Storage(cfg.data_dir)
    registry = Registry(cfg.data_dir / "registry.db")
    chats = ChatStore(cfg.data_dir / "chats.db")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)

    pipeline = Pipeline(DoclingParser(), build_chunker(cfg.chunking), embedder, store)
    worker = IngestWorker(storage, registry, pipeline)
    worker.start()   # resets stale PROCESSING rows on startup

    return {
        "cfg": cfg, "storage": storage, "registry": registry,
        "chats": chats,
        "store": store, "pipeline": pipeline, "search": Search(
            embedder, store, reranker=BGEReranker(cfg.reranker_model),
            candidates=cfg.candidates, top_k=cfg.top_k,
            score_floor=cfg.score_floor),
        "llm": llm, "answerer": Answerer(llm), "worker": worker,
    }


def format_eta(seconds: float | None) -> str:
    """A rough, human-friendly ' · ~N min remaining' suffix, or '' if unknown.

    Kept deliberately coarse — the estimate is approximate, so second-level
    precision would imply accuracy it doesn't have.
    """
    if seconds is None:
        return ""
    s = int(round(seconds))
    if s < 45:
        return " · finishing up"
    mins = max(round(s / 60), 1)
    if mins < 60:
        return f" · ~{mins} min remaining"
    hrs, rem = divmod(mins, 60)
    return (f" · ~{hrs} hr {rem} min remaining" if rem
            else f" · ~{hrs} hr remaining")


@st.cache_data(ttl=30)
def list_chat_models(ollama_url: str, exclude: str) -> list[str]:
    """Chat-capable models pulled in Ollama, excluding the embedding model."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        # Exclude the embedding model and any tagged variant of it
        # ("bge-m3", "bge-m3:latest", ...) — only chat models belong here.
        base = exclude.split(":")[0]
        return sorted(
            m["name"] for m in resp.json().get("models", [])
            if not m["name"].startswith(base)
        )
    except Exception:
        return []


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

SIDEBAR_HEADER_CSS = """
<style>
[class*="st-key-remove_container_"] button:hover,
[class*="st-key-del_chat_container_"] button:hover {
    background-color: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
}
/* Make the Settings/Documents expander labels read like st.header */
div[data-testid="stExpander"] summary p {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}
</style>
"""

with st.sidebar:
    st.title("ragnar - local RAG chat app")
    st.markdown(SIDEBAR_HEADER_CSS, unsafe_allow_html=True)

    with st.expander("Settings", expanded=False):
        available_models = list_chat_models(
            svc["cfg"].ollama_url, svc["cfg"].embedding_model
        ) or [svc["cfg"].llm_model]
        default_model = (
            svc["cfg"].llm_model if svc["cfg"].llm_model in available_models
            else available_models[0]
        )
        selected_model = st.selectbox(
            "Model", options=available_models,
            index=available_models.index(default_model),
        )
        selected_temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="0 = deterministic, always the most likely answer. Higher "
                 "values allow more varied, less predictable phrasing.",
        )

        st.markdown("**Retrieval**")
        selected_floor = st.slider(
            "Similarity floor", min_value=0.0, max_value=1.0,
            value=float(svc["cfg"].score_floor), step=0.05,
            help="How relevant a document chunk must be to be used. Higher = "
                 "stricter (refuses more, safer against wrong answers); lower "
                 "= more lenient (answers more, riskier on off-topic "
                 "questions). Advanced setting — the default is tuned for "
                 "typical use.",
        )

        st.markdown("**Chunking**")
        default_tokens = svc["cfg"].chunking.get("target_tokens", 500)
        default_overlap_pct = round(
            100 * svc["cfg"].chunking.get("overlap_tokens", 50) / default_tokens
        )

        chunk_size = st.number_input(
            "Chunk size (tokens)", min_value=100, max_value=2000,
            value=default_tokens, step=50,
        )
        overlap_pct = st.number_input(
            "Overlap (%)", min_value=0, max_value=50,
            value=default_overlap_pct, step=1,
        )
        st.caption(
            "Chunk size and overlap apply to text only. Tables (PDF tables "
            "and Excel sheets) follow the separate row-based rule below — "
            "token size doesn't apply to tabular data."
        )
        table_rows = st.number_input(
            "Table rows per chunk", min_value=5, max_value=200,
            value=svc["cfg"].chunking.get("table_rows_per_group", 20), step=5,
            help="How many table/spreadsheet rows go into one chunk, "
                 "independent of the chunk size setting above. A wide table "
                 "with long cells may read better with fewer rows per chunk; "
                 "a narrow table can fit more.",
        )
        st.caption(
            "Chunking applies to documents uploaded from now on. Documents "
            "already indexed keep the chunking they were ingested with."
        )

        if st.button("Re-chunk all documents at the current size"):
            # Set the chunker first, before any doc is re-queued, so the
            # worker can't pick one up under the old settings.
            svc["pipeline"].set_chunker(StructuralChunker(
                target_tokens=chunk_size,
                overlap_tokens=round(chunk_size * overlap_pct / 100),
                rows_per_group=table_rows,
            ))
            requeued = missing = 0
            for doc in svc["registry"].all():
                if doc.status.value != "done":
                    continue
                if svc["storage"].restore_to_inbox(doc.filename, doc.doc_id):
                    svc["registry"].requeue(doc.doc_id)
                    requeued += 1
                else:
                    missing += 1
            st.success(f"Re-queued {requeued} document(s) for re-chunking")
            if missing:
                st.warning(f"{missing} skipped — original file not found")
            st.rerun()

    overlap_tokens = round(chunk_size * overlap_pct / 100)
    svc["pipeline"].set_chunker(StructuralChunker(
        target_tokens=chunk_size, overlap_tokens=overlap_tokens,
        rows_per_group=table_rows,
    ))

    with st.expander("Documents", expanded=False):
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded = st.file_uploader(
            "Upload", type=["pdf", "xlsx", "docx"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded and st.button("Upload", type="primary"):
            for file in uploaded:
                target = svc["storage"].inbox / file.name
                target.write_bytes(file.getbuffer())
                svc["registry"].add(svc["storage"].doc_id(target), file.name,
                                    target.stat().st_size)
            # Force the uploader widget to reset to empty on the next render.
            st.session_state.uploader_key += 1
            st.rerun()

        # Files placed directly in the watched folder (outside the browser
        # upload widget) have no registry entry yet - only show the manual
        # ingest option when there's actually something like that to pick up.
        externally_dropped = [
            p for p in svc["storage"].pending_files()
            if svc["registry"].get(svc["storage"].doc_id(p)) is None
        ]
        if externally_dropped:
            st.caption(
                f"{len(externally_dropped)} file(s) found in the watched "
                "folder, not added via upload"
            )
            if st.button("Ingest inbox"):
                for path in externally_dropped:
                    svc["registry"].add(svc["storage"].doc_id(path), path.name,
                                        path.stat().st_size)
                st.success(f"Queued {len(externally_dropped)} file(s)")
                st.rerun()

        @st.fragment(run_every="2s")
        def status_strip():
            processing, queued, eta = svc["registry"].ingest_eta()

            if processing or queued:
                st.info(
                    f"Indexing — {processing} in progress, "
                    f"{queued} queued{format_eta(eta)}"
                )

            docs = svc["registry"].all()

            def _select_all_changed():
                value = st.session_state.get("select_all_docs", True)
                for d in docs:
                    st.session_state[f"sel_{d.doc_id}"] = value

            if docs:
                st.checkbox("Select all", value=True, key="select_all_docs",
                            on_change=_select_all_changed)
                st.caption(
                    "Unchecked documents are excluded from answers — only "
                    "checked ones are searched."
                )

            for doc in docs:
                icon = {"queued": "⏳", "processing": "⚙️",
                        "done": "✅", "failed": "❌"}[doc.status.value]

                col_check, col_view, col_remove = st.columns([0.6, 4.4, 1])
                with col_check:
                    st.checkbox(f"Include {doc.filename}", value=True,
                                key=f"sel_{doc.doc_id}",
                                label_visibility="collapsed")
                with col_view:
                    if st.button(f"{icon} {doc.filename}", key=f"view_{doc.doc_id}",
                                 use_container_width=True):
                        show_key = f"show_md_{doc.doc_id}"
                        st.session_state[show_key] = not st.session_state.get(show_key, False)
                with col_remove:
                    with st.container(key=f"remove_container_{doc.doc_id}"):
                        if st.button("✕", key=f"rm_{doc.doc_id}",
                                     help=f"Remove {doc.filename}"):
                            svc["store"].delete_by_doc(doc.doc_id)
                            svc["storage"].remove_converted(doc.doc_id)
                            svc["registry"].remove(doc.doc_id)
                            st.rerun()

                if doc.error:
                    err_col, retry_col = st.columns([4, 1])
                    with err_col:
                        st.caption(f"↳ {doc.error}")
                    if doc.status.value == "failed":
                        with retry_col:
                            if st.button("🔄", key=f"retry_{doc.doc_id}",
                                         help=f"Retry ingesting {doc.filename}"):
                                # Failed documents are never archived - the
                                # original is still sitting in inbox, so
                                # requeuing alone is enough to retry.
                                svc["registry"].requeue(doc.doc_id)
                                st.rerun()

                if doc.status.value == "done" and st.session_state.get(f"show_md_{doc.doc_id}"):
                    markdown = svc["storage"].read_markdown(doc.doc_id)
                    st.markdown(markdown or "_Not yet converted_")

        status_strip()

    with st.expander("Chats", expanded=False):
        # Conversations auto-save as they happen; this panel is for revisiting
        # and managing them. "New chat" just clears the working conversation -
        # the previous one is already persisted, so nothing is lost.
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()

        saved_chats = svc["chats"].all()
        if not saved_chats:
            st.caption("No saved chats yet — ask a question to start one.")

        for chat in saved_chats:
            is_current = chat.chat_id == st.session_state.get("current_chat_id")
            col_open, col_del = st.columns([5, 1])
            with col_open:
                label = ("▸ " if is_current else "") + chat.title
                if st.button(label, key=f"open_chat_{chat.chat_id}",
                             use_container_width=True):
                    st.session_state.messages = chat.messages
                    st.session_state.current_chat_id = chat.chat_id
                    st.rerun()
            with col_del:
                with st.container(key=f"del_chat_container_{chat.chat_id}"):
                    if st.button("✕", key=f"del_chat_{chat.chat_id}",
                                 help="Delete this chat"):
                        svc["chats"].delete(chat.chat_id)
                        if is_current:
                            st.session_state.messages = []
                            st.session_state.current_chat_id = None
                        st.rerun()

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
    svc["llm"].model = selected_model
    svc["llm"].temperature = selected_temperature
    svc["search"].score_floor = selected_floor

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        outcome = svc["search"].find(question, doc_ids=doc_ids_filter)
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
            text = st.write_stream(
                svc["answerer"].stream(question, outcome.results)
            )
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
