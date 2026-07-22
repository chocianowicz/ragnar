import uuid
from pathlib import Path

import httpx
import streamlit as st

from core.config import Config
from ingestion.parser import DoclingParser
from ingestion.chunkers.registry import build_chunker
from ingestion.pipeline import Pipeline
from ingestion.storage import Storage
from ingestion.registry_db import Registry
from ingestion.worker import IngestWorker
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from retrieval.reranker import BGEReranker
from generation.llm import OllamaLLM
from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from generation.guards import should_refuse_aggregation, aggregation_refusal


@st.cache_resource
def build_services():
    cfg = Config()
    storage = Storage(cfg.data_dir)
    registry = Registry(cfg.data_dir / "registry.db")

    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)

    pipeline = Pipeline(DoclingParser(), build_chunker(cfg.chunking), embedder, store)
    worker = IngestWorker(storage, registry, pipeline)
    worker.start()   # resets stale PROCESSING rows on startup

    return {
        "cfg": cfg, "storage": storage, "registry": registry,
        "store": store, "search": Search(embedder, store, reranker=BGEReranker(),
                                          candidates=cfg.candidates, top_k=cfg.top_k,
                                          score_floor=cfg.score_floor),
        "llm": llm, "worker": worker,
    }


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

if not svc["registry"].all():
    st.info("No documents indexed yet. Upload one to get started.")

st.title("Ragnar")

with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader(
        "Upload", type=["pdf", "xlsx", "docx"], accept_multiple_files=True
    )
    if uploaded and st.button("Queue for ingestion"):
        for file in uploaded:
            target = svc["storage"].inbox / file.name
            target.write_bytes(file.getbuffer())
            svc["registry"].add(svc["storage"].doc_id(target), file.name)
        st.success(f"Queued {len(uploaded)} file(s)")
        st.rerun()

    if st.button("Ingest inbox"):
        queued = 0
        for path in svc["storage"].pending_files():
            svc["registry"].add(svc["storage"].doc_id(path), path.name)
            queued += 1
        st.success(f"Queued {queued} file(s)")
        st.rerun()

    @st.fragment(run_every="2s")
    def status_strip():
        counts = svc["registry"].counts()
        processing = counts.get("processing", 0)
        queued = counts.get("queued", 0)

        if processing or queued:
            st.info(f"Indexing — {processing} in progress, {queued} queued")

        for doc in svc["registry"].all():
            icon = {"queued": "⏳", "processing": "⚙️",
                    "done": "✅", "failed": "❌"}[doc.status.value]
            st.write(f"{icon} {doc.filename}")
            if doc.error:
                st.caption(f"↳ {doc.error}")

            if doc.status.value == "done":
                with st.expander(f"View {doc.filename}"):
                    markdown = svc["storage"].read_markdown(doc.doc_id)
                    st.markdown(markdown or "_Not yet converted_")

            if st.button("Remove", key=f"rm_{doc.doc_id}"):
                svc["store"].delete_by_doc(doc.doc_id)
                svc["storage"].remove_converted(doc.doc_id)
                svc["registry"].remove(doc.doc_id)
                st.rerun()

    status_strip()

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        outcome = svc["search"].find(question)

        if outcome.refused:
            text = "I could not find anything relevant in the documents."
            st.markdown(text)
            citations = []

            related_labels = []
            for r in outcome.related:
                label = r.chunk.citation_label()
                if label not in related_labels:
                    related_labels.append(label)

            if related_labels:
                with st.expander("Related documents you might check"):
                    for label in related_labels:
                        st.caption(label)
        elif should_refuse_aggregation(question, outcome.results):
            text = aggregation_refusal(outcome.results)
            st.warning(text)
            citations = []
        else:
            excerpts = [(r.chunk.citation_label(), r.chunk.text)
                        for r in outcome.results]
            text = st.write_stream(
                svc["llm"].stream(
                    SYSTEM_PROMPT, build_user_prompt(question, excerpts)
                )
            )
            citations = []
            for r in outcome.results:
                label = r.chunk.citation_label()
                if label not in citations:
                    citations.append(label)

            with st.expander("Sources"):
                for citation in citations:
                    st.caption(citation)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "citations": citations}
    )
