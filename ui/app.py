import uuid
from pathlib import Path

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
from generation.llm import OllamaLLM
from generation.prompts import SYSTEM_PROMPT, build_user_prompt


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
        "store": store, "search": Search(embedder, store, cfg.candidates),
        "llm": llm, "worker": worker,
    }


svc = build_services()
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
        results = svc["search"].find(question)[: svc["cfg"].top_k]

        if not results:
            text = "I could not find anything relevant in the documents."
            st.markdown(text)
            citations = []
        else:
            excerpts = [(r.chunk.citation_label(), r.chunk.text)
                        for r in results]
            text = st.write_stream(
                svc["llm"].stream(
                    SYSTEM_PROMPT, build_user_prompt(question, excerpts)
                )
            )
            citations = []
            for r in results:
                label = r.chunk.citation_label()
                if label not in citations:
                    citations.append(label)

            with st.expander("Sources"):
                for citation in citations:
                    st.caption(citation)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "citations": citations}
    )
