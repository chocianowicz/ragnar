import uuid
from pathlib import Path

import streamlit as st

from core.config import Config
from ingestion.parser import DoclingParser
from ingestion.chunkers.fixed import FixedChunker
from ingestion.pipeline import Pipeline
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from generation.llm import OllamaLLM
from generation.answerer import Answerer
from generation.prompts import SYSTEM_PROMPT, build_user_prompt


@st.cache_resource
def build_services():
    cfg = Config()
    embedder = OllamaEmbedder(cfg.ollama_url, cfg.embedding_model)
    store = QdrantStore(cfg.qdrant_url, cfg.collection, cfg.embedding_dim)
    store.ensure_collection()
    llm = OllamaLLM(cfg.ollama_url, cfg.llm_model)
    return {
        "cfg": cfg,
        "pipeline": Pipeline(DoclingParser(), FixedChunker(),
                             embedder, store),
        "search": Search(embedder, store, cfg.candidates),
        "answerer": Answerer(llm),
        "llm": llm,
    }


svc = build_services()
st.title("Ragnar")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest"):
        inbox = svc["cfg"].data_dir / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        target = inbox / uploaded.name
        target.write_bytes(uploaded.getbuffer())

        with st.spinner(f"Ingesting {uploaded.name}…"):
            n = svc["pipeline"].ingest(target, uuid.uuid4().hex)
        st.success(f"Indexed {n} chunks")

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
