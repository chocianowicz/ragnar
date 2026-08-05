"""Service construction and shared UI helpers.

Kept apart from the panels and the page script so `build_services` (the
single cached wiring point) and the small formatting helpers can be reused
without importing Streamlit page logic.
"""
import httpx
import streamlit as st

from core.config import Config
from ingestion.parser import DoclingParser
from ingestion.chunkers.registry import build_chunker
from ingestion.pipeline import Pipeline
from ingestion.storage import Storage
from ingestion.registry_db import Registry
from ingestion.worker import IngestWorker
from history.chat_store import ChatStore
from retrieval.embedder import OllamaEmbedder
from retrieval.store import QdrantStore
from retrieval.search import Search
from retrieval.reranker import BGEReranker
from generation.llm import OllamaLLM
from generation.answerer import Answerer


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

    pipeline = Pipeline(
        DoclingParser(), build_chunker(cfg.chunking, embedder=embedder),
        embedder, store)
    worker = IngestWorker(storage, registry, pipeline)
    worker.start()   # resets stale PROCESSING rows on startup

    return {
        "cfg": cfg, "storage": storage, "registry": registry, "chats": chats,
        "store": store, "pipeline": pipeline, "embedder": embedder,
        "search": Search(
            embedder, store, reranker=BGEReranker(cfg.reranker_model),
            candidates=cfg.candidates, top_k=cfg.top_k,
            score_floor=cfg.score_floor),
        "answerer": Answerer(llm), "worker": worker,
    }


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
