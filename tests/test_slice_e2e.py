import pytest
import uuid
from pathlib import Path


@pytest.mark.integration
def test_pdf_ingested_then_answered_with_correct_citation():
    import os
    from core.config import Config
    from ingestion.parser import DoclingParser
    from ingestion.chunkers.registry import build_chunker
    from ingestion.pipeline import Pipeline
    from retrieval.embedder import OllamaEmbedder
    from retrieval.store import QdrantStore
    from retrieval.search import Search
    from generation.llm import OllamaLLM
    from generation.answerer import Answerer

    cfg = Config()
    collection = f"e2e_{uuid.uuid4().hex[:8]}"
    embedder = OllamaEmbedder(os.environ["OLLAMA_BASE_URL"], "bge-m3")
    store = QdrantStore(os.environ["QDRANT_URL"], collection, dim=1024)
    store.ensure_collection()

    try:
        pipeline = Pipeline(DoclingParser(), build_chunker(cfg.chunking), embedder, store)
        result = pipeline.ingest(Path("tests/fixtures/sample.pdf"), "doc1")
        assert result.chunk_count > 0

        search = Search(embedder, store, candidates=25)
        results = search.find("What is the service contract number?")
        assert results

        llm = OllamaLLM(os.environ["OLLAMA_BASE_URL"], "qwen2.5:14b")
        answer = Answerer(llm).answer(
            "What is the service contract number?", results[:5]
        )

        assert "SC-4471" in answer.text
        assert any("sample.pdf" in c for c in answer.citations)
    finally:
        store.drop_collection()
