from pathlib import Path
from typing import Protocol
from core.models import Chunk


class ParsedBlock(Protocol):
    text: str
    page: int | None
    is_table: bool


class DocumentParser(Protocol):
    def parse(self, path: Path): ...


class Chunker(Protocol):
    def chunk(self, parsed, doc_id: str, filename: str) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], limit: int) -> list: ...
    def delete_by_doc(self, doc_id: str) -> None: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk],
               top_k: int) -> list: ...
