from dataclasses import dataclass
from enum import Enum


class IngestStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Chunk:
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    page: int | None = None
    sheet: str | None = None
    is_table: bool = False
    low_confidence: bool = False
    is_summary: bool = False

    def citation_label(self) -> str:
        if self.page is not None:
            return f"{self.filename}, p. {self.page}"
        if self.sheet is not None:
            return f"{self.filename}, sheet {self.sheet}"
        return self.filename


@dataclass
class Document:
    doc_id: str
    filename: str
    status: IngestStatus = IngestStatus.QUEUED
    error: str | None = None
    chunk_count: int = 0


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
