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

    def key(self) -> str:
        """Identity of this chunk within the corpus.

        The store derives its point id from this, and the agentic layer
        deduplicates on it. One definition, so those two can never
        disagree about what "the same chunk" means.
        """
        return f"{self.doc_id}:{self.chunk_index}"

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
    folder_id: str | None = None   # None = Unfiled


@dataclass
class Folder:
    """A named group of documents.

    Identity is folder_id, not name: the name is a label the user renames
    freely, and a saved chat scope stores the id, so a rename never has to
    reach into chats.db.
    """
    folder_id: str
    name: str
    created_at: float


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class SavedChat:
    """A persisted conversation.

    `messages` mirrors the Streamlit chat history: a list of
    {"role", "content", "citations"} dicts, kept as plain dicts so it
    round-trips through JSON without a bespoke schema.
    """
    chat_id: str
    title: str
    messages: list[dict]
    created_at: float
    updated_at: float
    # Which folders and documents the conversation was asked under. None
    # means everything - a chat saved before scopes existed, or one asked
    # with nothing filtered.
    scope: dict | None = None
