import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny, PayloadSchemaType,
)

from core.models import Chunk, SearchResult

log = logging.getLogger(__name__)

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int = 1024,
                 client: QdrantClient | None = None,
                 upsert_batch: int = 128):
        self.collection = collection
        self.dim = dim
        # A document is upserted in one call; bound how large that call can
        # get, for the same reason the embedder batches.
        self.upsert_batch = upsert_batch
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim,
                                            distance=Distance.COSINE),
            )
        else:
            self._check_dimension()
        self._ensure_doc_id_index()

    def _check_dimension(self) -> None:
        """Fail loudly when the collection was built for a different model.

        Changing models.embedding in config.yaml is the one knob likely to
        cause this, and without the check it surfaces much later as an
        opaque upsert error attached to whichever document happened to be
        ingesting — rather than as the configuration problem it is.
        """
        params = self._client.get_collection(self.collection).config.params
        vectors = params.vectors
        size = getattr(vectors, "size", None)
        if size is None and isinstance(vectors, dict):
            sizes = {v.size for v in vectors.values()}
            size = sizes.pop() if len(sizes) == 1 else None
        if size is not None and size != self.dim:
            raise ValueError(
                f"Collection '{self.collection}' stores {size}-dimensional "
                f"vectors but the configured embedding model produces "
                f"{self.dim}. Changing the embedding model requires "
                f"re-indexing: drop the collection and re-ingest."
            )

    def _ensure_doc_id_index(self) -> None:
        """Index the one payload field that is ever filtered on.

        search() filters by doc_id whenever the user scopes the question to
        a subset of documents, and delete_by_doc filters on it during every
        re-ingest. Unindexed, both scan the whole collection.

        Idempotent, so existing collections gain the index on next startup.
        """
        try:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            # Already present, or a server too old to care. Neither is
            # worth failing startup over — the index is an optimisation,
            # not a correctness requirement — but don't swallow it silently
            # either, or a collection that never gets one looks identical
            # to one that has it.
            log.debug("doc_id payload index not created: %s", exc)

    def drop_collection(self) -> None:
        self._client.delete_collection(self.collection)

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        # Deterministic: re-upserting the same chunk overwrites rather than
        # duplicating.
        return str(uuid.uuid5(NAMESPACE, f"{chunk.doc_id}:{chunk.chunk_index}"))

    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            # zip() would silently drop the tail, indexing a document with
            # some of its chunks missing and no sign anything went wrong.
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors"
            )
        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page,
                    "sheet": chunk.sheet,
                    "is_table": chunk.is_table,
                    "low_confidence": chunk.low_confidence,
                    "is_summary": chunk.is_summary,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), self.upsert_batch):
            self._client.upsert(
                collection_name=self.collection,
                points=points[start:start + self.upsert_batch],
            )

    def search(self, vector: list[float], limit: int,
               doc_ids: list[str] | None = None) -> list[SearchResult]:
        # doc_ids=None means unfiltered (search everything). An explicit
        # empty list means "nothing selected" - short-circuit rather than
        # ask Qdrant to match against zero ids, which is a degenerate query.
        if doc_ids is not None and not doc_ids:
            return []

        query_filter = None
        if doc_ids is not None:
            query_filter = Filter(must=[
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            ])

        hits = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        return [
            SearchResult(
                chunk=Chunk(
                    doc_id=h.payload["doc_id"],
                    filename=h.payload["filename"],
                    text=h.payload["text"],
                    chunk_index=h.payload["chunk_index"],
                    page=h.payload.get("page"),
                    sheet=h.payload.get("sheet"),
                    is_table=h.payload.get("is_table", False),
                    low_confidence=h.payload.get("low_confidence", False),
                    is_summary=h.payload.get("is_summary", False),
                ),
                score=h.score,
            )
            for h in hits
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ]),
        )
