import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny, PayloadSchemaType,
    SparseVectorParams, SparseVector, Modifier, Prefetch, FusionQuery,
    Fusion,
)

from core.models import Chunk, SearchResult
from retrieval import sparse

log = logging.getLogger(__name__)

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Vector names used by a hybrid collection. A collection created before
# hybrid retrieval has a single unnamed vector instead, and Qdrant will not
# let a sparse vector be added to it afterwards — so both shapes are
# supported here and the collection itself says which one it is.
DENSE = "dense"
SPARSE = "sparse"


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int = 1024,
                 client: QdrantClient | None = None,
                 upsert_batch: int = 128, hybrid: bool = True):
        self.collection = collection
        self.dim = dim
        # A document is upserted in one call; bound how large that call can
        # get, for the same reason the embedder batches.
        self.upsert_batch = upsert_batch
        # What to build when creating a collection. Whether an *existing*
        # collection is hybrid is a property of that collection, not of
        # this flag — see is_hybrid.
        self.hybrid = hybrid
        self._client = client or QdrantClient(url=url)
        self._is_hybrid: bool | None = None

    @property
    def is_hybrid(self) -> bool:
        """Whether the live collection carries named dense+sparse vectors.

        Read from the collection rather than assumed, so a collection built
        before hybrid retrieval keeps working unchanged instead of failing
        on every upsert with a vector-name error.
        """
        if self._is_hybrid is None:
            try:
                vectors = self._client.get_collection(
                    self.collection).config.params.vectors
                self._is_hybrid = isinstance(vectors, dict) and DENSE in vectors
            except Exception:
                self._is_hybrid = False
        return self._is_hybrid

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            if self.hybrid:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        DENSE: VectorParams(size=self.dim,
                                            distance=Distance.COSINE)},
                    sparse_vectors_config={
                        # IDF is applied server-side at query time, so the
                        # stored vectors stay plain term frequencies and the
                        # statistics stay correct as documents come and go.
                        SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
                )
            else:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim,
                                                distance=Distance.COSINE),
                )
            self._is_hybrid = self.hybrid
        else:
            self._is_hybrid = None          # re-read from the live collection
            self._check_dimension()
            if self.hybrid and not self.is_hybrid:
                log.warning(
                    "Collection '%s' predates hybrid retrieval and has no "
                    "sparse vectors; lexical matching is off. Qdrant cannot "
                    "add one in place — migrate with "
                    "`python -m retrieval.migrate`.", self.collection,
                )
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
            # Named vectors: the dense one is what `dim` describes.
            named = vectors.get(DENSE) or next(iter(vectors.values()), None)
            size = getattr(named, "size", None)
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

    @staticmethod
    def _vector_payload(chunk: Chunk, dense: list[float], hybrid: bool):
        """The vector field for one point, in whichever shape the
        collection expects."""
        if not hybrid:
            return dense
        indices, values = sparse.encode(chunk.text)
        return {
            DENSE: dense,
            SPARSE: SparseVector(indices=indices, values=values),
        }

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
        hybrid = self.is_hybrid
        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=self._vector_payload(chunk, vector, hybrid),
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
               doc_ids: list[str] | None = None,
               text: str | None = None) -> list[SearchResult]:
        """Nearest chunks to `vector`.

        On a hybrid collection, passing the query `text` as well runs a
        lexical search beside the dense one and fuses the two by reciprocal
        rank. That is what finds an exact identifier: a dense vector ranked
        the chunk holding CN code 31022100 at 291 on the real corpus, while
        a lexical match puts it first, because the token is either present
        or it is not.

        Note the score then means something different. Fused results carry
        an RRF score — a function of rank in each list, not a cosine
        similarity — so it is comparable within one result set and nowhere
        else. Nothing downstream depends on its absolute value: the
        reranker replaces it, and the similarity floor is applied to rerank
        scores. The `use_reranker=False` path was already documented as
        returning uncalibrated scores, and this makes it more so.
        """
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

        hits = self._query(vector, limit, query_filter, text)

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

    def _query(self, vector, limit, query_filter, text):
        """Dense-only, or dense+lexical fused, depending on the collection
        and whether the caller supplied query text."""
        indices, values = sparse.encode(text) if text else ([], [])

        if not (self.is_hybrid and indices):
            return self._client.query_points(
                collection_name=self.collection,
                query=vector,
                using=DENSE if self.is_hybrid else None,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ).points

        return self._client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=vector, using=DENSE, limit=limit,
                         filter=query_filter),
                Prefetch(query=SparseVector(indices=indices, values=values),
                         using=SPARSE, limit=limit, filter=query_filter),
            ],
            # Reciprocal rank fusion rather than score fusion: the two
            # scales are not comparable (cosine similarity against
            # IDF-weighted term overlap), so combining them by rank is the
            # only honest option without calibrating both first.
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    def delete_by_doc(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ]),
        )
