"""Rebuild a pre-hybrid collection with dense and lexical vectors.

Qdrant will not add a sparse vector to an existing collection, so hybrid
retrieval needs a new one. It does not need a re-ingest: every chunk's text
is already in its payload and the dense vectors can be read back out, so
this copies points across and computes the lexical half locally — no
parsing, no embedding model, no Ollama, no re-reading the originals.

    python -m retrieval.migrate            # report what would happen
    python -m retrieval.migrate --apply    # build the new collection

Nothing is deleted or overwritten. The new collection is built alongside
the old one under a new name, and the last step is a line you change in
config.yaml — so the old index stays exactly as it was, and reverting is
changing that line back.
"""
from __future__ import annotations

import argparse
import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from core.config import Config
from retrieval import sparse
from retrieval.store import QdrantStore, DENSE, SPARSE

BATCH = 256


def _iter_points(client: QdrantClient, collection: str, batch: int = BATCH):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=batch, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            return
        yield points
        if offset is None:
            return


def _dense_of(point):
    """The dense vector, whether the source stored it named or bare."""
    vector = point.vector
    if isinstance(vector, dict):
        return vector.get(DENSE) or next(iter(vector.values()))
    return vector


def migrate(cfg: Config, target: str | None = None,
            apply: bool = False) -> int:
    client = QdrantClient(url=cfg.qdrant_url)
    source = cfg.collection
    target = target or f"{source}_hybrid"

    info = client.get_collection(source)
    total = info.points_count
    vectors = info.config.params.vectors
    already = isinstance(vectors, dict) and DENSE in vectors

    print(f"source '{source}': {total} points, "
          f"{'already hybrid' if already else 'dense only'}")
    if already:
        print("Nothing to do.")
        return 0

    if not apply:
        print(f"Would build '{target}' with {total} points, computing each "
              f"lexical vector from the chunk text already in the payload.")
        print(f"'{source}' would not be modified.")
        print("Re-run with --apply to do it.")
        return 0

    existing = {c.name for c in client.get_collections().collections}
    if target in existing:
        raise SystemExit(
            f"'{target}' already exists. Drop it first if it is a failed "
            f"attempt, or pass --target with another name."
        )

    QdrantStore(cfg.qdrant_url, target, cfg.embedding_dim, client=client,
                hybrid=True).ensure_collection()

    moved = 0
    started = time.time()
    for batch in _iter_points(client, source):
        points = []
        for p in batch:
            indices, values = sparse.encode(p.payload.get("text", ""))
            points.append(PointStruct(
                id=p.id,
                vector={DENSE: _dense_of(p),
                        SPARSE: SparseVector(indices=indices, values=values)},
                payload=p.payload,
            ))
        client.upsert(collection_name=target, points=points)
        moved += len(points)
        print(f"  {moved}/{total}", end="\r", flush=True)

    copied = client.get_collection(target).points_count
    print(f"\ncopied {copied}/{total} points in {time.time() - started:.1f}s")

    if copied != total:
        raise SystemExit(
            f"point count mismatch ({copied} != {total}). '{source}' is "
            f"untouched and still the live collection; inspect or drop "
            f"'{target}'."
        )

    print()
    print("Done. To switch over, set this in config.yaml:")
    print(f"    storage:\n      collection: {target}")
    print(f"Then restart the app. '{source}' is unchanged — revert by "
          f"putting the old name back, and drop '{target}' when you are "
          f"satisfied either way.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="build the new collection (default: report only)")
    parser.add_argument("--target", default=None,
                        help="name for the new collection "
                             "(default: <source>_hybrid)")
    args = parser.parse_args()
    return migrate(Config(), target=args.target, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
