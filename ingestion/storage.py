import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ingestion.parser import Block, ParsedDocument


class Storage:
    """Owns the on-disk layout.

    data/ is the source of truth; Qdrant is a rebuildable index. Originals
    are never deleted automatically.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.originals = self.root / "originals"
        self.converted = self.root / "converted"
        for directory in (self.inbox, self.originals, self.converted):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def doc_id(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def doc_id_for_bytes(data: bytes) -> str:
        """The same id as doc_id(), for content not yet written to disk.

        Lets an upload be named by its id before it lands in the inbox,
        instead of being written under a caller-supplied name and hashed
        afterwards.
        """
        return hashlib.sha256(data).hexdigest()

    def inbox_path(self, doc_id: str, filename: str) -> Path:
        """Where this document waits to be ingested.

        Keyed by doc_id, not by filename. The registry is keyed by content
        hash, so a filename-keyed inbox lets two different files that happen
        to share a name collide: the second write replaces the first's bytes
        while both ids sit in the queue, and the first id is then ingested
        from the second file's content - indexed, marked done, and cited
        under a hash that does not describe it.

        Mirrors archive()'s naming so the two directories read alike.
        """
        name = Path(filename)
        return self.inbox / f"{name.stem}.{doc_id[:8]}{name.suffix}"

    def archive(self, path: Path, doc_id: str,
                filename: str | None = None) -> Path:
        """Move an ingested file out of the inbox and into originals.

        `filename` is the display name. It has to be passed explicitly now
        that the inbox names files by doc_id: deriving the archive name from
        path.stem would fold that id into the name a second time, and
        archived_path() would no longer find what this wrote.
        """
        target = self.archived_path(filename or path.name, doc_id)
        shutil.move(str(path), str(target))
        return target

    def archived_path(self, filename: str, doc_id: str) -> Path:
        """Where archive() put the original for this (filename, doc_id)."""
        name = Path(filename)
        return self.originals / f"{name.stem}.{doc_id[:8]}{name.suffix}"

    def restore_to_inbox(self, filename: str, doc_id: str) -> bool:
        """Copy an archived original back into the inbox for re-ingestion.

        Returns False if the archived original is missing (e.g. removed by
        hand), so the caller can report which documents can't be re-chunked.
        """
        src = self.archived_path(filename, doc_id)
        if not src.exists():
            return False
        shutil.copy(str(src), str(self.inbox_path(doc_id, filename)))
        return True

    # Stated rather than inherited from the process locale. The corpus is
    # Polish and English, and converted/ is the cache a re-index would have
    # to be rebuilt from, so this is the wrong place to let an environment
    # variable decide whether "ż" survives a round trip.
    ENCODING = "utf-8"

    def write_converted(self, doc_id: str, markdown: str) -> None:
        (self.converted / f"{doc_id}.md").write_text(
            markdown, encoding=self.ENCODING)

    def read_markdown(self, doc_id: str) -> str | None:
        path = self.converted / f"{doc_id}.md"
        return path.read_text(encoding=self.ENCODING) if path.exists() else None

    def remove_converted(self, doc_id: str) -> None:
        (self.converted / f"{doc_id}.md").unlink(missing_ok=True)
        (self.converted / f"{doc_id}.blocks.json").unlink(missing_ok=True)

    def write_parsed(self, doc_id: str, parsed: ParsedDocument) -> None:
        """Cache the parsed blocks so a chunking-only change can skip the
        (expensive) Docling parse.

        Markdown is not duplicated here: it is already on disk via
        write_converted, keyed by the same doc_id, and the chunker reads only
        blocks. Written with the same explicit encoding as the markdown, for
        the same reason — the corpus is Polish and English.

        Parsing is deterministic for a given file and parser, so this is safe
        to reuse; it is *not* safe to reuse across a change to the parser
        itself (a new OCR setting, say), which is why the file is keyed by
        doc_id alone and must be deleted to force a re-parse.
        """
        payload = {
            "low_confidence": parsed.low_confidence,
            "blocks": [asdict(b) for b in parsed.blocks],
        }
        (self.converted / f"{doc_id}.blocks.json").write_text(
            json.dumps(payload), encoding=self.ENCODING)

    def read_parsed(self, doc_id: str) -> ParsedDocument | None:
        """The cached blocks for this doc_id, or None if there is no usable
        cache and the caller has to parse.

        None rather than an exception on anything unusable: a re-chunk is
        exactly the operation that must not fail, and a cache miss costs only
        the time it used to cost. That covers a file from an older format,
        which must not be trusted to have this shape — `markdown` in
        particular is reconstructed empty, so a caller that wants markdown
        reads it from write_converted's file, not from here.
        """
        path = self.converted / f"{doc_id}.blocks.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding=self.ENCODING))
            blocks = [Block(**b) for b in payload["blocks"]]
            low_confidence = bool(payload["low_confidence"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return ParsedDocument(markdown="", blocks=blocks,
                              low_confidence=low_confidence)

    def pending_files(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())
