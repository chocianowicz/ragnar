import hashlib
import shutil
from pathlib import Path


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

    def archive(self, path: Path, doc_id: str) -> Path:
        target = self.originals / f"{path.stem}.{doc_id[:8]}{path.suffix}"
        shutil.move(str(path), str(target))
        return target

    def write_converted(self, doc_id: str, markdown: str) -> None:
        (self.converted / f"{doc_id}.md").write_text(markdown)

    def read_markdown(self, doc_id: str) -> str | None:
        path = self.converted / f"{doc_id}.md"
        return path.read_text() if path.exists() else None

    def remove_converted(self, doc_id: str) -> None:
        for suffix in (".md", ".json"):
            path = self.converted / f"{doc_id}{suffix}"
            path.unlink(missing_ok=True)

    def pending_files(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())
