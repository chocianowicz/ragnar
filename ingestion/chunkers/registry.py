from ingestion.chunkers.fixed import FixedChunker
from ingestion.chunkers.structural import StructuralChunker

# The ONLY place a component is selected by name. Deliberately narrow:
# chunking is the one component with an experiment attached (Ragas eval,
# a much later task), which is what justifies swappability here and
# nowhere else in this codebase.
CHUNKERS = {
    "fixed": FixedChunker,
    "structural": StructuralChunker,
}


def build_chunker(config: dict):
    name = config.get("strategy", "structural")
    if name not in CHUNKERS:
        raise ValueError(
            f"Unknown chunker '{name}'. Available: {sorted(CHUNKERS)}"
        )

    if name == "fixed":
        return FixedChunker()
    return StructuralChunker(
        target_tokens=config.get("target_tokens", 500),
        overlap_tokens=config.get("overlap_tokens", 50),
        rows_per_group=config.get("table_rows_per_group", 20),
    )
