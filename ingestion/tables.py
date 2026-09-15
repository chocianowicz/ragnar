def _compact(line: str) -> str:
    """Strip the cosmetic column padding from one markdown table line.

    Docling pads every cell out to its column's widest value, which is for
    human eyes and costs real money here: the header of a table with
    sentence-length descriptions ran to ~1,270 characters of mostly spaces,
    and that header is repeated in every group. At one row per group it was
    two thirds of the embedded text, so every chunk of the table embedded
    to nearly the same vector regardless of the row it carried — the
    boilerplate drowning the only part that distinguishes them.

    Padding carries no information a reader or an embedder needs, so it
    goes before anything is measured or grouped.
    """
    if "|" not in line:
        return line.strip()

    cells = line.split("|")
    # A fully delimited row splits to empty leading and trailing cells.
    lead = cells[0].strip() == ""
    trail = len(cells) > 1 and cells[-1].strip() == ""
    inner = cells[(1 if lead else 0):(-1 if trail else None)]

    stripped = [c.strip() for c in inner]
    if stripped and set("".join(stripped)) <= set("-:"):
        stripped = ["---" for _ in stripped]      # separator row

    # Single spaces, not zero: this stays valid, readable markdown, which
    # is what the model sees and what a citation quotes back.
    body = " | ".join(stripped)
    return f"{'| ' if lead else ''}{body}{' |' if trail else ''}"


def chunk_table_markdown(table: str, rows_per_group: int = 20,
                         max_chars: int | None = None) -> list[str]:
    """Split a markdown table into row groups, repeating the header.

    A table row without its header is semantically meaningless to an
    embedder, so the header goes into every group.

    Blank rows are treated as natural section boundaries where present;
    otherwise groups are bounded by `rows_per_group` and, when given, by
    `max_chars`.

    Both bounds are needed because rows differ wildly in width. Row count
    alone was the only bound for a long time, and on a table whose cells
    hold sentence-length descriptions, twenty rows came to roughly 14,000
    characters — seven times the prose chunk target. Chunks that size
    retrieve poorly (one embedding averaged over far too much) and get
    truncated by the reranker anyway. `max_chars` bounds the group, and
    whichever limit is reached first ends it.

    A single row wider than `max_chars` is still emitted on its own: the
    header plus one row is the smallest unit that means anything, and
    splitting inside a row would produce exactly the headerless fragment
    this function exists to avoid.
    """
    lines = [_compact(line) for line in table.strip().splitlines()]
    if len(lines) < 2:
        return []

    header, separator = lines[0], lines[1]
    body = lines[2:]
    if not any(line.strip() for line in body):
        return []

    # Prefer blank-row boundaries when the table has them.
    sections: list[list[str]] = []
    current: list[str] = []
    for line in body:
        if not line.strip():
            if current:
                sections.append(current)
                current = []
        else:
            current.append(line)
    if current:
        sections.append(current)

    overhead = len(header) + len(separator) + 2   # two joining newlines
    groups: list[str] = []

    for section in sections:
        current: list[str] = []
        size = overhead

        for row in section:
            too_many = len(current) >= rows_per_group
            too_wide = (
                max_chars is not None
                and current
                and size + len(row) + 1 > max_chars
            )
            if too_many or too_wide:
                groups.append("\n".join([header, separator, *current]))
                current, size = [], overhead

            current.append(row)
            size += len(row) + 1

        if current:
            groups.append("\n".join([header, separator, *current]))

    return groups
