def chunk_table_markdown(table: str, rows_per_group: int = 20) -> list[str]:
    """Split a markdown table into row groups, repeating the header.

    A table row without its header is semantically meaningless to an
    embedder, so the header goes into every group.

    Blank rows are treated as natural section boundaries where present;
    otherwise groups are fixed-size.
    """
    lines = table.strip().splitlines()
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

    groups: list[str] = []
    for section in sections:
        for start in range(0, len(section), rows_per_group):
            rows = section[start:start + rows_per_group]
            groups.append("\n".join([header, separator, *rows]))

    return groups
