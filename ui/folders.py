"""Folder logic, kept out of the Streamlit panel so it can be tested.

Everything here is a pure function over Documents and Folders. The panel
renders what these return; the registry stores what they validate.
"""
from __future__ import annotations

from core.models import Document, Folder

# Unfiled is not a row. It is how the panel renders folder_id IS NULL, and
# how a saved scope names that group - real ids are uuid4, so the literal
# cannot collide with one. Without it Unfiled would be the one folder that
# does not pick up documents added after a chat was saved.
UNFILED = "unfiled"

UNFILED_LABEL = "Unfiled"

NAME_MAX = 60


class FolderNameError(ValueError):
    """A folder name the user has to fix before anything is written."""


def validate_name(name: str, existing: list[Folder],
                  allow_id: str | None = None) -> str:
    """The name to store, or raise with a message fit to show the user.

    allow_id is the folder being renamed, which is allowed to keep its own
    name - otherwise renaming a folder to what it is already called would
    collide with itself.
    """
    cleaned = name.strip()
    if not cleaned:
        raise FolderNameError("A folder needs a name.")
    if len(cleaned) > NAME_MAX:
        raise FolderNameError(f"Keep it to {NAME_MAX} characters or fewer.")
    if cleaned.casefold() == UNFILED_LABEL.casefold():
        raise FolderNameError(
            '"Unfiled" is reserved - it is where documents sit when they '
            'are in no folder.'
        )
    for folder in existing:
        if folder.folder_id == allow_id:
            continue
        if folder.name.casefold() == cleaned.casefold():
            raise FolderNameError(
                f'There is already a folder called "{folder.name}".')
    return cleaned


def folder_state(docs: list[Document],
                 checked_ids: set[str]) -> tuple[bool, str]:
    """(is_checked, "3/12") for one folder's documents.

    Streamlit checkboxes have no indeterminate state, so a partly-ticked
    folder reads as unchecked and the count carries the truth. An empty
    folder satisfies "all checked" vacuously and contributes no ids, so it
    renders checked and changes nothing.
    """
    total = len(docs)
    checked = sum(1 for d in docs if d.doc_id in checked_ids)
    return checked == total, f"{checked}/{total}"


def _group(docs: list[Document]) -> dict[str, list[Document]]:
    """Documents by folder id, with Unfiled under the sentinel."""
    grouped: dict[str, list[Document]] = {}
    for doc in docs:
        grouped.setdefault(doc.folder_id or UNFILED, []).append(doc)
    return grouped


def group_for_display(docs: list[Document],
                      all_folders: list[Folder]) -> list[tuple]:
    """(folder_id, label, documents) in the order the panel draws them.

    Named folders first, in the order the registry returns them, then
    Unfiled: it is where things land rather than somewhere the user chose,
    so it belongs at the bottom. Empty named folders are included - a user
    who has just made one needs to see it to file into it.
    """
    by_folder = _group(docs)
    rows = [(f.folder_id, f.name, by_folder.get(f.folder_id, []))
            for f in all_folders]
    rows.append((UNFILED, UNFILED_LABEL, by_folder.get(UNFILED, [])))
    return rows


def scope_from_selection(docs: list[Document],
                         checked_ids: set[str]) -> dict:
    """What to store for a chat asked under this selection.

    Whole folders are stored by id so they pick up documents added later.
    Anything else is stored as the document ids themselves.

    "Everything checked" is stored as {"all": True} rather than as every
    folder, so it stays distinguishable from an empty selection - which
    means "refuse", not "search everything".
    """
    checked = {d.doc_id for d in docs if d.doc_id in checked_ids}
    if docs and len(checked) == len(docs):
        return {"v": 1, "all": True}

    folders, loose = [], set(checked)
    for folder_id, group in _group(docs).items():
        if group and all(d.doc_id in checked for d in group):
            folders.append(folder_id)
            loose -= {d.doc_id for d in group}
    return {"v": 1, "folders": sorted(folders), "docs": sorted(loose)}


def selection_from_scope(scope: dict | None,
                         docs: list[Document]) -> set[str]:
    """Which documents to tick when a chat is reopened.

    None is a chat saved before scopes existed and means everything.
    {"all": True} means everything. Anything else is resolved against the
    documents that exist NOW, so a folder picks up what was added to it
    since - and an empty scope stays empty, which is a refusal, not a
    licence to search the whole corpus.
    """
    if scope is None or scope.get("all"):
        return {d.doc_id for d in docs}

    wanted = set(scope.get("folders") or [])
    selected = {d.doc_id for d in docs if (d.folder_id or UNFILED) in wanted}
    known = {d.doc_id for d in docs}
    return selected | (set(scope.get("docs") or []) & known)
