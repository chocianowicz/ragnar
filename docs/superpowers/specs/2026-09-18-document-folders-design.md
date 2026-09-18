# Document folders

**Status:** approved design, not yet planned
**Date:** 2026-09-18

## The problem

The Documents panel is a flat list with a checkbox per document. That works
at ten documents and stops working at fifty. A user whose corpus spans
several clients, projects or topics wants to ask a question of one of them,
and today the only way is to untick everything else by hand.

Scoped retrieval already exists: `doc_ids` threads through `store.py`,
`search.py` and `agentic.py`, and `ui/app.py` already builds a filter from
the ticked documents. What is missing is a way to name a group of documents
and tick it in one action.

## What this delivers

Folders in the Documents panel. Create, rename and delete them; file a
document into one; tick a folder to scope answers to it. A chat remembers
the folders it was scoped to.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Can a document be in several folders? | No, exactly one | A folder is "which client is this", not a label. One column, not a join table |
| Nested folders? | No | Streamlit has no tree widget, and one level covers client / project / topic |
| How does scoping work? | The same checkbox as today, at folder level | The user asked for folder selection to work exactly like document selection |
| How does a document get filed? | A folder picker on its own row | Keeps the checkbox meaning one thing |
| Deleting a folder? | Documents fall back to Unfiled | Deleting a folder is filing, not data loss |
| Is the scope remembered? | Per chat | "This chat is about Acme" is the useful unit |

### Why the picker is per row, not a bulk move

The obvious alternative is to tick documents and press "Move to…". It was
rejected because the tick already means *"search this"*. Giving it a second
meaning makes filing two documents require unticking every other one first,
which silently narrows what the next question can see — the user fixes their
filing and breaks their next answer, with no visible link between the two.

A picker on each row costs one control per row and keeps the checkbox
meaning exactly one thing.

## Data model

Both changes are additive; no existing column changes type or meaning.

```
registry.db
  documents.folder  TEXT                          -- NULL = Unfiled
  folders(name TEXT PRIMARY KEY, created_at REAL) -- new

chats.db
  chats.folders     TEXT                          -- JSON list of folder names
```

`documents.folder` goes into the existing `_MIGRATIONS` list in
`ingestion/registry_db.py`, so a database from an earlier release gains the
column when it is opened.

**The `folders` table is not redundant.** Deriving the folder list from
`SELECT DISTINCT folder FROM documents` would make an empty folder
impossible to represent, so "New folder" would have nothing to create and a
user could not make a folder before having something to put in it.

**`chats.db` has no migration mechanism today** — `history/chat_store.py`
creates its table and never alters it. This feature adds the same
`_MIGRATIONS` + `_migrate()` pattern that `registry_db.py` already uses,
so `chats.folders` lands on existing databases and later columns are cheap.

### Scope is stored as folder names, not document ids

A chat scoped to "Acme Corp" should see documents added to that folder
after the chat was saved. Storing the resolved `doc_id` list would freeze
the scope at save time and quietly exclude new documents.

The cost is that renaming a folder has to reach into `chats.db`.

## Operations

| Action | Effect |
|---|---|
| New folder | `INSERT INTO folders`. Empty folders are legal |
| Rename | `UPDATE folders`, `UPDATE documents`, then `UPDATE chats` |
| Delete | `DELETE FROM folders`, `UPDATE documents SET folder = NULL` |
| Move | `UPDATE documents SET folder = ?` |

Rename spans two databases, so it cannot be one transaction. The order is
registry first, chats last. If the chats update fails, a saved chat points
at a folder name that no longer exists — which resolves to no folders, and
so degrades to searching everything rather than to an error or an empty
result.

## Retrieval: unchanged

Nothing in `retrieval/` or `generation/` changes. `ui/app.py` still builds
`selected_doc_ids` from the ticked documents, and ticking a folder ticks its
documents. The existing rule that a fully-selected corpus sends
`doc_ids=None` still holds, so an unscoped question costs exactly what it
costs today.

This also means the refusal guarantee is untouched: the floor and
aggregation guards run where they always did.

## UI

Three levels of the same control the panel already uses:

```
☑ Select all

▾ ☐ Acme Corp                                  3/12
    ☑ ✅ nda.pdf              [ Acme Corp ▾ ]   ✕
    ☑ ✅ sow.pdf              [ Acme Corp ▾ ]   ✕
    ☐ ✅ invoice.xlsx         [ Acme Corp ▾ ]   ✕
▸ ☑ Northwind Ltd                             12/12
▸ ☑ Unfiled                                   32/32

[ New folder ]  [ Rename ]  [ Delete ]
```

### Streamlit has no indeterminate checkbox

A folder that is partly ticked cannot render as a half-filled box. The
folder checkbox is therefore **checked only when every document in it is
checked**, and the `3/12` count carries the real state. Ticking an
unchecked folder ticks all of its documents.

This is a platform limit, recorded here so it is not later mistaken for a
defect.

## Code layout

`ui/panels/documents.py` is 183 lines and this work would roughly double it.
Folder operations and the selection logic move to a new `ui/folders.py`,
leaving the panel responsible for rendering.

The selection rule becomes a pure function:

```python
def folder_state(docs, checked_ids) -> tuple[bool, str]:
    """(is_checked, "3/12") for one folder's documents."""
```

It has no Streamlit dependency and is tested directly.

## Testing

Registry (`ingestion/registry_db.py`):
- the column is added to a database created before it existed
- move sets the folder; delete sets its documents back to NULL
- rename updates both the folder row and its documents

Chat store (`history/chat_store.py`):
- the column is added to a database created before it existed
- a saved scope round-trips
- a scope naming a folder that no longer exists resolves to everything

Pure logic (`ui/folders.py`):
- `folder_state` for none, some and all documents checked

## Out of scope

Deliberately excluded, and easy to add later:

- nested folders
- a document in more than one folder
- folder colours or icons
- auto-filing by filename or content
- a folder picker on the upload widget, so documents land filed

The last is genuinely useful and pairs well with the per-row picker, but it
is not what was asked for.
