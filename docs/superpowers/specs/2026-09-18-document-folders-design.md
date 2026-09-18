# Document folders

**Status:** approved design, revised after review
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
| How does scoping work? | The same checkbox as today, at folder level | Folder selection should work exactly like document selection |
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
  documents.folder_id  TEXT                     -- NULL = Unfiled
  folders(
    folder_id   TEXT PRIMARY KEY,               -- uuid4, stable across renames
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at  REAL NOT NULL
  )

chats.db
  chats.scope  TEXT                             -- JSON, see "Saved scope"
```

`documents.folder_id` goes into the existing `_MIGRATIONS` list in
`ingestion/registry_db.py`, so a database from an earlier release gains the
column when it is opened.

**The `folders` table is not redundant.** Deriving the folder list from the
documents table would make an empty folder impossible to represent, so "New
folder" would have nothing to create and a user could not make a folder
before having something to put in it.

**`chats.db` has no migration mechanism today** — `history/chat_store.py`
creates its table and never alters it. This feature adds the same
`_MIGRATIONS` + `_migrate()` pattern that `registry_db.py` already uses, so
`chats.scope` lands on existing databases and later columns are cheap.

### Folders are referenced by id, not by name

A folder's identity is its `folder_id`. The name is a label the user can
change freely.

This is what keeps renaming cheap: **rename is one `UPDATE folders SET
name = ?` inside `registry.db` and nothing else**. Nothing in `chats.db`
stores a name, so there is no cross-database write, no ordering question and
no half-applied rename. An earlier draft of this design stored names in the
chat scope and had to reach into a second database on every rename; that is
gone.

### `Document` carries the folder

`core/models.Document` gains `folder_id: str | None = None`, and
`Registry._row_to_doc` selects it. Every caller that groups documents by
folder — the panel, and `folder_state` below — reads it from the `Document`
it already has, so no extra query is needed per row.

## Folder names

Validated in one place, `ui/folders.py`, before any write:

| Rule | Behaviour on failure |
|---|---|
| Trimmed of leading/trailing whitespace | silently trimmed |
| Not empty after trimming | inline error, no write |
| At most 60 characters | inline error, no write |
| Unique, case-insensitively (`COLLATE NOCASE`) | inline error naming the clash |
| Not the reserved word `Unfiled`, any case | inline error explaining it is reserved |

Duplicates are caught before the insert, so the `UNIQUE` constraint is a
backstop rather than the error path — a `sqlite3.IntegrityError` reaching
the UI would surface as a Streamlit traceback.

### Why `Unfiled` is reserved

`Unfiled` is not a row. It is how the panel renders `folder_id IS NULL`.
Nothing can rename or delete it, and those buttons are disabled when it is
the selected folder.

A real folder named "Unfiled" would be unambiguous to the code — different
`folder_id` — and completely ambiguous to the user, who would see two
identical rows with different contents and different capabilities. Reserving
the name costs one check and removes the whole class of confusion.

## Operations

| Action | Effect |
|---|---|
| New folder | Validate, then `INSERT INTO folders` with a fresh uuid4 |
| Rename | Validate, then `UPDATE folders SET name = ?` — registry only |
| Delete | `UPDATE documents SET folder_id = NULL` then `DELETE FROM folders`, one transaction |
| Move | `UPDATE documents SET folder_id = ?` |

All four run on the registry's existing lock-guarded connection. This
matters: the ingestion worker thread writes to the same database (status
transitions, chunk counts), so folder writes use `Registry._write` like
every other mutation rather than opening a connection of their own.

### A re-uploaded document inherits its folder

`doc_id` is a content hash, so correcting a typo in a filed document and
re-uploading it mints a **new, unrelated `doc_id`** with no folder — the
filing would silently vanish exactly when a user expects it to persist.

On `Registry.add`, if the new document's filename matches an existing
document that has a folder, the new one inherits that `folder_id`. Where
several match, the most recently added wins. This is a convenience, not an
invariant: a user can always move it afterwards.

## Saved scope

The runtime filter is an arbitrary set of document ids. A saved scope must
round-trip that faithfully *and* let a folder pick up documents added after
the chat was saved. It is therefore a JSON object, not a list of names:

```json
{"v": 1, "all": true}
{"v": 1, "folders": ["9f2c…"], "docs": ["a1b2…"]}
```

**Saving.** If every document is ticked, store `{"all": true}` — the
"everything" case, which must stay distinguishable. Otherwise store the ids
of folders whose documents are *all* ticked, plus the ids of any remaining
ticked documents not covered by those folders.

**Restoring.** `{"all": true}`, or a `NULL` column on a chat saved before
this feature, ticks everything. Otherwise the ticked set is every document
currently in the listed folders, plus the listed document ids. Folder ids
that no longer exist are dropped; document ids that no longer exist are
dropped.

Because folder membership is resolved at restore time, a document added to
Acme Corp after the chat was saved comes back ticked.

### An empty scope means refuse, and must keep meaning that

`ui/app.py` already treats an explicit empty selection as a genuine
instruction: the comment at line 154 reads *"which may be an empty list,
correctly refusing"*, and `retrieval/store.py:219` short-circuits an empty
`doc_ids` to no results rather than querying Qdrant.

So `{"v": 1, "folders": [], "docs": []}` means **nothing is selected, refuse**
— it is not the same as `{"all": true}`, and not the same as a `NULL`
column. Collapsing those three onto one representation would silently turn a
deliberate refusal into an answer drawn from the whole corpus, which is the
one failure this product exists to prevent.

This is why the column stores an object with an explicit `all` flag rather
than a bare list, where empty would have been ambiguous.

## Retrieval: unchanged

Nothing in `retrieval/` or `generation/` changes. `ui/app.py` still builds
`selected_doc_ids` from the ticked documents, and ticking a folder ticks its
documents. The existing rule that a fully-selected corpus sends
`doc_ids=None` still holds, so an unscoped question costs exactly what it
costs today.

The refusal guarantee is untouched: the floor and aggregation guards run
where they always did.

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

A partly-ticked folder cannot render as a half-filled box. The folder
checkbox is **checked only when every document in it is checked**, and the
`3/12` count carries the real state. Ticking an unchecked folder ticks all
of its documents.

This is a platform limit, recorded so it is not later mistaken for a defect.

### Ticking a folder

Folder checkboxes use an `on_change` callback that writes every
`sel_{doc_id}` key for that folder, mirroring the existing
`_select_all_changed` at `ui/panels/documents.py:100-107`. The pattern is
already in the file; this adds a per-folder version of it.

### A newly uploaded document arrives ticked

`sel_{doc_id}` does not exist for a document that was not present on the
last render, and the existing `st.checkbox(..., value=True)` default makes
it ticked. So uploading into a folder the user had deliberately left
partly ticked widens that folder's selection without the user touching it.

This is the current behaviour for the flat list and is kept deliberately: a
document you just uploaded being excluded from your next question is the
more surprising of the two options. The `3/12` count makes the change
visible.

### The picker when no folders exist

On a fresh install the only entry is `Unfiled`, so the row picker lists
`Unfiled` plus a `New folder…` entry that opens the same name prompt as the
panel button. Without it, filing the first document would require finding an
unrelated button first.

## Code layout

`ui/panels/documents.py` is 183 lines and this work would roughly double it.
Folder operations, name validation and the selection logic move to a new
`ui/folders.py`, leaving the panel responsible for rendering.

The selection rule becomes a pure function:

```python
def folder_state(docs, checked_ids) -> tuple[bool, str]:
    """(is_checked, "3/12") for one folder's documents."""
```

It has no Streamlit dependency and is tested directly.

## Testing

Registry (`ingestion/registry_db.py`):
- `folder_id` is added to a database created before the column existed
- move sets it; delete returns its documents to NULL and removes the row
- rename changes the name and leaves `folder_id` untouched
- a re-uploaded file inherits the folder of the same-named document

Names (`ui/folders.py`):
- empty, whitespace-only, over-length, duplicate-differing-in-case, and
  `unfiled` in any case are all rejected before any write
- a valid name is trimmed and stored

Chat store (`history/chat_store.py`):
- `scope` is added to a database created before the column existed
- `{"all": true}` round-trips and restores everything
- a folders+docs scope round-trips
- **an empty scope restores as nothing ticked, not as everything** — the
  regression that would reverse the refusal guarantee
- a `NULL` scope on a pre-feature chat restores as everything
- a scope naming a deleted folder drops it and keeps the rest
- a folder that gained a document since saving restores with it ticked

Pure logic (`ui/folders.py`):
- `folder_state` for none, some and all documents checked
- `folder_state` for an empty folder

## Out of scope

Deliberately excluded, and easy to add later:

- nested folders
- a document in more than one folder
- folder colours or icons
- auto-filing by filename or content
- a folder picker on the upload widget, so documents land filed

The last is genuinely useful and pairs well with the per-row picker, but it
is not what was asked for.
