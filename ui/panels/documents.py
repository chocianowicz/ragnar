import streamlit as st

from ui.services import format_eta
from ui import folders, selection, sources

STATUS_ICONS = {"queued": "⏳", "processing": "⚙️", "done": "✅", "failed": "❌"}


def render(svc) -> None:
    with st.expander("Documents", expanded=False):
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded = st.file_uploader(
            "Upload", type=["pdf", "xlsx", "docx"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded and st.button("Upload", type="primary"):
            for file in uploaded:
                # Hash before writing, so the file lands under its own
                # doc_id. Naming the inbox entry after the upload instead
                # lets two files that share a name overwrite each other,
                # and the survivor gets indexed under both ids.
                data = file.getbuffer()
                doc_id = svc["storage"].doc_id_for_bytes(data)
                target = svc["storage"].inbox_path(doc_id, file.name)
                target.write_bytes(data)
                svc["registry"].add(doc_id, file.name, target.stat().st_size)
            # Force the uploader widget to reset to empty on the next render.
            st.session_state.uploader_key += 1
            st.rerun()

        # Files placed directly in the watched folder (outside the browser
        # upload widget) have no registry entry yet - only show the manual
        # ingest option when there's actually something like that to pick up.
        externally_dropped = [
            p for p in svc["storage"].pending_files()
            if svc["registry"].get(svc["storage"].doc_id(p)) is None
        ]
        if externally_dropped:
            st.caption(
                f"{len(externally_dropped)} file(s) found in the watched "
                "folder, not added via upload"
            )
            if st.button("Ingest inbox"):
                for path in externally_dropped:
                    svc["registry"].add(svc["storage"].doc_id(path), path.name,
                                        path.stat().st_size)
                st.success(f"Queued {len(externally_dropped)} file(s)")
                st.rerun()

        _status_strip(svc)


# The strip polls so an in-progress ingest shows its countdown without the
# user touching anything. Nothing is ingesting most of the time, though,
# and a two-second poll for the life of the session is the only thing
# keeping a fully idle app busy. Two fragments, one interval each; the
# caller picks based on whether the queue is actually doing anything.
IDLE_POLL = "30s"
BUSY_POLL = "2s"


def _status_strip(svc) -> None:
    processing, queued, _ = svc["registry"].ingest_eta()
    if processing or queued:
        _status_strip_busy(svc)
    else:
        _status_strip_idle(svc)


@st.fragment(run_every=BUSY_POLL)
def _status_strip_busy(svc) -> None:
    _, queued, _ = svc["registry"].ingest_eta()
    _status_strip_body(svc)
    # The parent chose this fragment, so only the parent can hand back to
    # the idle one. Rerun the app once when the queue drains — a single
    # extra rerun per ingest, which also refreshes everything else that was
    # waiting on the document becoming available.
    if not queued and not svc["registry"].counts().get("processing"):
        st.rerun(scope="app")


@st.fragment(run_every=IDLE_POLL)
def _status_strip_idle(svc) -> None:
    _status_strip_body(svc)


def _status_strip_body(svc) -> None:
    processing, queued, eta = svc["registry"].ingest_eta()

    if processing or queued:
        st.info(
            f"Indexing — {processing} in progress, "
            f"{queued} queued{format_eta(eta)}"
        )

    docs = svc["registry"].all()
    all_folders = svc["registry"].folders()

    def _select_all_changed():
        value = st.session_state.get("select_all_docs", True)
        for d in docs:
            selection.set_selected(d.doc_id, value)

    if docs:
        st.checkbox("Select all", value=True, key="select_all_docs",
                    on_change=_select_all_changed)
        st.caption(
            "Unchecked documents and folders are excluded from answers — "
            "only checked ones are searched."
        )

    checked_ids = selection.selected_ids(docs)
    for folder_id, label, group in folders.group_for_display(docs,
                                                             all_folders):
        _render_folder(svc, folder_id, label, group, checked_ids, all_folders)

    _render_folder_controls(svc, all_folders)


def _render_folder(svc, folder_id, label, group, checked_ids,
                   all_folders) -> None:
    """One folder's header row, then its documents."""
    is_checked, count = folders.folder_state(group, checked_ids)

    def _folder_changed(folder_id=folder_id, group=group):
        value = st.session_state.get(f"folder_sel_{folder_id}", True)
        for d in group:
            selection.set_selected(d.doc_id, value)

    # Streamlit ignores value= once a widget's key is in session state, so
    # a derived checkbox has to be written INTO session state before it
    # renders. Without this the folder box freezes at whatever it showed
    # first while the count moves beneath it.
    st.session_state[f"folder_sel_{folder_id}"] = is_checked

    # Folders exist because the list got long, so each one can be folded
    # away. Open by default: a panel that hides everything on first sight
    # is worse than a long one. The state is per session, not stored.
    open_key = f"folder_open_{folder_id}"
    is_open = st.session_state.setdefault(open_key, True)

    # The checkbox leads, as it does on every document row. The rest is one
    # bordered body: clicking anywhere on the arrow, name or count folds
    # the folder, and the ⋯ sits inside the same body on the right.
    col_check, col_body = st.columns([0.5, 5.25], vertical_alignment="center")
    with col_check:
        st.checkbox(f"Include {label}", key=f"folder_sel_{folder_id}",
                    on_change=_folder_changed, label_visibility="collapsed")
    with col_body:
        # Columns, not a horizontal container: the sidebar is narrow enough
        # that a horizontal container wraps the ⋯ onto a line of its own.
        with st.container(key=f"rowbox_folder_{folder_id}", border=True):
            col_fold, col_menu = st.columns([4, 1],
                                            vertical_alignment="center",
                                            gap="small")
            with col_fold:
                # The count is what a folded folder says about itself, so
                # it stays in the label where it is visible either way.
                arrow = "▾" if is_open else "▸"
                if st.button(f"{arrow}  **{label}**  ·  {count}",
                             key=f"fold_{folder_id}", width="stretch",
                             help=("Hide these documents" if is_open
                                   else f"Show {len(group)} document(s)")):
                    st.session_state[open_key] = not is_open
                    st.rerun()
            with col_menu:
                # Unfiled is the absence of a folder, not a row: there is
                # nothing to rename and nothing to delete.
                if folder_id != folders.UNFILED:
                    _folder_menu(svc, folder_id, label, group, all_folders)

    if not is_open:
        return
    # One keyed container for the whole group, so the tree rule in APP_CSS
    # runs from the first document to the last and stops there.
    with st.container(key=f"folder_docs_{folder_id}"):
        if not group:
            st.caption("*empty*")
        for doc in group:
            _render_document(svc, doc, all_folders)


def _folder_menu(svc, folder_id, label, group, all_folders) -> None:
    """Rename and delete, behind the folder's ⋯."""
    with st.popover("⋯", width="stretch"):
        new_name = st.text_input("Rename to", value=label,
                                 key=f"rename_{folder_id}")
        if st.button("Rename", key=f"rename_go_{folder_id}",
                     use_container_width=True):
            try:
                clean = folders.validate_name(new_name, all_folders,
                                              allow_id=folder_id)
            except folders.FolderNameError as exc:
                st.error(str(exc))
            else:
                svc["registry"].rename_folder(folder_id, clean)
                st.rerun()

        st.divider()
        held = len(group)
        st.caption(
            f"{held} document(s) move to Unfiled." if held
            else "This folder is empty."
        )
        if st.button("Delete folder", key=f"del_folder_{folder_id}",
                     use_container_width=True):
            svc["registry"].delete_folder(folder_id)
            st.rerun()


def _document_menu(svc, doc, all_folders) -> None:
    """Move and remove, behind the document's ⋯.

    Moving is a list of buttons rather than a dropdown: a dropdown with a
    key remembers the value it showed, so renaming or deleting a folder
    would leave it pointing at a name that no longer exists. A button has
    nothing to remember.
    """
    with st.popover("⋯", width="stretch"):
        st.caption("Move to")
        targets = [(None, folders.UNFILED_LABEL)]
        targets += [(f.folder_id, f.name) for f in all_folders]
        movable = [(fid, name) for fid, name in targets if fid != doc.folder_id]

        if movable:
            for target_id, name in movable:
                if st.button(name, key=f"mv_{doc.doc_id}_{target_id}",
                             use_container_width=True):
                    svc["registry"].set_folder(doc.doc_id, target_id)
                    st.rerun()
        else:
            st.caption("No other folder yet — make one below.")

        st.divider()
        if st.button("Remove document", key=f"rm_{doc.doc_id}",
                     use_container_width=True,
                     help=f"Delete {doc.filename} from the index"):
            svc["store"].delete_by_doc(doc.doc_id)
            svc["storage"].remove_converted(doc.doc_id)
            # Also drop the served copy, or the file stays downloadable by
            # URL after the menu says it is gone.
            sources.unpublish(doc.doc_id, doc.filename)
            svc["registry"].remove(doc.doc_id)
            st.rerun()


def _render_document(svc, doc, all_folders) -> None:
    icon = STATUS_ICONS[doc.status.value]

    # Same shape as the folder header: checkbox first, then one bordered
    # body holding the name (click to preview) and the ⋯.
    col_check, col_body = st.columns([0.5, 5.25], vertical_alignment="center")
    with col_check:
        # value= seeds the widget only on the run that creates it, which
        # is exactly what is needed when a folded folder is opened again
        # and the key is gone. remember() then syncs the store from the
        # widget, so a click here is what survives.
        st.checkbox(f"Include {doc.filename}",
                    value=selection.is_selected(doc.doc_id),
                    key=f"sel_{doc.doc_id}",
                    label_visibility="collapsed")
        selection.remember(doc.doc_id)
    with col_body:
        with st.container(key=f"rowbox_doc_{doc.doc_id}", border=True):
            col_view, col_menu = st.columns([4, 1],
                                            vertical_alignment="center",
                                            gap="small")
            with col_view:
                if st.button(f"{icon} {doc.filename}",
                             key=f"view_{doc.doc_id}", width="stretch"):
                    show_key = f"show_md_{doc.doc_id}"
                    st.session_state[show_key] = not st.session_state.get(
                        show_key, False)
            with col_menu:
                _document_menu(svc, doc, all_folders)

    # Outside the columns, as they were in the flat list: the viewer needs
    # the full width, not a fifth of it.
    if doc.error:
        err_col, retry_col = st.columns([4, 1])
        with err_col:
            st.caption(f"↳ {doc.error}")
        if doc.status.value == "failed":
            with retry_col:
                if st.button("🔄", key=f"retry_{doc.doc_id}",
                             help=f"Retry ingesting {doc.filename}"):
                    # Failed documents are never archived - the original
                    # is still sitting in inbox, so requeuing alone is
                    # enough to retry.
                    svc["registry"].requeue(doc.doc_id)
                    st.rerun()

    if doc.status.value == "done" and st.session_state.get(f"show_md_{doc.doc_id}"):
        _render_viewer(svc, doc)


def _render_folder_controls(svc, all_folders) -> None:
    """The New folder button, under the list."""
    st.divider()
    with st.popover("➕  New folder", use_container_width=True):
        with st.form("new_folder", clear_on_submit=True):
            name = st.text_input("Folder name", placeholder="e.g. Acme Corp")
            submitted = st.form_submit_button("Create",
                                              use_container_width=True)
        if submitted:
            try:
                clean = folders.validate_name(name, all_folders)
            except folders.FolderNameError as exc:
                st.error(str(exc))
            else:
                svc["registry"].create_folder(clean)
                st.rerun()


def _render_viewer(svc, doc) -> None:
    """The converted document inline, plus the original to take away.

    Citations open their own reader page in a new tab instead (see
    render_document_page in ui/app.py); this is the panel's own preview,
    for browsing what has been indexed.

    The original is a download rather than a direct open: the app runs in a
    container with no desktop and no access to the host's file
    associations, so anything that shells out to `open` can only fail
    silently. A download hands the real file to the browser, which does
    have a PDF viewer.
    """
    original = svc["storage"].archived_path(doc.filename, doc.doc_id)
    if original.exists():
        st.download_button(
            "⬇ Download the original file",
            data=original.read_bytes(),
            file_name=doc.filename,
            key=f"dl_{doc.doc_id}",
            help="Open it in your own PDF or spreadsheet viewer",
        )
    else:
        st.caption("Original file not found — showing the converted text only")

    markdown = svc["storage"].read_markdown(doc.doc_id)
    st.markdown(markdown or "_Not yet converted_")
