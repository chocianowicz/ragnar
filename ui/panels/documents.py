import streamlit as st

from ui.services import format_eta
from ui import folders, sources

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
            st.session_state[f"sel_{d.doc_id}"] = value

    if docs:
        st.checkbox("Select all", value=True, key="select_all_docs",
                    on_change=_select_all_changed)
        st.caption(
            "Unchecked documents and folders are excluded from answers — "
            "only checked ones are searched."
        )

    checked_ids = {
        d.doc_id for d in docs
        if st.session_state.get(f"sel_{d.doc_id}", True)
    }
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
            st.session_state[f"sel_{d.doc_id}"] = value

    # Streamlit ignores value= once a widget's key is in session state, so
    # a derived checkbox has to be written INTO session state before it
    # renders. Without this the folder box freezes at whatever it showed
    # first while the count moves beneath it.
    st.session_state[f"folder_sel_{folder_id}"] = is_checked

    col_check, col_name, col_count = st.columns([0.6, 4.4, 1])
    with col_check:
        st.checkbox(f"Include {label}", key=f"folder_sel_{folder_id}",
                    on_change=_folder_changed, label_visibility="collapsed")
    with col_name:
        st.markdown(f"**{label}**")
    with col_count:
        st.caption(count)

    if not group:
        st.caption("&nbsp;&nbsp;&nbsp;&nbsp;_empty_", unsafe_allow_html=True)
    for doc in group:
        _render_document(svc, doc, all_folders)


NEW_FOLDER = "New folder…"


def _folder_picker(svc, doc, all_folders) -> None:
    """The control that files a document.

    Includes a New folder… entry so the first document can be filed
    without hunting for the panel button first — on a fresh install there
    are no folders to pick from at all.
    """
    names = [folders.UNFILED_LABEL] + [f.name for f in all_folders]
    names.append(NEW_FOLDER)
    current = next((f.name for f in all_folders
                    if f.folder_id == doc.folder_id), folders.UNFILED_LABEL)
    picked = st.selectbox(
        f"Folder for {doc.filename}", names, index=names.index(current),
        key=f"folder_of_{doc.doc_id}", label_visibility="collapsed",
    )
    if picked == current:
        return
    if picked == NEW_FOLDER:
        st.session_state["new_folder_for"] = doc.doc_id
        del st.session_state[f"folder_of_{doc.doc_id}"]
        st.rerun()
    target = next((f.folder_id for f in all_folders if f.name == picked),
                  None)
    svc["registry"].set_folder(doc.doc_id, target)
    st.rerun()


def _render_document(svc, doc, all_folders) -> None:
    icon = STATUS_ICONS[doc.status.value]

    col_check, col_view, col_folder, col_remove = st.columns(
        [0.6, 3.0, 1.4, 1])
    with col_check:
        st.checkbox(f"Include {doc.filename}", value=True,
                    key=f"sel_{doc.doc_id}",
                    label_visibility="collapsed")
    with col_view:
        if st.button(f"{icon} {doc.filename}", key=f"view_{doc.doc_id}",
                     use_container_width=True):
            show_key = f"show_md_{doc.doc_id}"
            st.session_state[show_key] = not st.session_state.get(show_key, False)
    with col_folder:
        _folder_picker(svc, doc, all_folders)
    with col_remove:
        with st.container(key=f"remove_container_{doc.doc_id}"):
            if st.button("✕", key=f"rm_{doc.doc_id}",
                         help=f"Remove {doc.filename}"):
                svc["store"].delete_by_doc(doc.doc_id)
                svc["storage"].remove_converted(doc.doc_id)
                # Also drop the served copy, or the file stays
                # downloadable by URL after the delete button says it
                # is gone.
                sources.unpublish(doc.doc_id, doc.filename)
                svc["registry"].remove(doc.doc_id)
                st.rerun()

    # Outside the columns, as they were in the flat list: the viewer needs
    # the full width, not a quarter of it.
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


def _forget_pickers() -> None:
    """Drop every row picker's remembered value.

    A selectbox with a key redisplays its stored VALUE, not its index, and
    Streamlit validates that value against the current options. Renaming or
    deleting a folder changes the options, so a picker still holding the
    old name would raise or snap to the wrong entry. Clearing the keys
    makes every picker re-derive from the document's folder_id, which is
    the truth.
    """
    for key in [k for k in st.session_state if k.startswith("folder_of_")]:
        del st.session_state[key]


def _render_folder_controls(svc, all_folders) -> None:
    """New / Rename / Delete, under the list.

    Rename and Delete act on a chosen folder rather than on ticked rows:
    the checkboxes mean "search this", and reusing them to mean "act on
    this" is exactly the overloading this design avoids.
    """
    st.divider()
    pending = st.session_state.get("new_folder_for")
    if pending:
        st.caption("Name the new folder — the document moves into it.")

    with st.form("new_folder", clear_on_submit=True):
        cols = st.columns([4, 1])
        with cols[0]:
            name = st.text_input("New folder", placeholder="e.g. Acme Corp",
                                 label_visibility="collapsed")
        with cols[1]:
            submitted = st.form_submit_button("Add", use_container_width=True)
    if submitted:
        try:
            clean = folders.validate_name(name, all_folders)
        except folders.FolderNameError as exc:
            st.error(str(exc))
        else:
            new_id = svc["registry"].create_folder(clean)
            if pending:
                svc["registry"].set_folder(pending, new_id)
                del st.session_state["new_folder_for"]
            st.rerun()

    if not all_folders:
        return

    names = [f.name for f in all_folders]
    chosen = st.selectbox("Folder to rename or delete", names,
                          key="folder_admin_target")
    target = next(f for f in all_folders if f.name == chosen)

    cols = st.columns([3, 1, 1])
    with cols[0]:
        new_name = st.text_input("Rename to", value=target.name,
                                 key=f"rename_{target.folder_id}",
                                 label_visibility="collapsed")
    with cols[1]:
        if st.button("Rename", use_container_width=True):
            try:
                clean = folders.validate_name(new_name, all_folders,
                                              allow_id=target.folder_id)
            except folders.FolderNameError as exc:
                st.error(str(exc))
            else:
                svc["registry"].rename_folder(target.folder_id, clean)
                _forget_pickers()
                st.rerun()
    with cols[2]:
        held = sum(1 for d in svc["registry"].all()
                   if d.folder_id == target.folder_id)
        if st.button("Delete", use_container_width=True,
                     help=f"{held} document(s) move to Unfiled"):
            svc["registry"].delete_folder(target.folder_id)
            _forget_pickers()
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
