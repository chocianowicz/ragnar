import streamlit as st

from ui.services import format_eta

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
                target = svc["storage"].inbox / file.name
                target.write_bytes(file.getbuffer())
                svc["registry"].add(svc["storage"].doc_id(target), file.name,
                                    target.stat().st_size)
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


@st.fragment(run_every="2s")
def _status_strip(svc) -> None:
    processing, queued, eta = svc["registry"].ingest_eta()

    if processing or queued:
        st.info(
            f"Indexing — {processing} in progress, "
            f"{queued} queued{format_eta(eta)}"
        )

    docs = svc["registry"].all()

    def _select_all_changed():
        value = st.session_state.get("select_all_docs", True)
        for d in docs:
            st.session_state[f"sel_{d.doc_id}"] = value

    if docs:
        st.checkbox("Select all", value=True, key="select_all_docs",
                    on_change=_select_all_changed)
        st.caption(
            "Unchecked documents are excluded from answers — only "
            "checked ones are searched."
        )

    for doc in docs:
        icon = STATUS_ICONS[doc.status.value]

        col_check, col_view, col_remove = st.columns([0.6, 4.4, 1])
        with col_check:
            st.checkbox(f"Include {doc.filename}", value=True,
                        key=f"sel_{doc.doc_id}",
                        label_visibility="collapsed")
        with col_view:
            if st.button(f"{icon} {doc.filename}", key=f"view_{doc.doc_id}",
                         use_container_width=True):
                show_key = f"show_md_{doc.doc_id}"
                st.session_state[show_key] = not st.session_state.get(show_key, False)
        with col_remove:
            with st.container(key=f"remove_container_{doc.doc_id}"):
                if st.button("✕", key=f"rm_{doc.doc_id}",
                             help=f"Remove {doc.filename}"):
                    svc["store"].delete_by_doc(doc.doc_id)
                    svc["storage"].remove_converted(doc.doc_id)
                    svc["registry"].remove(doc.doc_id)
                    st.rerun()

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
            markdown = svc["storage"].read_markdown(doc.doc_id)
            st.markdown(markdown or "_Not yet converted_")
