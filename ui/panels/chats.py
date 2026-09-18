import streamlit as st

from ui import folders


def render(svc, jobs=None) -> None:
    with st.expander("Chats", expanded=False):
        # Conversations auto-save as they happen; this panel is for revisiting
        # and managing them. "New chat" just clears the working conversation -
        # the previous one is already persisted, so nothing is lost.
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()

        saved_chats = svc["chats"].all()
        if not saved_chats:
            st.caption("No saved chats yet — ask a question to start one.")
        elif jobs is not None and jobs.any_running():
            st.caption("⏳ marks a conversation still working on an answer.")

        for chat in saved_chats:
            is_current = chat.chat_id == st.session_state.get("current_chat_id")
            col_open, col_del = st.columns([5, 1])
            thinking = jobs is not None and jobs.running(chat.chat_id)
            with col_open:
                # An answer can be running in a conversation you are not
                # looking at, so the list has to say which ones are still
                # working — otherwise the only way to find out is to open
                # each one.
                marker = "⏳ " if thinking else ("▸ " if is_current else "")
                label = marker + chat.title
                if st.button(label, key=f"open_chat_{chat.chat_id}",
                             use_container_width=True):
                    st.session_state.messages = chat.messages
                    st.session_state.current_chat_id = chat.chat_id
                    # Restore the folders this conversation was asked
                    # under. Resolved against the documents that exist
                    # now, so a folder picks up what was added to it
                    # since — and an empty scope stays empty, which is a
                    # refusal rather than a licence to search everything.
                    docs = svc["registry"].all()
                    ticked = folders.selection_from_scope(chat.scope, docs)
                    for doc in docs:
                        st.session_state[f"sel_{doc.doc_id}"] = (
                            doc.doc_id in ticked)
                    st.rerun()
            with col_del:
                with st.container(key=f"del_chat_container_{chat.chat_id}"):
                    if st.button("✕", key=f"del_chat_{chat.chat_id}",
                                 help="Delete this chat",
                                 disabled=thinking):
                        svc["chats"].delete(chat.chat_id)
                        if is_current:
                            st.session_state.messages = []
                            st.session_state.current_chat_id = None
                        st.rerun()
