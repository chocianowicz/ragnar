"""The selection store: which documents are ticked.

It exists because Streamlit deletes a widget's session-state key when the
widget is not rendered, and a folded folder renders no checkboxes. These
tests drive st.session_state directly, which works outside a script run.
"""
import pytest
import streamlit as st

from core.models import Document
from ui import selection


@pytest.fixture(autouse=True)
def clean_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _docs(*doc_ids):
    return [Document(doc_id=d, filename=f"{d}.pdf") for d in doc_ids]


def test_an_unseen_document_is_selected():
    """The flat list always included a document it had not been told
    about; uploading one and having it excluded would be the surprise."""
    assert selection.is_selected("never-seen") is True


def test_set_selected_writes_both_the_store_and_the_widget():
    selection.set_selected("a", False)

    assert selection.is_selected("a") is False
    # The widget key too, so a checkbox already on screen redraws right.
    assert st.session_state["sel_a"] is False


def test_remember_copies_a_widget_value_into_the_store():
    st.session_state["sel_a"] = False

    selection.remember("a")

    assert selection.is_selected("a") is False


def test_remember_ignores_a_document_with_no_widget():
    selection.remember("a")

    assert selection.is_selected("a") is True


def test_a_folded_folder_keeps_its_unticked_documents():
    """The bug this module exists to prevent.

    Untick a document, fold its folder away — Streamlit drops the widget
    key — and the document must stay unticked. Reading the widget key
    directly would default it back to True and silently return the
    document to the search.
    """
    selection.set_selected("a", False)

    del st.session_state["sel_a"]          # what folding does

    assert selection.is_selected("a") is False
    assert selection.selected_ids(_docs("a", "b")) == {"b"}


def test_unfolding_seeds_the_checkbox_from_the_store():
    """On the run that recreates the widget its key is absent, so
    st.checkbox(value=...) is honoured — that value comes from here."""
    selection.set_selected("a", False)
    del st.session_state["sel_a"]

    assert selection.is_selected("a") is False


def test_selected_ids_only_names_documents_it_was_given():
    selection.set_selected("gone", True)

    assert selection.selected_ids(_docs("a")) == {"a"}


def test_a_click_survives_the_rerun_that_follows_it():
    """Streamlit writes the new value into the widget key before the
    rerun; remember() then has to take it, not overwrite it."""
    selection.set_selected("a", True)

    st.session_state["sel_a"] = False      # the click
    selection.remember("a")

    assert selection.is_selected("a") is False
