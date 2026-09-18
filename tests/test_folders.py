import pytest

from core.models import Document, Folder
from ui.folders import (
    UNFILED, FolderNameError, validate_name, folder_state,
    group_for_display, scope_from_selection, selection_from_scope,
)


def _folders(*names):
    return [Folder(folder_id=f"id-{n}", name=n, created_at=0.0)
            for n in names]


def _docs(*specs):
    """specs are (doc_id, folder_id) pairs."""
    return [Document(doc_id=d, filename=f"{d}.pdf", folder_id=f)
            for d, f in specs]


# --- names ---------------------------------------------------------------

def test_a_valid_name_is_trimmed():
    assert validate_name("  Acme Corp  ", _folders()) == "Acme Corp"


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_an_empty_name_is_rejected(name):
    with pytest.raises(FolderNameError):
        validate_name(name, _folders())


def test_an_overlong_name_is_rejected():
    with pytest.raises(FolderNameError):
        validate_name("x" * 61, _folders())


def test_a_name_of_exactly_the_limit_is_allowed():
    assert validate_name("x" * 60, _folders()) == "x" * 60


def test_a_duplicate_is_rejected_regardless_of_case():
    with pytest.raises(FolderNameError, match="already"):
        validate_name("acme corp", _folders("Acme Corp"))


@pytest.mark.parametrize("name", ["Unfiled", "unfiled", "UNFILED"])
def test_unfiled_is_reserved(name):
    """It renders folder_id IS NULL, so a real folder of that name would
    show two identical rows with different contents."""
    with pytest.raises(FolderNameError, match="reserved"):
        validate_name(name, _folders())


def test_renaming_a_folder_to_its_own_name_is_allowed():
    existing = _folders("Acme Corp")
    assert validate_name("Acme Corp", existing,
                         allow_id="id-Acme Corp") == "Acme Corp"


# --- checkbox state ------------------------------------------------------

def test_folder_state_all_checked():
    docs = _docs(("a", "f1"), ("b", "f1"))
    assert folder_state(docs, {"a", "b"}) == (True, "2/2")


def test_folder_state_some_checked():
    """Streamlit has no indeterminate checkbox, so the count carries it."""
    docs = _docs(("a", "f1"), ("b", "f1"), ("c", "f1"))
    assert folder_state(docs, {"a"}) == (False, "1/3")


def test_folder_state_none_checked():
    docs = _docs(("a", "f1"), ("b", "f1"))
    assert folder_state(docs, set()) == (False, "0/2")


def test_an_empty_folder_is_vacuously_checked():
    """It contributes no ids, so it cannot change an answer either way."""
    assert folder_state([], set()) == (True, "0/0")


# --- display order -------------------------------------------------------

def test_unfiled_is_drawn_last():
    docs = _docs(("a", "id-Acme"), ("b", None))
    rows = group_for_display(docs, _folders("Acme"))
    assert [label for _, label, _ in rows] == ["Acme", "Unfiled"]


def test_an_empty_folder_is_still_drawn():
    """A folder just created has to be visible to file into."""
    rows = group_for_display([], _folders("Acme"))
    assert [label for _, label, _ in rows] == ["Acme", "Unfiled"]


def test_documents_land_under_their_folder():
    docs = _docs(("a", "id-Acme"), ("b", None))
    rows = group_for_display(docs, _folders("Acme"))
    assert [d.doc_id for d in rows[0][2]] == ["a"]
    assert [d.doc_id for d in rows[1][2]] == ["b"]


# --- saving a scope ------------------------------------------------------

def test_everything_checked_saves_as_all():
    docs = _docs(("a", "f1"), ("b", None))
    assert scope_from_selection(docs, {"a", "b"}) == {"v": 1, "all": True}


def test_a_whole_folder_saves_as_its_id():
    docs = _docs(("a", "f1"), ("b", "f1"), ("c", "f2"))
    scope = scope_from_selection(docs, {"a", "b"})
    assert scope == {"v": 1, "folders": ["f1"], "docs": []}


def test_unfiled_saves_as_the_sentinel():
    docs = _docs(("a", None), ("b", None), ("c", "f1"))
    scope = scope_from_selection(docs, {"a", "b"})
    assert scope == {"v": 1, "folders": [UNFILED], "docs": []}


def test_a_partial_folder_saves_its_documents():
    docs = _docs(("a", "f1"), ("b", "f1"))
    scope = scope_from_selection(docs, {"a"})
    assert scope == {"v": 1, "folders": [], "docs": ["a"]}


def test_nothing_checked_saves_as_an_empty_scope():
    docs = _docs(("a", "f1"))
    assert scope_from_selection(docs, set()) == {"v": 1, "folders": [],
                                                 "docs": []}


# --- restoring a scope ---------------------------------------------------

def test_all_restores_everything():
    docs = _docs(("a", "f1"), ("b", None))
    assert selection_from_scope({"v": 1, "all": True}, docs) == {"a", "b"}


def test_no_scope_restores_everything():
    """A chat saved before this feature had no scope, and meant everything."""
    docs = _docs(("a", "f1"), ("b", None))
    assert selection_from_scope(None, docs) == {"a", "b"}


def test_an_empty_scope_restores_nothing():
    """The regression that would reverse the refusal guarantee.

    ui/app.py sends an explicit empty doc_ids list, and store.py treats it
    as "refuse" rather than "search everything". An empty saved scope has
    to keep meaning exactly that.
    """
    docs = _docs(("a", "f1"), ("b", None))
    scope = {"v": 1, "folders": [], "docs": []}
    assert selection_from_scope(scope, docs) == set()


def test_a_folder_that_gained_a_document_restores_with_it():
    """Membership is resolved at restore time, not frozen at save time."""
    scope = {"v": 1, "folders": ["f1"], "docs": []}
    docs = _docs(("a", "f1"), ("new", "f1"))
    assert selection_from_scope(scope, docs) == {"a", "new"}


def test_unfiled_gains_documents_the_same_way():
    """Unfiled is a peer of every other folder, so it must behave like one."""
    scope = {"v": 1, "folders": [UNFILED], "docs": []}
    docs = _docs(("a", None), ("new", None), ("c", "f1"))
    assert selection_from_scope(scope, docs) == {"a", "new"}


def test_a_deleted_folder_is_dropped_and_the_rest_kept():
    scope = {"v": 1, "folders": ["gone", "f1"], "docs": ["also-gone"]}
    docs = _docs(("a", "f1"))
    assert selection_from_scope(scope, docs) == {"a"}


def test_a_scope_survives_a_round_trip():
    docs = _docs(("a", "f1"), ("b", "f1"), ("c", None))
    checked = {"a", "b"}

    restored = selection_from_scope(scope_from_selection(docs, checked), docs)

    assert restored == checked
