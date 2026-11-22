"""Per-visitor sandboxes: identities, what each visitor can see, and the reset."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentkit import auth
from agentkit.contract import AgentError
from agentkit.files import FileStore
from agentkit.sandbox import new_sandbox, persona, sandbox_of, scoped, valid_sandbox, visible_sandboxes
from agentkit.vectors import LocalStore
from conftest import FakeRedis

A, B = "aaaa1111bbbb", "cccc2222dddd"


def test_scoped_identities():
    assert scoped("asha", A) == f"asha~{A}" and scoped("asha", None) == "asha"
    assert sandbox_of(f"asha~{A}") == A and sandbox_of("asha") == ""
    assert persona(f"ravi~{A}") == "ravi"
    assert visible_sandboxes(f"asha~{A}") == {"", A}
    assert valid_sandbox(new_sandbox())
    for bad in ("../x", "AB", "a" * 40, "abc def1"):
        assert not valid_sandbox(bad)
        with pytest.raises(ValueError):
            scoped("asha", bad)
    with pytest.raises(ValueError):
        sandbox_of("asha~../../etc")


def test_vector_search_sees_global_and_own_sandbox_only(tmp_path):
    store = LocalStore(root=tmp_path)
    vec = np.ones((1, 4), dtype=np.float32) / 2
    store.upsert("c", ["g:0"], vec, [{"doc_id": "g", "text": "sample"}])  # global: no sandbox field
    store.upsert("c", ["a:0"], vec, [{"doc_id": "a", "text": "A's", "sandbox": A}])
    store.upsert("c", ["b:0"], vec, [{"doc_id": "b", "text": "B's", "sandbox": B}])
    seen = {h.payload["doc_id"] for h in store.search("c", vec[0], top_k=10, visible={"", A})}
    assert seen == {"g", "a"}
    assert {h.payload["doc_id"] for h in store.search("c", vec[0], top_k=10, visible={""})} == {"g"}
    assert len(store.search("c", vec[0], top_k=10)) == 3  # no filter: internal callers


def test_reingesting_right_after_a_delete_does_not_duplicate_chunks(tmp_path):
    store = LocalStore(root=tmp_path)
    vecs = np.ones((2, 4), dtype=np.float32) / 2
    store.upsert("c", ["d:0", "d:1"], vecs, [{"doc_id": "d", "text": "old"}] * 2)
    store.delete_doc("c", "d")  # what ingestion does, milliseconds apart
    store.upsert("c", ["d:0"], vecs[:1], [{"doc_id": "d", "text": "new"}])
    assert [c["text"] for c in store.chunks("c", "d")] == ["new"]


@pytest.fixture
def kb(monkeypatch):
    """Fake Redis behind the KB catalog, and a clean vector collection."""
    import agents.kb_common as kb_common
    from agentkit.vectors import get_store

    r = FakeRedis()
    monkeypatch.setattr(kb_common, "get_redis", lambda: r)
    store = get_store()
    for doc_id in ("g", "a", "b"):
        store.delete_doc(kb_common.COLLECTION, doc_id)
    return r, store


def _index(r, store, doc_id, sandbox, file_id, texts):
    from agents.kb_common import COLLECTION, save_doc

    payloads = [{"doc_id": doc_id, "title": doc_id.upper(), "file_id": file_id, "page": 1, "heading": "h",
                 "text": t, "order": i, **({"sandbox": sandbox} if sandbox else {})} for i, t in enumerate(texts)]
    store.upsert(COLLECTION, [f"{doc_id}:{i}" for i in range(len(texts))], np.ones((len(texts), 4), np.float32) / 2,
                 payloads)
    save_doc(doc_id, {"title": doc_id.upper(), "file_id": file_id, "sandbox": sandbox})


def test_catalog_visibility_and_purge(kb):
    from agents.kb_common import purge_sandbox, visible_docs

    r, store = kb
    _index(r, store, "g", "", "f-g", ["global"])
    _index(r, store, "a", A, "f-a", ["mine"])
    _index(r, store, "b", B, "f-b", ["theirs"])
    assert set(visible_docs(f"asha~{A}")) == {"g", "a"}
    assert set(visible_docs("asha")) == {"g"}
    assert purge_sandbox(A) == 1
    assert set(visible_docs(f"asha~{A}")) == {"g"} and set(visible_docs(f"asha~{B}")) == {"g", "b"}
    assert store.chunks("manuals", "a") == [] and store.chunks("manuals", "b")
    with pytest.raises(ValueError):
        purge_sandbox("")  # the samples are never purged


class Ctx:
    def __init__(self, user, files):
        self.user, self.files = user, files


def test_whole_document_mode_returns_reading_order_within_a_budget(kb, monkeypatch):
    from agents.kb_extraction import app as kbe

    r, store = kb
    files = FileStore(redis_client=FakeRedis())
    mine = files.put(f"asha~{A}", "mine.pdf", b"x")
    theirs = files.put(f"asha~{B}", "theirs.pdf", b"y")
    _index(r, store, "a", A, mine.file_id, ["one " * 10, "two " * 10, "three " * 10])
    monkeypatch.setattr(kbe, "DOCUMENT_CHARS", 90)
    chunks = kbe.whole_document(mine.file_id, Ctx(f"asha~{A}", files))
    assert [c["order"] for c in chunks] == [0, 1]  # in order, and the third would exceed the budget
    unindexed = files.put(f"asha~{A}", "new.pdf", b"z")
    assert kbe.whole_document(unindexed.file_id, Ctx(f"asha~{A}", files)) is None
    with pytest.raises(AgentError):
        kbe.whole_document(theirs.file_id, Ctx(f"asha~{A}", files))  # another visitor's file


def test_catalog_endpoint_lists_only_what_the_caller_can_search(kb):
    from agents.kb_extraction.app import app

    r, store = kb
    _index(r, store, "g", "", "f-g", ["global"])
    _index(r, store, "a", A, "f-a", ["mine"])
    _index(r, store, "b", B, "f-b", ["theirs"])
    token = auth.issue(f"ravi~{A}", ["viewer"])
    rows = TestClient(app).get("/catalog", headers={"Authorization": f"Bearer {token}"}).json()
    assert [(d["doc_id"], d["shared_sample"], d["added_by_you"]) for d in rows] == [("g", True, False), ("a", False, True)]
    assert TestClient(app).get("/catalog").status_code == 401


def test_choosing_a_document_scopes_the_answer_to_it():
    from agentkit.vectors import Hit
    from agents.kb_extraction.app import _resolve_choice, ambiguous_equipment, scope_hits

    hits = [Hit(0.61, {"doc_id": "doc-policy", "title": "Travel policy", "equipment": [], "text": "x"}),
            Hit(0.60, {"doc_id": "doc-guide", "title": "Expense guide", "equipment": [], "text": "y"})]
    options = ambiguous_equipment(hits)
    assert [o["title"] for o in options] == ["Travel policy", "Expense guide"]
    chosen = _resolve_choice("the expense guide", options)
    assert [h.payload["doc_id"] for h in scope_hits(hits, set(chosen))] == ["doc-guide"]


def test_ingestion_ids_never_collide_across_sandboxes():
    from agents.kb_ingestion.app import doc_ids

    g0, d0 = doc_ids(b"pdf", "")
    assert g0 == d0  # the global index uses the content id
    g, a = doc_ids(b"pdf", A)
    _, b = doc_ids(b"pdf", B)
    assert g == doc_ids(b"pdf", "")[0] and len({g, a, b}) == 3


def test_each_sandbox_publishes_to_its_own_library(monkeypatch):
    from agents.docs_agent import tools

    monkeypatch.delenv("DOCS_SIMULATE_GRAPH_FAILURE", raising=False)
    doc_id = tools.publish("Travel policy - Document brief", None, [], "# body\n", sandbox=A)
    assert doc_id == "TP-901"  # documents about no equipment get their own range
    assert doc_id in tools.load_graph(A)["documents"]
    assert doc_id not in tools.load_graph(B)["documents"] and doc_id not in tools.load_graph()["documents"]
    assert tools.impact(["KP-200-115"], "KP-200", B)[0]["id"] == "TP-101"  # B still has the samples
    assert tools.drop_library(A) and not tools.drop_library(A)
    assert "TP-901" not in tools.load_graph(A)["documents"]  # a fresh copy of the samples


def test_reset_removes_one_sandbox_and_keeps_everything_else(kb, monkeypatch, tmp_path):
    import orchestrator.sandbox as sb
    from agents.docs_agent import tools
    from orchestrator.history import HistoryStore

    r, store = kb
    monkeypatch.setattr(sb, "get_redis", lambda: r)
    files = FileStore(root=tmp_path / "files", redis_client=FakeRedis())
    history = HistoryStore(root=tmp_path / "history")
    up = files.put(f"asha~{A}", "kv80.pdf", b"x")
    keep_b = files.put(f"asha~{B}", "other.pdf", b"y")
    keep_lib = files.put("library", "manual.pdf", b"z")
    r.set(f"pi:tree:{up.file_id}", "[]")
    history.create(f"asha~{A}", "hello")
    history.create(f"ravi~{A}", "hi")
    history.create(f"asha~{B}", "other visitor")
    _index(r, store, "a", A, up.file_id, ["mine"])
    tools.library_dir(A)
    removed = sb.reset(A, ["asha", "ravi", "admin"], files, history)
    assert removed == {"chats": 2, "files": 1, "kb_documents": 1, "library": True}
    assert not files.can_read(up.file_id, f"asha~{A}") and r.get(f"pi:tree:{up.file_id}") is None
    assert files.can_read(keep_b.file_id, f"asha~{B}") and files.can_read(keep_lib.file_id, f"asha~{A}")
    assert [c["owner"] for c in history.list_for(f"asha~{B}")] == [f"asha~{B}"]
    with pytest.raises(ValueError):
        files.purge_owner("library")


def test_login_signs_in_to_a_sandbox():
    from orchestrator.app import app

    client = TestClient(app)
    res = client.post("/api/login", json={"username": "asha", "password": "demo", "sandbox": A}).json()
    assert res["user"] == f"asha~{A}" and res["persona"] == "asha" and res["roles"] == ["author"]
    assert auth.verify(res["token"]).user == f"asha~{A}"
    assert client.post("/api/login", json={"username": "asha", "password": "demo"}).json()["user"] == "asha"
    bad = client.post("/api/login", json={"username": "asha", "password": "demo", "sandbox": "../x"})
    assert bad.status_code == 400
    local = auth.issue("asha", ["author"])
    assert client.post("/api/sandbox/reset", headers={"Authorization": f"Bearer {local}"}).status_code == 400
