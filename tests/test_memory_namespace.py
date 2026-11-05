"""Explicit target identity and legacy-outbox guards; no database/model calls."""
import asyncio
import ctypes
import ctypes.util
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from core.memory_commits import MemoryCommitConflict, MemoryOutbox, memory_payload, reconcile_memory
from core.postgres_state_store import PostgresStateStore
from core.shared_memory import MemoryTrace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
from engine_stdio import _outbox_for_store
from test_memory_commits import FaultStore


BASE = "host=db.example port=5432 dbname=memory user=alice password='PRIVATE_PASSWORD'"


@pytest.fixture(autouse=True)
def local_conninfo_parser(monkeypatch):
    try:
        import psycopg
    except ImportError:
        # Exercise libpq's real DSN/URI parser without requiring a server or
        # psycopg. Only this optional-dependency adapter is replaced locally.
        library = ctypes.util.find_library("pq")
        if library is None:
            pytest.skip("psycopg or libpq is needed for connection-string parsing")
        libpq = ctypes.CDLL(library)

        class Option(ctypes.Structure):
            _fields_ = [(key, ctypes.c_char_p) for key in
                        ("keyword", "envvar", "compiled", "val", "label", "dispchar")] + [
                            ("dispsize", ctypes.c_int)]

        libpq.PQconninfoParse.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        libpq.PQconninfoParse.restype = ctypes.POINTER(Option)
        libpq.PQconninfoFree.argtypes = [ctypes.POINTER(Option)]
        libpq.PQconninfoFree.restype = None
        libpq.PQfreemem.argtypes = [ctypes.c_void_p]
        libpq.PQfreemem.restype = None

        def parse(dsn):
            error = ctypes.c_char_p()
            options = libpq.PQconninfoParse(dsn.encode(), ctypes.byref(error))
            if not options:
                if error:
                    libpq.PQfreemem(error)
                raise ValueError("Invalid test connection string")
            result, index = {}, 0
            try:
                while options[index].keyword:
                    option = options[index]
                    if option.val is not None:
                        result[option.keyword.decode()] = option.val.decode()
                    index += 1
            finally:
                libpq.PQconninfoFree(options)
            return result

        psycopg = SimpleNamespace(conninfo=SimpleNamespace(conninfo_to_dict=parse))
    monkeypatch.setattr(PostgresStateStore, "_require_psycopg", lambda self: psycopg)


def store(options=None, *, password="PRIVATE_PASSWORD", prefix="probe"):
    dsn = BASE.replace("PRIVATE_PASSWORD", password)
    if options is not None:
        dsn += f" options='{options}'"
    return PostgresStateStore(dsn, table_prefix=prefix)


def bound_fake(target):
    fake = FaultStore()
    fake.memory_namespace = target.memory_namespace
    fake.memory_legacy_namespaces = target.memory_legacy_namespaces
    return fake


def payload(answer="Source A"):
    return memory_payload([MemoryTrace("Project Alpha source", answer, qid="source",
                                      retrieval_scope="Alpha", scope_aliases=("Alpha",))], [])


def test_no_options_preserves_legacy_hash_and_password_rotation():
    original = store()
    assert original.memory_namespace == "5187962fb4e2d6a655380e03b9f371bb53f931f9dccc1ac396e1a9d0c1b95487"
    assert original.memory_namespace == store("").memory_namespace == store(password="ROTATED_PASSWORD").memory_namespace
    assert original.memory_legacy_namespaces == ()
    assert original.memory_namespace != store(prefix="another").memory_namespace


def test_explicit_options_bind_schema_and_keep_password_out_of_identity():
    a, b = store("-c search_path=alpha"), store("-c search_path=beta")
    assert a.memory_namespace != b.memory_namespace != store().memory_namespace
    assert a.memory_namespace == store("-c search_path=alpha", password="ROTATED_PASSWORD").memory_namespace
    assert a.memory_legacy_namespaces == b.memory_legacy_namespaces == (store().memory_namespace,)
    # Exact options strings intentionally remain distinct, even for equivalent
    # spellings: this is target binding, not a PostgreSQL schema interpreter.
    assert a.memory_namespace != store("-csearch_path=alpha").memory_namespace


def test_keyword_and_uri_options_use_the_same_decoded_target_identity():
    uri = "postgresql://alice:DIFFERENT_PASSWORD@db.example:5432/memory?options=-c%20search_path%3Dalpha"
    target = PostgresStateStore(uri, table_prefix="probe")
    assert target.memory_namespace == store("-c search_path=alpha").memory_namespace
    assert target.memory_legacy_namespaces == store("-c search_path=alpha").memory_legacy_namespaces


def test_connection_receives_the_same_explicit_options_bound_by_the_namespace(monkeypatch):
    target = store("-c search_path=alpha")
    calls = []
    connection = object()

    def connect(dsn):
        calls.append(dsn)
        return connection

    adapter = SimpleNamespace(connect=connect, conninfo=target._require_psycopg().conninfo)
    monkeypatch.setattr(target, "_require_psycopg", lambda: adapter)
    assert target._connect() is connection and calls == [target.dsn]
    assert target.memory_namespace != store("-c search_path=beta").memory_namespace


def test_same_file_session_and_operation_stay_separate_across_targets(tmp_path, monkeypatch):
    path = tmp_path / "outbox.sqlite3"
    monkeypatch.setenv("DUALBRAIN_MEMORY_OUTBOX_PATH", str(path))
    a, b = bound_fake(store("-c search_path=alpha")), bound_fake(store("-c search_path=beta"))
    first = _outbox_for_store(a)
    first.prepare("s", "operation", payload(), request={"question": "A"}, result={"answer": "Source A"})
    other = _outbox_for_store(b)
    assert other.get("s", "operation") is None and other.entries("s", pending_only=True) == []
    assert asyncio.run(reconcile_memory(other, b, "s")) == [] and b.applied == 0
    with pytest.raises(MemoryCommitConflict, match="target does not match"):
        asyncio.run(reconcile_memory(first, b, "s"))
    assert b.applied == 0

    other.prepare("s", "operation", payload("Source B"), result={"answer": "Source B"})
    asyncio.run(reconcile_memory(other, b, "s"))
    restarted = _outbox_for_store(a)
    receipt = asyncio.run(reconcile_memory(restarted, a, "s"))[0]
    assert [trace.answer for trace in a.traces] == ["Source A"]
    assert [trace.answer for trace in b.traces] == ["Source B"]
    assert restarted.get("s", "operation")["result"] == {"answer": "Source A"}
    assert asyncio.run(reconcile_memory(_outbox_for_store(a), a, "s", operation_id="operation")) == [receipt]
    assert a.applied == b.applied == 1
    assert b"search_path" not in path.read_bytes() and b"PRIVATE_PASSWORD" not in path.read_bytes()


@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("operation", ["get", "entries", "prepare", "acknowledge", "reconcile"])
def test_legacy_rows_block_recovery_and_new_writes_without_mutating_either_namespace(
        tmp_path, monkeypatch, acknowledged, operation):
    path = tmp_path / "outbox.sqlite3"
    monkeypatch.setenv("DUALBRAIN_MEMORY_OUTBOX_PATH", str(path))
    target = bound_fake(store("-c search_path=alpha"))
    guarded = _outbox_for_store(target)
    # Add an old entry after opening the guarded object, proving the guard runs
    # for each operation rather than only when the file is opened.
    legacy = MemoryOutbox(path, store().memory_namespace)
    entry = legacy.prepare("s", "old", payload(), result={"answer": "Original saved response"})
    receipt = {"operation_id": "old", "payload_sha256": entry["payload_sha256"], "state": "committed"}
    if acknowledged:
        legacy.acknowledge("s", "old", receipt)
    before = legacy.entries("s")
    actions = {
        "get": lambda: guarded.get("s", "old"),
        "entries": lambda: guarded.entries("s", pending_only=True),
        "prepare": lambda: guarded.prepare("s", "new", payload("New")),
        "acknowledge": lambda: guarded.acknowledge("s", "old", receipt),
        "reconcile": lambda: asyncio.run(reconcile_memory(guarded, target, "s")),
    }
    with pytest.raises(MemoryCommitConflict, match="Legacy outbox target is ambiguous") as caught:
        actions[operation]()
    assert "search_path" not in str(caught.value) and "PRIVATE_PASSWORD" not in str(caught.value)
    assert legacy.entries("s") == before and target.applied == 0
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM memory_outbox WHERE namespace=?", (target.memory_namespace,)).fetchone()[0] == 0
    # An unrelated session may coexist without guessing the old session's target.
    guarded.prepare("other-session", "fresh", payload("Fresh"))
    assert len(guarded.entries("other-session")) == 1


def test_no_options_legacy_response_remains_readable(tmp_path, monkeypatch):
    path = tmp_path / "outbox.sqlite3"
    monkeypatch.setenv("DUALBRAIN_MEMORY_OUTBOX_PATH", str(path))
    legacy = MemoryOutbox(path, store().memory_namespace)
    old = legacy.prepare("s", "old", payload(), result={"answer": "Original saved response"})
    target = bound_fake(store(password="ROTATED_PASSWORD"))
    assert _outbox_for_store(target).get("s", "old") == old


@pytest.mark.parametrize("acknowledged", [False, True])
def test_reconcile_enforces_store_legacy_guards_with_default_outbox_constructor(tmp_path, acknowledged):
    path = tmp_path / "outbox.sqlite3"
    legacy = MemoryOutbox(path, store().memory_namespace)
    entry = legacy.prepare("s", "old", payload(), result={"answer": "Original response"})
    if acknowledged:
        legacy.acknowledge("s", "old", {"operation_id": "old", "state": "committed",
                                       "payload_sha256": entry["payload_sha256"]})
    before = legacy.entries("s")
    target = bound_fake(store("-c search_path=alpha"))
    direct = MemoryOutbox(path, target.memory_namespace)
    assert direct.legacy_namespaces == ()
    with pytest.raises(MemoryCommitConflict, match="Legacy outbox target is ambiguous"):
        asyncio.run(reconcile_memory(direct, target, "s"))
    assert target.applied == 0 and legacy.entries("s") == before
    # A failed binding keeps the required guard, preventing subsequent code
    # from treating the old response as absent and preparing a replacement.
    with pytest.raises(MemoryCommitConflict, match="Legacy outbox target is ambiguous"):
        direct.get("s", "old")
    with pytest.raises(MemoryCommitConflict, match="Legacy outbox target is ambiguous"):
        direct.prepare("s", "new", payload("Replacement"))
    assert legacy.entries("s") == before


def test_default_outbox_constructor_can_reconcile_a_fresh_bound_target(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    target = bound_fake(store("-c search_path=alpha"))
    direct = MemoryOutbox(path, target.memory_namespace)
    direct.prepare("s", "fresh", payload())
    receipt = asyncio.run(reconcile_memory(direct, target, "s"))[0]
    assert direct.legacy_namespaces == target.memory_legacy_namespaces
    restarted = MemoryOutbox(path, target.memory_namespace)
    assert asyncio.run(reconcile_memory(restarted, target, "s", operation_id="fresh")) == [receipt]
    assert target.applied == 1
