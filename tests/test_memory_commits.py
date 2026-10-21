"""Model-free recovery tests. The fake store does not establish PostgreSQL atomicity."""
import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.memory_commits import (
    MemoryCommitConflict, MemoryCommitPending, MemoryOutbox, decode_payload,
    memory_payload, payload_fingerprint, reconcile_memory,
)
from core.shared_memory import MemoryTrace
from test_conversation_memory import memory, retrieve, write
from engine_stdio import _handle_process, _handle_process_stream, _prepare_memory_request


class FaultStore:
    memory_namespace = "isolated-test-store"

    def __init__(self, traces=()):
        self.traces, self.receipts = list(traces), {}
        self.fault, self.applied = None, 0

    async def commit_memory_operation(self, sid, op_id, payload):
        fingerprint = payload_fingerprint(payload)
        previous = self.receipts.get((sid, op_id))
        if previous:
            if previous["payload_sha256"] != fingerprint:
                raise MemoryCommitConflict("different memory payload")
            return deepcopy(previous)
        if self.fault == "before":
            raise OSError("before commit")
        traces, updates = decode_payload(payload)
        candidate = list(self.traces)
        for updated in updates:
            indexes = [i for i, t in enumerate(candidate) if (t.qid, t.timestamp, t.question, t.answer)
                       == (updated.qid, updated.timestamp, updated.question, updated.answer)]
            if len(indexes) != 1 or candidate[indexes[0]].evidence_status == "superseded":
                raise MemoryCommitConflict("stale target")
            candidate[indexes[0]] = updated
        receipt = {"operation_id": op_id, "payload_sha256": fingerprint, "state": "committed",
                   "memory_ids": list(range(len(candidate) + 1, len(candidate) + len(traces) + 1)),
                   "superseded_ids": [t.qid for t in updates]}
        self.traces = candidate + traces
        self.receipts[sid, op_id] = deepcopy(receipt)
        self.applied += 1
        if self.fault == "after":
            raise OSError("commit acknowledgement lost")
        return receipt

    async def memory_commit_receipt(self, sid, op_id):
        return deepcopy(self.receipts.get((sid, op_id)))

    async def load_session(self, sid, *, memory_limit=None, episode_limit=None):
        return list(self.traces[-memory_limit:] if memory_limit else self.traces), []


def new_session(tmp_path, store):
    m = memory()
    m.track_persistence = True
    m.past_qas = list(store.traces)
    calls = []

    class Controller:
        async def process(self, question, **kwargs):
            calls.append(question)
            m.begin_turn(question, kwargs["qid"], kwargs.get("memory_context"))
            m.store({"Q": question, "A": "7 days"}, qid=kwargs["qid"])
            return "7 days"

    return SimpleNamespace(
        memory=m, controller=Controller(), session_id="s", state_store=store,
        memory_outbox=MemoryOutbox(tmp_path / "outbox.sqlite3", store.memory_namespace),
        telemetry=SimpleNamespace(events=[], clear=lambda: None), remember_trace=lambda *a, **kw: None,
        persisted_memory=len(store.traces), hippocampus=SimpleNamespace(episodes=[]), persisted_episodes=0,
        generation_calls=calls,
    )


def correction(op="op-one", qid="new", targets=("old",)):
    return {"qid": qid, "question": "Project Alpha deadline is now seven days", "memory_operation_id": op,
            "memory_context": {"id": "Alpha", "aliases": ["Alpha"], "remember": True, "supersedes": list(targets)}}


def process(session, tmp_path, params, stream=False):
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache=OrderedDict(), vision_cache_limit=1)
    if stream:
        options["emit_event"] = lambda *a: None
    return asyncio.run((_handle_process_stream if stream else _handle_process)(session, params, **options))


@pytest.fixture
def store():
    m = memory()
    write(m, "Alpha", "old", "14 days")
    return FaultStore(m.past_qas)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fault", ["before", "after"])
def test_restart_retries_exact_generation_once(tmp_path, store, fault, stream):
    session = new_session(tmp_path, store)
    params = correction()
    store.fault = fault
    with pytest.raises(MemoryCommitPending, match="op-one"):
        process(session, tmp_path, params, stream)
    entry = session.memory_outbox.get("s", "op-one")
    frozen_payload = deepcopy(entry["payload"])
    assert entry["receipt"] is None
    assert len(store.traces) == (1 if fault == "before" else 2)
    assert store.traces[0].evidence_status == ("unverified" if fault == "before" else "superseded")

    # A fresh session and outbox object simulate loss of every in-process queue.
    restarted = new_session(tmp_path, store)
    store.fault = None
    result = process(restarted, tmp_path, params, stream)
    original_receipt = deepcopy(store.receipts["s", "op-one"])
    assert result["memory_commit"] == original_receipt
    assert result["replayed"] and result["answer"] == "7 days"
    assert not restarted.generation_calls and len(session.generation_calls) == 1
    assert store.applied == 1 and len(store.traces) == 2
    assert restarted.memory_outbox.get("s", "op-one")["payload"] == frozen_payload
    assert [t.qid for t in retrieve(restarted.memory, "Project Alpha deadline?")] == ["new"]

    # A legitimate newer correction must stay active when an older receipt is retried.
    process(restarted, tmp_path, correction("op-two", "newer", ("new",)), stream)
    before = list(store.traces)
    replay = process(restarted, tmp_path, params, stream)
    assert replay["memory_commit"] == original_receipt
    assert store.traces == before and store.applied == 2
    assert [t.qid for t in retrieve(restarted.memory, "Project Alpha deadline?")] == ["newer"]


@pytest.mark.parametrize("change", ["question", "targets"])
def test_conflicting_request_does_not_even_reconcile_pending_write(tmp_path, store, change):
    session = new_session(tmp_path, store)
    store.fault = "before"
    with pytest.raises(MemoryCommitPending):
        process(session, tmp_path, correction())
    conflicting = correction()
    if change == "question":
        conflicting["question"] = "Project Alpha deadline is nine days"
    else:
        conflicting["memory_context"]["supersedes"] = []
    store.fault = None
    with pytest.raises(MemoryCommitConflict):
        process(session, tmp_path, conflicting)
    assert store.applied == 0 and len(store.traces) == 1
    assert len(session.generation_calls) == 1


@pytest.mark.parametrize("change", ["answer", "targets"])
def test_outbox_payload_is_immutable(tmp_path, store, change):
    outbox = MemoryOutbox(tmp_path / "outbox.sqlite3", "test")
    source = MemoryTrace("Project Alpha deadline", "7 days", qid="new", retrieval_scope="Alpha", scope_aliases=("Alpha",))
    payload = memory_payload([source], [replace(store.traces[0], evidence_status="superseded")])
    outbox.prepare("s", "op", payload)
    changed = deepcopy(payload)
    if change == "answer":
        changed["traces"][0]["answer"] = "9 days"
    else:
        changed["updates"] = []
    with pytest.raises(MemoryCommitConflict):
        outbox.prepare("s", "op", changed)
    assert outbox.get("s", "op")["payload_sha256"] == payload_fingerprint(payload)
    assert store.applied == 0


@pytest.mark.parametrize("reason", ["length", None])
def test_incomplete_generation_never_reaches_outbox(tmp_path, store, reason):
    from core.llm_client import LLMConfig
    from core.models import LeftBrainModel
    from test_dual_brain import make_stage_controller
    session = new_session(tmp_path, store)
    model = LeftBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY"))

    async def partial(*args, **kwargs):
        return "7 days except", reason

    model._llm_client._openai_style_stream_call = partial
    session.controller = make_stage_controller(memory=session.memory, left_model=model)
    with pytest.raises(RuntimeError, match="Primary source generation failed"):
        process(session, tmp_path, correction())
    assert not session.memory_outbox.entries("s") and store.applied == 0
    assert [t.answer for t in retrieve(session.memory, "Project Alpha deadline?")] == ["14 days"]


def test_missing_outbox_and_reset_receipts_never_regenerate(tmp_path, store):
    session = new_session(tmp_path, store)
    process(session, tmp_path, correction())
    no_journal = new_session(tmp_path / "different-host", store)
    with pytest.raises(MemoryCommitConflict, match="original outbox"):
        asyncio.run(_prepare_memory_request(no_journal, correction()))
    assert not no_journal.generation_calls and store.applied == 1
    store.receipts.clear()  # Explicit database reset: historical local receipts are not reapplied.
    with pytest.raises(MemoryCommitConflict, match="no longer matches"):
        asyncio.run(reconcile_memory(session.memory_outbox, store, "s", operation_id="op-one"))
    assert store.applied == 1


def test_engine_creation_recovers_outbox_before_exposing_memory(tmp_path, store, monkeypatch):
    from engine_stdio import EngineSession
    session = new_session(tmp_path, store)
    store.fault = "after"
    with pytest.raises(MemoryCommitPending):
        process(session, tmp_path, correction())
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    monkeypatch.setenv("DUALBRAIN_MEMORY_OUTBOX_PATH", str(tmp_path / "outbox.sqlite3"))
    store.fault = None

    async def restart():
        restored = await EngineSession.create("s", state_store=store)
        try:
            assert len(restored.memory.past_qas) == 2
            assert restored.memory.past_qas[0].evidence_status == "superseded"
            assert restored.memory.past_qas[-1].answer == "7 days"
            assert not restored.memory_outbox.entries("s", pending_only=True)
        finally:
            await restored.close()

    asyncio.run(restart())
    assert store.applied == 1


def test_configured_durable_memory_never_falls_back_to_transient(monkeypatch):
    import engine_stdio

    async def fail(_):
        raise OSError("database unavailable")

    monkeypatch.setenv("DUALBRAIN_PG_DSN", "NOT_A_DSN")
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    monkeypatch.setattr("core.postgres_state_store.PostgresStateStore.ensure_schema", fail)
    with pytest.raises(RuntimeError, match="recovery requires PostgreSQL"):
        asyncio.run(engine_stdio.main())


def test_conflicting_request_cannot_trigger_startup_recovery(tmp_path, store, monkeypatch):
    from engine_stdio import EngineSession
    session = new_session(tmp_path, store)
    store.fault = "before"
    with pytest.raises(MemoryCommitPending):
        process(session, tmp_path, correction())
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    monkeypatch.setenv("DUALBRAIN_MEMORY_OUTBOX_PATH", str(tmp_path / "outbox.sqlite3"))
    store.fault = None
    conflicting = correction(targets=())
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(EngineSession.create("s", state_store=store, memory_request=conflicting))
    assert store.applied == 0 and len(store.traces) == 1


def test_persistence_refresh_preserves_live_followup_focus(tmp_path, store):
    session = new_session(tmp_path, store)
    process(session, tmp_path, correction())
    assert [t.qid for t in retrieve(session.memory, "その締切は？")] == ["new"]
