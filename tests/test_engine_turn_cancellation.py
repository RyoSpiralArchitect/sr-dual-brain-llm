"""Cancellation through the real engine/controller and durable-memory boundary."""
import asyncio
from collections import OrderedDict
import threading
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import engine_stdio as engine
from core.turn_lifecycle import TurnDispatcher, begin_turn_commit
from test_turn_lifecycle import until
from test_memory_commits import FaultStore, new_session, correction
from test_conversation_memory import memory, write


def options(tmp_path):
    return dict(blob_root=tmp_path, blob_index={}, vision_cache=OrderedDict(), vision_cache_limit=1)


class QueueTransport:
    def __init__(self):
        self.input, self.output = asyncio.Queue(), []
        self.closed = False

    async def read(self):
        return await self.input.get()

    async def write(self, message):
        self.output.append(message)

    async def close(self):
        self.closed = True

    def send(self, rid, method, **params):
        self.input.put_nowait({"id": rid, "method": method, "params": params})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_live_reader_cancels_real_controller_without_storing_partial_source(tmp_path, monkeypatch, stream):
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    session = await engine.EngineSession.create("s")
    # Existing independent source and focus must survive the abandoned write.
    write(session.memory, "Alpha", "old-alpha", "14 days")
    write(session.memory, "Beta", "prior", "29 days")
    prior = list(session.memory.past_qas)
    previous_focus = session.memory._focus_scope
    started, stopped = asyncio.Event(), asyncio.Event()

    async def blocking_source(*args, on_delta=None, **kwargs):
        if on_delta:
            on_delta("partial source")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if on_delta:
                on_delta("STALE-AFTER-CANCEL")
            stopped.set()

    session.left.generate_source_answer = blocking_source
    async def existing(*args, **kwargs):
        return session
    monkeypatch.setattr(engine.EngineSession, "create", existing)
    transport = QueueTransport()
    monkeypatch.setattr(engine, "_StdioTransport", lambda: transport)
    monkeypatch.setenv("DUALBRAIN_BLOB_DIR", str(tmp_path))
    main = asyncio.create_task(engine.main())
    try:
        transport.send("turn", "process_stream" if stream else "process", session_id="s",
                       question="Project Alpha stores a seven day deadline", qid="abandoned",
                       system2_mode="off", leading_brain="left",
                       memory_context={"id": "Alpha", "aliases": ["Alpha"], "remember": True,
                                       "supersedes": ["old-alpha"]})
        await asyncio.wait_for(started.wait(), 3.)
        # This health must arrive while generation is still blocked.
        transport.send("health", "health")
        await until(lambda: any(m["id"] == "health" for m in transport.output))
        health = next(m["result"] for m in transport.output if m["id"] == "health")
        assert health["rpc"]["pending"] == 1 and not stopped.is_set()
        transport.send("control", "cancel", request_id="turn")
        await until(lambda: any(m["id"] == "turn" and "ok" in m for m in transport.output))
        assert stopped.is_set()
        assert session.memory.past_qas == prior
        assert not session.hippocampus.episodes
        assert session.memory._focus_scope == previous_focus
        assert not session.memory.primary_write_pending and not session.memory._pending_supersedes
        assert "abandoned" not in session.trace_cache
        assert not any("STALE-AFTER-CANCEL" in str(m) for m in transport.output)
        terminal = next(m for m in transport.output if m["id"] == "turn" and "ok" in m)
        assert terminal["error"]["type"] == "Cancelled"
        assert not terminal["error"]["may_have_committed"]
        # Reuse the session after cancellation. No dead worker or held serial slot.
        transport.send("trace", "get_trace", session_id="s", qid="abandoned")
        await until(lambda: any(m["id"] == "trace" for m in transport.output))
        assert next(m for m in transport.output if m["id"] == "trace")["ok"]
        transport.send("fresh-attempt", "process", session_id="s", qid="abandoned", question="こんにちは")
        await until(lambda: any(m["id"] == "fresh-attempt" for m in transport.output))
        assert next(m for m in transport.output if m["id"] == "fresh-attempt")["ok"]
    finally:
        transport.input.put_nowait(None)
        await asyncio.wait_for(main, 3.)
        assert transport.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_cancel_during_threaded_commit_preserves_receipt_and_serializes_next_turn(tmp_path, stream):
    m = memory()
    write(m, "Alpha", "old", "14 days")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class ThreadStore(FaultStore):
        async def commit_memory_operation(self, sid, op_id, payload):
            def transaction():
                started.set()
                release.wait(5.)
                finished.set()
            await asyncio.to_thread(transaction)
            return await super().commit_memory_operation(sid, op_id, payload)

    store = ThreadStore(m.past_qas)
    session = new_session(tmp_path, store)
    original = session.controller.process
    async def process(*args, **kwargs):
        # Production DualBrainController does this before its first fact write.
        begin_turn_commit()
        return await original(*args, **kwargs)
    session.controller.process = process
    responses, next_observations = [], []
    async def handle(req):
        if req["id"] == "first":
            opts = options(tmp_path)
            if stream:
                opts["emit_event"] = lambda *a: None
            result = await (engine._handle_process_stream if stream else engine._handle_process)(session, correction(), **opts)
            return {"id": "first", "ok": True, "result": result}
        next_observations.append((finished.is_set(), len(store.traces), store.applied))
        return {"id": req["id"], "ok": True}

    dispatcher = TurnDispatcher(handle, responses.append)
    try:
        dispatcher.submit({"id": "first"})
        await until(started.is_set)
        dispatcher.submit({"id": "next"})
        assert dispatcher.cancel("first")["deferred"]
        await asyncio.sleep(.02)
        assert not responses and not next_observations and not finished.is_set()
        release.set()
        await until(lambda: len(responses) == 2)
        assert next_observations == [(True, 2, 1)]
        assert responses[0]["error"]["may_have_committed"]
        entry = session.memory_outbox.get("s", "op-one")
        assert entry["receipt"]["state"] == "committed"
        assert session.memory.past_qas[0].evidence_status == "superseded"
    finally:
        release.set()
        await dispatcher.close()


@pytest.mark.asyncio
async def test_turn_waits_for_orphan_consultation_cleanup_before_next_session_work(tmp_path, monkeypatch):
    from core.turn_lifecycle import create_turn_task
    session = await engine.EngineSession.create("s")
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def blocking_right(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
    session.right.deepen = blocking_right
    async def process(*args, **kwargs):
        create_turn_task(session.callosum.ask_detail({"type": "ASK_DETAIL", "question": "block", "qid": "q"}))
        await asyncio.Event().wait()
    session.controller.process = process
    responses = []
    async def handle(req):
        return await engine._dispatch_request(req, sessions={"s": session}, state_store=None,
                                              blob_pipe=None, send=responses.append, **options(tmp_path))
    dispatcher = TurnDispatcher(handle, responses.append)
    try:
        dispatcher.submit({"id": "turn", "method": "process", "params": {"session_id": "s", "question": "block"}})
        await asyncio.wait_for(started.wait(), 2.)
        dispatcher.cancel("turn")
        await asyncio.wait_for(cleaning.wait(), 2.)
        dispatcher.submit({"id": "next", "method": "get_trace", "params": {"session_id": "s", "qid": "q"}})
        await asyncio.sleep(.02)
        assert not responses and session.callosum.snapshot()["running"] == 1
        dispatcher.cancel("turn")  # Do not interrupt provider finally a second time.
        release.set()
        await until(lambda: len(responses) == 2)
        assert responses[0]["error"]["type"] == "Cancelled" and responses[1]["ok"]
        assert session.callosum.snapshot()["running"] == 0 and not session.worker_task.done()
    finally:
        release.set()
        await dispatcher.close()
        await session.close()
