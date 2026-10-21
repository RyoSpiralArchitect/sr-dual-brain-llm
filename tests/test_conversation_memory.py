import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))

from core.memory_answerability import AnswerabilityGate
from core.memory_evidence import bounded_evidence, request_memory_context
from core.shared_memory import MemoryTrace, SharedMemory
from test_memory_answerability import Checks


def memory():
    return SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Checks()))


def write(m, name, qid, answer, supersedes=()):
    m.begin_turn(f"Project {name} deadline", qid, {"id": name, "aliases": [name],
                 "remember": True, "supersedes": list(supersedes)})
    m.store({"Q": f"Project {name} deadline", "A": answer}, qid=qid)


def retrieve(m, question, qid="probe"):
    m.begin_turn(question, qid)
    asyncio.run(m.prepare_retrieval(question))
    return m.retrieve_related_traces(question)


def test_correction_followup_and_repeat_keep_primary_sources_and_cache():
    m = memory()
    write(m, "Alpha", "old", "14 days")
    write(m, "Beta", "beta", "29 days")
    write(m, "Alpha", "new", "7 days", ["old"])
    assert m.past_qas[0].evidence_status == "superseded"
    question = "Project Alpha deadline?"
    assert [t.qid for t in retrieve(m, question)] == ["new"]
    m.store({"Q": question, "A": "7 days"}, qid="probe")
    assert m.past_qas[-1].recalled_answer
    assert [t.qid for t in retrieve(m, "その締切は？", "followup")] == ["new"]
    assert "Current conversation reference:" in m.retrieve_related("その締切は？")
    calls = m.answerability_gate.backend.calls
    assert [t.qid for t in retrieve(m, question, "repeat")] == ["new"]
    assert m.answerability_gate.backend.calls == calls
    assert all(t.evidence_status == "unverified" for t in m.past_qas[1:])


@pytest.mark.parametrize("interruption", ["Project Alpha and Beta deadlines?", "What is the weather?",
    "Project Unknown deadline?", "Project Alpha and Project Unknown deadlines?",
    "プロジェクト Alpha と プロジェクト Unknown の締切は？",
    "プロジェクトAlphaとプロジェクトUnknownの締切は？"])
def test_ambiguous_or_unrelated_turn_clears_focus(interruption):
    m = memory()
    write(m, "Alpha", "a", "7 days")
    write(m, "Beta", "b", "29 days")
    retrieve(m, "Project Alpha deadline?")
    retrieve(m, interruption, "interruption")
    assert not retrieve(m, "その締切は？", "ambiguous")
    assert m.retrieval_question("その締切は？") == "その締切は？"


@pytest.mark.parametrize("metadata", [
    {"id": "Alpha", "aliases": ["Alpha"], "remember": True, "supersedes": ["b"]},
    {"id": "Alpha", "aliases": ["Alpha"], "remember": True, "supersedes": ["absent"]},
    {"id": "Alpha", "aliases": ["Alpha"], "remember": "true"},
    {"id": "Alpha", "aliases": ["Alpha"], "supersedes": ["a"]},
])
def test_invalid_updates_are_rejected_before_replacing_sources(metadata):
    m = memory()
    write(m, "Alpha", "a", "7 days")
    write(m, "Beta", "b", "29 days")
    original = list(m.past_qas)
    with pytest.raises(ValueError):
        m.begin_turn("Project Alpha deadline", "new", metadata)
    assert m.past_qas == original


def test_begin_turn_does_not_supersede_until_answer_is_stored():
    m = memory()
    write(m, "Alpha", "a", "14 days")
    m.begin_turn("Project Alpha correction", "new", {"id": "Alpha", "aliases": ["Alpha"],
                 "remember": True, "supersedes": ["a"]})
    assert m.past_qas[0].evidence_status == "unverified"
    assert m.retrieval_question("unrelated question") == "unrelated question"


def test_pending_correction_hides_old_approvals_without_mutating_persisted_status():
    m = memory()
    write(m, "Alpha", "old", "OBSOLETE_14")
    m.bind_seed("old", "old-seed")
    question = "Project Alpha deadline?"
    assert retrieve(m, question)[0].qid == "old"
    m.begin_turn("Project Alpha deadline is now seven days", "new", {
        "id":"Alpha", "aliases":["Alpha"], "remember":True, "supersedes":["old"]})
    assert m.past_qas[0].evidence_status == "unverified"
    assert not m.answerability_candidates(question) and not m.retrieve_related(question)
    assert not m.evidence_for_seeds(["old-seed"], question=question)
    m.begin_turn(question, "after-cancelled-correction")
    assert m.retrieve_related_traces(question)[0].qid == "old"


@pytest.mark.parametrize("expired", [False, True])
def test_corrections_reject_stale_or_expired_targets(expired):
    from dataclasses import replace
    m = memory()
    write(m, "Alpha", "old", "14 days")
    if expired:
        m.past_qas[0] = replace(m.past_qas[0], valid_until=0)
    else:
        write(m, "Alpha", "current", "7 days", ["old"])
    before = list(m.past_qas)
    with pytest.raises(ValueError, match="current source"):
        write(m, "Alpha", "stale-update", "3 days", ["old"])
    assert m.past_qas == before
    if not expired:
        assert [t.qid for t in retrieve(m, "Project Alpha deadline?")] == ["current"]


def test_engine_components_cannot_reintroduce_old_or_answer_only_memory(monkeypatch):
    from engine_stdio import EngineSession
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    prompts = []

    class Capture:
        async def complete(self, prompt, **kwargs):
            prompts.append(prompt + "\n" + str(kwargs.get("system", "")))
            return "CURRENT_7" if len(prompts) == 1 else "DERIVED_SENTINEL"

    async def run():
        session = await EngineSession.create("channel-regression")
        try:
            m, c = session.memory, session.controller
            m.answerability_gate = AnswerabilityGate(Checks(), max_question_chars=2048, max_answer_chars=2048)
            write(m, "Alpha", "old", "OBSOLETE_14")
            c.prefrontal_cortex.remember_working_memory(question="Project Alpha deadline", answer="OBSOLETE_14", qid="old")
            c.hippocampus.index_episode("old", "Project Alpha deadline", "OBSOLETE_14")
            c.left_model._llm_client = Capture()
            original_routing = c._run_routing_stage
            async def route(**kwargs):
                result = await original_routing(**kwargs)
                result.decision.action = 0
                return result
            c._run_routing_stage = route
            await c.process("Project Alpha deadline is now seven days", qid="new", leading_brain="left", system2_mode="off",
                memory_context={"id":"Alpha", "aliases":["Alpha"], "remember":True, "supersedes":["old"]})
            await c.process("Project Alpha deadline?", qid="probe", leading_brain="left", system2_mode="off")
            await c.process("Project Alpha deadline?", qid="repeat", leading_brain="left", system2_mode="off")
            await c.process("その締切をもう一度教えて。", qid="followup", leading_brain="left", system2_mode="off")
            assert len(prompts) == 4
            assert all("OBSOLETE_14" not in p for p in prompts)
            assert "CURRENT_7" in prompts[1] and "CURRENT_7" in prompts[2]
            assert "CURRENT_7" in prompts[3]
            assert all("DERIVED_SENTINEL" not in p for p in prompts[1:])
            unsafe_parts = {k:"UNAPPROVED" for k in ("working_memory", "hippocampal", "replay", "schema", "pitfalls")}
            frame = c._apply_director_memory_control(question="Project Alpha deadline?", context_parts=unsafe_parts,
                control={"memory":{"working_memory":"keep", "long_term":"keep"}})
            assert "CURRENT_7" in frame.context and "OBSOLETE_14" not in frame.context
            assert "DERIVED_SENTINEL" not in frame.context
            assert all(not unsafe_parts[k] for k in ("working_memory", "hippocampal", "replay", "schema", "pitfalls"))
        finally:
            await session.close()

    asyncio.run(run())


def test_cortical_line_gating_keeps_complete_checked_conditions():
    from test_dual_brain import make_stage_controller
    from core.prefrontal_cortex import PrefrontalCortex
    m = memory()
    answer = "Project Alpha standard limit: 5kg.\n" + "Other policy.\n" * 60 + "Europe exception: 2kg."
    write(m, "Alpha", "a", answer)
    q = "Project Alpha limit?"
    retrieve(m, q)
    c = make_stage_controller(memory=m, prefrontal_cortex=PrefrontalCortex())
    context, _, parts = c._compose_context(q)
    assert answer in context
    rebuilt, _ = c._rebuild_context_from_parts(question=q, context_parts=dict(parts, working_memory="UNAPPROVED"))
    assert answer in rebuilt and "UNAPPROVED" not in rebuilt


def test_interactive_server_retains_legacy_memory_with_stdio_flag_exported(monkeypatch):
    import run_server
    monkeypatch.setenv("DUALBRAIN_MEMORY_CONVERSATION_CONTEXT", "1")
    memories = []
    questions = iter(["Project Alpha deadline?", "Project Alpha deadline?"])

    def input_question(*_):
        try:
            return next(questions)
        except StopIteration:
            raise EOFError from None

    class Controller:
        def __init__(self, **kwargs):
            self.memory = kwargs["memory"]
            memories.append(self.memory)
        async def process(self, question, **kwargs):
            if self.memory.past_qas:
                assert "SERVER_MEMORY" in self.memory.retrieve_related(question)
            self.memory.store({"Q":question, "A":"SERVER_MEMORY"})
            return "SERVER_MEMORY"

    async def worker(*_): pass
    monkeypatch.setattr("builtins.input", input_question)
    monkeypatch.setattr(run_server, "DualBrainController", Controller)
    monkeypatch.setattr(run_server, "right_worker", worker)
    with pytest.raises(EOFError):
        asyncio.run(run_server.main())
    assert not memories[0].conversation_context and len(memories[0].past_qas) == 2
    assert not any(t.recalled_answer for t in memories[0].past_qas)


@pytest.mark.parametrize("failure", ["timeout", "stream", "empty", "unconfigured"])
def test_failed_production_source_generation_keeps_previous_fact(failure):
    from core.llm_client import LLMConfig
    from core.models import LeftBrainModel
    from test_dual_brain import make_stage_controller
    m = memory()
    m.track_persistence = True
    write(m, "Alpha", "old", "14 days")
    m.pending_memory_writes.clear()

    class Failure:
        async def complete(self, *args, **kwargs):
            if failure == "empty":
                return ""
            raise TimeoutError("PROVIDER_SECRET")

    class StreamFailure:
        async def complete_stream(self, *args, **kwargs):
            if kwargs.get("on_delta"):
                result = kwargs["on_delta"]("partial")
                if asyncio.iscoroutine(result):
                    await result
            raise TimeoutError("PROVIDER_SECRET")

    model = LeftBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY"))
    model._llm_client = None if failure == "unconfigured" else StreamFailure() if failure == "stream" else Failure()
    controller = make_stage_controller(memory=m, left_model=model)
    with pytest.raises(RuntimeError, match="Primary source generation") as error:
        asyncio.run(controller.process("Project Alpha deadline is now seven days", qid="new", leading_brain="left",
            system2_mode="off", on_delta=lambda text: None,
            memory_context={"id":"Alpha", "aliases":["Alpha"], "remember":True, "supersedes":["old"]}))
    assert "PROVIDER_SECRET" not in str(error.value)
    assert len(m.past_qas) == 1 and m.past_qas[0].evidence_status == "unverified"
    assert not m.pending_memory_writes and not m.pending_memory_updates
    assert [t.qid for t in retrieve(m, "Project Alpha deadline?")] == ["old"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasons", [
    ("length", "length", "length"),
    (None,),
    ("length", None),
    ("content_filter",),
    ("stop",),
    ("length", "stop"),
])
def test_source_promotion_requires_terminal_completion(monkeypatch, stream, reasons):
    from core.llm_client import LLMConfig
    from core.models import LeftBrainModel
    from test_dual_brain import make_stage_controller

    m = memory()
    m.track_persistence = True
    write(m, "Alpha", "old", "14 days")
    m.pending_memory_writes.clear()
    original = list(m.past_qas)
    model = LeftBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY", max_continuations=2))
    client = model._llm_client
    calls, deltas = [], []

    def response():
        reason = reasons[len(calls)]
        calls.append(reason)
        return "7 days" if len(calls) == 1 else " including weekends", reason

    async def post_json(*args, **kwargs):
        text, reason = response()
        return {"choices": [{"message": {"content": text}, "finish_reason": reason}]}

    async def stream_call(messages, *, on_delta, **kwargs):
        text, reason = response()

        async def events():
            yield json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": reason}]})
            yield "[DONE]"

        return await client._consume_openai_style_stream(events(), on_delta=on_delta)

    monkeypatch.setattr(client, "_post_json", post_json)
    monkeypatch.setattr(client, "_openai_style_stream_call", stream_call)
    if not stream:
        # Exercise the model's nonstreaming client adapter as well.
        model._llm_client = SimpleNamespace(complete=client.complete)
    controller = make_stage_controller(memory=m, left_model=model)

    async def correct():
        return await controller.process(
            "Project Alpha deadline is now seven days", qid="new", leading_brain="left",
            system2_mode="off", on_delta=deltas.append,
            memory_context={"id": "Alpha", "aliases": ["Alpha"], "remember": True, "supersedes": ["old"]},
        )

    if reasons[-1] == "stop":
        asyncio.run(correct())
        assert m.past_qas[0].evidence_status == "superseded"
        assert m.pending_memory_updates and m.pending_memory_writes
        sources = retrieve(m, "Project Alpha deadline?")
        assert [t.qid for t in sources] == ["new"]
        assert sources[0].answer == "7 days" + (" including weekends" if len(reasons) > 1 else "")
    else:
        with pytest.raises(RuntimeError, match="Primary source generation failed"):
            asyncio.run(correct())
        assert m.past_qas == original
        assert not m.pending_memory_writes and not m.pending_memory_updates
        assert [t.qid for t in retrieve(m, "Project Alpha deadline?")] == ["old"]
        assert m.past_qas[0].answer == "14 days"
    assert calls == list(reasons)
    if stream:
        assert deltas  # Partial output can reach the caller without becoming a source.


def test_failed_primary_generation_cancels_preview_and_observer_tasks():
    from core.llm_client import LLMConfig
    from core.models import LeftBrainModel
    from test_dual_brain import make_stage_controller
    m = memory()
    write(m, "Alpha", "old", "14 days")
    started, stopped = set(), set()

    async def blocked(name):
        started.add(name)
        try:
            await asyncio.Future()
        finally:
            stopped.add(name)

    class Failure:
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise TimeoutError("simulated outage")

    class Observer:
        async def advise(self, **kwargs):
            await blocked("observer")

    async def preview(*args, **kwargs):
        await blocked("preview")

    model = LeftBrainModel(LLMConfig("openai", "fake", "NOT_A_KEY"))
    model._llm_client = Failure()
    controller = make_stage_controller(memory=m, left_model=model)
    controller.executive_model = Observer()
    controller.right_model.generate_lead = preview

    async def run():
        with pytest.raises(RuntimeError, match="Primary source generation"):
            await controller.process("Project Alpha deadline is now seven days", qid="new", leading_brain="right",
                system2_mode="off", executive_mode="observe",
                memory_context={"id":"Alpha", "aliases":["Alpha"], "remember":True, "supersedes":["old"]})
        assert started == stopped == {"preview", "observer"}

    asyncio.run(run())


def test_long_complete_source_keeps_late_exception_and_rejects_oversized_units():
    m = memory()
    answer = "Standard: 5kg. " + "Memo. " * 120 + "Europe exception: 2kg."
    write(m, "Alpha", "a", answer)
    m.bind_seed("a", "seed-a")
    q = "Project Alpha Europe limit?"
    assert [t.answer for t in retrieve(m, q)] == [answer]
    evidence = m.evidence_for_seeds(["seed-a"], question=q)
    assert evidence[0]["answer"] == answer and not evidence[0]["truncated"]
    request = {"memory_evidence": evidence, "memory_evidence_format": "complete_v1"}
    assert "Europe exception: 2kg." in request_memory_context(request)
    assert not bounded_evidence([dict(evidence[0], answer="x" * 2049)], complete=True)
    legacy = SharedMemory(answerability_gate=AnswerabilityGate(Checks()))
    legacy.store(m.past_qas[0])
    asyncio.run(legacy.prepare_retrieval(q))
    assert not legacy.retrieve_related_traces(q)


def test_persistence_queue_keeps_evicted_writes_and_acknowledges_only_success(tmp_path):
    from engine_stdio import _persist_session_memory
    from core.memory_commits import MemoryOutbox
    from test_memory_commits import FaultStore
    m = memory()
    m.track_persistence = True
    m.max_items = 1
    write(m, "Alpha", "a", "7 days")
    write(m, "Beta", "b", "29 days")
    store = FaultStore()
    store.fault = "before"
    session = SimpleNamespace(state_store=store, memory=m, session_id="s", persisted_memory=0,
                              hippocampus=SimpleNamespace(episodes=[]), persisted_episodes=0,
                              memory_outbox=MemoryOutbox(tmp_path / "outbox.sqlite3", store.memory_namespace))
    with pytest.raises(RuntimeError):
        asyncio.run(_persist_session_memory(session, "b"))
    pending = session.memory_outbox.entries("s", pending_only=True)
    assert len(pending) == 1 and len(pending[0]["payload"]["traces"]) == 2
    store.fault = None
    asyncio.run(_persist_session_memory(session, "b"))
    assert [t.qid for t in store.traces] == ["a", "b"] and not m.pending_memory_writes
    assert not session.memory_outbox.entries("s", pending_only=True)


def test_loaded_recalled_answers_cannot_become_primary_sources():
    m = memory()
    m.store(MemoryTrace("Project Alpha deadline", "wrong", qid="derived", recalled_answer=True,
                        retrieval_scope="Alpha", scope_aliases=("Alpha",)))
    assert not retrieve(m, "Project Alpha deadline?")
    with pytest.raises(ValueError):
        SharedMemory().begin_turn("Q", "q", {"id": "Alpha", "aliases": ["Alpha"]})


@pytest.mark.parametrize("stream", [False, True])
def test_rpc_passes_explicit_write_and_correction_metadata(tmp_path, stream):
    from collections import OrderedDict
    from engine_stdio import _handle_process, _handle_process_stream
    m = memory()

    class Controller:
        async def process(self, question, **kwargs):
            m.begin_turn(question, kwargs["qid"], kwargs.get("memory_context"))
            m.store({"Q": question, "A": "source"}, qid=kwargs["qid"])
            return "source"

    telemetry = SimpleNamespace(events=[], clear=lambda: None)
    session = SimpleNamespace(controller=Controller(), memory=m, session_id="s", telemetry=telemetry,
                              state_store=None, remember_trace=lambda *a, **kw: None)
    handler = _handle_process_stream if stream else _handle_process
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache=OrderedDict(), vision_cache_limit=1)
    if stream:
        options["emit_event"] = lambda *args: None
    for qid, old in [("old", []), ("new", ["old"])]:
        result = asyncio.run(handler(session, {"qid": qid, "question": "Project Alpha deadline",
            "memory_context": {"id":"Alpha", "aliases":["Alpha"], "remember":True, "supersedes":old}}, **options))
        assert result["answer"] == "source"
    assert m.past_qas[0].evidence_status == "superseded"
    assert m.past_qas[1].retrieval_scope == "Alpha" and not m.past_qas[1].recalled_answer


def test_persisted_correction_updates_metadata_in_same_transaction_as_new_source(monkeypatch):
    import json
    from core.postgres_state_store import PostgresStateStore
    sql_calls, commits = [], []

    class Cursor:
        rowcount = 1
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def execute(self, sql, params=()): sql_calls.append((sql, params))

    cursor = Cursor()

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def cursor(self): return cursor
        def commit(self): commits.append(True)

    store = PostgresStateStore("NOT_A_DSN")
    monkeypatch.setattr(store, "_connect", Connection)
    m = memory()
    m.track_persistence = True
    write(m, "Alpha", "old", "14 days")
    m.pending_memory_writes.clear()  # Prior successful persistence.
    write(m, "Alpha", "new", "7 days", ["old"])
    asyncio.run(store.commit_memory_changes("s", list(m.pending_memory_writes.values()), list(m.pending_memory_updates.values())))
    updates = [p for sql, p in sql_calls if sql.startswith("UPDATE srdb_shared_memory")]
    inserts = [p for sql, p in sql_calls if "INSERT INTO srdb_shared_memory" in sql]
    assert len(updates) == len(inserts) == len(commits) == 1
    assert json.loads(updates[0][0])["evidence_status"] == "superseded"
    assert updates[0][2] == "old" and inserts[0][1] == "new"
    cursor.rowcount = 0
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(store.commit_memory_changes("s", [], list(m.pending_memory_updates.values())))
    assert len(commits) == 1
