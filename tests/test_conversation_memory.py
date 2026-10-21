import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("interruption", ["Project Alpha and Beta deadlines?", "What is the weather?", "Project Unknown deadline?"])
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


def test_persistence_queue_keeps_evicted_writes_and_acknowledges_only_success():
    from engine_stdio import _persist_session_memory
    m = memory()
    m.track_persistence = True
    m.max_items = 1
    write(m, "Alpha", "a", "7 days")
    write(m, "Beta", "b", "29 days")
    seen = []

    class Store:
        fail = True
        async def commit_memory_changes(self, sid, traces, updates):
            if self.fail:
                raise RuntimeError("simulated storage failure")
            seen.extend(t.qid for t in traces)

    store = Store()
    session = SimpleNamespace(state_store=store, memory=m, session_id="s", persisted_memory=0,
                              hippocampus=SimpleNamespace(episodes=[]), persisted_episodes=0)
    with pytest.raises(RuntimeError):
        asyncio.run(_persist_session_memory(session, "b"))
    assert len(m.pending_memory_writes) == 2
    store.fail = False
    asyncio.run(_persist_session_memory(session, "b"))
    assert seen == ["a", "b"] and not m.pending_memory_writes


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
