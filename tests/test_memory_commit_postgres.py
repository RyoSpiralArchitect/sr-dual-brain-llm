"""Real PostgreSQL integration tests; use only an isolated test database.

Run with DUALBRAIN_TEST_PG_DSN set and requirements-pg.txt installed. No model calls.
"""
import asyncio
from copy import deepcopy
from dataclasses import replace
import os
import subprocess
import sys
import uuid

import pytest

from core.memory_commits import MemoryCommitConflict, MemoryCommitPending, MemoryOutbox, memory_payload, reconcile_memory
from core.postgres_state_store import PostgresStateStore
from core.shared_memory import MemoryTrace
from test_conversation_memory import memory, retrieve


@pytest.fixture
def pg_store():
    dsn = os.environ.get("DUALBRAIN_TEST_PG_DSN")
    if not dsn:
        pytest.skip("isolated PostgreSQL DSN not configured")
    pytest.importorskip("psycopg")
    store = PostgresStateStore(dsn, table_prefix="retry_test_" + uuid.uuid4().hex[:12])
    asyncio.run(store.ensure_schema())
    try:
        yield store
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            for name in ("memory_commits", "shared_memory", "hippocampal_episodes", "telemetry_events",
                         "schema_memory", "consolidation_state", "sessions"):
                cur.execute(f"DROP TABLE IF EXISTS {store._table(name)} CASCADE")


def source(answer, ts, qid="shared-qid"):
    return MemoryTrace("Project Alpha deadline", answer, qid=qid, timestamp=ts,
                       retrieval_scope="Alpha", scope_aliases=("Alpha",))


def load(store):
    return asyncio.run(store.load_session("s"))[0]


def count_receipts(store):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {store._table('memory_commits')}")
        return cur.fetchone()[0]


@pytest.mark.parametrize("fault", ["before", "after"])
def test_real_commit_fault_restart_and_old_receipt_replay(pg_store, tmp_path, monkeypatch, fault):
    store = pg_store
    old, new, newest = source("14 days", 1), source("7 days", 2), source("3 days", 3)
    asyncio.run(store.append_memory_traces("s", [old]))
    payload = memory_payload([new], [replace(old, evidence_status="superseded")])
    path = tmp_path / "outbox.sqlite3"
    outbox = MemoryOutbox(path, store.memory_namespace)
    outbox.prepare("s", "first", payload)
    connect = store._connect

    class FaultConnection:
        def __init__(self): self.conn = connect()
        def __enter__(self): self.conn.__enter__(); return self
        def __exit__(self, *args): return self.conn.__exit__(*args)
        def cursor(self): return self.conn.cursor()
        def commit(self):
            if fault == "before":
                raise OSError("injected before transaction commit")
            self.conn.commit()
            raise OSError("injected lost commit acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", FaultConnection)
        with pytest.raises(MemoryCommitPending):
            asyncio.run(reconcile_memory(outbox, store, "s"))
    if fault == "before":
        assert load(store) == [old] and count_receipts(store) == 0
    else:
        assert len(load(store)) == 2 and count_receipts(store) == 1
        assert load(store)[0].evidence_status == "superseded"
    durable_receipt = asyncio.run(store.memory_commit_receipt("s", "first"))

    restarted = MemoryOutbox(path, store.memory_namespace)
    receipt = asyncio.run(reconcile_memory(restarted, store, "s"))[0]
    assert durable_receipt is None or receipt == durable_receipt
    assert len(load(store)) == 2 and count_receipts(store) == 1
    assert receipt["memory_ids"] and receipt["superseded_ids"]

    # Distinct revisions intentionally share session_id + qid; exact targets include ts/text.
    next_payload = memory_payload([newest], [replace(new, evidence_status="superseded")])
    asyncio.run(store.commit_memory_operation("s", "second", next_payload))
    before = load(store)
    assert asyncio.run(store.commit_memory_operation("s", "first", payload)) == receipt
    assert asyncio.run(reconcile_memory(restarted, store, "s", operation_id="first")) == [receipt]
    assert load(store) == before and count_receipts(store) == 2
    m = memory()
    m.past_qas = load(store)
    assert [t.answer for t in retrieve(m, "Project Alpha deadline?")] == ["3 days"]


@pytest.mark.parametrize("change", ["answer", "targets"])
def test_real_conflicting_operation_never_mutates_memory(pg_store, change):
    old, new = source("14 days", 1), source("7 days", 2)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    payload = memory_payload([new], [replace(old, evidence_status="superseded")])
    receipt = asyncio.run(pg_store.commit_memory_operation("s", "operation", payload))
    before = load(pg_store)
    conflict = deepcopy(payload)
    if change == "answer":
        conflict["traces"][0]["answer"] = "9 days"
    else:
        conflict["updates"] = []
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(pg_store.commit_memory_operation("s", "operation", conflict))
    assert load(pg_store) == before and count_receipts(pg_store) == 1
    assert asyncio.run(pg_store.memory_commit_receipt("s", "operation")) == receipt


def test_concurrent_duplicate_and_competing_corrections(pg_store):
    old = source("14 days", 1)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    payload = memory_payload([source("7 days", 2)], [replace(old, evidence_status="superseded")])

    async def duplicate():
        return await asyncio.gather(*(pg_store.commit_memory_operation("s", "same", payload) for _ in range(4)))

    receipts = asyncio.run(duplicate())
    assert all(receipt == receipts[0] for receipt in receipts)
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 1
    # A different operation cannot supersede an already replaced source.
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(pg_store.commit_memory_operation("s", "stale", payload))
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 1


def test_legacy_duplicate_qids_remain_allowed(pg_store):
    asyncio.run(pg_store.append_memory_traces("s", [source("one", 1), source("two", 2)]))
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 0


def test_existing_episode_sql_roundtrip_with_quoted_leading_column(pg_store):
    import numpy as np
    from core.temporal_hippocampal_indexing import EpisodicTrace
    episode = EpisodicTrace("episode", "question", "answer", np.ones(128), leading="left")
    asyncio.run(pg_store.upsert_episodes("s", [episode]))
    # Exercise both initial insertion and the existing ON CONFLICT path.
    asyncio.run(pg_store.upsert_episodes("s", [replace(episode, leading="right")]))
    assert asyncio.run(pg_store.load_session("s"))[1][0].leading == "right"
    assert asyncio.run(pg_store.load_session("s", episode_limit=1))[1][0].leading == "right"
    assert asyncio.run(pg_store.search_episodes("s", np.ones(128)))[0][1].leading == "right"
    assert asyncio.run(pg_store.fetch_episodes_after_id("s", after_id=0, limit=5))[0]["leading"] == "right"


def test_process_exit_after_commit_recovers_only_latest_revision(pg_store, tmp_path):
    old, new, newest = source("14 days", 1), source("7 days", 2), source("3 days", 3)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    path = tmp_path / "outbox.sqlite3"
    payload = memory_payload([new], [replace(old, evidence_status="superseded")])
    MemoryOutbox(path, pg_store.memory_namespace).prepare("s", "first", payload)
    script = """
import asyncio, os, sys
from core.memory_commits import MemoryOutbox
from core.postgres_state_store import PostgresStateStore
store = PostgresStateStore(sys.argv[1], table_prefix=sys.argv[2])
entry = MemoryOutbox(sys.argv[3], store.memory_namespace).get('s', 'first')
asyncio.run(store.commit_memory_operation('s', 'first', entry['payload']))
os._exit(23)  # Commit is durable, but no outbox acknowledgement can be written.
"""
    result = subprocess.run(
        [sys.executable, "-c", script, pg_store.dsn, pg_store.table_prefix, str(path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 23, result.stderr
    receipt = asyncio.run(pg_store.memory_commit_receipt("s", "first"))
    assert receipt and count_receipts(pg_store) == 1
    asyncio.run(pg_store.commit_memory_operation(
        "s", "newer", memory_payload([newest], [replace(new, evidence_status="superseded")]))
    )
    restarted_store = PostgresStateStore(pg_store.dsn, table_prefix=pg_store.table_prefix)
    restarted_outbox = MemoryOutbox(path, restarted_store.memory_namespace)
    assert restarted_outbox.get("s", "first")["receipt"] is None
    assert asyncio.run(reconcile_memory(restarted_outbox, restarted_store, "s")) == [receipt]
    assert len(load(restarted_store)) == 3 and count_receipts(restarted_store) == 2
    m = memory()
    m.past_qas = load(restarted_store)
    assert [t.answer for t in retrieve(m, "Project Alpha deadline?")] == ["3 days"]


def test_concurrent_different_operations_cannot_both_replace_one_source(pg_store):
    old = source("14 days", 1)
    asyncio.run(pg_store.append_memory_traces("s", [old]))

    async def race():
        return await asyncio.gather(*(
            pg_store.commit_memory_operation("s", op, memory_payload(
                [source(answer, ts)], [replace(old, evidence_status="superseded")]))
            for op, answer, ts in (("one", "7 days", 2), ("two", "3 days", 3))
        ), return_exceptions=True)

    results = asyncio.run(race())
    assert sum(isinstance(result, MemoryCommitConflict) for result in results) == 1
    assert sum(isinstance(result, dict) for result in results) == 1
    assert len(load(pg_store)) == 2 and count_receipts(pg_store) == 1
    assert sum(t.evidence_status != "superseded" for t in load(pg_store)) == 1


def test_invalid_later_target_rolls_back_earlier_supersession(pg_store):
    old = source("14 days", 1)
    asyncio.run(pg_store.append_memory_traces("s", [old]))
    payload = memory_payload([source("7 days", 2)], [
        replace(old, evidence_status="superseded"),
        replace(old, qid="missing", evidence_status="superseded"),
    ])
    with pytest.raises(MemoryCommitConflict):
        asyncio.run(pg_store.commit_memory_operation("s", "invalid", payload))
    assert load(pg_store) == [old] and count_receipts(pg_store) == 0


def test_primary_loader_filters_before_limit_and_keeps_legacy_history(pg_store):
    current = source('14 days', 1, 'primary')
    traces = [current, source('EXPIRED', 2, 'expired'), source('SUPERSEDED', 3, 'superseded')]
    traces[1] = replace(traces[1], valid_until=0)
    traces[2] = replace(traces[2], evidence_status='superseded')
    traces += [replace(source('DERIVED', 100+i, f'derived-{i}'), recalled_answer=True) for i in range(32)]
    asyncio.run(pg_store.append_memory_traces('s', traces))
    # The legacy/history API still exposes recent QA, including derived answers.
    assert asyncio.run(pg_store.load_session('s', memory_limit=1))[0][0].recalled_answer
    assert asyncio.run(pg_store.load_conversation_sources('s', limit=1)) == [current]
    assert not asyncio.run(pg_store.load_conversation_sources('other', limit=1))
    assert not asyncio.run(pg_store.load_conversation_sources('s', limit=0))
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            asyncio.run(pg_store.load_conversation_sources('s', limit=invalid))
    assert len(load(pg_store)) == len(traces)


def test_primary_loader_has_stable_ties_and_preserves_legacy_default_metadata(pg_store):
    a, b = source('first', 1, 'first'), source('second', 1, 'second')
    asyncio.run(pg_store.append_memory_traces('s', [a, b]))
    # Existing rows without origin metadata retain their historical primary default.
    with pg_store._connect() as conn:
        conn.execute(f"UPDATE {pg_store._table('shared_memory')} SET evidence_metadata = '{{}}'::jsonb WHERE qid='first'")
    assert [t.qid for t in asyncio.run(pg_store.load_conversation_sources('s', limit=1))] == ['second']
    assert [t.qid for t in asyncio.run(pg_store.load_conversation_sources('s', limit=2))] == ['first', 'second']


def test_real_engine_restart_and_correction_replay_retain_buried_primary(pg_store, tmp_path, monkeypatch):
    from engine_stdio import EngineSession, _persist_session_memory, _prepare_memory_request, _memory_request
    original = source('14 days', 1, 'old-source')
    noise = [replace(source('DERIVED', 100+i, f'derived-{i}'), recalled_answer=True) for i in range(1030)]
    asyncio.run(pg_store.append_memory_traces('s', [original, *noise]))
    monkeypatch.setenv('DUALBRAIN_MEMORY_CONVERSATION_CONTEXT', '1')
    monkeypatch.setenv('DUALBRAIN_MEMORY_OUTBOX_PATH', str(tmp_path / 'outbox.sqlite3'))
    params = {'qid': 'new-source', 'question': 'Project Alpha deadline is now seven days',
              'memory_operation_id': 'source-correction',
              'memory_context': {'id': 'Alpha', 'aliases': ['Alpha'], 'remember': True, 'supersedes': ['old-source']}}

    async def run():
        first = await EngineSession.create('s', state_store=pg_store)
        try:
            assert first.memory.past_qas == [original]
            first.memory.begin_turn(params['question'], params['qid'], params['memory_context'])
            # Fixture ingestion validates persistence; no model answer is fabricated.
            first.memory.store({'Q': params['question'], 'A': '7 days'}, qid=params['qid'])
            receipt = await _persist_session_memory(first, params['qid'], operation_id=params['memory_operation_id'],
                request=_memory_request(params), result={'answer': '7 days'})
            assert [t.qid for t in first.memory.past_qas] == ['new-source']
        finally:
            await first.close()
        second = await EngineSession.create('s', state_store=pg_store)
        try:
            replay = await _prepare_memory_request(second, params)
            assert replay['replayed'] and replay['memory_commit'] == receipt
            assert [t.qid for t in second.memory.past_qas] == ['new-source']
            second.memory.begin_turn('Project Alpha deadline?', 'probe')
            assert [t.answer for t in second.memory.retrieve_related_traces('Project Alpha deadline?')] == ['7 days']
        finally:
            await second.close()
    asyncio.run(run())
    assert len(load(pg_store)) == 1032 and count_receipts(pg_store) == 1


@pytest.mark.parametrize('selected', [None, ['conditions', 'cold-conditions']])
def test_persistent_answer_runner_real_store_pairing_and_snapshot_reuse(pg_store, tmp_path, monkeypatch, selected):
    import benchmark_persistent_answers as bench
    import json
    from types import SimpleNamespace
    from core.llm_client import LLMConfig
    fixture = bench.read_json(bench.FIXTURE)
    fixture['episodes'] = [ep for ep in fixture['episodes'] if ep['split'] == 'development']
    config = LLMConfig('openai', 'fake-model', 'NEVER_PERSIST_THIS_KEY', auto_continue=False)
    args = SimpleNamespace(phase='development', output_dir=tmp_path / 'first', judge_model='fake-judge',
                           in_process=True, development_probes=selected)
    args.output_dir.mkdir()
    journal = bench.Journal(config, {}, max_calls=64)
    async def post(url, payload, headers, **kwargs):
        if payload.get('response_format', {}).get('type') == 'json_schema':
            packet = json.loads(payload['messages'][-1]['content'])
            text = json.dumps({'scores': {label: {c: {'verdict': 'pass', 'quote': answer}
                               for c in bench.CRITERIA} for label, answer in packet['answers'].items()}})
        elif payload.get('response_format'):
            packet = json.loads(payload['messages'][-1]['content'])
            text = json.dumps({'decisions': [{'id': r['id'], 'supported': False, 'quote': ''} for r in packet['records']]})
        elif 'verdict' in str(payload['messages']):
            text = json.dumps({'verdict': 'ok', 'issues': [], 'fixes': [], 'critic_sum': 'No defect', 'confidence_r': .9})
        else:
            text = '確認できる記録がありません。'
        return {'model': payload['model'], 'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    journal.transport._post_json = post
    report = asyncio.run(bench.run(args, fixture, config, journal))
    assert report['complete'] and report['grader_failures'] == 0
    expected = len(selected) if selected else 5
    assert report['summaries']['left']['probes'] == report['summaries']['collaborative']['probes'] == expected
    assert len(report['grades']) == 2*expected
    if selected:
        assert {row['id'] for row in report['rows']} == set(selected)
    assert report['usage_by_role']['right']['calls'] > 0
    assert 'NEVER_PERSIST_THIS_KEY' not in json.dumps(journal.data)
    sources = bench.read_json(args.output_dir / 'sources.json')
    args.output_dir = tmp_path / 'second'
    args.output_dir.mkdir()
    reused = bench.Journal(config, {}, max_calls=64)
    reused.transport._post_json = post
    repeated = asyncio.run(bench.run(args, fixture, config, reused, sources))
    assert repeated['sources'] == report['sources']
    assert 'seed' not in repeated['usage_by_role']
    # Spawned workers have fresh interpreters and may only use exact recorded
    # requests; a missing request fails instead of reaching the fake/live client.
    args.output_dir = tmp_path / 'spawned-replay'
    args.output_dir.mkdir()
    args.in_process = False
    frozen = bench.Journal(config, {}, frozen=journal.data, max_calls=64)
    replayed = asyncio.run(bench.run(args, fixture, config, frozen))
    assert replayed['fresh_processes'] and replayed['provider_calls'] == 0
    assert replayed['grades'] == report['grades']
    for arm in bench.ARMS:
        pids = {row['worker_pid'] for row in replayed['rows'] if row['arm'] == arm}
        assert len(pids) == 2
