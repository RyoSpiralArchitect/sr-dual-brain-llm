#!/usr/bin/env python3
"""Model-free PostgreSQL retention, cold recall, correction and latency experiment.

Requires DUALBRAIN_TEST_PG_DSN for an isolated DB. Writes only a unique disposable
prefix; never uses the application DSN. Authored source selection is measured,
not natural-language understanding or final-answer quality.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.memory_commits import canonical_json
from core.postgres_state_store import PostgresStateStore
from core.shared_memory import MemoryTrace, SharedMemory
from engine_stdio import EngineSession, _persist_session_memory, _prepare_memory_request, _memory_request


FACTS = 8
TABLES = ('memory_commits', 'shared_memory', 'hippocampal_episodes', 'telemetry_events',
          'schema_memory', 'consolidation_state', 'sessions')


def fixture(history):
    sources = [MemoryTrace(f'Project Atlas{i} deadline', f'{14+i} calendar days; Europe only.',
                           qid=f'source-{i}', timestamp=1_000_000_000.+i,
                           retrieval_scope=f'Atlas{i}', scope_aliases=(f'Atlas{i}',)) for i in range(FACTS)]
    derived = [MemoryTrace('Project Atlas0 deadline', 'DERIVED_DO_NOT_RECALL', qid=f'derived-{i}',
                           timestamp=1_000_000_100.+i, recalled_answer=True) for i in range(history)]
    decoys = [MemoryTrace('Project Atlas0 deadline', 'STALE_DO_NOT_RECALL', qid=f'decoy-{i}',
                          timestamp=1_100_000_000.+i, retrieval_scope='Atlas0', scope_aliases=('Atlas0',),
                          evidence_status='superseded' if i == 0 else 'unverified',
                          valid_until=None if i == 0 else 1.) for i in range(2)]
    return sources, [*sources, *derived, *decoys]


def probe(memory, sources, repetitions):
    hits, bad, samples = 0, [], []
    for i, source in enumerate(sources):
        question = f'Project Atlas{i} deadline?'
        selected = []
        for _ in range(repetitions):
            start = time.perf_counter()
            memory.begin_turn(question, 'probe')
            selected = memory.retrieve_related_traces(question)
            samples.append((time.perf_counter()-start)*1000)
        hits += any(t.qid == source.qid and t.answer == source.answer for t in selected)
        bad.extend(t.qid for t in selected if t.qid != source.qid)
    return {'hits': hits, 'expected': len(sources), 'recall': hits/len(sources),
            'unexpected_qids': sorted(set(bad)), 'resident_traces': len(memory.past_qas),
            'retrieval_ms': timing(samples)}


def timing(samples):
    ordered = sorted(samples)
    return {'median': statistics.median(ordered),
            'p95': ordered[min(len(ordered)-1, int(.95*len(ordered)))], 'samples': len(samples)}


async def correct(session, number, target):
    params = {'qid': f'correction-{number}', 'question': f'Project Atlas0 deadline is now {number} days',
              'memory_operation_id': f'correction-op-{number}',
              'memory_context': {'id': 'Atlas0', 'aliases': ['Atlas0'], 'remember': True, 'supersedes': [target]}}
    memory = session.memory
    memory.begin_turn(params['question'], params['qid'], params['memory_context'])
    # This is an authored fixture ingestion, not a generated source completion.
    answer = f'{number} calendar days; Europe only.'
    memory.store({'Q': params['question'], 'A': answer}, qid=params['qid'])
    receipt = await _persist_session_memory(session, params['qid'], operation_id=params['memory_operation_id'],
                                            request=_memory_request(params), result={'answer': answer})
    return params, receipt


async def case(store, history, repetitions):
    sources, traces = fixture(history)
    fixture_hash = hashlib.sha256(canonical_json([asdict(t) for t in traces]).encode()).hexdigest()
    sid = f'history-{history}'
    await store.append_memory_traces(sid, traces)
    warm = SharedMemory(conversation_context=True)
    start = time.perf_counter()
    for trace in traces:
        warm.store(trace)
    warm_build_ms = (time.perf_counter()-start)*1000
    row = {'derived_history': history, 'total_rows_before_correction': len(traces), 'fixture_sha256': fixture_hash,
           'warm_build_ms': warm_build_ms, 'warm': probe(warm, sources, repetitions)}
    cold_times, cold_results = [], []
    session = None
    for _ in range(3):
        start = time.perf_counter()
        loaded = await EngineSession.create(sid, state_store=store)
        cold_times.append((time.perf_counter()-start)*1000)
        cold_results.append(probe(loaded.memory, sources, repetitions))
        if session is not None:
            await session.close()
        session = loaded
    try:
        row['cold_start_ms'] = timing(cold_times)
        row['cold'] = cold_results[-1]
        row['cold_repeat_hits'] = [r['hits'] for r in cold_results]
        # Focus is deliberately not durable: a pronoun alone must not guess.
        previous = session.memory._focus_scope
        session.memory._focus_scope = None
        session.memory.begin_turn('その締切は？', 'pronoun-after-restart')
        row['unfocused_pronoun_withheld'] = not session.memory.retrieve_related_traces('その締切は？')
        session.memory._focus_scope = previous
        try:
            first, receipt = await correct(session, 7, 'source-0')
        except ValueError as exc:
            row['correction'] = {'accepted': False, 'error_type': type(exc).__name__}
        else:
            await correct(session, 3, 'correction-7')
            replay = await _prepare_memory_request(session, first)
            session.memory.begin_turn('Project Atlas0 deadline?', 'after-old-replay')
            selected = session.memory.retrieve_related_traces('Project Atlas0 deadline?')
            active = [t.qid for t in selected]
            current = active == ['correction-3']
            # A cancelled correction must not hide or supersede the committed source.
            focus = session.memory._focus_scope
            session.memory.begin_turn('Project Atlas0 deadline correction', 'abandoned',
                {'id': 'Atlas0', 'aliases': ['Atlas0'], 'remember': True, 'supersedes': ['correction-3']})
            hidden = not session.memory.retrieve_related_traces('Project Atlas0 deadline?')
            session.memory.abort_turn(focus)
            restored = [t.qid for t in session.memory.retrieve_related_traces('Project Atlas0 deadline?')]
            row['correction'] = {'accepted': True, 'old_receipt_replayed': bool(replay['replayed']),
                                 'same_receipt': replay['memory_commit'] == receipt,
                                 'latest_source_only': current, 'active_qids_after_old_replay': active,
                                 'pending_hidden': hidden, 'cancel_restores_current': restored == ['correction-3']}
        if row['warm']['unexpected_qids'] or row['cold']['unexpected_qids']:
            raise RuntimeError('Ineligible or wrong-scope fixture record reached retrieval')
    finally:
        await session.close()
    return row


async def run(args):
    dsn = os.environ.get('DUALBRAIN_TEST_PG_DSN')
    if not dsn:
        raise ValueError('DUALBRAIN_TEST_PG_DSN must identify an isolated test database')
    prefix = 'persistent_bench_' + uuid.uuid4().hex[:10]
    store = PostgresStateStore(dsn, table_prefix=prefix)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['DUALBRAIN_MEMORY_CONVERSATION_CONTEXT'] = '1'
    os.environ['DUALBRAIN_MEMORY_OUTBOX_PATH'] = str(args.output_dir / 'outbox.sqlite3')
    await store.ensure_schema()
    try:
        # Exclude one-time model/module initialization from session hydration.
        warmup = await EngineSession.create('benchmark-process-warmup')
        await warmup.close()
        rows = []
        for history in args.history_sizes:
            rows.append(await case(store, history, args.repetitions))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        with store._connect() as conn:
            version = conn.execute('SHOW server_version').fetchone()[0]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           cwd=Path(__file__).resolve().parents[2]).strip()
        return {'schema_version': 1, 'experiment': 'postgres_persistent_memory_retention',
                'source_revision': revision, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'postgres_version': version, 'provider_calls': 0, 'answer_quality_evaluated': False,
                'resident_budget': 1024, 'source_count': FACTS, 'unmeasured_process_warmups': 1, 'cold_starts_per_size': 3,
                'retrieval_repetitions': args.repetitions, 'rows': rows}
    finally:
        with store._connect() as conn:
            for name in TABLES:
                conn.execute(f'DROP TABLE IF EXISTS {store._table(name)} CASCADE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--history-sizes', type=int, nargs='+', default=[128, 1024, 8192, 32768])
    parser.add_argument('--repetitions', type=int, default=9)
    args = parser.parse_args()
    if args.repetitions < 1 or any(n < 0 for n in args.history_sizes):
        parser.error('history sizes must be nonnegative and repetitions positive')
    report = asyncio.run(run(args))
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
