#!/usr/bin/env python3
"""Fresh, bounded operational checks of #91 using real HTTP and PostgreSQL.

An acceptance pilot, not a before/after quality comparison. Exact replay uses
the original raw envelopes; invalid completions reach the production client.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import answer_assessment_v3 as assessment
import benchmark_memory_review as profile
import benchmark_module_gating as bench
from offline_provider_guard import forbid_provider_network
from core.llm_client import LLMClient, LLMConfig, load_llm_config
from core.postgres_state_store import PostgresStateStore
from core.shared_memory import MemoryTrace

base = bench.base
CALL_LIMIT = 128
REPLAY_ERRORS = {'CancelledError': asyncio.CancelledError, 'TimeoutError': TimeoutError,
                 'RuntimeError': RuntimeError, 'ValueError': ValueError, 'OSError': OSError}
STAGE = ContextVar('live_contract_stage', default=None)
FIXTURE = base.ROOT / 'examples/contract_live_verification.json'
IDS = ('caller_scope', 'period_missing', 'rate_missing', 'bare_retention',
       'reference_ambiguous', 'record_ambiguous', 'record_missing',
       'support_unknown', 'critic_on', 'critic_off')


def validate(fixture):
    if (fixture.get('schema_version') != 1
            or fixture.get('origin') != 'prospective_contract_operational_verification'
            or tuple(c['id'] for c in fixture.get('cases', [])) != IDS
            or tuple(c['id'] for c in fixture.get('durable_turns', [])) != (
                'write_initial', 'quote_initial', 'write_correction', 'quote_corrected', 'quote_pronoun')):
        raise ValueError('Require the complete fresh operational fixture')
    for case in fixture['cases']:
        if not case['question'].strip() or not case.get('manual_criterion'):
            raise ValueError('A question and prospective qualitative criterion are required')
        for record in case['records']:
            MemoryTrace(**record)


class AdmissionError(RuntimeError):
    pass


class RawJournal:
    """One entry per attempted HTTP occurrence, in admission order.

    Never reject or normalize a returned provider envelope here. That is the
    production completion validator's responsibility. The bounded sticky latch
    prevents a retry from spending again after a real transport failure.
    """
    def __init__(self, config, metadata, *, frozen=None, save=None, max_calls=CALL_LIMIT):
        if type(max_calls) is not int or not 1 <= max_calls <= CALL_LIMIT:
            raise ValueError('Invalid frozen call cap')
        self.max_calls, self.save, self.frozen = max_calls, save, frozen
        self.transport = None if frozen else LLMClient(config)
        self.data = {'metadata': metadata, 'complete': False, 'occurrences': [], 'denied_admissions': []}
        self.calls, self.position, self.events = 0, 0, []
        self.case, self.blocked, self.divergences = None, False, []
        if frozen:
            if not frozen.get('complete') or frozen.get('metadata') != metadata:
                raise ValueError('Journal metadata differs or run was incomplete')
            if (frozen.get('occurrences_sha256') != bench.digest(frozen['occurrences'])
                    or len(frozen['occurrences']) > max_calls):
                raise ValueError('Corrupt raw HTTP occurrence sequence')
            for row in frozen['occurrences']:
                if row.get('sha256') != self.row_digest(row):
                    raise ValueError('Corrupt raw HTTP occurrence')
                if row.get('error_type') and row['error_type'] not in REPLAY_ERRORS:
                    raise ValueError('Unsupported transport exception replay')

    @staticmethod
    def row_digest(row):
        return bench.digest({k: v for k, v in row.items() if k not in {'sha256', 'latency_ms'}})

    def checkpoint(self):
        if self.save:
            self.save(self.data)

    async def post(self, role, url, payload, headers, **kwargs):
        request = {'case': self.case, 'role': role,
                   'endpoint_sha256': hashlib.sha256(url.encode()).hexdigest(),
                   'payload': deepcopy(payload)}
        if self.blocked or len(self.events) >= self.max_calls:
            self.data['denied_admissions'].append({'request_sha256': bench.digest(request),
                                                  'reason': 'transport_blocked' if self.blocked else 'call_cap'})
            self.checkpoint()
            raise AdmissionError('Paid transport is closed')
        row = {'request': request}
        event = {'role': role, 'case': self.case, 'input_tokens': None, 'output_tokens': None,
                 'error_type': None, 'served_model': None}
        # Allocate before awaiting, so overlapping requests preserve admission
        # order independently of the order in which they finish.
        self.data['occurrences'].append(row)
        self.events.append(event)
        started = time.perf_counter()
        try:
            if self.frozen:
                index = self.position
                self.position += 1
                rows = self.frozen['occurrences']
                if index >= len(rows) or rows[index]['request'] != request:
                    self.divergences.append(index)
                    raise AdmissionError('Exact request occurrence differs')
                recorded = rows[index]
                if recorded.get('error_type'):
                    category = recorded['error_type']
                    row.update({k: v for k, v in recorded.items() if k != 'request'})
                    # Preserve the fixed retry marker without storing exception
                    # text, which may include credentials or provider content.
                    raise REPLAY_ERRORS[category]('Recorded transport failure ' + (row.get('retry_marker') or ''))
                response = deepcopy(recorded['response'])
            else:
                if self.transport is None:
                    raise AdmissionError('Live transport unavailable')
                self.calls += 1
                response = await self.transport._post_json(url, payload, headers, **kwargs)
            row['response'] = response
            return response
        except BaseException as exc:
            # Preserve the original safe receipt when a replay wrapper differs.
            row.setdefault('error_type', type(exc).__name__)
            if not self.frozen and not isinstance(exc, (asyncio.CancelledError, AdmissionError)):
                row['retry_marker'] = next((marker for marker in ('rate limit', 'rate_limited', '429', 'timeout')
                                            if marker in str(exc).lower()), None)
                status = re.search(r' returned (\d{3}):', str(exc))
                if status:
                    row['http_status'] = int(status.group(1))
            if not isinstance(exc, (asyncio.CancelledError, AdmissionError)):
                self.blocked = True
            raise
        finally:
            row['latency_ms'] = (time.perf_counter() - started) * 1000
            row['sha256'] = self.row_digest(row)
            response = row.get('response')
            usage = response.get('usage') if isinstance(response, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            for field, key in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')):
                value = usage.get(key)
                event[field] = value if type(value) is int and value >= 0 else None
            event.update(error_type=row.get('error_type'),
                         served_model=response.get('model') if isinstance(response, dict) else None)
            self.checkpoint()

    def finish(self):
        if self.frozen and (self.divergences or self.position != len(self.frozen['occurrences'])
                            or self.data['denied_admissions'] != self.frozen.get('denied_admissions', [])):
            raise ValueError('Exact occurrence coverage differs')
        self.data.update(complete=True, live_provider_calls=self.calls,
                         transport_blocked=self.blocked, usage=bench.usage(self.events),
                         occurrences_sha256=bench.digest(self.data['occurrences']))
        self.checkpoint()


def code_digest():
    paths = sorted(set((base.ROOT / 'core').glob('*.py')) | set((base.ROOT / 'scripts').glob('*.py')))
    return bench.digest([(str(p.relative_to(base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])


def metadata(fixture, config):
    return {'experiment': fixture['origin'], 'version': 1, 'fixture_sha256': bench.digest(fixture),
            'code_sha256': code_digest(), 'config': bench.public_config(config), 'call_limit': CALL_LIMIT,
            'runtime': profile.environment_contract(), 'transport': 'production_nonstream_json',
            'torch_available': bench.TORCH_AVAILABLE,
            'semantic_grading': 'not_assessed_manual_observations_only',
            'source_diagnostics': assessment.VERSION}


async def make_session(sid, store, config, journal, clock, *, seed=False):
    logical_time = clock[0]
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(int(bench.digest([sid, seed])[:8], 16))
    session, clients = await base.session_for(sid, store, config, journal, 'current', clock, seed=seed)
    clients = [StageClient(config, journal, role) for role in ('left', 'right')]
    session.left._llm_client, session.right._llm_client = clients
    for model, method, stage in (
        (session.left, 'generate_answer', 'source_write' if seed else 'draft'),
        (session.left, 'integrate_info_async', 'integration'),
        (session.right, 'deepen', 'detail'),
        (session.right, 'criticise_reasoning', 'critic')):
        track_stage(model, method, stage)
    clock[0] = logical_time
    return session, clients


class StageClient(base.RecordedClient):
    def __init__(self, config, journal, role):
        super().__init__(config, journal, role)
        async def post(url, payload, headers, **kwargs):
            stage = STAGE.get() or role
            if stage == 'critic' and payload['messages'][-1]['content'].startswith('Your previous response'):
                stage = 'critic_repair'
            return await journal.post(stage, url, payload, headers, **kwargs)
        self.client._post_json = post


def track_stage(model, method, stage):
    original = getattr(model, method)
    async def tracked(*args, **kwargs):
        token = STAGE.set(stage)
        try:
            return await original(*args, **kwargs)
        finally:
            STAGE.reset(token)
    setattr(model, method, tracked)


def source_snapshot(memory, question):
    traces = memory.retrieve_related_traces(question)
    _, cue = memory._retrieval_scope_cue(memory.retrieval_question(question))
    records, checks = [], {}
    gate = memory.answerability_gate
    for trace in traces:
        source = memory._answerability_source(trace)
        records.append({**source, 'recalled_answer': trace.recalled_answer})
        checks[trace.qid] = {
            'current': memory._available(trace, time.time()) and not trace.recalled_answer
                       and trace.evidence_status in {'verified', 'unverified'},
            'complete': not source['truncated'] and bool(source['question'].strip())
                        and bool(source['answer'].strip())
                        and len(source['question']) <= (gate.max_question_chars if gate else 2048)
                        and len(source['answer']) <= (gate.max_answer_chars if gate else 2048),
            'scope': not cue.ambiguous and trace.retrieval_scope in cue.scopes if cue.scopes else None,
            'source_approved': gate.decision(memory.retrieval_question(question), source,
                                            intent=memory.replay_intent(question)) if gate else None,
        }
    condition = memory.missing_evidence(question)
    intent = memory.replay_intent(question)
    return {'available_context': {'reference': memory.reference_resolution(question),
                'approved_records': records, 'source_checks_by_qid': checks,
                'evaluation_time': time.time()},
            'source_qids': [t.qid for t in traces],
            'condition': condition.to_dict() if condition else None,
            'intent': intent.identity() if intent else None}


def accounting(metrics, telemetry):
    system2 = metrics.get('system2') or {}
    execution = next((e for e in reversed(telemetry) if e.get('event') == 'system2_execution'), {})
    attempts = execution.get('attempts') or []
    rounds = sorted({a['round'] for a in attempts})
    latest = attempts[-1] if attempts else {}
    returned = latest.get('outcome') == 'returned'
    return {'mode': system2.get('mode'), 'enabled': system2.get('enabled'),
            **{k: system2.get(k) for k in ('used', 'critic_attempts', 'critic_responses', 'attempted_rounds',
                                          'review_completed', 'external_review_completed', 'rounds')},
            'attempts': attempts,
            'consistent': (system2.get('used') == bool(attempts)
                and system2.get('critic_attempts') == len(attempts)
                and system2.get('critic_responses') == sum(a['outcome'] == 'returned' for a in attempts)
                and system2.get('attempted_rounds') == rounds
                and system2.get('review_completed') == (latest.get('review_valid') if returned else False)
                and system2.get('external_review_completed') == (latest.get('external_review_valid') if returned else False)
                and system2.get('rounds') == len(rounds))}


async def turn(session, clients, case, journal, options):
    journal.case = case['id']
    before, started = len(journal.events), time.perf_counter()
    prior_failures = [c.failed for c in clients]
    row = {'id': case['id'], 'status': 'unavailable', 'answer': None, 'question': case['question']}
    captures = []
    preparations = []
    prepare = session.memory.prepare_retrieval
    generate = session.left.generate_answer
    async def observed_prepare(question, **kwargs):
        result = await prepare(question, **kwargs)
        preparations.append(deepcopy(result))
        captures.append(source_snapshot(session.memory, question))
        return result
    async def observed_generate(*args, **kwargs):
        if not captures:
            captures.append(source_snapshot(session.memory, case['question']))
        return await generate(*args, **kwargs)
    session.memory.prepare_retrieval = observed_prepare
    session.left.generate_answer = observed_generate
    try:
        params = {'question': case['question'], 'qid': case['id'], 'leading_brain': 'left',
                  'system2_mode': case.get('system2_mode', 'auto'),
                  'executive_mode': 'off', 'executive_observer_mode': 'off', 'return_telemetry': True}
        for name in ('memory_context', 'memory_operation_id'):
            if name in case:
                params[name] = case[name]
        result = await base._handle_process(session, params, **options)
        await session.callosum.wait_idle()
        answer = result.get('answer')
        if isinstance(answer, str) and answer.strip():
            row.update(status='ok', answer=answer)
        row['system2'] = accounting(result.get('metrics') or {}, result.get('telemetry') or [])
        receipt = result.get('memory_commit') or {}
        row['commit'] = {k: receipt.get(k) for k in ('state', 'operation_id')}
        row['metrics'] = result.get('metrics')
    except Exception as exc:
        row['error_type'] = type(exc).__name__
    finally:
        await session.callosum.wait_idle()
        session.memory.prepare_retrieval = prepare
        session.left.generate_answer = generate
    row.update(captures[-1] if captures else source_snapshot(session.memory, case['question']))
    row['preparations'] = preparations
    row['client_failures'] = [{'role': c.role, 'count': c.failed - n,
                               'types': c.failure_types[n:]} for c, n in zip(clients, prior_failures)]
    row['usage'] = bench.usage(journal.events[before:])
    row['calls_by_role'] = dict(Counter(e['role'] for e in journal.events[before:]))
    row['elapsed_ms'] = (time.perf_counter() - started) * 1000
    row['stages_available'] = row['usage']['failed_calls'] == 0 and not any(c['count'] for c in row['client_failures'])
    # Answer availability is independent of an invalid critic recovered locally.
    return row


def assess_row(case, row, target=None):
    packet = {'requested_operation': case.get('requested_operation', 'fact_lookup'),
              'available_context': row['available_context']}
    if target is not None:
        packet['replay_target'] = {k: getattr(target, k) for k in ('qid', 'retrieval_scope', 'answer')}
    row['diagnostics'] = assessment.public_diagnostics(assessment.diagnostics(packet, row['answer']))
    expected = case.get('expected_sources')
    condition = row.get('condition') or {}
    row['contracts'] = {
        'answer_available': row['status'] == 'ok',
        'accounting': (row.get('system2') or {}).get('consistent', False),
        'source_selection': row['source_qids'] == expected if expected is not None else None,
        'missing_condition': condition.get('action') == case['expected_action'] if 'expected_action' in case else None,
        'scope_no_leak': not any(value in (row['answer'] or '') for value in case['forbidden_values'])
                         if case.get('forbidden_values') else None,
        'exact_replay': row['diagnostics']['exact_match']['verdict'] == 'pass'
                        if case.get('target_qid') else None,
        'durable_commit': (row['commit'].get('state') == 'committed'
                           and row['commit'].get('operation_id') == case['memory_operation_id'])
                           if 'memory_operation_id' in case else None,
        'explicit_critic_mode': ((row.get('system2') or {}).get('used') is (case['system2_mode'] == 'on'))
                               if case.get('id') in {'critic_on', 'critic_off'} else None,
        'healthy_external_review': (row.get('system2') or {}).get('external_review_completed') is True
                                    if case.get('id') == 'critic_on' else None,
    }


async def durable_trajectory(fixture, config, journal, clock, options, store, save):
    rows = []
    session = None
    clients = []
    for index, case in enumerate(fixture['durable_turns']):
        clock[0] = fixture['clock'] + index
        journal.case = case['id']
        seed = bool(case.get('write'))
        if case.get('reopen', True):
            if session:
                await session.close()
            session, clients = await make_session('contract-live:durable', store, config, journal, clock, seed=seed)
        try:
            target_id = case.get('target_qid')
            target = next((t for t in session.memory.past_qas if t.qid == target_id), None)
            row = await turn(session, clients, case, journal, options)
            active = await store.load_conversation_sources(session.session_id)
            row['active_primary_qids'] = [t.qid for t in active]
            row['source_sha256'] = bench.digest([asdict(t) for t in active])
            if seed:
                written = [t for t in active if t.qid == case['id']]
                row['primary_write_complete'] = (len(written) == 1 and not written[0].recalled_answer
                    and written[0].answer == row['answer'] and not session.memory._answerability_source(written[0])['truncated'])
            assess_row(case, row, target)
            row['contracts']['active_primaries'] = row['active_primary_qids'] == case['expected_active']
            if seed:
                row['contracts']['primary_write_complete'] = row['primary_write_complete']
            rows.append(row)
            save(row)
        except BaseException:
            if session:
                await session.close()
            raise
    if session:
        await session.close()
    return rows


def canonical(report):
    ignored = {'metrics', 'elapsed_ms'}
    return {'rows': [{k: v for k, v in row.items() if k not in ignored} for row in report['rows']],
            'usage': report['usage'], 'served_models': report['served_models']}


async def run(output_dir, fixture, config, journal, *, dsn):
    validate(fixture)
    base.enforce_runtime()
    report = {'schema_version': 1, 'metadata': metadata(fixture, config), 'complete': False,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.ROOT.parent, text=True).strip(),
              'rows': []}
    save = lambda _row=None: bench.checkpoint(output_dir / 'report.partial.json', report)
    clock = [fixture['clock']]
    options = dict(blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    env = profile.environment()
    env.update(DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir / 'outbox.sqlite3'), DUALBRAIN_PG_PERSIST_TELEMETRY='0')
    # A fresh private prefix is allocated before patching UUIDs; replay never
    # shares application tables or an application outbox with the live run.
    store = PostgresStateStore(dsn, table_prefix='contract_live_' + secrets.token_hex(6))
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(
                uuid4=lambda: uuid.UUID(int=base.UUID_RNG.getrandbits(128)))):
        try:
            await store.ensure_schema()
            async def run_durable():
                return await durable_trajectory(fixture, config, journal, clock, options, store,
                                               lambda row: (report['rows'].append(row), save()))
            await run_durable()
            for index, case in enumerate(fixture['cases']):
                clock[0] = fixture['clock'] + 20 + index
                journal.case = case['id']
                session, clients = await make_session('contract-live:' + case['id'], None, config, journal, clock)
                try:
                    for record in case['records']:
                        session.memory.store(MemoryTrace(**record))
                    row = await turn(session, clients, case, journal, options)
                    assess_row(case, row)
                    report['rows'].append(row)
                    save()
                finally:
                    await session.close()
                print(json.dumps({'case': case['id'], 'status': row['status'],
                                  'contracts': row['contracts'], 'usage': row['usage']}), flush=True)
            journal.finish()
            report.update(complete=True, live_provider_calls=journal.calls, usage=bench.usage(journal.events),
                          calls_by_role=dict(Counter(e['role'] for e in journal.events)),
                          served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}))
            report['contract_failures'] = [{'case': r['id'], 'contract': key}
                                          for r in report['rows'] for key, value in r['contracts'].items() if value is False]
            report['canonical_sha256'] = bench.digest(canonical(report))
            save()
            return report
        finally:
            # Drop only this invocation's explicitly owned table namespace.
            with store._connect() as conn:
                for name in base.TABLES:
                    conn.execute(f'DROP TABLE IF EXISTS {store._table(name)} CASCADE')


def public_summary(report):
    return {k: v for k, v in report.items() if k != 'rows'} | {
        'rows': [{k: v for k, v in row.items() if k not in {'answer', 'question', 'available_context', 'intent', 'metrics'}}
                 | {'answer_sha256': bench.digest(row['answer'])} for row in report['rows']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--replay-api-base', help='Optional public endpoint for exact replay of a custom endpoint')
    parser.add_argument('--max-calls', type=int, choices=(CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        fixture = bench.read_json(FIXTURE)
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze code and cases in a clean commit before paid calls')
        dsn = os.environ.get('DUALBRAIN_TEST_PG_DSN')
        if not dsn:
            raise ValueError('An isolated DUALBRAIN_TEST_PG_DSN is required')
        model = args.model or frozen['metadata']['config']['model']
        config = load_llm_config(default_model=model) if not frozen else LLMConfig('openai', model, 'REPLAY_ONLY')
        if config is None or config.provider != 'openai':
            raise ValueError('An OpenAI-compatible configuration is required')
        if frozen:
            config = replace(config, api_base=args.replay_api_base)
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        journal = RawJournal(config, metadata(fixture, config), frozen=frozen,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            with forbid_provider_network() as audit:
                report = asyncio.run(run(args.output_dir, fixture, config, journal, dsn=dsn))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report = asyncio.run(run(args.output_dir, fixture, config, journal, dsn=dsn))
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': report['complete'], 'live_provider_calls': journal.calls,
                          'usage': report['usage'], 'contract_failures': report['contract_failures']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Operational verification incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
