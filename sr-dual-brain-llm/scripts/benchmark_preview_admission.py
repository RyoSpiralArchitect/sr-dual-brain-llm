#!/usr/bin/env python3
"""One fresh natural trajectory with predeclared transport-event replay.

Operational/source contracts only: no model semantic grade, paired savings
estimate or reuse of the #97 trace as an independent holdout.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_natural_memory as natural
from scheduled_http_journal import ScheduledJournal

pilot = natural.pilot
FIXTURE = pilot.base.ROOT / 'examples/preview_admission_trajectory.json'
CALL_LIMIT = 48


def metadata(fixture, config):
    return pilot.metadata(fixture, config) | {
        'experiment': fixture['origin'], 'call_limit': CALL_LIMIT,
        'policy': 'unmodified_production_policy_opt_in_all_auto',
        'drafts': 'native_generation_no_substitution',
        'semantic_grading': 'not_assessed_operational_source_contracts_only',
        'transport_schedule': 'ordered_start_response_error_and_owner_cancellation_v1',
        'replay_feedback': 'measured_latency_only_verify_all_non_timing_inputs',
        'repeat_rule': 'one_fixed_order_no_resampling_rescue_or_paid_diagnostics',
        'comparison': 'none_no_quality_latency_or_savings_effect_estimate',
    }


def validate(fixture):
    if (fixture.get('schema_version') != 1
            or fixture.get('origin') != 'prospective_preview_admission_trajectory'
            or tuple(t['id'] for t in fixture.get('turns', [])) != (
                'period_initial', 'backup_initial', 'ambiguous_replay', 'selected_replay',
                'period_corrected', 'old_selection', 'restart_selection', 'relisted_replay',
                'current_period', 'bare_parameter', 'explicit_rate_missing', 'explicit_period_missing')):
        raise ValueError('Require the complete fixed preview trajectory')


def canonical(report):
    return {key: report[key] for key in ('usage', 'served_models', 'feedback_sha256',
                                         'transport_events_sha256', 'grades')} | {
        'rows': [{k: v for k, v in row.items() if k not in
                  {'elapsed_ms', 'choice_elapsed_ms', 'restart_elapsed_ms'}} for row in report['rows']]}


def public_summary(report):
    return {k: v for k, v in report.items() if k != 'rows'} | {
        'rows': [{k: v for k, v in row.items() if k not in
                  {'answer', 'question', 'available_context', 'intent', 'memory_text'}} | {
                      'answer_sha256': pilot.bench.digest(row.get('answer'))} for row in report['rows']]}


async def run(output_dir, fixture, config, journal, dsn):
    validate(fixture)
    pilot.base.enforce_runtime()
    report = {'schema_version': 1, 'metadata': metadata(fixture, config), 'complete': False,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                  cwd=pilot.base.ROOT.parent, text=True).strip(), 'rows': [], 'grades': []}
    save = lambda: pilot.bench.checkpoint(output_dir / 'report.partial.json', report)
    clock = [fixture['clock']]
    store = pilot.PostgresStateStore(dsn, table_prefix='preview_admission_' + secrets.token_hex(6))
    options = dict(blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    env = pilot.profile.environment()
    env.update(DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir / 'outbox.sqlite3'),
               DUALBRAIN_PG_PERSIST_TELEMETRY='0')
    session = None
    schema_ready = False
    refs = {}
    tape = natural.FeedbackTape(journal)
    started = time.perf_counter()
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(
                uuid4=lambda: uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        try:
            await store.ensure_schema()
            schema_ready = True
            session = await natural.open_session(store, config, journal, clock, tape)
            for index, turn in enumerate(fixture['turns']):
                clock[0] = fixture['clock'] + index
                journal.case = turn['id']
                active = await store.load_conversation_sources(natural.SESSION_ID)
                row = {'id': turn['id'], 'status': 'not_executed', 'write': turn['memory_context']['remember'],
                       'answer': None, 'usage': pilot.bench.usage([]), 'contracts': {'turn_executed': False}}
                if not set(turn.get('requires', [])) <= {t.qid for t in active}:
                    row['reason'] = 'dependency_unavailable'
                else:
                    context = deepcopy(turn['memory_context'])
                    choice_ms = restart_ms = 0.
                    try:
                        choice = turn.get('choice_before_restart') or turn.get('choice')
                        if choice:
                            ref, choice_ms = await natural.choose(session,
                                {'scope': context['id'], **choice}, options, journal)
                            refs[choice['save_as']] = ref
                        if turn.get('reopen'):
                            restarting = time.perf_counter()
                            await session.close()
                            session = await natural.open_session(store, config, journal, clock, tape)
                            restart_ms = (time.perf_counter() - restarting) * 1000.
                        if turn.get('use_ref'):
                            context['record_ref'] = refs[turn['use_ref']]
                        elif turn.get('choice'):
                            context['record_ref'] = refs[turn['choice']['save_as']]
                    except (KeyError, ValueError):
                        row['reason'] = 'choice_unavailable'
                    else:
                        row = await natural.execute_turn(session, turn, context, store, options, journal)
                        row.update(choice_elapsed_ms=choice_ms, restart_elapsed_ms=restart_ms)
                report['rows'].append(row)
                report['grades'].append({'id': turn['id'], 'status': 'not_assessed',
                                         'reason': 'operational_source_contracts_only'})
                save()
                print(json.dumps({'id': row['id'], 'status': row['status'], 'preview': row.get('preview'),
                                  'usage': row['usage'], 'contracts': row.get('contracts')}), flush=True)
            report['episode_elapsed_ms'] = (time.perf_counter() - started) * 1000.
        finally:
            if session:
                await session.close()
            if schema_ready:
                with store._connect() as conn:
                    for name in pilot.base.TABLES:
                        conn.execute(f'DROP TABLE IF EXISTS {store._table(name)} CASCADE')
    report['feedback_sha256'] = tape.finish()
    journal.finish()
    report.update(complete=True, live_provider_calls=journal.calls, usage=pilot.bench.usage(journal.events),
        calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
        served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}),
        transport_events_sha256=journal.data['transport_events_sha256'],
        transport_event_count=len(journal.data['transport_events']))
    report['trajectory_complete'] = all(r['status'] == 'ok' for r in report['rows'])
    report['contract_failures'] = [{'case': r['id'], 'contract': k} for r in report['rows']
                                   for k, v in r.get('contracts', {}).items() if v is False]
    report['canonical_sha256'] = pilot.bench.digest(canonical(report))
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--max-calls', type=int, choices=(CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        fixture = pilot.bench.read_json(FIXTURE)
        validate(fixture)
        frozen = pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'],
                cwd=pilot.base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze implementation, fixture and protocol before paid calls')
        model = args.model or frozen['metadata']['config']['model']
        config = pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig('openai', model, 'REPLAY_ONLY')
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        dsn = os.environ.get('DUALBRAIN_TEST_PG_DSN')
        if not dsn:
            raise ValueError('Isolated test PostgreSQL required')
        journal = ScheduledJournal(config, metadata(fixture, config), frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: pilot.bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            with pilot.forbid_provider_network() as audit:
                report = asyncio.run(run(args.output_dir, fixture, config, journal, dsn))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report = asyncio.run(run(args.output_dir, fixture, config, journal, dsn))
        pilot.bench.write_private(args.output_dir / 'report.json', report)
        pilot.bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        pilot.bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': journal.calls,
                          'usage': report['usage'], 'failures': report['contract_failures']}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Preview admission confirmation incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
