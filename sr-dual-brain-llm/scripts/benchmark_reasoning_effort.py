#!/usr/bin/env python3
"""Fixed fresh provider-default/low comparison on the completion-aware runtime."""
import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_task_goals as study
from core.task_goals import extract_task_plan
from engine_stdio import _dispatch_request

pilot = study.pilot
natural = study.natural
ScheduledJournal = study.ScheduledJournal
assess, assignment_exposure = study.assess, study.assignment_exposure
canonical, public_summary = study.canonical, study.public_summary
FIXTURE = pilot.base.ROOT / 'examples/reasoning_effort_comparison.json'
CALL_LIMIT, RUNTIME_LIMIT = 64, 59
PROFILES = ('provider_default', 'low')
IDS = ('lower_bound_search', 'conditional_tests', 'order_totals', 'inverse_measurements', 'direct_control')


def metadata(fixture, config, judge):
    return study.metadata(fixture, config, judge) | {
        'experiment': fixture['origin'], 'arms': {p: p for p in PROFILES},
        'goal_orchestration': 'auto_in_both_arms', 'left_output_ceiling_allocation': 8192,
        'effort_scope': 'ordinary_draft_integration_detail_only',
        'semantic_grading': 'fresh_effort_quality_uncalibrated_single_model',
    }


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_bounded_reasoning_effort_comparison'
            or tuple(c['id'] for c in fixture.get('cases', [])) != IDS):
        raise ValueError('Require all five frozen fresh cases')
    for case in fixture['cases']:
        plan = extract_task_plan(case['question'])
        if ((len(plan.goals) if plan else 0) != case['expected_goal_count']
                or sorted(case['arms']) != sorted(PROFILES)
                or sorted(case['judge_labels']) != ['A', 'B']
                or sorted(case['judge_labels'].values()) != sorted(PROFILES)
                or not case['reference'].strip()):
            raise ValueError('Require exact goals, profiles, blind labels and references')


def comparison(grades):
    # Reuse the unchanged criterion aggregator, explicitly translating condition
    # names for this fixed experiment rather than treating low as an AUTO gate.
    normalized = [{**g, **({'scores': {'off': g['scores']['provider_default'], 'auto': g['scores']['low']}}
                  if 'scores' in g else {})} for g in grades]
    result = study.comparison(normalized)
    rename = {'auto_only_pass': 'low_only_pass', 'off_only_pass': 'provider_default_only_pass'}
    result['totals'] = {group: {c: {rename.get(k, k): v for k, v in counts.items()}
        for c, counts in criteria.items()} for group, criteria in result['totals'].items()}
    result['pairs'] = [{**p, 'outcome': rename.get(p['outcome'], p['outcome'])} for p in result['pairs']]
    return result


async def run(output_dir, fixture, config, judge, journal):
    validate(fixture)
    pilot.base.enforce_runtime()
    report = {'schema_version': 1, 'metadata': metadata(fixture, config, judge), 'complete': False,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.base.ROOT.parent, text=True).strip(),
        'rows': [], 'grades': []}
    save = lambda: pilot.bench.checkpoint(output_dir / 'report.partial.json', report)
    clock = [fixture['clock']]
    env = pilot.profile.environment()
    env.update(DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS='2048',
               DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir / 'outbox.sqlite3'))
    journal.max_calls = RUNTIME_LIMIT
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(fixture['cases']):
            for arm in case['arms']:
                os.environ['DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE'] = 'auto'
                os.environ['DUALBRAIN_ANSWER_REASONING_EFFORT'] = arm
                os.environ['DUALBRAIN_ANSWER_TOKEN_BUDGET'] = '8192'
                clock[0] = fixture['clock'] + index
                journal.case = case['id'] + ':' + arm
                before = len(journal.events)
                started = time.perf_counter()
                row = {'case': case['id'], 'arm': arm, 'reasoning_profile': arm, 'question': case['question'], 'answer': None, 'status': 'unavailable'}
                session = None
                try:
                    # The same SID resets the same policy/RNG initialization in both fresh arms.
                    session, clients = await pilot.make_session('task-goals:' + case['id'], None, config, journal, clock, seed=False)
                    reply = await _dispatch_request({'id': journal.case, 'method': 'process', 'params': {
                        'session_id': session.session_id, 'question': case['question'], 'qid': journal.case,
                        'system2_mode': case['system2_mode'], 'executive_mode': 'off', 'executive_observer_mode': 'off',
                        'return_telemetry': True}}, sessions={session.session_id: session}, state_store=None,
                        blob_pipe=None, send=lambda *_: None, blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
                    if reply.get('ok'):
                        value = reply['result']
                        metrics = value['metrics']
                        row.update(status='ok', answer=value['answer'], task_goals=metrics.get('task_goals'),
                            answer_generation=metrics.get('answer_generation'),
                            policy_class=type(session.controller.policy).__name__, policy_action=metrics['policy']['action'],
                            system2=pilot.accounting(metrics, value['telemetry']), detail=metrics.get('detail'),
                            preview=metrics.get('preview'), gates=metrics['modules']['gates'],
                            plan_state_cleared=not session.controller.orchestrator.task_plans)
                    else:
                        row['error_type'] = (reply.get('error') or {}).get('type', 'EngineError')
                    row['client_failures'] = [{'role': c.role, 'types': list(c.failure_types)} for c in clients if c.failure_types]
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                finally:
                    if session:
                        await session.close()
                row['elapsed_ms'] = (time.perf_counter() - started) * 1000.
                occurrences = journal.data['occurrences'][before:]
                row['usage'] = pilot.bench.usage(journal.events[before:])
                row['calls_by_role'] = dict(pilot.Counter(e['role'] for e in journal.events[before:]))
                row['generation_receipts'] = natural.generation_observations(occurrences)
                row['provider_degraded'] = (any(t != 'CancelledError' for c in row.get('client_failures', []) for t in c['types'])
                    or any(e.get('error_type') is not None and not (e['role'] == 'right' and e['error_type'] == 'CancelledError')
                           for e in journal.events[before:])
                    or any(r['finish_reason'] != 'stop' and not (r['stage'] == 'right' and r['transport_error'] == 'CancelledError')
                           or not r['nonempty_text'] and r['transport_error'] != 'CancelledError' for r in row['generation_receipts']))
                row['provider_degraded'] |= any(s['status'] != 'complete' for s in
                    (row.get('answer_generation') or {}).get('stages', []))
                row['request_reasoning_effort_counts'] = dict(pilot.Counter(
                    o['request']['payload'].get('reasoning_effort', 'provider_default') for o in occurrences))
                row['assignment_request_counts'] = assignment_exposure(occurrences, extract_task_plan(case['question']))
                report['rows'].append(row)
                save()
                print(json.dumps({k: row.get(k) for k in ('case', 'arm', 'status', 'task_goals', 'usage', 'assignment_request_counts')}), flush=True)
    journal.max_calls = CALL_LIMIT
    for case in fixture['cases']:
        rows = {r['arm']: r for r in report['rows'] if r['case'] == case['id']}
        result = {'case': case['id'], 'status': 'not_assessed'}
        journal.case = 'judge:' + case['id']
        before = len(journal.events)
        if all(isinstance(r['answer'], str) and r['answer'].strip() for r in rows.values()) and len(rows) == 2:
            try:
                result.update(scores=await assess(case, rows, judge, journal), status='graded')
            except Exception as exc:
                result['error_type'] = type(exc).__name__
        else:
            result['reason'] = 'unavailable_answer_no_substitution'
        result['usage'] = pilot.bench.usage(journal.events[before:])
        result['runtime_qualified'] = all(r['status'] == 'ok' and not r['provider_degraded'] for r in rows.values())
        report['grades'].append(result)
        save()
    journal.finish()
    report['comparison'] = comparison(report['grades'])
    report.update(complete=True, live_provider_calls=journal.calls, usage=pilot.bench.usage(journal.events),
        calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
        served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}),
        transport_events_sha256=journal.data['transport_events_sha256'])
    report['canonical_sha256'] = pilot.bench.digest(canonical(report))
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--judge-model', default='gpt-6-sol')
    parser.add_argument('--max-calls', type=int, choices=(CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        fixture = pilot.bench.read_json(FIXTURE)
        validate(fixture)
        frozen = pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=pilot.base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze code, fixtures and protocol before paid calls')
        model = args.model or frozen['metadata']['config']['model']
        config = pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig('openai', model, 'REPLAY_ONLY')
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        judge = replace(config, model=args.judge_model)
        journal = ScheduledJournal(config, metadata(fixture, config, judge), frozen=frozen, max_calls=CALL_LIMIT,
            replay_wait_seconds=30.,
            save=lambda data: pilot.bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            with pilot.forbid_provider_network() as audit:
                report = asyncio.run(run(args.output_dir, fixture, config, judge, journal))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report = asyncio.run(run(args.output_dir, fixture, config, judge, journal))
        pilot.bench.write_private(args.output_dir / 'report.json', report)
        pilot.bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        pilot.bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'usage': report['usage'], 'calls': journal.calls}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Reasoning-effort comparison incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
