#!/usr/bin/env python3
"""Frozen cross-domain local-witness ablation with native routing and replay."""
import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_task_goals as study
from core.constraint_checks import CheckReport, check_constraints, current_constraints
from core.task_goals import extract_task_plan
from engine_stdio import _dispatch_request

pilot = study.pilot
ScheduledJournal = study.ScheduledJournal
FIXTURE = pilot.base.ROOT / 'examples/constraint_mixed_comparison.json'
PROTOCOL = pilot.base.ROOT.parent / 'docs/benchmarks/constraint-mixed-protocol.md'
ARMS = ('ablated', 'witnesses')
CALL_LIMIT, RUNTIME_LIMIT = 96, 86
IDS = ('three_sensor_scope', 'nonlinear_ambiguity', 'status_object', 'conditional_object',
       'dedup_call_boundary', 'cancellation_boundary', 'latest_event_row', 'silent_monitor',
       'customer_memo', 'literal_record')
public_summary = study.public_summary


def canonical(report):
    return study.canonical(report) | {'by_domain': report['by_domain']}


def metadata(fixture, config, judge):
    return study.metadata(fixture, config, judge) | {
        'experiment': fixture['origin'], 'call_limit': CALL_LIMIT, 'runtime_call_limit': RUNTIME_LIMIT,
        'arms': {'ablated': 'only_micro_entry_returns_empty_CheckReport',
                 'witnesses': 'unchanged_production_detector'},
        'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        'goal_orchestration': 'auto_in_both_arms', 'reasoning_effort': 'provider_default',
        'left_output_ceiling_allocation': 8192,
        'semantic_grading': 'mixed_constraint_quality_uncalibrated_single_model',
        'exact_probes': 'frozen_full_json_values_and_utf8_transcription_separate_from_runtime_checker',
    }


def validate(fixture):
    cases = fixture.get('cases', [])
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_mixed_constraint_ablation'
            or tuple(c['id'] for c in cases) != IDS):
        raise ValueError('Require the complete frozen ten-case mixture')
    for case in cases:
        plan = extract_task_plan(case['question'])
        if (sorted(case['arms']) != sorted(ARMS)
                or sorted(case['judge_labels']) != ['A', 'B']
                or sorted(case['judge_labels'].values()) != sorted(ARMS)
                or (len(plan.goals) if plan else 0) != case['expected_goal_count']
                or case['system2_mode'] not in {'auto', 'off'}
                or type(case['negative_control']) is not bool or not case['reference'].strip()):
            raise ValueError('Require complete scope, labels, criteria and mode allocation')
    if (sum(c['arms'][0] == 'witnesses' for c in cases) != 5
            or sum(c['judge_labels']['A'] == 'witnesses' for c in cases) != 5
            or pilot.Counter(c['domain'] for c in cases) != {
                'numeric': 2, 'json': 2, 'code': 2, 'sql': 1, 'evidence': 1, 'prose': 1, 'literal': 1}):
        raise ValueError('Require balanced fixed order and the complete domain mix')
    if {c['id'] for c in cases if c['negative_control']} != {
            'nonlinear_ambiguity', 'conditional_object', 'literal_record'}:
        raise ValueError('Require all prospective negative controls')
    if {c['id'] for c in cases if 'exact_probe' in c} != {
            'status_object', 'conditional_object', 'literal_record'}:
        raise ValueError('Require all three frozen exact probes')
    for case in cases:
        spec = case.get('exact_probe')
        if spec is not None and (set(spec) != {'kind', 'expected'}
                or spec['kind'] not in {'json_object', 'literal_text'}
                or not isinstance(spec['expected'], dict if spec['kind'] == 'json_object' else str)):
            raise ValueError('Invalid exact probe')


@contextmanager
def detector_condition(arm, observations):
    """Ablate the local adapter only, never substitute a draft/model judgment."""
    if arm not in ARMS:
        raise ValueError('Unknown detector condition')

    def observed(question, draft):
        report = check_constraints(question, draft) if arm == 'witnesses' else CheckReport()
        observations.append({'snapshot_sha256': pilot.bench.digest(draft),
            'condition': arm, **report.summary()})
        return report

    with patch('core.micro_critic.check_constraints', observed):
        yield


def exact_probe(case, answer):
    spec = case.get('exact_probe')
    if spec is None:
        return {'status': 'not_assessed'}
    if not isinstance(answer, str) or not answer.strip():
        return {'kind': spec['kind'], 'status': 'unavailable'}
    if spec['kind'] == 'literal_text':
        passed = answer.encode('utf-8') == spec['expected'].encode('utf-8')
    elif spec['kind'] == 'json_object':
        try:
            value = json.loads(answer, object_pairs_hook=study._unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
            passed = isinstance(value, dict) and value == spec['expected']
        except (ValueError, RecursionError):
            passed = False
    else:
        raise ValueError('Unknown frozen probe')
    return {'kind': spec['kind'], 'status': 'pass' if passed else 'fail'}


def first_request_hash(occurrences):
    if not occurrences:
        return None
    request = occurrences[0]['request']
    return pilot.bench.digest({k: request[k] for k in ('role', 'endpoint_sha256', 'payload')})


def comparison(grades):
    normalized = [{**g, **({'scores': {'off': g['scores']['ablated'], 'auto': g['scores']['witnesses']}}
                  if 'scores' in g else {})} for g in grades]
    result = study.comparison(normalized)
    rename = {'auto_only_pass': 'witnesses_only_pass', 'off_only_pass': 'ablated_only_pass'}
    result['totals'] = {group: {c: {rename.get(k, k): v for k, v in counts.items()}
        for c, counts in criteria.items()} for group, criteria in result['totals'].items()}
    result['pairs'] = [{**p, 'outcome': rename.get(p['outcome'], p['outcome'])} for p in result['pairs']]
    return result


async def turn(output_dir, case, arm, index, fixture, config, journal, clock):
    clock[0] = fixture['clock'] + index
    journal.case = case['id'] + ':' + arm
    before, started, observations = len(journal.events), time.perf_counter(), []
    row = {'case': case['id'], 'domain': case['domain'], 'negative_control': case['negative_control'],
           'arm': arm, 'question': case['question'], 'answer': None, 'status': 'unavailable'}
    session, clients = None, []
    # All workers are closed/drained before the patch is restored or the next
    # condition starts. Both arms use exactly the same session and request IDs.
    with detector_condition(arm, observations):
        try:
            session, clients = await pilot.make_session('constraint-mixed:' + case['id'], None,
                                                       config, journal, clock, seed=False)
            reply = await _dispatch_request({'id': journal.case, 'method': 'process', 'params': {
                'session_id': session.session_id, 'question': case['question'], 'qid': case['id'] + ':probe',
                'system2_mode': case['system2_mode'], 'executive_mode': 'off', 'executive_observer_mode': 'off',
                'return_telemetry': True}}, sessions={session.session_id: session}, state_store=None,
                blob_pipe=None, send=lambda *_: None, blob_root=output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
            if reply.get('ok'):
                value, metrics = reply['result'], reply['result']['metrics']
                system2 = metrics.get('system2') or {}
                row.update(status='ok', answer=value['answer'], task_goals=metrics.get('task_goals'),
                    answer_generation=metrics.get('answer_generation'), constraint_check=metrics.get('constraint_check'),
                    policy_class=type(session.controller.policy).__name__, policy_action=metrics['policy']['action'],
                    system2=pilot.accounting(metrics, value['telemetry']) | {
                        k: system2.get(k) for k in ('resolved', 'resolution_basis')},
                    detail=metrics.get('detail'), preview=metrics.get('preview'), gates=metrics['modules']['gates'],
                    plan_state_cleared=not session.controller.orchestrator.task_plans)
            else:
                row['error_type'] = (reply.get('error') or {}).get('type', 'EngineError')
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        finally:
            if session is not None:
                await session.close()
    occurrences = journal.data['occurrences'][before:]
    row.update(elapsed_ms=(time.perf_counter() - started) * 1000.,
        client_failures=[{'role': c.role, 'types': list(c.failure_types)} for c in clients if c.failure_types],
        usage=pilot.bench.usage(journal.events[before:]),
        calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events[before:])),
        generation_receipts=study.natural.generation_observations(occurrences),
        initial_request_sha256=first_request_hash(occurrences),
        detector_observations=observations, constraint_state_cleared=current_constraints() is None,
        final_local_diagnostic=check_constraints(case['question'], row['answer']).summary(),
        exact_probe=exact_probe(case, row['answer']),
        assignment_request_counts=study.assignment_exposure(occurrences, extract_task_plan(case['question'])),
        request_reasoning_effort_counts=dict(pilot.Counter(
            o['request']['payload'].get('reasoning_effort', 'provider_default') for o in occurrences)))
    row['provider_degraded'] = (
        any(t != 'CancelledError' for c in row['client_failures'] for t in c['types'])
        or any(e.get('error_type') is not None and not (e['role'] == 'right' and e['error_type'] == 'CancelledError')
               for e in journal.events[before:])
        or any(r['finish_reason'] != 'stop' and not (r['stage'] == 'right' and r['transport_error'] == 'CancelledError')
               or not r['nonempty_text'] and r['transport_error'] != 'CancelledError' for r in row['generation_receipts'])
        or any(s['status'] != 'complete' for s in (row.get('answer_generation') or {}).get('stages', [])))
    return row


async def run(output_dir, fixture, config, judge, journal):
    validate(fixture)
    pilot.base.enforce_runtime()
    report = {'schema_version': 1, 'metadata': metadata(fixture, config, judge), 'complete': False,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=pilot.base.ROOT.parent, text=True).strip(),
        'rows': [], 'grades': []}
    save = lambda: pilot.bench.checkpoint(output_dir / 'report.partial.json', report)
    clock = [fixture['clock']]
    env = pilot.profile.environment()
    env.update(DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS='2048', DUALBRAIN_ANSWER_REASONING_EFFORT='provider_default',
        DUALBRAIN_ANSWER_TOKEN_BUDGET='8192', DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE='auto',
        DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir / 'outbox.sqlite3'))
    journal.max_calls = RUNTIME_LIMIT
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(fixture['cases']):
            for arm in case['arms']:
                row = await turn(output_dir, case, arm, index, fixture, config, journal, clock)
                report['rows'].append(row)
                save()
                print(json.dumps({k: row.get(k) for k in ('case', 'arm', 'status', 'usage', 'constraint_check', 'exact_probe')}), flush=True)
    journal.max_calls = CALL_LIMIT
    for case in fixture['cases']:
        rows = {r['arm']: r for r in report['rows'] if r['case'] == case['id']}
        hashes = [r['initial_request_sha256'] for r in rows.values()]
        matched = len(hashes) == 2 and bool(hashes[0]) and hashes[0] == hashes[1]
        result = {'case': case['id'], 'domain': case['domain'], 'status': 'not_assessed',
                  'first_requests_matched': matched}
        journal.case, before = 'judge:' + case['id'], len(journal.events)
        if len(rows) == 2 and all(isinstance(r['answer'], str) and r['answer'].strip() for r in rows.values()):
            try:
                result.update(scores=await study.assess(case, rows, judge, journal), status='graded')
            except Exception as exc:
                result['error_type'] = type(exc).__name__
        else:
            result['reason'] = 'unavailable_answer_no_substitution'
        result['usage'] = pilot.bench.usage(journal.events[before:])
        result['runtime_qualified'] = matched and all(r['status'] == 'ok' and not r['provider_degraded']
            and isinstance(r['answer'], str) and bool(r['answer'].strip())
            and (r.get('answer_generation') or {}).get('terminal_complete') is True for r in rows.values())
        report['grades'].append(result)
        save()
    journal.finish()
    report['comparison'] = comparison(report['grades'])
    report['by_domain'] = {d: comparison([g for g in report['grades'] if g['domain'] == d])
                           for d in sorted({c['domain'] for c in fixture['cases']})}
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
    parser.add_argument('--replay-api-base')
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
            raise ValueError('Freeze source, fixture and protocol before paid calls')
        model = args.model or frozen['metadata']['config']['model']
        config = (pilot.load_llm_config(default_model=model) if not frozen
                  else pilot.LLMConfig('openai', model, 'REPLAY_ONLY', api_base=args.replay_api_base))
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        judge = replace(config, model=args.judge_model)
        journal = ScheduledJournal(config, metadata(fixture, config, judge), frozen=frozen, max_calls=CALL_LIMIT,
            replay_wait_seconds=30., save=lambda data: pilot.bench.checkpoint(args.output_dir / 'journal.partial.json', data))
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
        print('Mixed constraint comparison incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
