#!/usr/bin/env python3
"""Frozen four-arm native benchmark for completion work products and critic routing."""
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_replanning_stress as stress
import completion_benchmark_tasks as tasks
from core.task_goals import extract_task_plan

base = stress.base
ROOT = base.pilot.base.ROOT.parent
FIXTURE = ROOT/'sr-dual-brain-llm/examples/completion_mechanisms.json'
PROTOCOL = ROOT/'docs/benchmarks/completion-mechanisms-protocol.md'
MANIFEST = ROOT/'docs/benchmarks/completion-mechanisms-manifest.json'
ARMS = ('off', 'work_product', 'critic_aware', 'both')
ORDERS = (ARMS, ARMS[1:]+ARMS[:1], ARMS[2:]+ARMS[:2],
          ARMS[3:]+ARMS[:3], tuple(reversed(ARMS)))
CALL_LIMIT = 80
CAPS = {'draft': 4096, 'integration': 4096, 'integration_after_work_product': 3072,
        'work_product': 1024,
        'critic': 2048, 'critic_repair': 2048}


def validate_fixture(fixture):
    if (fixture != tasks.build_fixture() or len(fixture['cases']) != len(tasks.SPECS)
            or len({c['id'] for c in fixture['cases']}) != len(tasks.SPECS)):
        raise ValueError('Fresh fixture differs from its frozen generator')
    for case in fixture['cases']:
        plan = extract_task_plan(case['question'])
        if (plan is None or len(plan.goals) != 3 or len(case['question']) > 10000
                or case['expected'] != tasks.base.ORACLES[case['domain']](case['data'])
                or not tasks.base.grade(case, json.dumps(case['expected']))['exact']):
            raise ValueError('Question, goal or oracle contract differs')
        if case['domain'] == 'sql' and case['expected'] != tasks.base.sql_reference(case['data']):
            raise ValueError('Independent SQL references disagree')


def expected_manifest(fixture):
    return {'schema_version': 1, 'origin': tasks.ORIGIN,
            'fixture_sha256': base.pilot.bench.digest(fixture),
            'candidate_code_sha256': base.pilot.code_digest(),
            'grader_sha256': stress.file_hash(Path(tasks.base.__file__)),
            'protocol_sha256': stress.file_hash(PROTOCOL),
            'case_ids': [c['id'] for c in fixture['cases']],
            'arms': list(ARMS), 'order': [list(o) for o in ORDERS],
            'call_limit': CALL_LIMIT, 'output_ceiling_per_trial': base.OUTPUT_ALLOWANCE,
            'ordinary_left_allocation': 8192, 'serialized_request_caps': CAPS}


class Journal(stress.Journal):
    def __init__(self, *args, **kwargs):
        self.configuration_fault = None
        super().__init__(*args, **kwargs)

    async def post(self, role, url, payload, headers, **kwargs):
        if any('Prepare a bounded internal work product for this one goal only:'
               in str(message.get('content', '')) for message in payload.get('messages', [])):
            role = 'work_product'
        cap = payload.get('max_completion_tokens', payload.get('max_tokens'))
        product_admitted = any(row['request']['case'] == self.case
            and row['request']['role'] == 'work_product' for row in self.data['occurrences'])
        expected = (CAPS['integration_after_work_product']
            if role == 'integration' and product_admitted else CAPS.get(role))
        if self.configuration_fault is not None:
            raise base.pilot.AdmissionError('Admission closed after configuration fault')
        if expected is None or type(cap) is not int or cap != expected:
            self.configuration_fault = {'role': role, 'actual_cap': cap,
                                        'expected_cap': expected}
            self.data['configuration_fault'] = self.configuration_fault
            self.checkpoint()
            raise base.pilot.AdmissionError('Serialized request cap differs')
        return await super().post(role, url, payload, headers, **kwargs)


def score(row):
    return row['grade']['item_accuracy'] if row['answer_complete'] else 0.


def summarize(rows):
    arms = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        products = [p for r in group for p in (r.get('task_goals') or {}).get('work_products', [])]
        attempts = [a for r in group for a in (r.get('system2') or {}).get('attempts', [])]
        arms[arm] = {'trials': len(group),
            'exact': sum(r['answer_complete'] and r['grade']['exact'] for r in group),
            'complete': sum(r['answer_complete'] for r in group),
            'workflow_valid': sum(r['workflow_valid'] for r in group),
            'schema_valid': sum(r['grade']['schema_valid'] for r in group),
            'mean_item_accuracy': sum(map(score, group))/len(group) if group else None,
            'initial_complete': sum(r['initial_answer_complete'] for r in group),
            'work_products_attempted': len(products),
            'work_products_available': sum(p['status'] == 'available' for p in products),
            'external_review_completed': sum((r.get('system2') or {}).get('external_review_completed') is True
                                             for r in group),
            'critic_attempts': len(attempts)}
    pairs = []
    for case in [c['id'] for c in tasks.build_fixture()['cases']]:
        group = {r['arm']: r for r in rows if r['case'] == case}
        for i, reference in enumerate(ARMS):
            for candidate in ARMS[i+1:]:
                if reference in group and candidate in group:
                    a, b = group[reference], group[candidate]
                    delta = score(b) - score(a)
                    pairs.append({'case': case, 'reference': reference, 'candidate': candidate,
                                  'delta': delta, 'outcome': 'higher' if delta > 0 else
                                  'lower' if delta < 0 else 'same',
                                  'both_workflows_valid': a['workflow_valid'] and b['workflow_valid']})
    transitions = Counter()
    for row in rows:
        initial, final = row['initial_answer_complete'], row['answer_complete']
        if not initial and final:
            kind = 'incomplete_to_complete'
        elif not initial:
            kind = 'both_unavailable'
        elif not final:
            kind = 'regressed_to_unavailable'
        else:
            before, after = row['initial_grade']['item_accuracy'], row['grade']['item_accuracy']
            kind = 'semantic_repair' if after > before else 'semantic_regression' if after < before else 'unchanged_complete'
        transitions[(row['arm'], kind)] += 1
    return {'authority': 'descriptive_five_fresh_tasks_independent_samples_no_isolated_causal_claim',
            'arms': arms, 'pairs': pairs,
            'transitions': {arm: {kind: n for (a, kind), n in transitions.items() if a == arm}
                            for arm in ARMS}}


def canonical(report):
    return {'outcome': report['outcome'], 'summary': report['summary'],
            'usage': report['usage'], 'phase_receipt': report['phase_receipt'],
            'configuration_fault': report['configuration_fault'],
            'rows': [{k: v for k, v in row.items() if k != 'elapsed_ms'} for row in report['rows']]}


def public(report):
    result = stress.public(report)
    result['summary'] = report['summary']
    return result


async def run(output_dir, fixture, manifest, config, *, frozen=None, transport_factory=None):
    validate_fixture(fixture)
    if manifest != expected_manifest(fixture):
        raise ValueError('Fixture, grader, protocol or code differs from frozen manifest')
    base.pilot.base.enforce_runtime()
    budget = base.CampaignBudget()
    metadata = base.pilot.metadata(fixture, config) | {
        'experiment': tasks.ORIGIN, 'phase': 'evaluation', 'call_limit': CALL_LIMIT,
        'protocol_sha256': manifest['protocol_sha256'],
        'manifest_sha256': base.pilot.bench.digest(manifest), 'arms': list(ARMS),
        'order': [list(o) for o in ORDERS], 'serialized_request_caps': CAPS,
        'output_ceiling_per_trial': base.OUTPUT_ALLOWANCE,
        'ordinary_left_allocation': 8192, 'grader_sha256': manifest['grader_sha256'],
        'semantic_grading': 'exact_closed_world_no_paid_judge',
        'replay': 'recorded_transport_duration_and_order_actual_owner_cancellation'}
    report = {'schema_version': 1, 'complete': False, 'outcome': 'not_assessed',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'metadata': metadata, 'rows': []}
    journal = Journal(config, metadata, campaign=budget, frozen=frozen,
        max_calls=CALL_LIMIT, replay_wait_seconds=75.,
        save=lambda data: base.pilot.bench.checkpoint(output_dir/'journal.partial.json', data))
    if transport_factory is not None:
        journal.transport = transport_factory(journal)
    save = lambda: base.pilot.bench.checkpoint(output_dir/'report.partial.json', report)
    clock = [fixture['clock']]
    env = base.pilot.profile.environment() | {
        'DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS': '2048',
        'DUALBRAIN_ANSWER_REASONING_EFFORT': 'provider_default',
        'DUALBRAIN_ANSWER_TOKEN_BUDGET': '8192',
        'DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE': 'auto',
        'DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE': 'auto',
        'DUALBRAIN_MEMORY_OUTBOX_PATH': str(output_dir/'outbox.sqlite3')}
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(
                int=base.pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(fixture['cases']):
            for arm in ORDERS[index]:
                clock[0] = fixture['clock'] + index
                with patch.dict(os.environ, {
                    'DUALBRAIN_COMPLETION_WORK_PRODUCTS': '1' if arm in {'work_product', 'both'} else '0',
                    'DUALBRAIN_CRITIC_COMPLETION_AWARE': '1' if arm in {'critic_aware', 'both'} else '0'}):
                    row = await base.dual(case, arm, config, journal, output_dir, clock)
                row['tier'] = case['tier']
                report['rows'].append(row)
                save()
                print(json.dumps({'case': row['case'], 'arm': arm, 'complete': row['answer_complete'],
                                  'exact': row['grade']['exact'], 'calls': row['usage']['calls']}), flush=True)
                if budget.blocked or stress.study_cap_denied(journal) or journal.configuration_fault:
                    break
            if budget.blocked or stress.study_cap_denied(journal) or journal.configuration_fault:
                break
    journal.finish()
    report.update(complete=True, outcome='configuration_invalid' if journal.configuration_fault else
        'transport_blocked' if budget.blocked else 'budget_incomplete' if stress.study_cap_denied(journal)
        else 'comparisons_complete' if len(report['rows']) == len(fixture['cases'])*len(ARMS)
        else 'incomplete_unclassified', live_provider_calls=journal.calls,
        usage=base.pilot.bench.usage(journal.events), summary=summarize(report['rows']),
        request_caps=[{'role': role, 'cap': cap, 'admissions': n} for (role, cap), n in sorted(
            Counter((r['request']['role'], r['request']['payload'].get('max_completion_tokens',
                r['request']['payload'].get('max_tokens'))) for r in journal.data['occurrences']).items())],
        configuration_fault=journal.configuration_fault,
        phase_receipt={'requests': len(journal.events), 'events': len(journal.data['transport_events']),
            'transport_events_sha256': journal.data['transport_events_sha256'],
            'denied': journal.data['denied_admissions']})
    report['canonical_sha256'] = base.pilot.bench.digest(canonical(report))
    save()
    return report, frozen if frozen else journal.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--compare-report', type=Path)
    parser.add_argument('--max-calls', type=int, choices=(CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        read = base.pilot.bench.read_json
        fixture, manifest = read(FIXTURE), read(MANIFEST)
        validate_fixture(fixture)
        if manifest != expected_manifest(fixture):
            raise ValueError('Frozen study manifest mismatch')
        for path in (FIXTURE, PROTOCOL, MANIFEST):
            subprocess.run(['git', 'ls-files', '--error-unmatch', str(path.relative_to(ROOT))],
                           cwd=ROOT, check=True, capture_output=True)
        frozen = read(args.journal) if args.journal else None
        model = args.model or frozen['metadata']['config']['model']
        config = (base.pilot.load_llm_config(default_model=model) if frozen is None else
                  base.pilot.LLMConfig('openai', model, 'REPLAY_ONLY'))
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=4096, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'],
                cwd=ROOT, text=True).strip():
            raise ValueError('Freeze clean source, fixture, protocol and manifest before provider calls')
        if frozen:
            with base.pilot.forbid_provider_network() as audit:
                report, journal = asyncio.run(run(args.output_dir, fixture, manifest, config, frozen=frozen))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
            if args.compare_report is None:
                raise ValueError('Replay requires the live report for exact comparison')
            original = read(args.compare_report)
            if (report['canonical_sha256'] != original['canonical_sha256']
                    or report['usage'] != original['usage']
                    or report['phase_receipt'] != original['phase_receipt']
                    or report['request_caps'] != original['request_caps']
                    or report['unexpected_blocked_http_attempts'] != 0):
                raise ValueError('Replay differs from live canonical outcome or coverage')
        elif args.compare_report is not None:
            raise ValueError('Live acquisition cannot compare against prior answers')
        else:
            report, journal = asyncio.run(run(args.output_dir, fixture, manifest, config))
        for name, data in (('journal.json', journal), ('report.json', report), ('summary.json', public(report))):
            base.pilot.bench.write_private(args.output_dir/name, data)
        print(json.dumps({'outcome': report['outcome'], 'usage': report['usage'],
            'provider_calls': report['live_provider_calls'], 'canonical_sha256': report['canonical_sha256']}))
    except BaseException as exc:
        base.pilot.bench.checkpoint(args.output_dir/'failure.json',
                                    {'complete': False, 'error_type': type(exc).__name__})
        print('Completion benchmark incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
