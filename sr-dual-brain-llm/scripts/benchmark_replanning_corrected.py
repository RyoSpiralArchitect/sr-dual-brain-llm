#!/usr/bin/env python3
"""Independent corrected study using only never-sent frozen stress tasks."""
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

base = stress.base
ORIGIN = 'corrected_replanning_stress_v1'
FIXTURE = stress.FIXTURE.with_name('replanning_corrected.json')
PROTOCOL = stress.PROTOCOL.with_name('replanning-corrected-protocol.md')
PLAN = PROTOCOL.with_name('replanning-corrected-manifest.json')
CALL_LIMIT = 96
CAPS = {'draft': 4096, 'integration': 4096, 'critic': 2048, 'critic_repair': 2048,
        'self_initial': 4096, 'self_critique': 2048, 'self_revision': 4096, 'self_audit': 2048}


def exposed_ids(journal):
    return sorted({r['request']['case'].rsplit(':', 1)[0] for r in journal['occurrences']})


def build_fixture(pool, parent_journal):
    """Membership-only selection, independent of every score and model output."""
    excluded = exposed_ids(parent_journal)
    counts, cases = Counter(), []
    for case in pool['cases']:
        if case['id'] not in excluded and counts[case['domain']] < 2:
            counts[case['domain']] += 1
            cases.append(case)
    if counts != {domain: 2 for domain in stress.tasks.base.DOMAINS}:
        raise ValueError('Require two never-sent original cases per domain')
    return {'schema_version': 1, 'origin': ORIGIN, 'clock': pool['clock'],
            'parent_fixture_sha256': base.pilot.bench.digest(pool),
            'excluded_case_ids': excluded, 'cases': cases}


def make_plan(fixture, parent, journal, report_hash, journal_hash):
    return {'schema_version': 1, 'selection': 'first_two_unexposed_per_domain_in_original_order',
            'parent_report_sha256': report_hash, 'parent_journal_sha256': journal_hash,
            'parent_report_json_sha256': base.pilot.bench.digest(parent),
            'parent_journal_json_sha256': base.pilot.bench.digest(journal),
            'parent_canonical_sha256': parent['canonical_sha256'],
            'parent_admissions': parent['usage']['calls'],
            'parent_fixture_sha256': fixture['parent_fixture_sha256'],
            'fixture_sha256': base.pilot.bench.digest(fixture),
            'excluded_case_ids': fixture['excluded_case_ids'],
            'selected_case_ids': [c['id'] for c in fixture['cases']],
            'grader_sha256': stress.file_hash(Path(stress.tasks.base.__file__)),
            'candidate_code_sha256': base.pilot.code_digest(),
            'phase_call_limit': CALL_LIMIT, 'global_call_limit': stress.GLOBAL_LIMIT,
            'maximum_campaign_admissions': parent['usage']['calls'] + CALL_LIMIT}


def validate(fixture, pool, parent, journal, plan, config):
    stress.validate(pool)
    if (not parent['complete'] or parent['phase'] != 'main'
            or parent['outcome'] != 'transport_blocked' or not journal['complete']
            or not journal['transport_blocked']
            or parent['canonical_sha256'] != base.pilot.bench.digest(stress.canonical(parent))
            or parent['metadata']['fixture_sha256'] != base.pilot.bench.digest(pool)
            or parent['metadata']['config'] != base.pilot.bench.public_config(config)
            or parent['metadata']['grader_sha256'] != stress.file_hash(Path(stress.tasks.base.__file__))
            or journal['metadata'] != parent['metadata']
            or journal['occurrences_sha256'] != base.pilot.bench.digest(journal['occurrences'])
            or parent['usage'] != journal['usage']
            or type(parent['usage']['calls']) is not int or not 0 < parent['usage']['calls'] <= 32
            or parent['usage']['calls'] != len(journal['occurrences'])
            or parent['usage']['calls'] != sum(r['usage']['calls'] for r in parent['rows'])
            or parent['phase_receipt']['requests'] != len(journal['occurrences'])
            or parent['phase_receipt']['transport_events_sha256'] != journal['transport_events_sha256']
            or journal['transport_events_sha256'] != base.pilot.bench.digest(journal['transport_events'])
            or any(r['sha256'] != base.pilot.RawJournal.row_digest(r) for r in journal['occurrences'])
            or not set(exposed_ids(journal)) <= {c['id'] for c in pool['cases']}):
        raise ValueError('Original failed-study lineage differs')
    # This is a separate measurement after a configuration/deadline correction.
    # It never reopens a campaign stopped by auth/quota/provider-status errors.
    errors = [r for r in journal['occurrences'] if r.get('error_type')]
    if (not any(r['error_type'] == 'TimeoutError' for r in errors)
            or any(r['error_type'] not in {'TimeoutError', 'CancelledError'}
                   or r.get('http_status') is not None for r in errors)):
        raise ValueError('A provider-status/auth/quota fault cannot authorize this corrected study')
    if fixture != build_fixture(pool, journal):
        raise ValueError('Require the unchanged first two never-sent cases per domain')
    expected = make_plan(fixture, parent, journal, plan['parent_report_sha256'], plan['parent_journal_sha256'])
    if plan != expected:
        raise ValueError('Freeze the complete candidate/selection/lineage manifest before calls')


class Journal(stress.Journal):
    def __init__(self, *args, **kwargs):
        self.configuration_fault = None
        super().__init__(*args, **kwargs)

    async def post(self, role, url, payload, headers, **kwargs):
        cap = payload.get('max_completion_tokens', payload.get('max_tokens'))
        if self.configuration_fault is not None:
            raise base.pilot.AdmissionError('Study admission closed after a request ceiling mismatch')
        if role not in CAPS or type(cap) is not int or cap != CAPS[role]:
            self.configuration_fault = {'role': role, 'actual_cap': cap, 'expected_cap': CAPS.get(role)}
            self.data['configuration_fault'] = self.configuration_fault
            self.checkpoint()
            raise base.pilot.AdmissionError('Serialized request ceiling differs; study admission closed')
        return await super().post(role, url, payload, headers, **kwargs)


def request_caps(journal):
    counts = Counter((r['request']['role'], r['request']['payload'].get('max_completion_tokens',
        r['request']['payload'].get('max_tokens'))) for r in journal['occurrences'])
    return [{'role': role, 'cap': cap, 'admissions': n} for (role, cap), n in sorted(counts.items())]


def canonical(report):
    return stress.canonical(report) | {k: report[k] for k in
        ('prior_journal_sha256', 'request_caps', 'configuration_fault')}


async def run(output_dir, fixture, pool, config, parent, parent_journal, plan, *, frozen=None, transport_factory=None):
    validate(fixture, pool, parent, parent_journal, plan, config)
    base.pilot.base.enforce_runtime()
    budget = base.CampaignBudget()
    budget.admissions = plan['parent_admissions']
    metadata = base.pilot.metadata(fixture, config) | {
        'experiment': ORIGIN, 'phase': 'corrected', 'call_limit': CALL_LIMIT,
        'global_call_limit': stress.GLOBAL_LIMIT, 'prior_admissions': budget.admissions,
        'protocol_sha256': stress.file_hash(PROTOCOL), 'manifest_sha256': base.pilot.bench.digest(plan),
        'arms': list(stress.ARMS), 'serialized_request_caps': CAPS,
        'output_ceiling_per_trial': base.OUTPUT_ALLOWANCE, 'ordinary_left_allocation': 8192,
        'semantic_grading': 'exact_closed_world_no_paid_judge',
        'grader_sha256': plan['grader_sha256'],
        'replay': 'recorded_transport_duration_and_order_actual_owner_cancellation',
        'comparison': 'independent_native_system2_on_off_vs_auto_not_compute_matched'}
    report = {'schema_version': 1, 'complete': False, 'phase': 'corrected', 'outcome': 'not_assessed',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=base.pilot.base.ROOT.parent, text=True).strip(),
        'metadata': metadata, 'prior_report_sha256': plan['parent_report_sha256'],
        'prior_journal_sha256': plan['parent_journal_sha256'], 'rows': []}
    journal = Journal(config, metadata, campaign=budget, frozen=frozen, max_calls=CALL_LIMIT,
        replay_wait_seconds=75., save=lambda data: base.pilot.bench.checkpoint(output_dir/'journal.partial.json', data))
    if transport_factory is not None:
        journal.transport = transport_factory(journal)
    save = lambda: base.pilot.bench.checkpoint(output_dir/'report.partial.json', report)
    clock = [fixture['clock']]
    env = base.pilot.profile.environment() | {'DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS': '2048',
        'DUALBRAIN_ANSWER_REASONING_EFFORT': 'provider_default', 'DUALBRAIN_ANSWER_TOKEN_BUDGET': '8192',
        'DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE': 'auto',
        'DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE': 'off',
        'DUALBRAIN_MEMORY_OUTBOX_PATH': str(output_dir/'outbox.sqlite3')}
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(
                int=base.pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(fixture['cases']):
            for arm in stress.ORDERS[index % len(stress.ORDERS)]:
                clock[0] = fixture['clock'] + pool['cases'].index(case)
                with patch.dict(os.environ, {'DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE':
                        'auto' if arm == 'dual_replan' else 'off'}):
                    row = (await base.self_review(case, config, journal) if arm == 'self_review' else
                           await base.dual(case, arm, config, journal, output_dir, clock))
                row['tier'] = case['tier']
                report['rows'].append(row)
                save()
                print(json.dumps({'phase': 'corrected', 'case': row['case'], 'arm': arm,
                    'grade': row['grade'], 'workflow_valid': row['workflow_valid'],
                    'calls': row['usage']['calls']}), flush=True)
                if budget.blocked or stress.study_cap_denied(journal) or journal.configuration_fault:
                    break
            if budget.blocked or stress.study_cap_denied(journal) or journal.configuration_fault:
                break
    journal.finish()
    report.update(complete=True, outcome='configuration_invalid' if journal.configuration_fault else
        'transport_blocked' if budget.blocked else 'budget_incomplete' if stress.study_cap_denied(journal)
        else 'comparisons_complete', live_provider_calls=journal.calls,
        usage=base.pilot.bench.usage(journal.events), campaign_admissions=budget.admissions,
        summary=stress.summarize(report['rows']), request_caps=request_caps(journal.data),
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
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--parent-journal', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, choices=(stress.GLOBAL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        read = base.pilot.bench.read_json
        fixture, pool, plan = read(FIXTURE), read(stress.FIXTURE), read(PLAN)
        parent, parent_journal = read(args.parent_report), read(args.parent_journal)
        if (plan['parent_report_sha256'] != stress.file_hash(args.parent_report)
                or plan['parent_journal_sha256'] != stress.file_hash(args.parent_journal)):
            raise ValueError('Original failed-study bytes differ')
        frozen = read(args.journal) if args.journal else None
        model = args.model or frozen['metadata']['config']['model']
        config = (base.pilot.load_llm_config(default_model=model) if frozen is None else
                  base.pilot.LLMConfig('openai', model, 'REPLAY_ONLY'))
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=4096, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        validate(fixture, pool, parent, parent_journal, plan, config)
        for path in (PLAN, FIXTURE, PROTOCOL):
            subprocess.run(['git', 'ls-files', '--error-unmatch', str(path.relative_to(base.pilot.base.ROOT.parent))],
                           cwd=base.pilot.base.ROOT.parent, check=True, capture_output=True)
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'],
                cwd=base.pilot.base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze corrected source, inputs, manifest and protocol before paid output')
        if frozen:
            with base.pilot.forbid_provider_network() as audit:
                report, journal = asyncio.run(run(args.output_dir, fixture, pool, config,
                    parent, parent_journal, plan, frozen=frozen))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report, journal = asyncio.run(run(args.output_dir, fixture, pool, config, parent, parent_journal, plan))
        for name, data in (('journal.json', journal), ('report.json', report), ('summary.json', stress.public(report))):
            base.pilot.bench.write_private(args.output_dir/name, data)
        print(json.dumps({'outcome': report['outcome'], 'usage': report['usage'],
            'campaign_admissions': report['campaign_admissions'], 'provider_calls': report['live_provider_calls']}))
    except BaseException as exc:
        base.pilot.bench.checkpoint(args.output_dir/'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Corrected study incomplete: '+type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
