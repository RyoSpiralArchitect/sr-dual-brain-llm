#!/usr/bin/env python3
"""Untouched confirmation after an explicitly amended, baseline-only selection."""
import argparse
import asyncio
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

import benchmark_reasoning_frontier as base

ORIGIN = 'reasoning_frontier_headroom_followup_v1'
PROTOCOL = base.PROTOCOL.with_name('reasoning-frontier-followup-protocol.md')
MANIFEST = base.PROTOCOL.with_name('reasoning-frontier-followup-manifest.json')
RULE = 'first_level_with_3_eligible_and_1_semantic_error_of_4'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserved_code_digest():
    """Verify every original source file; only this new runner is excluded."""
    root = base.pilot.base.ROOT
    paths = (set((root/'core').glob('*.py')) | set((root/'scripts').glob('*.py'))) - {Path(__file__).resolve()}
    return base.pilot.bench.digest([(str(p.relative_to(root)), file_hash(p)) for p in sorted(paths)])


def select_level(rows):
    for level in range(1, 7):
        eligible = [r for r in rows if r['level'] == level and base.eligible_for_calibration(r)]
        if len(eligible) >= 3 and any(not r['grade']['exact'] for r in eligible):
            return level
    return None


def verify_source(source, manifest, fixture, config):
    """Reject changed calibration, grading, profiles, source or spent holdouts."""
    base.validate(fixture)
    if (manifest['experiment'] != ORIGIN or manifest['selection_rule'] != RULE
            or manifest['global_call_limit'] != base.GLOBAL_CALL_LIMIT
            or manifest['confirmation_call_limit'] != base.PHASE_CALL_LIMIT):
        raise ValueError('Unrecognized frozen follow-up manifest')
    if (not source['complete'] or source['selected_level'] is not None or source['evaluation']
            or source['outcome'] != 'no_eligible_frontier_within_frozen_levels'):
        raise ValueError('Require completed original calibration with untouched confirmation')
    if (source['source_revision'] != manifest['source_revision']
            or source['canonical_sha256'] != manifest['source_canonical_sha256']
            or base.pilot.bench.digest(base.canonical(source)) != source['canonical_sha256']):
        raise ValueError('Original calibration identity changed')
    meta = source['metadata']
    if (meta['fixture_sha256'] != base.pilot.bench.digest(fixture)
            or meta['fixture_sha256'] != manifest['fixture_sha256']
            or meta['code_sha256'] != preserved_code_digest()
            or meta['code_sha256'] != manifest['source_code_sha256']
            or meta['protocol_sha256'] != file_hash(base.PROTOCOL)
            or meta['protocol_sha256'] != manifest['source_protocol_sha256']
            or meta['config'] != base.pilot.bench.public_config(config)):
        raise ValueError('Original fixture, source, protocol or model profile changed')
    cases = {c['id']: c for c in fixture['cases'] if c['partition'] == 'calibration'}
    rows = source['calibration']
    if len(rows) != 24 or {r['case'] for r in rows} != set(cases):
        raise ValueError('Require all original calibration rows without replacement')
    for row in rows:
        case = cases[row['case']]
        if (row['arm'] != 'self_review' or row['domain'] != case['domain'] or row['level'] != case['level']
                or row['grade'] != base.tasks.grade(case, row['answer'])):
            raise ValueError('Calibration row or exact grader changed')
    if any(base.level_break([r for r in rows if r['level'] == level]) for level in range(1, 7)):
        raise ValueError('Original selection outcome is inconsistent')
    calls = source['usage']['calls']
    if (type(calls) is not int or not 0 < calls <= base.PHASE_CALL_LIMIT
            or calls != manifest['prior_admissions']
            or calls != sum(r['usage']['calls'] for r in rows)
            or set(source['phase_receipts']) != {'calibration'}
            or source['phase_receipts']['calibration']['requests'] != calls):
        raise ValueError('Original admission ledger changed')
    level = select_level(rows)
    if level is None or level != manifest['selected_level']:
        raise ValueError('Amended baseline-only selection changed')
    return level


def load_source(path, manifest, fixture, config):
    if file_hash(path) != manifest['source_report_sha256']:
        raise ValueError('Original report file changed')
    source = base.pilot.bench.read_json(path)
    verify_source(source, manifest, fixture, config)
    return source


def metadata(fixture, config, manifest):
    original = base.metadata(fixture, config, 'evaluation')
    return original | {
        'experiment': ORIGIN, 'calibration_rule': RULE,
        'protocol_sha256': file_hash(PROTOCOL),
        'original_protocol_sha256': original['protocol_sha256'],
        'source_manifest_sha256': base.pilot.bench.digest(manifest),
        'source_report_sha256': manifest['source_report_sha256'],
        'prior_admissions': manifest['prior_admissions'],
        'selection_status': 'amended_after_baseline_calibration_before_any_dual_or_holdout_output',
    }


def combined_usage(first, second):
    return {key: (None if first[key] is None or second[key] is None else first[key]+second[key])
            for key in first}


def canonical(report):
    return base.canonical(report) | {k: report[k] for k in (
        'source_manifest_sha256', 'campaign_admissions', 'campaign_usage')}


async def run(output_dir, fixture, config, source, manifest, *, frozen=None, transport_factory=None):
    level = verify_source(source, manifest, fixture, config)
    base.pilot.base.enforce_runtime()
    budget = base.CampaignBudget()
    budget.admissions = manifest['prior_admissions']
    meta = metadata(fixture, config, manifest)
    journal = base.PhaseJournal(config, meta, campaign=budget, frozen=frozen,
        max_calls=base.PHASE_CALL_LIMIT, replay_wait_seconds=30.,
        save=lambda data: base.pilot.bench.checkpoint(output_dir/'evaluation.journal.partial.json', data))
    if transport_factory is not None:
        journal.transport = transport_factory(journal)
    report = {'schema_version': 1, 'complete': False, 'selected_level': level, 'outcome': 'not_assessed',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=base.pilot.base.ROOT.parent, text=True).strip(),
        'metadata': meta, 'source_manifest_sha256': base.pilot.bench.digest(manifest),
        'calibration': [], 'evaluation': []}
    save = lambda: base.pilot.bench.checkpoint(output_dir/'report.partial.json', report)
    clock = [fixture['clock']]
    env = base.pilot.profile.environment()
    env.update(DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS='2048', DUALBRAIN_ANSWER_REASONING_EFFORT='provider_default',
        DUALBRAIN_ANSWER_TOKEN_BUDGET='8192', DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE='auto',
        DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir/'outbox.sqlite3'))
    cases = [c for c in fixture['cases'] if c['partition'] == 'holdout' and c['level'] in {level, level+1}]
    with patch.dict(os.environ, env, clear=True), patch('time.time', lambda: clock[0]), \
            patch('core.unconscious_field.uuid', SimpleNamespace(uuid4=lambda: uuid.UUID(int=base.pilot.base.UUID_RNG.getrandbits(128)))):
        for index, case in enumerate(cases):
            for arm in base.ORDERS[index % len(base.ORDERS)]:
                clock[0] = fixture['clock'] + fixture['cases'].index(case)
                row = (await base.self_review(case, config, journal) if arm == 'self_review' else
                       await base.dual(case, arm, config, journal, output_dir, clock))
                report['evaluation'].append(row); save()
                print(json.dumps({'phase': 'confirmation_followup', 'case': row['case'], 'arm': arm,
                    'grade': row['grade'], 'workflow_valid': row['workflow_valid'],
                    'calls': row['usage']['calls']}), flush=True)
    journal.finish()
    report['phase_receipts'] = {'evaluation': {'requests': len(journal.events),
        'events': len(journal.data['transport_events']),
        'transport_events_sha256': journal.data['transport_events_sha256'],
        'denied': journal.data['denied_admissions']}}
    report['summary'] = base.summarize(report['evaluation'])
    report.update(complete=True, outcome='transport_blocked' if budget.blocked else 'heldout_followup_complete',
        live_provider_calls=journal.calls, usage=base.pilot.bench.usage(journal.events),
        campaign_admissions=budget.admissions)
    report['campaign_usage'] = combined_usage(source['usage'], report['usage'])
    report['canonical_sha256'] = base.pilot.bench.digest(canonical(report))
    save()
    return report, frozen if frozen else journal.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal-dir', type=Path)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, choices=(base.GLOBAL_CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        fixture = base.pilot.bench.read_json(base.FIXTURE)
        manifest = base.pilot.bench.read_json(MANIFEST)
        frozen = (base.pilot.bench.read_json(args.journal_dir/'evaluation.journal.json')
                  if args.journal_dir else None)
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'],
                cwd=base.pilot.base.ROOT.parent, text=True).strip():
            raise ValueError('Freeze follow-up before any heldout output')
        model = args.model or frozen['metadata']['config']['model']
        config = (base.pilot.load_llm_config(default_model=model) if frozen is None else
                  base.pilot.LLMConfig('openai', model, 'REPLAY_ONLY'))
        if not config or config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        config = replace(config, model=model, max_output_tokens=4096, timeout_seconds=60,
            auto_continue=False, max_continuations=0)
        source = load_source(args.source_report, manifest, fixture, config)
        if frozen is not None:
            with base.pilot.forbid_provider_network() as audit:
                report, journal = asyncio.run(run(args.output_dir, fixture, config, source, manifest, frozen=frozen))
            report['unexpected_blocked_http_attempts'] = audit.blocked_requests
        else:
            report, journal = asyncio.run(run(args.output_dir, fixture, config, source, manifest))
        base.pilot.bench.write_private(args.output_dir/'evaluation.journal.json', journal)
        base.pilot.bench.write_private(args.output_dir/'report.json', report)
        base.pilot.bench.write_private(args.output_dir/'summary.json', base.public_summary(report))
        print(json.dumps({'complete': True, 'outcome': report['outcome'], 'usage': report['usage'],
            'campaign_admissions': report['campaign_admissions'], 'provider_calls': report['live_provider_calls']}))
    except BaseException as exc:
        base.pilot.bench.checkpoint(args.output_dir/'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Frontier follow-up incomplete: '+type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
