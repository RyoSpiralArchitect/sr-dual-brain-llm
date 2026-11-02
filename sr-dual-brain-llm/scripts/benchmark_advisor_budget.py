#!/usr/bin/env python3
"""Matched frozen-draft advisor timing pilot with observed, shared warm history."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid
from unittest.mock import patch

import benchmark_module_gating as bench
from core.advisor_timing import ADVISORS, finite_nonnegative, model_profile

FIXTURE = bench.base.ROOT / 'examples/advisor_budget_benchmark.json'
ARMS = ('untimed_advisors', 'timed_advisors')
HISTORY_POLICY = 'one_actual_warmup_per_case;recent_max_reseeded_age_zero_per_arm;no_outer_rpc_deadline'


def validate(fixture):
    cases = fixture['cases']
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_advisor_budget_pilot'
            or len(cases) != 3 or len({c['id'] for c in cases}) != 3):
        raise ValueError('Require the complete three-case prospective fixture')
    for case in cases:
        if (case['origin'] != 'model' or case['draft'] is not None or case['records']
                or not case['question'].strip() or not case['problem_id']
                or set(case['criteria']) != set(bench.base.CRITERIA)
                or case['executive_mode'] != 'observe' or case['executive_observer_mode'] != 'both'):
            raise ValueError('Require non-memory natural drafts with matched observer modes')


def environment(case, arm, source=None):
    env = bench.environment(case, 'all_auto', source)
    if arm in {'warmup', ARMS[0]}:
        env.update({bench.DEFAULT_REGISTRY[module].mode_env: 'on' for module in ADVISORS})
    elif arm not in {'draft', ARMS[1]}:
        raise ValueError('Unknown timing profile')
    return env


def models(engine):
    return {'director_advice': engine.director, 'executive_advice': engine.executive,
            'executive_metrics_observation': engine.executive}


def seed_history(engine, seed):
    if not isinstance(seed, dict) or set(seed) - ADVISORS:
        raise ValueError('Invalid observed history roles')
    for module, sample in seed.items():
        if (not isinstance(sample, dict) or set(sample) != {'ms', 'kind'}
                or not finite_nonnegative(sample['ms'])
                or sample['kind'] not in {'completed_max', 'censored_bound'}):
            raise ValueError('Invalid observed history sample')
        engine.hypothalamus.advisor_timing.record(module, model_profile(models(engine)[module]),
            sample['ms'], censored=sample['kind'] == 'censored_bound')


async def execute(case, arm, draft, seed, config, journal, clock, options):
    if arm == 'draft':
        return await bench.execute(case, arm, draft, config, journal, clock, options)
    before, started = len(journal.events), time.perf_counter()
    row = {'case': case['id'], 'problem_id': case['problem_id'], 'origin': 'model',
           'arm': arm, 'status': 'unavailable', 'answer': None, 'draft_sha256': bench.digest(draft),
           'history_sha256': bench.digest(seed)}
    engine = await bench.session(case, arm, config, journal, clock)
    try:
        seed_history(engine, seed)

        async def frozen(*_args, **_kwargs):
            return draft

        engine.left.generate_answer = frozen
        result = await bench.base._handle_process(engine, {
            'question': case['question'], 'qid': case['id'] + ':probe', 'leading_brain': 'left',
            'system2_mode': 'auto', 'executive_mode': case['executive_mode'],
            'executive_observer_mode': case['executive_observer_mode'],
        }, **options)
        await engine.callosum.wait_idle()
        if not isinstance(result['answer'], str) or not result['answer'].strip():
            raise ValueError('Missing final answer')
        row.update(status='ok', answer=result['answer'], metrics=result['metrics'])
    except Exception as exc:
        row.update(error_type=type(exc).__name__)
    finally:
        # Each role runs at most once per cell. Snapshot the actual resource
        # ledger, not a model completion duration reconstructed from receipts.
        row['observed_history'] = {}
        for module, model in models(engine).items():
            estimate = engine.hypothalamus.advisor_timing.estimate(module, model_profile(model))
            if estimate.ms is not None:
                row['observed_history'][module] = {'ms': estimate.ms, 'kind': estimate.kind}
        await engine.close()
    events = journal.events[before:]
    row.update(usage=bench.usage(events), calls_by_role=dict(Counter(e['role'] for e in events)),
               elapsed_ms=(time.perf_counter() - started) * 1000)
    return row


def summarize(report):
    grades = {(g['case'], g['arm']): g['scores'] for g in report['grades'] if 'arm' in g}
    arms = {}
    for arm in ('no_revision', *ARMS):
        rows = [r for r in report['rows'] if r['arm'] == arm]
        scores = [grades[(r['case'], arm)] for r in rows if (r['case'], arm) in grades]
        role_counts = Counter()
        for row in rows:
            role_counts.update(row['calls_by_role'])
        lifecycle = {}
        for module in sorted(ADVISORS):
            observations = [r.get('metrics', {}).get('modules', {}).get('advisor_timing', {}).get(module, {})
                            for r in rows]
            lifecycle[module] = {'admitted': sum(bool(o.get('admitted')) for o in observations),
                'started': sum(bool(o.get('started')) for o in observations),
                'usable_output': sum(bool(o.get('usable_output')) for o in observations),
                'outcomes': dict(Counter(o.get('outcome', 'unreached') for o in observations))}
        arms[arm] = {'planned': len(rows), 'available': sum(r['status'] == 'ok' for r in rows),
            'graded': len(scores), 'all_pass': sum(all(v['verdict'] == 'pass' for v in s.values()) for s in scores),
            'criterion_pass': {c: sum(s[c]['verdict'] == 'pass' for s in scores) for c in bench.base.CRITERIA},
            'usage': {k: bench.sum_usage([r['usage'] for r in rows], k) if rows else 0 for k in bench.usage([])},
            'calls_by_role': dict(role_counts), 'advisor_lifecycle': lifecycle}
    jointly = [c for c in report['case_order'] if all((c, arm) in grades for arm in ARMS)]
    return {'by_arm': arms, 'jointly_graded_pairs': len(jointly),
        'paired_timing_regressions': [c for c in jointly
            if all(v['verdict'] == 'pass' for v in grades[(c, ARMS[0])].values())
            and not all(v['verdict'] == 'pass' for v in grades[(c, ARMS[1])].values())],
        'criterion_regressions': {criterion: [c for c in jointly
            if grades[(c, ARMS[0])][criterion]['verdict'] == 'pass'
            and grades[(c, ARMS[1])][criterion]['verdict'] != 'pass'] for criterion in bench.base.CRITERIA},
        'grader_failures': sum('error_type' in g for g in report['grades'])}


async def run(args, fixture, config, journal):
    validate(fixture)
    bench.base.enforce_runtime()
    cases = list(fixture['cases'])
    random.Random(20261102).shuffle(cases)
    metadata = {k: v for k, v in journal.data.items() if k not in {'responses', 'complete'}}
    report = {'schema_version': 1, 'experiment': 'advisor_budget', 'complete': False,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bench.base.ROOT.parent, text=True).strip(),
        'metadata': metadata, 'case_order': [c['id'] for c in cases], 'drafts': {},
        'warmups': [], 'warm_history': {}, 'rows': [], 'grades': []}
    save = lambda: bench.checkpoint(args.output_dir / 'report.partial.json', report)
    clock, generation_events, warmup_events = [1793150000.], [], []
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch('time.time', lambda: clock[0]), patch('uuid.uuid4', lambda: uuid.UUID(int=bench.base.UUID_RNG.getrandbits(128))):
        for case in cases:
            before = len(journal.events)
            with patch.dict(os.environ, environment(case, 'draft'), clear=True):
                row = await execute(case, 'draft', None, {}, config, journal, clock, options)
            generation_events.extend(journal.events[before:])
            report['drafts'][case['id']] = {'text': row['answer'], 'status': row['status']}
            report['rows'].append({'case': case['id'], 'problem_id': case['problem_id'], 'origin': 'model',
                'arm': 'no_revision', 'status': row['status'], 'answer': row['answer'],
                'usage': bench.usage([]), 'calls_by_role': {}})
            save()
        for i, case in enumerate(cases):
            draft = report['drafts'][case['id']]['text']
            if draft is not None:
                before = len(journal.events)
                with patch.dict(os.environ, environment(case, 'warmup'), clear=True):
                    warmup = await execute(case, 'warmup', draft, {}, config, journal, clock, options)
                warmup_events.extend(journal.events[before:])
                report['warmups'].append(warmup)
                # Replay instantaneous HTTP receipts cannot reconstruct latency.
                # Restore the original, prospectively specified history inputs.
                seed = (journal.frozen['warm_history'][case['id']] if journal.frozen
                        else warmup['observed_history'])
                report['warm_history'][case['id']] = seed
                journal.data['warm_history'] = report['warm_history']
                if journal.save and journal.frozen is None:
                    journal.save(journal.data)
            else:
                seed = {}
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            for arm in order:
                if draft is None:
                    row = {'case': case['id'], 'problem_id': case['problem_id'], 'origin': 'model',
                        'arm': arm, 'status': 'unavailable', 'answer': None,
                        'usage': bench.usage([]), 'calls_by_role': {}}
                else:
                    with patch.dict(os.environ, environment(case, arm), clear=True):
                        row = await execute(case, arm, draft, seed, config, journal, clock, options)
                report['rows'].append(row)
                save()
                print(json.dumps({'case': case['id'], 'arm': arm, 'status': row['status'], 'usage': row['usage']}), flush=True)
        judge_before = len(journal.events)
        for case in cases:
            report['grades'].extend(await bench.grade(case, report['rows'], replace(config, model=args.judge_model), journal))
            save()
        report['judge_usage'] = bench.usage(journal.events[judge_before:])
    journal.finish()
    report.update(complete=True, provider_calls=journal.calls, frozen_replay=journal.frozen is not None,
        generation_usage=bench.usage(generation_events), warmup_usage=bench.usage(warmup_events),
        experiment_usage=bench.usage(journal.events), summaries=summarize(report))
    save()
    return report


def public_summary(report):
    grades = {(g['case'], g['arm']): g['scores'] for g in report['grades'] if 'arm' in g}
    def public_row(row):
        return {k: row.get(k) for k in ('case', 'arm', 'status', 'usage', 'calls_by_role', 'elapsed_ms', 'history_sha256')} | {
            'answer_sha256': bench.digest(row['answer']), 'modules': row.get('metrics', {}).get('modules', {}),
            'scores': {k: v['verdict'] for k, v in grades.get((row['case'], row['arm']), {}).items()}}
    return {k: report[k] for k in ('schema_version', 'experiment', 'complete', 'source_revision', 'metadata',
        'provider_calls', 'experiment_usage', 'generation_usage', 'warmup_usage', 'judge_usage', 'summaries', 'warm_history')} | {
        'rows': [public_row(r) for r in report['rows']], 'warmups': [public_row(r) for r in report['warmups']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--judge-model', default='gpt-6-sol')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if bool(args.model) == bool(args.journal) or not 1 <= args.max_calls <= 96:
            raise ValueError('Choose live or replay, with the prospective 96-attempt maximum')
        fixture = bench.read_json(FIXTURE)
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        model = args.model or frozen['model']
        config = bench.load_llm_config(default_model=model)
        if config is None:
            if args.model:
                raise ValueError('Live provider configuration required')
            config = bench.LLMConfig('openai', model, 'FROZEN_NO_KEY')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        if config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        code = [Path(__file__), Path(bench.__file__), Path(bench.base.__file__),
                bench.base.ROOT / 'scripts/benchmark_conversation_memory.py',
                bench.base.ROOT / 'scripts/engine_stdio.py', *sorted((bench.base.ROOT / 'core').glob('*.py'))]
        metadata = {'model': model, 'judge_model': args.judge_model, 'fixture_sha256': bench.digest(fixture),
            'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in code]),
            'judge_policy_sha256': bench.digest(bench.JUDGE), 'history_policy': HISTORY_POLICY,
            'config': bench.public_config(config), 'arms': list(ARMS), 'runtime': bench.COMMON,
            'call_limit': args.max_calls, 'torch_policy': bench.TORCH_AVAILABLE,
            'gate_policies': {k: v.policy_version for k, v in bench.DEFAULT_REGISTRY.items()}}
        journal = bench.Journal(config, metadata, frozen=frozen, max_calls=args.max_calls,
            save=lambda d: bench.checkpoint(args.output_dir / 'journal.partial.json', d))
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, fixture, config, journal))
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen if frozen else journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps(report['summaries'], indent=2))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Advisor pilot incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
