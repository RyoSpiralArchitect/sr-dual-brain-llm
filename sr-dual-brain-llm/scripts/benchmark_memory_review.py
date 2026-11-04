#!/usr/bin/env python3
"""Frozen obligation calibration and paired, retained-session memory review pilot."""
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

import advisor_assessment_v2 as v2
import advisor_assessment_v3 as v3
import benchmark_advice_boundaries as boundaries
import benchmark_module_gating as bench
from core.shared_memory import MemoryTrace
from core.thalamic_reticular_nucleus import DEFAULT_REGISTRY

CALIBRATION = bench.base.ROOT / 'examples/advice_obligation_calibration.json'
DIALOGUE = bench.base.ROOT / 'examples/memory_review_dialogue.json'
CAL_ARMS = ('criterion_v2', 'obligation_v3')
ARMS = ('baseline', 'memory_bridge')
CALL_LIMIT = 128  # Includes 16 calibration packets, all support/draft/review/repair/judge attempts.


def validate(calibration, dialogue):
    if (calibration.get('schema_version') != 1 or dialogue.get('schema_version') != 1
            or calibration.get('origin') != 'prospective_explicit_advice_obligations'
            or tuple(c['id'] for c in calibration['cases']) != (
                'valve_confirmation', 'measurement_units', 'monitor_attestation', 'monitor_actor_scope')
            or dialogue.get('origin') != 'prospective_memory_review_dialogue'
            or tuple(t['id'] for t in dialogue['turns']) != (
                'lookup', 'competing', 'corrected', 'scope_switch', 'comparison', 'expired')):
        raise ValueError('Complete prospective fixtures required')
    for case in calibration['cases']:
        obligations = v3.obligations_for(case, case['role'])
        if len(case['variants']) != 3:
            raise ValueError('Three independent variants required')
        for variant in case['variants'].values():
            v3.original.validate_fields(variant['fields'])
            if (set(variant['expected']) != set(v3.CRITERIA)
                    or set(variant['expected_obligations']) != set(obligations)
                    or any(v not in {'pass', 'fail', 'uncertain'} for v in
                           [*variant['expected'].values(), *variant['expected_obligations'].values()])):
                raise ValueError('Explicit complete expectations required')
    for turn in dialogue['turns']:
        if set(turn['criteria']) != set(bench.base.CRITERIA) or not turn['reference'].strip():
            raise ValueError('Complete dialogue criteria required')


def calibration_summary(rows):
    summary = {}
    index = {(r['case'], r['order'], r['arm']): r for r in rows if 'scores' in r}
    corrections, regressions = [], []
    for arm in CAL_ARMS:
        cells = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'planned': len(cells), 'valid': sum('scores' in r for r in cells),
            'invalid': sum('error_type' in r for r in cells),
            'attempted': sum(r['usage']['calls'] > 0 for r in cells),
            'unattempted': sum(r['usage']['calls'] == 0 for r in cells),
            'usage': {k: bench.sum_usage([r['usage'] for r in cells], k) for k in bench.usage([])},
            'criteria': {}, 'order_disagreements': []}
        for criterion in v3.CRITERIA:
            matches = [s[criterion]['verdict'] == r['expected'][name][criterion]
                       for r in cells if 'scores' in r for name, s in r['scores'].items()]
            counts = Counter(s[criterion]['verdict'] for r in cells if 'scores' in r for s in r['scores'].values())
            summary[arm]['criteria'][criterion] = {'planned': sum(len(r['expected']) for r in cells),
                'available': len(matches), 'agreements': sum(matches),
                'verdict_counts': {v: counts[v] for v in ('pass', 'fail', 'uncertain')}}
        summary[arm]['fully_matched_presentations'] = sum(
            all(s[c]['verdict'] == r['expected'][name][c] for c in v3.CRITERIA)
            for r in cells if 'scores' in r for name, s in r['scores'].items())
        summary[arm]['fully_matched_variants_both_orders'] = 0
        for case in sorted({r['case'] for r in cells}):
            pair = [index.get((case, order, arm)) for order in (0, 1)]
            if not all(pair):
                continue
            summary[arm]['fully_matched_variants_both_orders'] += sum(
                all(r['scores'][name][c]['verdict'] == r['expected'][name][c] for r in pair for c in v3.CRITERIA)
                for name in pair[0]['scores'])
            for name in pair[0]['scores']:
                for c in v3.CRITERIA:
                    if pair[0]['scores'][name][c]['verdict'] != pair[1]['scores'][name][c]['verdict']:
                        summary[arm]['order_disagreements'].append({'case': case, 'variant': name, 'criterion': c})
    for case, order in sorted({(r['case'], r['order']) for r in rows}):
        pair = [index.get((case, order, arm)) for arm in CAL_ARMS]
        if not all(pair):
            continue
        for name in pair[0]['scores']:
            for criterion in v3.CRITERIA:
                expected = pair[0]['expected'][name][criterion]
                old, new = [r['scores'][name][criterion]['verdict'] for r in pair]
                item = {'case': case, 'order': order, 'variant': name, 'criterion': criterion}
                if old != expected and new == expected:
                    corrections.append(item)
                elif old == expected and new != expected:
                    regressions.append(item)
    audits = [(v, r['expected_obligations'][name][oid]) for r in rows if r['arm'] == CAL_ARMS[1] and 'scores' in r
              for name, scores in r['scores'].items() for oid, v in scores['critical_coverage']['obligation_checks'].items()]
    return {'by_arm': summary, 'corrections': corrections, 'regressions': regressions,
            'errors': [{k: r[k] for k in ('case', 'order', 'arm', 'error_type')} for r in rows if 'error_type' in r],
            'obligation_checks': {'available': len(audits), 'agreements': sum(a == b for a, b in audits)}}


async def calibrate(fixture, config, journal, report, save):
    for index, case in enumerate(fixture['cases']):
        for order in (0, 1):
            labels, names = boundaries.labels_for(case, order)
            for arm in CAL_ARMS if (index + order) % 2 == 0 else CAL_ARMS[::-1]:
                before = len(journal.events)
                row = {'case': case['id'], 'order': order, 'arm': arm,
                    'expected': {n: v['expected'] for n, v in case['variants'].items()},
                    'expected_obligations': {n: v['expected_obligations'] for n, v in case['variants'].items()}}
                try:
                    scores = await (v2 if arm == CAL_ARMS[0] else v3).assess(case, labels, config, journal, role=arm)
                    row['scores'] = {names[label]: score for label, score in scores.items()}
                except Exception as exc:
                    row['error_type'] = recorded_error(journal.events[before:], exc)
                row['usage'] = bench.usage(journal.events[before:])
                report['calibration'].append(row)
                save()
                print(json.dumps({'phase': 'calibration', 'case': case['id'], 'arm': arm, 'order': order,
                                  'valid': 'scores' in row}), flush=True)


def recorded_error(events, exc):
    return next((e['error_type'] for e in events if e.get('error_type')), type(exc).__name__)


def environment():
    env = bench.environment({'records': True}, 'all_auto')
    for module in ('DIRECTOR_ADVICE', 'EXECUTIVE_ADVICE', 'EXECUTIVE_METRICS_OBSERVATION'):
        env['DUALBRAIN_MODULE_' + module + '_MODE'] = 'off'
    env.update(DUALBRAIN_MEMORY_CONVERSATION_CONTEXT='1', DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN='0')
    return env


async def make_session(arm, config, journal, clock, records):
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(20261104)
    engine, clients = await bench.base.session_for('memory-review:' + arm, None, config, journal,
                                                 'left' if arm == 'draft' else arm, clock)
    engine.left._llm_client = bench.RecordedClient(config, journal, 'draft' if arm == 'draft' else 'left')
    engine.right._llm_client = bench.RecordedClient(config, journal, 'right')
    # Deliberate causal baseline: remove only the new review cue in both context
    # and ACC. All other production routing, source guards and AUTO gates remain.
    if arm != 'memory_bridge':
        engine.memory.competing_answer_groups = lambda _question: ()
    for record in records:
        engine.memory.store(MemoryTrace(**record))
    return engine


def update_memory(memory, turn, clock):
    if 'add_record' in turn:
        memory.store(MemoryTrace(**turn['add_record']))
    if 'correction' in turn:
        c = turn['correction']
        memory.begin_turn(c['question'], c['qid'], {k: c[k] for k in ('id', 'aliases', 'supersedes')} | {'remember': True})
        memory.store({'Q': c['question'], 'A': c['answer']}, qid=c['qid'])
        memory.past_qas = [replace(t, timestamp=clock[0]) if t.qid == c['qid'] else t for t in memory.past_qas]


async def cell(engine, arm, turn, draft, journal, options):
    # Match the policy's stochastic draws by turn, rather than arm scheduling.
    seed = int(bench.digest(turn['id'] + ':memory-review')[:8], 16)
    random.seed(seed)
    bench.base.np.random.seed(seed)
    bench.base.UUID_RNG.seed(seed)
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
    before, started = len(journal.events), time.perf_counter()
    row = {'case': turn['id'], 'arm': arm, 'status': 'unavailable', 'answer': None,
           'draft_sha256': bench.digest(draft) if draft is not None else None, 'review_need': turn['review_need']}
    captured, routing = [], []
    prior_failures = sum(model._llm_client.failed for model in (engine.left, engine.right))
    original_draft, original_route = engine.left.generate_answer, engine.controller._run_routing_stage
    async def generate(*args, **kwargs):
        answer = await original_draft(*args, **kwargs) if arm == 'draft' else draft
        captured.append(answer)
        return answer
    async def route(**kwargs):
        result = await original_route(**kwargs)
        state = result.decision.state
        routing.append({'action': result.decision.action, 'system2': bool(result.system2_active),
            'reason': result.system2_reason, 'competition_groups': state.get('acc_memory_competition', 0),
            'acc_sources': list(result.acc_signal.sources) if result.acc_signal else [],
            'selective': state.get('selective_consultation')})
        return result
    engine.left.generate_answer, engine.controller._run_routing_stage = generate, route
    try:
        result = await bench.base._handle_process(engine, {'question': turn['question'], 'qid': 'probe-' + turn['id'],
            'leading_brain': 'left', 'system2_mode': 'off' if arm == 'draft' else 'auto',
            'executive_mode': 'off', 'executive_observer_mode': 'off'}, **options)
        await engine.callosum.wait_idle()
        if (any(e.get('error_type') for e in journal.events[before:]) or len(captured) != 1
                or sum(model._llm_client.failed for model in (engine.left, engine.right)) != prior_failures):
            raise ValueError('Provider or initial draft unavailable')
        answer = captured[0] if arm == 'draft' else result['answer']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Empty answer')
        row.update(status='ok', answer=answer)
    except Exception as exc:
        row['error_type'] = recorded_error(journal.events[before:], exc)
    finally:
        engine.left.generate_answer, engine.controller._run_routing_stage = original_draft, original_route
    question = turn['question']
    row.update(route=routing[-1] if routing else None,
        source_qids=[t.qid for t in engine.memory.retrieve_related_traces(question)],
        evidence_sha256=bench.digest([engine.memory._answerability_source(t)
                                     for t in engine.memory.retrieve_related_traces(question)]),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        usage=bench.usage(journal.events[before:]), calls_by_role=dict(Counter(e['role'] for e in journal.events[before:])))
    return row


def dialogue_summary(report):
    grades = {(g['case'], g['arm']): g['scores'] for g in report['grades'] if 'scores' in g}
    return {arm: {'planned': len(rows := [r for r in report['rows'] if r['arm'] == arm]),
        'available': sum(r['status'] == 'ok' for r in rows),
        'graded': sum((r['case'], arm) in grades for r in rows),
        'all_pass': sum(all(v['verdict'] == 'pass' for v in grades[r['case'], arm].values())
                        for r in rows if (r['case'], arm) in grades),
        'criterion_pass': {c: sum(grades[r['case'], arm][c]['verdict'] == 'pass'
                            for r in rows if (r['case'], arm) in grades) for c in bench.base.CRITERIA},
        'right_calls': sum(r['calls_by_role'].get('right', 0) for r in rows),
        'optional_lookup_right_calls': sum(r['calls_by_role'].get('right', 0) for r in rows if r['review_need'] == 'optional'),
        'elapsed_ms': [r['elapsed_ms'] for r in rows]} for arm in ('draft', *ARMS)}


async def run(args, calibration, dialogue, config, journal):
    validate(calibration, dialogue)
    bench.base.enforce_runtime()
    report = {'schema_version': 1, 'experiment': 'memory_review_bridge', 'complete': False,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bench.base.ROOT.parent, text=True).strip(),
        'calibration': [], 'rows': [], 'grades': []}
    save = lambda: bench.checkpoint(args.output_dir / 'report.partial.json', report)
    await calibrate(calibration, replace(config, model=args.judge_model), journal, report, save)
    clock = [dialogue['clock_start']]
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, environment(), clear=True), patch('time.time', lambda: clock[0]), \
            patch('uuid.uuid4', lambda: uuid.UUID(int=bench.base.UUID_RNG.getrandbits(128))):
        engines = {}
        try:
            for arm in ('draft', *ARMS):
                engines[arm] = await make_session(arm, config, journal, clock, dialogue['records'])
            for index, turn in enumerate(dialogue['turns']):
                clock[0] = dialogue['clock_start'] + index * dialogue['seconds_per_turn']
                for engine in engines.values():
                    update_memory(engine.memory, turn, clock)
                draft = await cell(engines['draft'], 'draft', turn, None, journal, options)
                report['rows'].append(draft)
                save()
                for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
                    if draft['status'] == 'ok':
                        row = await cell(engines[arm], arm, turn, draft['answer'], journal, options)
                    else:
                        row = {'case': turn['id'], 'arm': arm, 'status': 'unavailable', 'answer': None,
                            'error_type': 'SharedDraftUnavailable', 'usage': bench.usage([]), 'calls_by_role': {},
                            'elapsed_ms': None, 'review_need': turn['review_need']}
                    report['rows'].append(row)
                    save()
                    print(json.dumps({'phase': 'dialogue', 'case': turn['id'], 'arm': arm,
                        'status': row['status'], 'calls': row['usage']['calls'], 'route': row.get('route')}), flush=True)
        finally:
            for engine in engines.values():
                await engine.close()
    for turn in dialogue['turns']:
        report['grades'].extend(await bench.grade(turn, report['rows'], replace(config, model=args.judge_model), journal))
        save()
    journal.finish()
    report.update(complete=True, live_provider_calls=journal.calls, usage=bench.usage(journal.events),
        calibration_summary=calibration_summary(report['calibration']), dialogue_summary=dialogue_summary(report))
    save()
    return report


def public_summary(report):
    return {k: v for k, v in report.items() if k not in {'calibration', 'rows', 'grades'}} | {
        'rows': [{k: v for k, v in row.items() if k != 'answer'} for row in report['rows']],
        'grades': [{k: v for k, v in grade.items() if k != 'scores'} | {
            'verdicts': {c: s['verdict'] for c, s in grade.get('scores', {}).items()}} for grade in report['grades']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--judge-model', default='gpt-6-sol')
    parser.add_argument('--max-calls', type=int, choices=(CALL_LIMIT,), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        calibration, dialogue = bench.read_json(CALIBRATION), bench.read_json(DIALOGUE)
        validate(calibration, dialogue)
        frozen = bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=bench.base.ROOT.parent, text=True).strip():
            raise ValueError('Commit the complete experiment before paid calls')
        model = args.model or frozen['model']
        config = bench.load_llm_config(default_model=model) or (bench.LLMConfig('openai', model, 'REPLAY_ONLY') if frozen else None)
        if config is None or config.provider != 'openai':
            raise ValueError('Configured OpenAI-compatible provider required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        files = sorted(set((bench.base.ROOT / 'scripts').glob('*.py')) | set((bench.base.ROOT / 'core').glob('*.py')))
        metadata = {'model': model, 'judge_model': args.judge_model, 'config': bench.public_config(config),
            'call_limit': CALL_LIMIT, 'calibration_sha256': bench.digest(calibration), 'dialogue_sha256': bench.digest(dialogue),
            'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'rubric_sha256': {'v2': bench.digest(v2.INSTRUCTIONS), 'v3': bench.digest(v3.INSTRUCTIONS)},
            'arms': list(ARMS), 'calibration_arms': list(CAL_ARMS), 'runtime': environment_contract()}
        journal = bench.Journal(config, metadata, frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, calibration, dialogue, config, journal))
        report['metadata'] = metadata
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': report['live_provider_calls'], 'usage': report['usage']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Memory review pilot incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


def environment_contract():
    return {k: v for k, v in environment().items() if k.startswith('DUALBRAIN_') or k == 'REASONING_DIAL'}


if __name__ == '__main__':
    main()
