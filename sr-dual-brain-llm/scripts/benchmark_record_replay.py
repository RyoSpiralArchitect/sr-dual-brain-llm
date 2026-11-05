#!/usr/bin/env python3
"""Prospective repeated support factor, answer calibration and bound-receipt runtime pilot."""
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
from unittest.mock import patch

import answer_assessment_v2 as assessment
import benchmark_memory_review as previous
from core.memory_answerability import AnswerabilityGate, OpenAIAnswerability
from core.shared_memory import MemoryTrace, SharedMemory

bench = assessment.bench
ARMS = ('generic_contract', 'replay_contract')
RUBRICS = ('answer_v1', 'quotation_v2')
CALL_LIMIT = 96
CALIBRATION = bench.base.ROOT / 'examples/record_replay_calibration.json'
FIXTURE = bench.base.ROOT / 'examples/record_replay_benchmark.json'


def validate(calibration, fixture):
    if (calibration.get('schema_version') != 1 or fixture.get('schema_version') != 1
            or calibration.get('origin') != 'prospective_record_replay_answer_contract'
            or fixture.get('origin') != 'prospective_record_replay_support_factor'
            or fixture.get('support_repeats') != 3
            or tuple(c['id'] for c in calibration['cases']) != (
                'literal_unverified', 'corrected_quote', 'missing_current', 'ambiguous_record')
            or tuple(c['id'] for c in fixture['cases']) != (
                'explicit_quote', 'corrected_followup', 'scope_switch', 'expired', 'extra_clause', 'incomplete')):
        raise ValueError('Complete prospective fixtures required')
    for case in [*calibration['cases'], *fixture['cases']]:
        if (set(case['criteria']) != set(bench.base.CRITERIA)
                or case['requested_operation'] not in {'recorded_answer_replay', 'fact_lookup'}):
            raise ValueError('Explicit operation and three criteria required')
    for case in calibration['cases']:
        if len(case['variants']) != 3 or any(set(v['expected']) != set(bench.base.CRITERIA)
                                            for v in case['variants'].values()):
            raise ValueError('Three complete authored variants required')


def labels_for(case, order):
    names = list(case['variants'])
    random.Random(int(bench.digest(case['id'] + ':quotation-labels')[:8], 16)).shuffle(names)
    if order:
        names.reverse()
    mapping = dict(zip((chr(65 + i) for i in range(len(names))), names))
    return {label: {'answer': case['variants'][name]['answer']} for label, name in mapping.items()}, mapping


def calibration_summary(rows):
    result = {}
    indexed = {(r['case'], r['order'], r['arm']): r for r in rows if 'scores' in r}
    differences = []
    for arm in RUBRICS:
        cells = [r for r in rows if r['arm'] == arm]
        result[arm] = {'planned': len(cells), 'valid': sum('scores' in r for r in cells),
            'attempted': sum(r['usage']['calls'] > 0 for r in cells), 'criteria': {}, 'order_disagreements': []}
        result[arm]['usage'] = {k: bench.sum_usage([r['usage'] for r in cells], k) for k in bench.usage([])}
        result[arm]['unattempted'] = sum(r['usage']['calls'] == 0 for r in cells)
        result[arm]['invalid'] = sum('error_type' in r for r in cells)
        result[arm]['fully_matched_presentations'] = sum(
            all(s[c]['verdict'] == r['expected'][name][c] for c in bench.base.CRITERIA)
            for r in cells if 'scores' in r for name, s in r['scores'].items())
        for criterion in bench.base.CRITERIA:
            values = [s[criterion]['verdict'] for r in cells if 'scores' in r for s in r['scores'].values()]
            result[arm]['criteria'][criterion] = {'planned': len(cells) * 3, 'available': len(values),
                'agreements': sum(s[criterion]['verdict'] == r['expected'][name][criterion]
                    for r in cells if 'scores' in r for name, s in r['scores'].items()),
                'verdict_counts': dict(Counter(values))}
        for cid in sorted({r['case'] for r in cells}):
            pair = [indexed.get((cid, order, arm)) for order in (0, 1)]
            if not all(pair):
                continue
            for name in pair[0]['scores']:
                for criterion in bench.base.CRITERIA:
                    if pair[0]['scores'][name][criterion]['verdict'] != pair[1]['scores'][name][criterion]['verdict']:
                        result[arm]['order_disagreements'].append({'case': cid, 'variant': name, 'criterion': criterion})
    for cid, order in sorted({(r['case'], r['order']) for r in rows}):
        pair = [indexed.get((cid, order, arm)) for arm in RUBRICS]
        if not all(pair):
            continue
        for name in pair[0]['scores']:
            for criterion in bench.base.CRITERIA:
                old, new = [r['scores'][name][criterion]['verdict'] for r in pair]
                expected = pair[0]['expected'][name][criterion]
                if old != new:
                    differences.append({'case': cid, 'order': order, 'variant': name, 'criterion': criterion,
                        'v1': old, 'v2': new, 'expected': expected,
                        'corrected': old != expected and new == expected, 'regressed': old == expected and new != expected})
    return {'by_arm': result, 'paired_differences': differences,
        'errors': [{k: r[k] for k in ('case', 'order', 'arm', 'error_type')} for r in rows if 'error_type' in r]}


def configure(memory, case, arm, clock):
    if arm == ARMS[0]:
        memory.replay_intent = lambda _question: None
    for record in case['records']:
        memory.store(MemoryTrace(**record))
    if 'correction' in case:
        previous.update_memory(memory, case, clock)
    if case.get('focus'):
        memory.begin_turn(case['focus'], 'setup-focus')


async def support_cell(case, arm, repeat, config, journal, clock):
    backend = OpenAIAnswerability(config, config.model)
    async def post(url, payload, headers, **kwargs):
        return await journal.post('support_' + arm, url, payload, headers, **kwargs)
    backend.client._post_json = post
    memory = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(backend))
    configure(memory, case, arm, clock)
    memory.begin_turn(case['question'], 'probe')
    before, started = len(journal.events), time.perf_counter()
    result = await memory.prepare_retrieval(case['question'])
    traces = memory.retrieve_related_traces(case['question'])
    row = {'case': case['id'], 'arm': arm, 'repeat': repeat, 'result': result['answerability'],
        'source_qids': [t.qid for t in traces], 'expected_source_qids': case['expected_source_qids'],
        'intent': memory.replay_intent(case['question']) is not None,
        'elapsed_ms': (time.perf_counter() - started) * 1000, 'usage': bench.usage(journal.events[before:]),
        'receipt': {'backend': memory.answerability_gate.identity,
                    'cache': dict(memory.answerability_gate.cache)}}
    return row


async def runtime_cell(case, arm, receipt, config, journal, clock, options, *, system2_mode='auto'):
    before, started = len(journal.events), time.perf_counter()
    logical_time = clock[0]
    seed = int(bench.digest(case['id'] + ':replay-runtime')[:8], 16)
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
    engine, _ = await bench.base.session_for(case['id'] + ':' + arm, None, config, journal, arm, clock)
    clock[0] = logical_time
    for model, role in ((engine.left, 'left'), (engine.right, 'right')):
        model._llm_client = bench.RecordedClient(config, journal, role)
    configure(engine.memory, case, arm, clock)
    gate = engine.memory.answerability_gate
    # Reuse only repeat zero, including false or missing decisions. Each key
    # binds exact normalized query/source/intent/policy/backend. No new support
    # call or selected best-of-three decision is permitted during runtime.
    async def unavailable(_request):
        raise RuntimeError('No prospectively measured support receipt')
    gate.backend.evaluate = unavailable
    original_prepare = engine.memory.prepare_retrieval
    imported = []
    async def prepare(question, **kwargs):
        query = engine.memory.retrieval_question(question)
        intent = engine.memory.replay_intent(question)
        for trace in engine.memory.answerability_candidates(question):
            source = engine.memory._answerability_source(trace)
            key = gate._key(query, source, intent)
            if gate.identity == receipt['backend'] and key in receipt['cache']:
                gate.cache[key] = receipt['cache'][key]
                imported.append(key)
        return await original_prepare(question, **kwargs)
    engine.memory.prepare_retrieval = prepare
    random.seed(seed)
    bench.base.np.random.seed(seed)
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
    row = {'case': case['id'], 'arm': arm, 'status': 'unavailable', 'answer': None}
    routes = []
    original_route = engine.controller._run_routing_stage
    async def route(**kwargs):
        result = await original_route(**kwargs)
        routes.append({'action': result.decision.action, 'system2': result.system2_active,
            'selective': result.decision.state.get('selective_consultation')})
        return result
    engine.controller._run_routing_stage = route
    try:
        result = await bench.base._handle_process(engine, {'question': case['question'], 'qid': 'probe',
            'leading_brain': 'left', 'system2_mode': system2_mode, 'executive_mode': 'off',
            'executive_observer_mode': 'off'}, **options)
        await engine.callosum.wait_idle()
        if any(e.get('error_type') for e in journal.events[before:]) or any(
                model._llm_client.failed for model in (engine.left, engine.right)):
            raise ValueError('Provider stage unavailable')
        row.update(status='ok', answer=result['answer'])
    except Exception as exc:
        row['error_type'] = previous.recorded_error(journal.events[before:], exc)
    finally:
        traces = engine.memory.retrieve_related_traces(case['question'])
        row.update(available_context={'reference': engine.memory.reference_resolution(case['question']),
            'approved_records': [engine.memory._answerability_source(t) for t in traces]},
            source_qids=[t.qid for t in traces], receipt_keys=sorted(set(imported)),
            route=routes[-1] if routes else None)
        row['system2_mode'] = system2_mode
        row['exact_approved_answer'] = (row['answer'] == traces[0].answer
            if row['status'] == 'ok' and case['requested_operation'] == 'recorded_answer_replay' and len(traces) == 1 else None)
        targets = [t for t in engine.memory.past_qas if t.qid in case['expected_source_qids']]
        row['exact_stored_answer'] = (row['answer'] == targets[0].answer
            if row['status'] == 'ok' and case['requested_operation'] == 'recorded_answer_replay' and len(targets) == 1 else None)
        await engine.close()
    row.update(usage=bench.usage(journal.events[before:]),
        calls_by_role=dict(Counter(e['role'] for e in journal.events[before:])),
        elapsed_ms=(time.perf_counter() - started) * 1000)
    return row


async def run(args, calibration, fixture, config, journal):
    validate(calibration, fixture)
    bench.base.enforce_runtime()
    report = {'schema_version': 1, 'complete': False, 'experiment': 'record_replay_intent',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bench.base.ROOT.parent, text=True).strip(),
        'calibration': [], 'support': [], 'rows': [], 'grades': []}
    save = lambda: bench.checkpoint(args.output_dir / 'report.partial.json', report)
    judge = replace(config, model=args.judge_model)
    for index, case in enumerate(calibration['cases']):
        for order in (0, 1):
            labels, names = labels_for(case, order)
            for arm in RUBRICS if (index + order) % 2 == 0 else RUBRICS[::-1]:
                before = len(journal.events)
                row = {'case': case['id'], 'order': order, 'arm': arm,
                       'expected': {n: v['expected'] for n, v in case['variants'].items()}}
                records = case['available_context']['approved_records']
                row['literal_equal'] = {n: (v['answer'] == records[0]['answer']
                    if case['requested_operation'] == 'recorded_answer_replay' and len(records) == 1 else None)
                    for n, v in case['variants'].items()}
                try:
                    scores = await assessment.assess(case, labels, judge, journal, modern=arm == RUBRICS[1], role=arm)
                    row['scores'] = {names[label]: score for label, score in scores.items()}
                except Exception as exc:
                    row['error_type'] = previous.recorded_error(journal.events[before:], exc)
                row['usage'] = bench.usage(journal.events[before:])
                report['calibration'].append(row)
                save()
                print(json.dumps({'phase': 'calibration', 'case': case['id'], 'arm': arm, 'order': order,
                                  'valid': 'scores' in row}), flush=True)
    clock = [fixture['clock']]
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, previous.environment(), clear=True), patch('time.time', lambda: clock[0]):
        for index, case in enumerate(fixture['cases']):
            for repeat in range(fixture['support_repeats']):
                for arm in ARMS if (index + repeat) % 2 == 0 else ARMS[::-1]:
                    row = await support_cell(case, arm, repeat, config, journal, clock)
                    report['support'].append(row)
                    save()
            if not case['execute']:
                continue
            for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
                receipt = next(r['receipt'] for r in report['support']
                               if r['case'] == case['id'] and r['arm'] == arm and r['repeat'] == 0)
                row = await runtime_cell(case, arm, receipt, config, journal, clock, options)
                report['rows'].append(row)
                save()
                print(json.dumps({'phase': 'runtime', 'case': case['id'], 'arm': arm,
                    'status': row['status'], 'exact': row['exact_approved_answer'], 'usage': row['usage']}), flush=True)
    for case in fixture['cases']:
        rows = [r for r in report['rows'] if r['case'] == case['id'] and r['status'] == 'ok']
        if not rows:
            continue
        # Actual exposure can differ between policy arms. Grade each answer with
        # its own context, not another arm's approval or the whole fixture.
        for row in rows:
            packet_case = {**case, 'available_context': row['available_context']}
            before = len(journal.events)
            try:
                scores = await assessment.assess(packet_case, {'A': row}, judge, journal, role='answer_judge_v2')
                report['grades'].append({'case': case['id'], 'arm': row['arm'], 'scores': scores['A']})
            except Exception as exc:
                report['grades'].append({'case': case['id'], 'arm': row['arm'],
                    'error_type': previous.recorded_error(journal.events[before:], exc)})
            save()
    journal.finish()
    report.update(complete=True, live_provider_calls=journal.calls, usage=bench.usage(journal.events),
                  calibration_summary=calibration_summary(report['calibration']))
    save()
    return report


def public_summary(report):
    return {k: v for k, v in report.items() if k not in {'calibration', 'support', 'rows', 'grades'}} | {
        'support': [{k: v for k, v in row.items() if k != 'receipt'} for row in report['support']],
        'rows': [{k: v for k, v in row.items() if k not in {'answer', 'available_context', 'receipt_keys'}} | {
                    'receipt_keys_sha256': bench.digest(row['receipt_keys'])} for row in report['rows']],
        'grades': [{k: v for k, v in g.items() if k != 'scores'} | {
                    'verdicts': {c: s['verdict'] for c, s in g.get('scores', {}).items()}} for g in report['grades']]}


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
        calibration, fixture = bench.read_json(CALIBRATION), bench.read_json(FIXTURE)
        validate(calibration, fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=bench.base.ROOT.parent, text=True).strip():
            raise ValueError('Commit the prospective experiment before paid calls')
        model = args.model or frozen['model']
        config = bench.load_llm_config(default_model=model) or (bench.LLMConfig('openai', model, 'REPLAY_ONLY') if frozen else None)
        if config is None or config.provider != 'openai':
            raise ValueError('Configured OpenAI-compatible provider required')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        files = sorted(set((bench.base.ROOT / 'core').glob('*.py')) | set((bench.base.ROOT / 'scripts').glob('*.py')))
        metadata = {'model': model, 'judge_model': args.judge_model, 'config': bench.public_config(config),
            'call_limit': CALL_LIMIT, 'calibration_sha256': bench.digest(calibration), 'fixture_sha256': bench.digest(fixture),
            'arms': list(ARMS), 'rubrics': list(RUBRICS), 'support_repeats': fixture['support_repeats'],
            'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'answer_rubric_sha256': bench.digest(assessment.INSTRUCTIONS), 'runtime': previous.environment_contract()}
        journal = bench.Journal(config, metadata, frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, calibration, fixture, config, journal))
        report['metadata'] = metadata
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': report['live_provider_calls'], 'usage': report['usage']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Record replay pilot incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
