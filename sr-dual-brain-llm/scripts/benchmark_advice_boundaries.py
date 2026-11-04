#!/usr/bin/env python3
"""Prospective paired v1/v2 criterion calibration; every attempt is journaled."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import subprocess

import advisor_assessment as v1
import advisor_assessment_v2 as v2

bench = v1.bench
ARMS = ('structured_v1', 'criterion_v2')
ORDERS = (0, 1)
CASE_IDS = ('barrier_position', 'invoice_decoration', 'limiter_check_count',
            'inventory_authority', 'sensor_qualification', 'observer_access')
FIXTURE = bench.base.ROOT / 'examples/advice_boundary_calibration.json'
CALL_LIMIT = len(CASE_IDS) * len(ARMS) * len(ORDERS)


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_advice_criterion_boundaries'
            or tuple(case['id'] for case in fixture['cases']) != CASE_IDS):
        raise ValueError('Complete independent boundary fixture required')
    for case in fixture['cases']:
        if (case['role'] not in {'director', 'executive', 'metrics'} or len(case['variants']) != 3
                or any(not isinstance(case[k], str) or not case[k].strip()
                       for k in ('question', 'reference', 'obligations'))):
            raise ValueError('Invalid boundary case')
        for variant in case['variants'].values():
            v1.validate_fields(variant['fields'])
            if (set(variant['expected']) != set(v1.CRITERIA)
                    or any(value not in {'pass', 'fail', 'uncertain'} for value in variant['expected'].values())):
                raise ValueError('Complete fixed expected verdicts required')


def labels_for(case, order):
    variants = list(case['variants'])
    random.Random(int(bench.digest(case['id'] + ':boundary-labels')[:8], 16)).shuffle(variants)
    if order == 1:
        variants.reverse()
    labels = {chr(65 + i): {'role': case['role'], 'fields': case['variants'][name]['fields'],
        'input_facts': case.get('input_facts', {}), 'final_answer': case.get('final_answer', '')}
        for i, name in enumerate(variants)}
    return labels, dict(zip(labels, variants))


def summarize(rows):
    result = {}
    for arm in ARMS:
        cells = [r for r in rows if r['arm'] == arm]
        available = [r for r in cells if 'scores' in r]
        result[arm] = {'planned_packets': len(cells), 'valid_packets': len(available),
            'failed_packets': sum('error_type' in r for r in cells), 'graded_outputs': sum(len(r['scores']) for r in available),
            'attempted_packets': sum(r['usage']['calls'] > 0 for r in cells),
            'unattempted_packets': sum(r['usage']['calls'] == 0 for r in cells),
            'criteria': {}, 'label_order_disagreements': []}
        for criterion in v1.CRITERIA:
            matches = [s[criterion]['verdict'] == r['expected'][name][criterion]
                       for r in available for name, s in r['scores'].items()]
            counts = Counter(s[criterion]['verdict'] for r in available for s in r['scores'].values())
            result[arm]['criteria'][criterion] = {
                'planned_checks': sum(len(r['expected']) for r in cells), 'available_checks': len(matches),
                'agreements': sum(matches), 'disagreements': len(matches) - sum(matches),
                'verdict_counts': {v: counts[v] for v in ('pass', 'fail', 'uncertain')}}
        indexed = {(r['case'], r['order']): r for r in available}
        for case_id in CASE_IDS:
            if not all((case_id, order) in indexed for order in ORDERS):
                continue
            a, b = (indexed[case_id, order] for order in ORDERS)
            for name in a['scores']:
                for criterion in v1.CRITERIA:
                    if a['scores'][name][criterion]['verdict'] != b['scores'][name][criterion]['verdict']:
                        result[arm]['label_order_disagreements'].append({'case': case_id, 'variant': name, 'criterion': criterion})
        result[arm]['fully_matched_presentations'] = sum(
            all(s[c]['verdict'] == r['expected'][name][c] for c in v1.CRITERIA)
            for r in available for name, s in r['scores'].items())
        result[arm]['fully_matched_variants_both_orders'] = sum(
            all(indexed[case, order]['scores'][name][c]['verdict'] == indexed[case, order]['expected'][name][c]
                for order in ORDERS for c in v1.CRITERIA)
            for case in CASE_IDS if all((case, order) in indexed for order in ORDERS)
            for name in indexed[case, 0]['scores'])
    differences = []
    indexed = {(r['case'], r['order'], r['arm']): r for r in rows if 'scores' in r}
    for case in CASE_IDS:
        for order in ORDERS:
            if not all((case, order, arm) in indexed for arm in ARMS):
                continue
            a, b = (indexed[case, order, arm] for arm in ARMS)
            for name in a['scores']:
                for criterion in v1.CRITERIA:
                    old, new = a['scores'][name][criterion]['verdict'], b['scores'][name][criterion]['verdict']
                    expected = a['expected'][name][criterion]
                    if old != new:
                        differences.append({'case': case, 'order': order, 'variant': name, 'criterion': criterion,
                            'expected': expected, 'v1': old, 'v2': new,
                            'v2_corrected': old != expected and new == expected,
                            'v2_regressed': old == expected and new != expected})
    return {'by_arm': result, 'paired_differences': differences}


async def run(args, fixture, config, journal):
    validate(fixture)
    bench.base.enforce_runtime()
    report = {'schema_version': 1, 'experiment': 'prospective_advice_boundaries', 'complete': False,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bench.base.ROOT.parent, text=True).strip(),
        'frozen_replay': journal.frozen is not None, 'rows': []}
    save = lambda: bench.checkpoint(args.output_dir / 'report.partial.json', report)
    for index, case in enumerate(fixture['cases']):
        expected = {name: variant['expected'] for name, variant in case['variants'].items()}
        for order in ORDERS:
            labels, names = labels_for(case, order)
            # Paired input labels match within each comparison; both arm orders
            # occur in every case. Expected/variant identifiers never enter HTTP.
            arms = ARMS if (index + order) % 2 == 0 else ARMS[::-1]
            for arm in arms:
                before = len(journal.events)
                row = {'case': case['id'], 'order': order, 'arm': arm, 'expected': expected}
                try:
                    assessor = v1.assess if arm == ARMS[0] else v2.assess
                    scores = await assessor(case, labels, config, journal, role=arm)
                    row['scores'] = {names[label]: score for label, score in scores.items()}
                except Exception as exc:
                    # A frozen transport raises a generic exception. Preserve
                    # the original recorded error rather than relabel it during
                    # replay; local JSON/schema errors have no transport error.
                    recorded = next((e['error_type'] for e in journal.events[before:] if e.get('error_type')), None)
                    row['error_type'] = recorded or type(exc).__name__
                row['usage'] = bench.usage(journal.events[before:])
                report['rows'].append(row)
                save()
                print(json.dumps({k: row[k] for k in ('case', 'order', 'arm')} |
                    {'valid': 'scores' in row, 'calls': row['usage']['calls']}), flush=True)
    journal.finish()
    report.update(complete=True, live_provider_calls=journal.calls, usage=bench.usage(journal.events),
        summary=summarize(report['rows']))
    save()
    return report


def public_summary(report):
    result = {k: v for k, v in report.items() if k != 'rows'}
    result['rows'] = [{k: v for k, v in row.items() if k != 'scores'} | {
        'scores': {name: {c: {k: s[k] for k in ('verdict', 'field_path', 'interpretation', 'basis') if k in s}
                          for c, s in criteria.items()} for name, criteria in row.get('scores', {}).items()}}
        for row in report['rows']]
    return result


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
        fixture = bench.read_json(FIXTURE)
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git', 'status', '--porcelain'], cwd=bench.base.ROOT.parent, text=True).strip():
            raise ValueError('Commit the experiment before paid calls')
        model = args.model or frozen['model']
        config = bench.load_llm_config(default_model=model)
        if config is None:
            if args.model:
                raise ValueError('Configured live provider required')
            config = bench.LLMConfig('openai', model, 'REPLAY_ONLY')
        config = replace(config, model=model, max_output_tokens=4096, timeout_seconds=60,
                         auto_continue=False, max_continuations=0)
        if config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        files = sorted(set((bench.base.ROOT / 'scripts').glob('*.py')) | set((bench.base.ROOT / 'core').glob('*.py')))
        metadata = {'model': model, 'config': bench.public_config(config), 'call_limit': CALL_LIMIT,
            'fixture_sha256': bench.digest(fixture), 'arms': list(ARMS), 'orders': list(ORDERS),
            'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'rubric_sha256': {'v1': bench.digest(v1.INSTRUCTIONS), 'v2': bench.digest(v2.INSTRUCTIONS)},
            'v2_version': v2.VERSION, 'evidence': 'synthetic_independent_calibration;no_production_turns'}
        journal = bench.Journal(config, metadata, frozen=frozen, max_calls=CALL_LIMIT,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, fixture, config, journal))
        report['metadata'] = metadata
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': report['live_provider_calls']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Boundary study incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
