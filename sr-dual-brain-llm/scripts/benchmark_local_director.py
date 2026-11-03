#!/usr/bin/env python3
"""Prospective trusted local Director controls plus structured judge calibration."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import advisor_assessment as assessment
import benchmark_advisor_confidence as confidence
from core.advisor_timing import AdvisorTiming
from core.dual_brain import DualBrainController

recovery = confidence.recovery
bench = confidence.bench
ARMS = ('compact_remote', 'local_facts')
PHASES = ('initial', 'opportunity', 'confirmation', 'followup', 'release')
FIXTURE = bench.base.ROOT / 'examples/local_director_benchmark.json'
SYSTEM2_MODES = ('on', 'auto', 'on', 'auto', 'on')


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_local_director'
            or fixture.get('idle_before_opportunity_seconds') != 16
            or tuple(c['id'] for c in fixture['cases']) != PHASES
            or tuple(c.get('system2_mode') for c in fixture['cases']) != SYSTEM2_MODES):
        raise ValueError('Require the complete independent local-control sequence')
    for case in fixture['cases']:
        if (not case['question'].strip() or case['records'] or case['rpc_seconds'] != 120
                or (case['executive_mode'], case['executive_observer_mode']) != ('observe', 'both')
                or set(case['criteria']) != set(bench.base.CRITERIA)
                or set(case['advice_obligations']) != {'director', 'executive', 'metrics'}
                or any(not isinstance(v, str) or not v.strip() for v in case['advice_obligations'].values())):
            raise ValueError('Invalid task, caller or bounded-control contract')


def _append_advice(engine, *, role, fields, facts, source, usable):
    text = '\n'.join(k + ': ' + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True))
                     for k, v in fields.items() if v)
    engine._benchmark_advice.append({'role': role, 'fields': fields, 'text': text,
        'input_facts': facts, 'source': source, 'structurally_usable': bool(usable)})


def prepare_engine(engine, arm):
    controller = engine.controller
    # The common historical session constructor pins local steering off.
    controller._local_director_plan = (DualBrainController._local_director_plan.__get__(controller)
                                       if arm == 'local_facts' else lambda **_kw: None)
    engine._benchmark_advice, engine._benchmark_phase_order = [], []
    engine._benchmark_local_stage_ms, engine._benchmark_current_local = None, False
    engine._benchmark_routing = None
    for model, is_director in ((engine.director, True), (engine.executive, False)):
        original = model.advise

        async def advise(*a, _original=original, _director=is_director, **kw):
            result = await _original(*a, **kw)
            work = getattr(asyncio.current_task(), '_advisor_work', None)
            if getattr(result, 'source', '') == 'llm' and (work is None or work.stop_elapsed_ms is None):
                role = ('director' if _director else 'metrics' if 'Observer report (internal;' in kw.get('context', '') else 'executive')
                facts = ({'signals_json': json.dumps(kw.get('signals') or {}, ensure_ascii=False)[:1200],
                          'context': (kw.get('context') or '')[:1600]} if role == 'director'
                         else {'observer_context': (kw.get('context') or '')[:1600]} if role == 'metrics' else {})
                _append_advice(engine, role=role, fields=assessment.structured_output(result), facts=facts,
                    source='llm', usable=DualBrainController._advisor_usable(result))
            return result

        model.advise = advise

    original_stage = controller._run_pre_draft_director_stage

    async def stage(**kw):
        started = time.perf_counter()
        result = await original_stage(**kw)
        payload = result.director_payload or {}
        engine._benchmark_current_local = payload.get('source') == 'local_facts'
        if engine._benchmark_current_local:
            plan = payload['local_plan']
            engine._benchmark_local_stage_ms = (time.perf_counter() - started) * 1000.
            fields = assessment.structured_output(SimpleNamespace(memo=payload.get('memo', ''),
                control=result.director_control or {}, directives={}, mix_in=''))
            _append_advice(engine, role='director', fields=fields,
                facts={'trusted_controls': plan['controls'], 'trusted_reasons': plan['reasons']},
                source='local_facts', usable=bool(result.director_control))
            engine._benchmark_phase_order.append('local_controls_applied')
        return result

    controller._run_pre_draft_director_stage = stage
    original_draft = controller._run_left_draft_stage

    async def draft(**kw):
        engine._benchmark_phase_order.append('draft_entered')
        return await original_draft(**kw)

    controller._run_left_draft_stage = draft
    original_route = controller._run_routing_stage

    async def route(**kw):
        result = await original_route(**kw)
        engine._benchmark_phase_order.append('routing_complete')
        state = result.decision.state
        engine._benchmark_routing = {'action': result.decision.action,
            'system2_active': bool(result.system2_active),
            'director_consult': state.get('director_consult'),
            'director_consult_blocked': state.get('director_consult_blocked'),
            'director_memory_working': state.get('director_memory_working'),
            'director_memory_long_term': state.get('director_memory_long_term')}
        return result

    controller._run_routing_stage = route


def after_cell(row, engine):
    row['advisor_outputs'] = sorted(engine._benchmark_advice, key=lambda o: (o['role'], o['source'], o['text']))
    order = list(engine._benchmark_phase_order)
    row['local_director_stage_ms'] = engine._benchmark_local_stage_ms
    row['phase_order'] = order
    row['local_before_draft'] = ('local_controls_applied' in order and 'draft_entered' in order
        and order.index('local_controls_applied') < order.index('draft_entered'))
    row['routing'] = engine._benchmark_routing
    engine._benchmark_advice.clear()
    engine._benchmark_phase_order.clear()
    engine._benchmark_local_stage_ms, engine._benchmark_current_local = None, False
    engine._benchmark_routing = None


async def run(args, fixture, config, journal, *, idle=None, ledger_clock=None):
    validate(fixture)
    before = len(journal.events)
    calibration = await assessment.calibrate(replace(config, model=args.judge_model), journal)
    calibration_usage = bench.usage(journal.events[before:])
    experiment = recovery.SessionExperiment(name='trusted_local_director', arms=ARMS, phases=PHASES,
        validate_fixture=validate, make_timing=lambda _arm, clock: AdvisorTiming(clock=clock),
        make_observer=confidence.ObservedConfidence, prepare_engine=prepare_engine,
        after_cell=after_cell, grade_advice=assessment.grade_report, compact_delivery=True)
    report = await recovery.run(args, fixture, config, journal, idle=idle, ledger_clock=ledger_clock, experiment=experiment)
    report['calibration'], report['calibration_usage'] = calibration, calibration_usage
    return report


def public_summary(report):
    # Calibration rows contain private quoted field context, just like advice grades.
    public = recovery.public_summary({k: v for k, v in report.items() if k != 'calibration'})
    public['calibration_summary'] = report['calibration']['summary']
    public['advice_summary'] = {}
    for arm in ARMS:
        rows = [row for row in report['rows'] if row['arm'] == arm]
        public['advice_summary'][arm] = {}
        for role in ('director', 'executive', 'metrics'):
            role_outputs = [output for row in rows for output in row['advisor_outputs'] if output['role'] == role]
            role_grades = [grade for grade in report['advice_grades'] if grade.get('arm') == arm and grade.get('role') == role]
            sources = {}
            for source in ('llm', 'local_facts'):
                outputs = [output for output in role_outputs if output['source'] == source]
                grades = [grade['scores'] for grade in role_grades if grade.get('source') == source]
                first = next((row for row in rows if any(o['role'] == role and o['source'] == source and o['structurally_usable'] for o in row['advisor_outputs'])), None)
                sources[source] = {'available': len(outputs), 'graded': len(grades),
                    'structurally_usable': sum(o['structurally_usable'] for o in outputs),
                    'first_usable_turn': first['session_turn'] if first else None,
                    'criterion_outcomes': {criterion: {verdict: sum(g[criterion]['verdict'] == verdict for g in grades)
                        for verdict in ('pass', 'fail', 'uncertain')} for criterion in assessment.CRITERIA},
                    'all_pass': sum(all(g[c]['verdict'] == 'pass' for c in assessment.CRITERIA) for g in grades)}
            public['advice_summary'][arm][role] = {'requested': len(rows), 'available': len(role_outputs),
                'missing_exposure': len(rows) - len(role_outputs), 'sources': sources}
    public['advice_grader_failures'] = sum('error_type' in grade for grade in report['advice_grades'])
    for row, raw in zip(public['rows'], report['rows']):
        row.update({key: raw[key] for key in ('local_director_stage_ms', 'local_before_draft', 'phase_order', 'routing')})
        row['cell_elapsed_ms'] = raw['elapsed_ms']
        row['advisor_output_counts'] = {source: {role: sum(o['source'] == source and o['role'] == role for o in raw['advisor_outputs'])
            for role in ('director', 'executive', 'metrics')} for source in ('llm', 'local_facts')}
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--judge-model', default='gpt-6-sol')
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if not 1 <= args.max_calls <= 96:
            raise ValueError('Prospective maximum is 96 attempts')
        fixture = bench.read_json(FIXTURE)
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        model = args.model or frozen['model']
        config = bench.load_llm_config(default_model=model)
        if config is None:
            if args.model:
                raise ValueError('Configured live provider required')
            config = bench.LLMConfig('openai', model, 'REPLAY_ONLY')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        if config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        files = [Path(__file__), Path(assessment.__file__), Path(confidence.__file__), Path(recovery.__file__),
            Path(recovery.sessions.__file__), Path(recovery.sessions.budget.__file__), Path(bench.__file__),
            Path(bench.base.__file__), bench.base.ROOT / 'scripts/benchmark_conversation_memory.py',
            bench.base.ROOT / 'scripts/engine_stdio.py', *sorted((bench.base.ROOT / 'core').glob('*.py'))]
        metadata = {'model': model, 'judge_model': args.judge_model, 'config': bench.public_config(config),
            'fixture_sha256': bench.digest(fixture), 'calibration_fixture_sha256': bench.digest(bench.read_json(assessment.FIXTURE)),
            'code_sha256': bench.digest([(str(p.relative_to(bench.base.ROOT)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'advice_judge_sha256': bench.digest(assessment.INSTRUCTIONS), 'call_limit': args.max_calls, 'arms': list(ARMS),
            'advice_judge_max_output_tokens': assessment.MAX_OUTPUT_TOKENS,
            'torch_policy': bench.TORCH_AVAILABLE, 'history_policy': 'retained_sessions;both_confirmed_v4;recorded_resource_reads_only_in_replay',
            'local_policy': {'baseline': 'compact_remote', 'treatment': 'trusted_local_facts',
                'system2_modes': list(SYSTEM2_MODES), 'calibration_pairs': 6, 'classification_calls': 0},
            'gate_policies': {key: spec.policy_version for key, spec in bench.DEFAULT_REGISTRY.items()}}
        journal = recovery.sessions.SessionJournal(config, metadata, frozen=frozen, max_calls=args.max_calls,
            save=lambda data: bench.checkpoint(args.output_dir / 'journal.partial.json', data))
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, fixture, config, journal))
        bench.write_private(args.output_dir / 'report.json', report)
        bench.write_private(args.output_dir / 'journal.json', frozen or journal.data)
        bench.write_private(args.output_dir / 'summary.json', public_summary(report))
        print(json.dumps({'complete': True, 'live_provider_calls': report['live_provider_calls'], 'transport_calls': report['transport_calls']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir / 'failure.json', {'complete': False, 'error_type': type(exc).__name__})
        print('Local Director study incomplete: ' + type(exc).__name__, file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
