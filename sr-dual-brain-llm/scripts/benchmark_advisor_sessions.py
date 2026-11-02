#!/usr/bin/env python3
"""Natural advisor history in persistent EngineSessions; controlled/live/replay."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import uuid
from unittest.mock import patch

import benchmark_advisor_budget as budget
from core.advisor_timing import AdvisorTiming, LatencyEstimate, finite_nonnegative
from core.turn_lifecycle import TurnDispatcher
from engine_stdio import _extract_metrics

bench = budget.bench
ARMS = budget.ARMS
FIXTURE = bench.base.ROOT / 'examples/advisor_session_benchmark.json'
PHASES = ('cold', 'warm', 'warm_again', 'expired', 'recovered_warm', 'bounded_rpc')
PRIMARY = ContextVar('benchmark_primary_generation', default=False)


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_advisor_session_sequence'
            or fixture.get('idle_before_expired_seconds') != 61
            or tuple(c['id'] for c in fixture['cases']) != PHASES):
        raise ValueError('Require the complete prospective session sequence')
    for c in fixture['cases']:
        if (not c['question'].strip() or c['records'] or c['executive_mode'] != 'observe'
                or c['executive_observer_mode'] != 'both' or set(c['criteria']) != set(bench.base.CRITERIA)
                or c['rpc_seconds'] != (20 if c['id'] == 'bounded_rpc' else 120)):
            raise ValueError('Invalid sequence input or caller controls')


class SessionJournal(bench.Journal):
    async def post(self, role, *args, **kwargs):
        return await super().post('primary' if role == 'left' and PRIMARY.get() else role, *args, **kwargs)


class ObservedTiming:
    """Record fact reads without seeding live history; replay original inputs.

    Replayed response time is not live latency. Only replay substitutes recorded
    reads; live/controlled record() always advances the underlying real ledger.
    """
    def __init__(self, ledger):
        self.ledger = ledger
        self.reads, self.frozen, self.position = [], None, 0

    def begin(self, frozen=None):
        self.reads, self.frozen, self.position = [], frozen, 0

    def record(self, *args, **kwargs):
        self.ledger.record(*args, **kwargs)

    def estimate(self, module, profile):
        if self.frozen is None:
            estimate = self.ledger.estimate(module, profile)
        else:
            if self.position >= len(self.frozen):
                raise ValueError('Missing recorded timing read')
            row = self.frozen[self.position]
            if row['module'] != module or row['profile'] != profile:
                raise ValueError('Recorded timing identity differs')
            values = row['estimate']
            if (set(values) != {'ms', 'kind', 'completed_samples', 'censored_samples'}
                    or values['ms'] is not None and not finite_nonnegative(values['ms'])
                    or values['kind'] not in {'unknown', 'completed_max', 'censored_bound'}
                    or any(type(values[k]) is not int or not 0 <= values[k] <= 16
                           for k in ('completed_samples', 'censored_samples'))
                    or values['completed_samples'] + values['censored_samples'] > 16
                    or (values['ms'] is None) != (values['kind'] == 'unknown')
                    or (values['kind'] == 'unknown' and (values['completed_samples'] or values['censored_samples']))
                    or (values['kind'] == 'completed_max' and not values['completed_samples'])
                    or (values['kind'] == 'censored_bound' and not values['censored_samples'])):
                raise ValueError('Invalid recorded timing fact')
            estimate = LatencyEstimate(**values)
        self.position += 1
        self.reads.append({'module': module, 'profile': profile, 'estimate': asdict(estimate)})
        return estimate

    def finish(self):
        if self.frozen is not None and self.position != len(self.frozen):
            raise ValueError('Recorded timing read coverage differs')


class ControlledTransport:
    """Explicit offline timing intervention, never a live provider measurement."""
    def __init__(self):
        self.phase = 'cold'
        self.completed_roles = []

    async def post(self, _url, payload, _headers, **_kwargs):
        messages = payload['messages']
        system = messages[0]['content']
        metrics = any('Observer report (internal;' in str(m.get('content', '')) for m in messages)
        try:
            packet = json.loads(messages[-1]['content'])
        except (ValueError, TypeError):
            packet = {}
        if isinstance(packet, dict) and 'records' in packet:
            role, content = 'support', {'decisions': [{'id':r['id'], 'supported':True, 'quote':r['answer']}
                                                       for r in packet['records']]}
        elif 'Director' in system:
            role, content = 'director', {'memo': 'Controlled guidance.', 'control': {'consult': 'auto'}, 'confidence': .9}
        elif 'Executive Reasoning' in system:
            role = 'executive_metrics' if metrics else 'executive'
            content = {'memo': 'Controlled guidance.', 'directives': {}, 'mix_in': '', 'confidence': .9}
        elif 'critic' in system.lower():
            role, content = 'right', {'verdict': 'ok', 'issues': [], 'fixes': [], 'critic_sum': 'No issues.', 'confidence_r': .9}
        else:
            role, content = 'primary', 'Controlled answer.'
        slow = self.phase in {'cold', 'warm'}
        delay = {'director': 1.8 if slow else .005, 'executive_metrics': 2.7 if slow else .005,
                 'executive': 1.35, 'primary': .02 if self.phase == 'bounded_rpc' else 1.45,
                 'right': .005, 'support': .005}[role]
        await asyncio.sleep(delay)
        self.completed_roles.append(role)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': content if isinstance(content, str) else json.dumps(content)}}],
            'model': 'controlled', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}


async def rpc(engine, params, options, timeout):
    reply = asyncio.get_running_loop().create_future()

    async def handle(request):
        result = await bench.base._handle_process(engine, params, **options)
        return {'id': request['id'], 'ok': True, 'result': result}

    dispatcher = TurnDispatcher(handle, reply.set_result, timeout_seconds=timeout)
    try:
        dispatcher.submit({'id': params['qid'], 'method': 'process'})
        return await reply
    finally:
        await dispatcher.close()


def seed_policy(case_id):
    seed = int(bench.digest(case_id)[:8], 16)
    bench.base.UUID_RNG.seed(seed)
    random.seed(seed)
    bench.base.np.random.seed(seed)
    if bench.TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)


async def execute(engine, case, arm, index, inputs, journal, options, *, controlled=False):
    before, started, captured = len(journal.events), time.perf_counter(), []
    row = {'case': case['id'], 'arm': arm, 'origin': 'model', 'status': 'unavailable',
           'answer': None, 'session_turn': index + 1, 'rpc_seconds': case['rpc_seconds']}
    original = engine.left.generate_answer

    async def primary(*args, **kwargs):
        token = PRIMARY.set(True)
        try:
            answer = await original(*args, **kwargs)
            captured.append(answer)
            return answer
        finally:
            PRIMARY.reset(token)

    engine.left.generate_answer = primary
    timing = engine.hypothalamus.advisor_timing
    timing.begin(inputs['timing_reads'] if journal.frozen else None)
    params = {'qid': 'advisor-sequence:' + case['id'], 'question': case['question'],
              'leading_brain': 'left', 'system2_mode': 'off' if controlled else 'auto',
              'executive_mode': case['executive_mode'], 'executive_observer_mode': case['executive_observer_mode']}
    try:
        response = await rpc(engine, params, options, case['rpc_seconds'])
        if response['ok']:
            result = response['result']
            if not isinstance(result['answer'], str) or not result['answer'].strip():
                raise ValueError('Missing final answer')
            row.update(status='ok', answer=result['answer'])
        else:
            row['error_type'] = response['error']['type']
            row['may_have_committed'] = response['error'].get('may_have_committed', False)
        await engine.callosum.wait_idle()
    except Exception as exc:
        row['error_type'] = type(exc).__name__
    finally:
        engine.left.generate_answer = original
    timing.finish()
    events = journal.events[before:]
    row.update(metrics=_extract_metrics(engine.telemetry.events), usage=bench.usage(events),
        calls_by_role=dict(Counter(e['role'] for e in events)), elapsed_ms=(time.perf_counter() - started) * 1000,
        timing_reads=list(timing.reads), pending_advisors=sum(not t.done() for t in engine._benchmark_advisors),
        memory_sources=len(engine.memory.past_qas))
    row['draft'] = captured[0] if len(captured) == 1 and not any(e.get('error_type') for e in events if e['role'] == 'primary') else None
    if not journal.frozen:
        inputs['timing_reads'] = row['timing_reads']
    return row


async def run(args, fixture, config, journal, *, transport=None, idle=None):
    validate(fixture)
    bench.base.enforce_runtime()
    controlled = transport is not None
    report = {'schema_version': 1, 'experiment': 'advisor_session_sequence', 'complete': False,
        'evidence': 'controlled' if controlled else 'live', 'frozen_replay': journal.frozen is not None,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=bench.base.ROOT.parent, text=True).strip(),
        'metadata': {k:v for k,v in journal.data.items() if k not in {'responses', 'complete'}},
        'case_order': list(PHASES), 'rows': [], 'grades': [], 'idle_intervals': []}
    save = lambda: bench.checkpoint(args.output_dir / 'report.partial.json', report)
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    engines, clock, offset = {}, [time.time()], [0.]
    journal.data['cell_inputs'] = {}
    try:
        for arm in ARMS:
            with patch.dict(os.environ, budget.environment(fixture['cases'][0], arm), clear=True):
                engines[arm] = await bench.session(fixture['cases'][0], arm, config, journal, clock)
            engine = engines[arm]
            if controlled:
                engine.controller.policy = bench.base.SkipConsult()
                routing = engine.controller._run_routing_stage

                async def left_only(*, _routing=routing, **kwargs):
                    result = await _routing(**kwargs)
                    result.decision.action = 0
                    return result

                engine.controller._run_routing_stage = left_only
                engine.hypothalamus.advisor_timing = AdvisorTiming(clock=lambda: time.monotonic() + offset[0])
            engine.hypothalamus.advisor_timing = ObservedTiming(engine.hypothalamus.advisor_timing)
            engine._benchmark_advisors = []
            original = engine.controller._start_advisor

            def tracked(*a, _original=original, _engine=engine, **kw):
                task = _original(*a, **kw)
                _engine._benchmark_advisors.append(task)
                return task

            engine.controller._start_advisor = tracked
        for index, source_case in enumerate(fixture['cases']):
            case = dict(source_case)
            if case['id'] == 'expired':
                start = time.monotonic()
                if controlled:
                    offset[0] += fixture['idle_before_expired_seconds']
                elif journal.frozen is None:
                    print(json.dumps({'idle_seconds': fixture['idle_before_expired_seconds']}), flush=True)
                    await (idle or asyncio.sleep)(fixture['idle_before_expired_seconds'])
                report['idle_intervals'].append({'before': case['id'], 'requested_seconds': 61,
                    'observed_seconds': time.monotonic() - start, 'simulated': controlled, 'replay_skipped': journal.frozen is not None})
            if controlled:
                transport.phase = case['id']
                if case['id'] == 'recovered_warm':
                    case['rpc_seconds'] = 3.
                elif case['id'] == 'bounded_rpc':
                    case['rpc_seconds'] = .4
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
                key = case['id'] + ':' + arm
                inputs = (journal.frozen['cell_inputs'][key] if journal.frozen else
                          {'wall_time': time.time(), 'timing_reads': []})
                journal.data['cell_inputs'][key] = inputs
                clock[0] = inputs['wall_time']
                seed_policy(case['id'])
                completed_before = len(transport.completed_roles) if controlled else 0
                with patch.dict(os.environ, budget.environment(case, arm), clear=True), \
                     patch('time.time', lambda: clock[0]), \
                     patch('uuid.uuid4', lambda: uuid.UUID(int=bench.base.UUID_RNG.getrandbits(128))):
                    row = await execute(engines[arm], case, arm, index, inputs, journal, options, controlled=controlled)
                if controlled:
                    row['controlled_completed_roles'] = transport.completed_roles[completed_before:]
                report['rows'].append(row)
                save()
                print(json.dumps({'case': case['id'], 'arm': arm, 'status': row['status'], 'usage': row['usage']}), flush=True)
        before = len(journal.events)
        if not controlled:
            candidates = list(report['rows']) + [dict(r, arm=r['arm'] + '_draft', status='ok', answer=r['draft'])
                                               for r in report['rows'] if r['draft'] is not None]
            for case in fixture['cases']:
                report['grades'].extend(await bench.grade(case, candidates, replace(config, model=args.judge_model), journal))
                save()
        report['judge_usage'] = bench.usage(journal.events[before:])
        journal.finish()
        report.update(complete=True, transport_calls=journal.calls, live_provider_calls=0 if controlled else journal.calls,
            experiment_usage=bench.usage(journal.events), summaries=budget.summarize(report))
        save()
        return report
    finally:
        for engine in engines.values():
            await engine.close()


def public_summary(report):
    grades = {(g['case'],g['arm']):g['scores'] for g in report['grades'] if 'arm' in g}
    rows = []
    for row in report['rows']:
        modules = row['metrics']['modules']
        rows.append({k:row.get(k) for k in ('case','arm','status','session_turn','rpc_seconds','error_type',
            'may_have_committed','usage','calls_by_role','elapsed_ms','pending_advisors','memory_sources','timing_reads')} | {
            'answer_sha256':bench.digest(row['answer']), 'draft_sha256':bench.digest(row['draft']),
            'advisor_gates':{k:v for k,v in modules['gates'].items() if k in budget.ADVISORS},
            'advisor_timing':modules['advisor_timing'],
            'scores':{k:v['verdict'] for k,v in grades.get((row['case'],row['arm']),{}).items()},
            'draft_scores':{k:v['verdict'] for k,v in grades.get((row['case'],row['arm']+'_draft'),{}).items()}})
        if 'controlled_completed_roles' in row:
            rows[-1]['controlled_completed_roles'] = row['controlled_completed_roles']
    return {k:report[k] for k in ('schema_version','experiment','complete','evidence','frozen_replay','source_revision',
        'metadata','transport_calls','live_provider_calls','experiment_usage','judge_usage','summaries','idle_intervals')} | {'rows':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal', type=Path)
    mode.add_argument('--controlled', action='store_true')
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
        if frozen and frozen.get('controlled'):
            raise ValueError('Controlled timing traces require real timers; journal replay is for live traces')
        model = args.model or (frozen['model'] if frozen else 'controlled-advisor')
        config = (bench.LLMConfig('openai', model, 'OFFLINE_ONLY') if args.controlled else bench.load_llm_config(default_model=model))
        if config is None:
            if args.model:
                raise ValueError('Configured live provider required')
            config = bench.LLMConfig('openai', model, 'REPLAY_ONLY')
        config = replace(config, model=model, max_output_tokens=2048, timeout_seconds=60, auto_continue=False, max_continuations=0)
        if config.provider != 'openai':
            raise ValueError('OpenAI-compatible provider required')
        files = [Path(__file__), Path(budget.__file__), Path(bench.__file__), Path(bench.base.__file__),
                 bench.base.ROOT/'scripts/benchmark_conversation_memory.py', bench.base.ROOT/'scripts/engine_stdio.py',
                 *sorted((bench.base.ROOT/'core').glob('*.py'))]
        metadata = {'model':model, 'judge_model':args.judge_model, 'fixture_sha256':bench.digest(fixture),
            'code_sha256':bench.digest([(str(p.relative_to(bench.base.ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'config':bench.public_config(config), 'call_limit':args.max_calls, 'arms':list(ARMS),
            'history_policy':'same_EngineSession_per_arm;no_seeding;recorded_reads_only_in_replay',
            'controlled':args.controlled, 'torch_policy':bench.TORCH_AVAILABLE,
            'gate_policies':{k:v.policy_version for k,v in bench.DEFAULT_REGISTRY.items()}}
        journal = SessionJournal(config, metadata, frozen=frozen, max_calls=args.max_calls,
            save=lambda d:bench.checkpoint(args.output_dir/'journal.partial.json',d))
        transport = ControlledTransport() if args.controlled else None
        if transport:
            journal.transport._post_json = transport.post
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args, fixture, config, journal, transport=transport))
        bench.write_private(args.output_dir/'report.json',report)
        bench.write_private(args.output_dir/'journal.json',frozen or journal.data)
        bench.write_private(args.output_dir/'summary.json',public_summary(report))
        print(json.dumps({'complete':True,'transport_calls':report['transport_calls'],'live_provider_calls':report['live_provider_calls']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Session pilot incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
