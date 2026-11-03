#!/usr/bin/env python3
"""Bounded recovery versus v2-equivalent history in persistent sessions."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from unittest.mock import patch

import benchmark_advisor_sessions as sessions
from core.advisor_timing import AdvisorTiming, LatencyEstimate, RecoveryStatus, finite_nonnegative

bench = sessions.bench
ARMS = ('ttl_only_auto', 'recovery_auto')
PHASES = ('initial', 'opportunity', 'followup', 'relapse', 'capped')
FIXTURE = bench.base.ROOT / 'examples/advisor_recovery_benchmark.json'


def validate(fixture):
    if (fixture.get('schema_version') != 1 or fixture.get('origin') != 'prospective_bounded_advisor_recovery'
            or fixture.get('idle_before_opportunity_seconds') != 16
            or tuple(c['id'] for c in fixture['cases']) != PHASES):
        raise ValueError('Require the complete independent recovery sequence')
    for case in fixture['cases']:
        if (not case['question'].strip() or case['records'] or case['rpc_seconds'] != 120
                or case['executive_mode'] != 'observe' or case['executive_observer_mode'] != 'both'
                or set(case['criteria']) != set(bench.base.CRITERIA)):
            raise ValueError('Invalid matched inputs or caller modes')


class ObservedRecovery:
    """Record real resource reads/claims; replay them without fake live learning.

    Role/method occurrence matching allows independent async completion order.
    No live history is seeded. Replay leases are opaque recorded capabilities;
    cached response time cannot legitimately retrain a latency estimate.
    """
    LATENCY_KINDS = frozenset({'unknown','completed_max','censored_bound'})
    FINISH_OUTCOMES = frozenset({'recovered','still_slow','cancelled','unusable','profile_unavailable'})

    def __init__(self, ledger):
        self.ledger, self.leases = ledger, {}

    def begin(self, frozen=None):
        self.reads, self.frozen, self.invalid = [], frozen, False
        self.expected, self.counts = defaultdict(list), Counter()
        if self.leases:
            raise ValueError('Previous cell has an unfinished probe')
        if frozen is not None:
            if not isinstance(frozen, list):
                raise ValueError('Invalid recorded resource inputs')
            for row in frozen:
                self.expected[(row['method'], row['module'], row['profile'])].append(row['result'])

    def _call(self, method, module, profile, live):
        key = method, module, profile
        try:
            if self.frozen is None:
                value = live()
            else:
                value = self.expected[key][self.counts[key]]
            self.counts[key] += 1
            self.reads.append({'method':method, 'module':module, 'profile':profile, 'result':value})
            return value
        except Exception:
            self.invalid = True
            raise ValueError('Recorded resource coverage differs') from None

    def record(self, *args, **kwargs):
        self.ledger.record(*args, **kwargs)

    def estimate(self, module, profile):
        value = self._call('estimate',module,profile,lambda:asdict(self.ledger.estimate(module,profile)))
        try:
            estimate = LatencyEstimate(**value)
            if (estimate.ms is not None and not finite_nonnegative(estimate.ms)
                    or estimate.kind not in self.LATENCY_KINDS
                    or any(type(n) is not int or not 0 <= n <= 16 for n in (estimate.completed_samples,estimate.censored_samples))):
                raise ValueError
            return estimate
        except Exception:
            self.invalid = True
            raise ValueError('Invalid recorded latency fact') from None

    def recovery_status(self, module, profile, budget):
        value = self._call('status',module,profile,lambda:asdict(self.ledger.recovery_status(module,profile,budget)))
        try:
            status = RecoveryStatus(**value)
            if (type(status.allowed) is not bool or status.reason not in {
                    'no_history','disabled','unknown_budget','budget_too_small','estimate_fits','in_flight','rate_limited','cooldown','ready'}
                    or status.retry_after_ms is not None and not finite_nonnegative(status.retry_after_ms)):
                raise ValueError
            return status
        except Exception:
            self.invalid = True
            raise ValueError('Invalid recorded recovery fact') from None

    def start_probe(self, module, profile, budget, *, kind='probe'):
        actual = []

        def start():
            token = self.ledger.start_probe(module,profile,budget,kind=kind)
            actual.append(token)
            return token is not None

        claimed = self._call('claim' if kind == 'probe' else 'confirm',module,profile,start)
        if type(claimed) is not bool:
            self.invalid = True
            raise ValueError('Invalid recorded probe claim')
        token = (actual[0] if self.frozen is None else object()) if claimed else None
        if token is not None:
            self.leases[token] = (module,profile,kind)
        return token

    def finish_probe(self, token, elapsed, **kwargs):
        if token not in self.leases:
            self.invalid = True
            raise ValueError('Invalid probe identity')
        module,profile,kind = self.leases.pop(token)
        outcome = self._call('finish' if kind == 'probe' else 'confirm_finish',module,profile,lambda:self.ledger.finish_probe(token,elapsed,**kwargs))
        if outcome not in self.FINISH_OUTCOMES:
            self.invalid = True
            raise ValueError('Invalid recorded probe outcome')
        return outcome

    def finish(self):
        if (self.invalid or self.leases or self.frozen is not None
                and dict(self.counts) != {k:len(v) for k,v in self.expected.items()}):
            raise ValueError('Resource read/claim coverage differs')


class ControlledTransport:
    def __init__(self, regime):
        self.regime, self.phase = regime, 'initial'

    async def post(self, _url, payload, _headers, **_kwargs):
        messages, system = payload['messages'], payload['messages'][0]['content']
        try:
            packet = json.loads(messages[-1]['content'])
        except (TypeError,ValueError):
            packet = {}
        slow = self.regime == 'slow' or self.phase in {'initial','relapse','capped'}
        if isinstance(packet,dict) and 'records' in packet:
            delay, content = .002, {'decisions':[{'id':r['id'],'supported':True,'quote':r['answer']} for r in packet['records']]}
        elif 'Director' in system:
            delay,content = (1.8 if slow else .005), {'memo':'Controlled guidance.','control':{'consult':'auto'},'confidence':.9}
        elif 'Executive Reasoning' in system:
            metrics = any('Observer report (internal;' in str(m.get('content','')) for m in messages)
            delay = (2.7 if slow else .005) if metrics else .01
            content = {'memo':'Controlled guidance.','directives':{},'mix_in':'','confidence':.9}
        else:
            delay,content = .03,'Controlled answer.'
        await asyncio.sleep(delay)
        return {'choices':[{'finish_reason':'stop','message':{'content':content if isinstance(content,str) else json.dumps(content)}}],
                'model':'controlled','usage':{'prompt_tokens':10,'completion_tokens':5}}


def summarize(report, *, arms=ARMS, phases=PHASES):
    grades = {(g['case'],g['arm']):g['scores'] for g in report['grades'] if 'arm' in g}
    summary = {}
    for arm in arms:
        rows = [r for r in report['rows'] if r['arm']==arm]
        role_counts = Counter()
        for row in rows:
            role_counts.update(row['calls_by_role'])
        scores = [grades[(r['case'],arm)] for r in rows if (r['case'],arm) in grades]
        summary[arm] = {'planned':len(rows),'available':sum(r['status']=='ok' for r in rows),'graded':len(scores),
            'all_pass':sum(all(v['verdict']=='pass' for v in s.values()) for s in scores),
            'criterion_pass':{c:sum(s[c]['verdict']=='pass' for s in scores) for c in bench.base.CRITERIA},
            'usage':{k:bench.sum_usage([r['usage'] for r in rows],k) for k in bench.usage([])},
            'calls_by_role':dict(role_counts),'advisors':{}}
        for module in sessions.budget.ADVISORS:
            events = [r['metrics']['modules']['advisor_timing'].get(module,{}) for r in rows]
            summary[arm]['advisors'][module] = {'started':sum(bool(e.get('started')) for e in events),
                'usable_output':sum(bool(e.get('usable_output')) for e in events),
                'outcomes':dict(Counter(e.get('outcome','unreached') for e in events)),
                'probe_starts':sum(bool(e.get('recovery_probe_started')) for e in events),
                'confirmation_starts':sum(bool(e.get('recovery_confirmation_started')) for e in events),
                'probe_outcomes':dict(Counter(e.get('recovery_outcome','unreached') for e in events))}
    paired = [c for c in phases if all((c,arm) in grades for arm in arms)]
    return {'by_arm':summary,'jointly_graded_pairs':len(paired),
        'criterion_regressions':{criterion:[c for c in paired if grades[(c,arms[0])][criterion]['verdict']=='pass'
            and grades[(c,arms[1])][criterion]['verdict']!='pass'] for criterion in bench.base.CRITERIA},
        'grader_failures':sum('error_type' in g for g in report['grades'])}


def _v3_timing(arm, clock):
    return AdvisorTiming(clock=clock,recovery_enabled=arm=='recovery_auto',confirmation_enabled=False)


@dataclass(frozen=True)
class SessionExperiment:
    """Explicit benchmark configuration; production gates are not reconfigured."""
    name: str = 'bounded_advisor_recovery'
    arms: tuple = ARMS
    phases: tuple = PHASES
    validate_fixture: object = validate
    make_timing: object = _v3_timing
    make_observer: object = ObservedRecovery
    prepare_engine: object = None
    after_cell: object = None
    grade_advice: object = None
    compact_delivery: bool = False


async def run(args, fixture, config, journal, *, transport=None, idle=None, ledger_clock=None, experiment=None):
    experiment = experiment or SessionExperiment()
    experiment.validate_fixture(fixture)
    bench.base.enforce_runtime()
    report = {'schema_version':1,'experiment':experiment.name,'complete':False,
        'evidence':'controlled_'+transport.regime if transport else 'live','frozen_replay':journal.frozen is not None,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=bench.base.ROOT.parent,text=True).strip(),
        'metadata':{k:v for k,v in journal.data.items() if k not in {'responses','complete'}},
        'rows':[],'grades':[],'advice_grades':[],'idle_intervals':[]}
    save = lambda:bench.checkpoint(args.output_dir/'report.partial.json',report)
    engines,clock,offset = {},[time.time()],[0.]
    history_clock = ledger_clock or time.monotonic
    options = dict(blob_root=args.output_dir,blob_index={},vision_cache={},vision_cache_limit=0)
    journal.data['cell_inputs'] = {}
    try:
        for arm in experiment.arms:
            with patch.dict(os.environ,bench.environment(fixture['cases'][0],'all_auto'),clear=True):
                engine = await bench.session(fixture['cases'][0],arm,config,journal,clock)
            engines[arm] = engine
            # Old studies retain their original full request contracts even
            # when the production AUTO implementation evolves.
            if not experiment.compact_delivery:
                engine.controller._advisor_model = lambda _module, model:model
            engine.hypothalamus.advisor_timing = experiment.make_observer(experiment.make_timing(arm,lambda:history_clock()+offset[0]))
            engine._benchmark_advisors = []
            start = engine.controller._start_advisor

            def tracked(*a,_start=start,_engine=engine,**kw):
                task = _start(*a,**kw)
                _engine._benchmark_advisors.append(task)
                return task

            engine.controller._start_advisor = tracked
            if transport:
                engine.controller.policy = bench.base.SkipConsult()
                routing = engine.controller._run_routing_stage

                async def left_only(*,_routing=routing,**kw):
                    result = await _routing(**kw)
                    result.decision.action = 0
                    return result

                engine.controller._run_routing_stage = left_only
            if experiment.prepare_engine:
                experiment.prepare_engine(engine,arm)
        for index,case in enumerate(fixture['cases']):
            if case['id']=='opportunity':
                before = time.monotonic()
                if transport:
                    offset[0] += 16.
                elif journal.frozen is None:
                    print(json.dumps({'idle_seconds':16}),flush=True)
                    await (idle or asyncio.sleep)(16.)
                report['idle_intervals'].append({'before':'opportunity','requested_seconds':16,
                    'observed_seconds':time.monotonic()-before,'simulated':bool(transport),'replay_skipped':journal.frozen is not None})
            if transport:
                transport.phase = case['id']
            for arm in (experiment.arms if index%2==0 else experiment.arms[::-1]):
                if transport and hasattr(transport,'prepare_cell'):
                    transport.prepare_cell(arm,case,index)
                key = case['id']+':'+arm
                inputs = journal.frozen['cell_inputs'][key] if journal.frozen else {'wall_time':time.time(),'timing_reads':[]}
                journal.data['cell_inputs'][key] = inputs
                clock[0] = inputs['wall_time']
                sessions.seed_policy(case['id'])
                with patch.dict(os.environ,bench.environment(case,'all_auto'),clear=True), \
                     patch('time.time',lambda:clock[0]), \
                     patch('uuid.uuid4',lambda:uuid.UUID(int=bench.base.UUID_RNG.getrandbits(128))):
                    row = await sessions.execute(engines[arm],case,arm,index,inputs,journal,options,controlled=bool(transport))
                if experiment.after_cell:
                    experiment.after_cell(row,engines[arm])
                if transport and hasattr(transport,'decorate_row'):
                    transport.decorate_row(row,arm,case,index)
                report['rows'].append(row)
                save()
                print(json.dumps({'case':case['id'],'arm':arm,'status':row['status'],'usage':row['usage']}),flush=True)
        before = len(journal.events)
        if not transport:
            candidates = list(report['rows'])+[dict(r,arm=r['arm']+'_draft',status='ok',answer=r['draft']) for r in report['rows'] if r['draft'] is not None]
            for case in fixture['cases']:
                report['grades'].extend(await bench.grade(case,candidates,replace(config,model=args.judge_model),journal))
                save()
        report['answer_judge_usage'] = bench.usage(journal.events[before:])
        advice_before = len(journal.events)
        if not transport and experiment.grade_advice:
            await experiment.grade_advice(report,fixture['cases'],replace(config,model=args.judge_model),journal)
            save()
        report['advice_judge_usage'] = bench.usage(journal.events[advice_before:])
        journal.finish()
        report.update(complete=True,transport_calls=journal.calls,live_provider_calls=0 if transport else journal.calls,
            experiment_usage=bench.usage(journal.events),judge_usage=bench.usage(journal.events[before:]),
            summaries=summarize(report,arms=experiment.arms,phases=experiment.phases))
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
        rows.append({k:row.get(k) for k in ('case','arm','status','session_turn','error_type','usage','calls_by_role','pending_advisors','memory_sources')} | {
            'answer_sha256':bench.digest(row['answer']),'draft_sha256':bench.digest(row['draft']),
            'advisor_gates':{k:{f:v.get(f) for f in ('mode','enabled','reason','policy_version')} for k,v in modules['gates'].items() if k in sessions.budget.ADVISORS},
            'advisor_timing':modules['advisor_timing'],
            'scores':{k:v['verdict'] for k,v in grades.get((row['case'],row['arm']),{}).items()},
            'draft_scores':{k:v['verdict'] for k,v in grades.get((row['case'],row['arm']+'_draft'),{}).items()}})
    return {k:v for k,v in report.items() if k not in {'rows','grades','advice_grades'}} | {'rows':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model')
    mode.add_argument('--journal',type=Path)
    mode.add_argument('--controlled',choices=('recovery','slow'))
    parser.add_argument('--judge-model',default='gpt-6-sol')
    parser.add_argument('--max-calls',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        if not 1 <= args.max_calls <= 96:
            raise ValueError('Prospective maximum is 96 attempts')
        fixture = bench.read_json(FIXTURE)
        validate(fixture)
        frozen = bench.read_json(args.journal) if args.journal else None
        if frozen and frozen.get('controlled'):
            raise ValueError('Controlled scenarios require real timers; replay is for live journals')
        model = args.model or (frozen['model'] if frozen else 'controlled-recovery')
        config = bench.LLMConfig('openai',model,'OFFLINE_ONLY') if args.controlled else bench.load_llm_config(default_model=model)
        if config is None:
            if args.model:
                raise ValueError('Configured live provider required')
            config = bench.LLMConfig('openai',model,'REPLAY_ONLY')
        config = replace(config,model=model,max_output_tokens=2048,timeout_seconds=60,auto_continue=False,max_continuations=0)
        if config.provider != 'openai':
            raise ValueError('OpenAI-compatible configuration required')
        files = [Path(__file__),Path(sessions.__file__),Path(sessions.budget.__file__),Path(bench.__file__),Path(bench.base.__file__),
                 bench.base.ROOT/'scripts/benchmark_conversation_memory.py',bench.base.ROOT/'scripts/engine_stdio.py',*sorted((bench.base.ROOT/'core').glob('*.py'))]
        metadata = {'model':model,'judge_model':args.judge_model,'config':bench.public_config(config),
            'fixture_sha256':bench.digest(fixture),'code_sha256':bench.digest([(str(p.relative_to(bench.base.ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'call_limit':args.max_calls,'controlled':args.controlled,'arms':list(ARMS),'torch_policy':bench.TORCH_AVAILABLE,
            'history_policy':'same_sessions;no_seeding;v2_equivalent_baseline;recorded_resource_reads_and_claims_only_in_replay',
            'gate_policies':{k:v.policy_version for k,v in bench.DEFAULT_REGISTRY.items()}}
        journal = sessions.SessionJournal(config,metadata,frozen=frozen,max_calls=args.max_calls,
            save=lambda d:bench.checkpoint(args.output_dir/'journal.partial.json',d))
        transport = ControlledTransport(args.controlled) if args.controlled else None
        if transport:
            journal.transport._post_json = transport.post
        if frozen:
            journal.transport = None
        report = asyncio.run(run(args,fixture,config,journal,transport=transport))
        bench.write_private(args.output_dir/'report.json',report)
        bench.write_private(args.output_dir/'journal.json',frozen or journal.data)
        bench.write_private(args.output_dir/'summary.json',public_summary(report))
        print(json.dumps({'complete':True,'live_provider_calls':report['live_provider_calls'],'transport_calls':report['transport_calls']}))
    except BaseException as exc:
        bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Recovery study incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
