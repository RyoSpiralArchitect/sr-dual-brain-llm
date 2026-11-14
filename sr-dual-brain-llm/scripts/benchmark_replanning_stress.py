#!/usr/bin/env python3
"""Frozen hard tasks, real native replanning ablation and duration-aware replay."""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_reasoning_frontier as base
import replanning_stress_tasks as tasks
from core.task_goals import extract_task_plan

FIXTURE=base.pilot.base.ROOT/'examples/replanning_stress.json'
PROTOCOL=base.PROTOCOL.with_name('replanning-stress-protocol.md')
ARMS=('self_review','dual_no_replan','dual_replan')
ORDERS=(ARMS,ARMS[1:]+ARMS[:1],ARMS[2:]+ARMS[:2],
    (ARMS[0],ARMS[2],ARMS[1]),(ARMS[2],ARMS[1],ARMS[0]),(ARMS[1],ARMS[0],ARMS[2]))
GLOBAL_LIMIT=128
PHASE_LIMITS={'main':96,'confirmation':32}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(fixture):
    ids={f'{phase}:{tier}:{domain}' for phase in PHASE_LIMITS
        for tier in ((0,1) if phase=='main' else (1,)) for domain in tasks.base.DOMAINS}
    if (fixture.get('schema_version')!=1 or fixture.get('origin')!=tasks.ORIGIN
            or len(fixture['cases'])!=12 or {c['id'] for c in fixture['cases']}!=ids):
        raise ValueError('Require all twelve frozen stress/confirmation tasks')
    for case in fixture['cases']:
        plan=extract_task_plan(case['question'])
        if (plan is None or len(plan.goals)!=3 or len(case['question'])>10000
                or case['expected']!=tasks.base.ORACLES[case['domain']](case['data'])
                or not tasks.base.grade(case,json.dumps(case['expected']))['exact']):
            raise ValueError('Task/reference/goal contract differs')
        if case['domain']=='sql' and case['expected']!=tasks.base.sql_reference(case['data']):
            raise ValueError('Independent SQL references disagree')


def metadata(fixture,config,phase,prior,plan=None):
    return base.pilot.metadata(fixture,config) | {
        'experiment':tasks.ORIGIN,'phase':phase,'call_limit':PHASE_LIMITS[phase],
        'global_call_limit':GLOBAL_LIMIT,'prior_admissions':prior['usage']['calls'] if prior else 0,
        'protocol_sha256':file_hash(PROTOCOL),'arms':list(ARMS if phase=='main' else (ARMS[0],ARMS[2])),
        'output_ceiling_per_trial':base.OUTPUT_ALLOWANCE,'self_caps':list(base.SELF_CAPS),
        'native_generation_cap':4096,'native_critic_cap':2048,'ordinary_left_allocation':8192,
        'semantic_grading':'exact_closed_world_no_paid_judge',
        'replay':'recorded_transport_duration_and_order_actual_owner_cancellation',
        'comparison':'post_110_native_system2_on_replanning_off_vs_auto_not_compute_matched',
        'prior_canonical_sha256':prior['canonical_sha256'] if prior else None,
        'confirmation_plan_sha256':base.pilot.bench.digest(plan) if plan else None,
        'grader_sha256':file_hash(Path(tasks.base.__file__)),
    }


class Journal(base.PhaseJournal):
    def __init__(self,*args,max_recorded_latency_ms=75000,**kwargs):
        if type(max_recorded_latency_ms) is not int or not 1 <= max_recorded_latency_ms <= 150000:
            raise ValueError('Invalid recorded duration bound')
        self.max_recorded_latency_ms=max_recorded_latency_ms
        self.starts={}
        super().__init__(*args,**kwargs)
        if self.frozen and any(type(r.get('latency_ms')) not in (float,int)
                or not math.isfinite(r['latency_ms']) or not 0<=r['latency_ms']<=self.max_recorded_latency_ms
                for r in self.frozen['occurrences']):
            raise ValueError('Recorded duration exceeds the frozen replay bound')

    async def post(self,role,url,payload,headers,**kwargs):
        if self.campaign.admissions>=GLOBAL_LIMIT:
            self.trial_denied=True
            self.data['denied_admissions'].append({'reason':'study_global_call_cap',
                'request_sha256':base.pilot.bench.digest({'case':self.case,'role':role,'payload':payload})})
            self.checkpoint()
            raise base.pilot.AdmissionError('Study admission closed')
        return await super().post(role,url,payload,headers,**kwargs)

    def _transport_admitted(self,index):
        self.starts[index]=asyncio.get_running_loop().time()
        super()._transport_admitted(index)

    async def _wait_recorded_transport(self,index):
        await super()._wait_recorded_transport(index)
        # A new local critic window can expire between transport returns. Pure
        # order-only replay would give retries/repair time they did not have.
        # Delay recorded responses/errors; cancellation still comes from its
        # real owner through the unchanged ScheduledJournal contract.
        recorded=self.frozen['occurrences'][index]
        delay=recorded['latency_ms']/1000.-(asyncio.get_running_loop().time()-self.starts[index])
        if delay>0:
            await asyncio.sleep(delay)


def score(row):
    return row['grade']['item_accuracy'] if row['answer_complete'] else 0.


def study_cap_denied(journal):
    return any(d['reason'] in {'call_cap','study_global_call_cap'} for d in journal.data['denied_admissions'])


def summarize(rows):
    arms={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        arms[arm]={'trials':len(group),'exact':sum(r['answer_complete'] and r['grade']['exact'] for r in group),
            'complete':sum(r['answer_complete'] for r in group),'workflow_valid':sum(r['workflow_valid'] for r in group),
            'schema_valid':sum(r['grade']['schema_valid'] for r in group),
            'mean_item_accuracy':sum(map(score,group))/len(group) if group else None}
    pairs=[]
    for case in sorted({r['case'] for r in rows}):
        group={r['arm']:r for r in rows if r['case']==case}
        for reference,candidate in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])):
            if reference in group and candidate in group:
                a,b=group[reference],group[candidate]
                delta=score(b)-score(a)
                pairs.append({'case':case,'reference':reference,'candidate':candidate,'delta':delta,
                    'outcome':'higher' if delta>0 else 'lower' if delta<0 else 'same',
                    'both_workflows_valid':a['workflow_valid'] and b['workflow_valid']})
    return {'authority':'descriptive_authored_tasks_no_statistical_or_isolated_causal_claim','arms':arms,'pairs':pairs}


def canonical(report):
    return {k:report[k] for k in ('phase','outcome','summary','usage','phase_receipt','campaign_admissions',
            'prior_report_sha256')} | {'rows':[{k:v for k,v in row.items() if k!='elapsed_ms'} for row in report['rows']]}


def public(report):
    result={k:v for k,v in report.items() if k!='rows'}
    fields=('case','domain','level','tier','arm','answer_complete','workflow_valid','grade','one_shot_grade',
        'initial_grade','initial_answer_complete','stages','error_type','elapsed_ms','usage','calls_by_role',
        'output_reserved','output_cap_denied','generation_receipts','policy_class','plan_state_cleared',
        'answer_generation','task_goals','gates')
    result['rows']=[]
    for row in report['rows']:
        safe={k:row[k] for k in fields if k in row}
        safe.update(answer_sha256=base.pilot.bench.digest(row['answer']),
            initial_answer_sha256=base.pilot.bench.digest(row.get('initial_answer')))
        if 'system2' in row:
            safe['system2']={k:row['system2'][k] for k in ('mode','enabled','used','critic_attempts',
                'critic_responses','review_completed','external_review_completed','rounds','resolved','resolution_basis')
                if k in row['system2']}
            safe['system2']['attempts']=[{k:a[k] for k in ('round','source','outcome','review_valid',
                'external_review_valid','error_type','critic_status','critic_kind') if k in a}
                for a in row['system2'].get('attempts',[])]
        result['rows'].append(safe)
    return result


def validate_prior(prior,fixture,config,plan):
    if (prior['phase']!='main' or not prior['complete'] or prior['outcome']!='comparisons_complete'
            or prior['metadata']['fixture_sha256']!=base.pilot.bench.digest(fixture)
            or prior['metadata']['config']!=base.pilot.bench.public_config(config)
            or prior['canonical_sha256']!=base.pilot.bench.digest(canonical(prior))
            or prior['metadata']['grader_sha256']!=file_hash(Path(tasks.base.__file__))
            or type(prior['usage']['calls']) is not int or not 0<prior['usage']['calls']<=96
            or prior['usage']['calls']!=prior['phase_receipt']['requests']
            or prior['usage']['calls']!=sum(r['usage']['calls'] for r in prior['rows'])
            or len(prior['rows'])!=24):
        raise ValueError('Require immutable complete main study with an unchanged profile/fixture')
    if (plan.get('prior_canonical_sha256')!=prior['canonical_sha256']
            or plan.get('prior_admissions')!=prior['usage']['calls']
            or plan.get('candidate_code_sha256')!=base.pilot.code_digest()):
        raise ValueError('Freeze the candidate/lineage confirmation plan before calls')


async def run(output_dir,fixture,config,phase,*,prior=None,prior_hash=None,plan=None,frozen=None,transport_factory=None):
    validate(fixture);base.pilot.base.enforce_runtime()
    if phase=='confirmation':
        validate_prior(prior,fixture,config,plan)
        if not prior_hash or plan['prior_report_sha256']!=prior_hash:
            raise ValueError('Original main report bytes changed')
    elif prior is not None:
        raise ValueError('Main comparison cannot reuse prior paid output')
    budget=base.CampaignBudget()
    budget.admissions=prior['usage']['calls'] if prior else 0
    report={'schema_version':1,'complete':False,'phase':phase,'outcome':'not_assessed',
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.pilot.base.ROOT.parent,text=True).strip(),
        'metadata':metadata(fixture,config,phase,prior,plan),'prior_report_sha256':prior_hash,'rows':[]}
    journal=Journal(config,report['metadata'],campaign=budget,frozen=frozen,max_calls=PHASE_LIMITS[phase],
        replay_wait_seconds=75.,save=lambda data:base.pilot.bench.checkpoint(output_dir/'journal.partial.json',data))
    if transport_factory is not None:journal.transport=transport_factory(journal)
    save=lambda:base.pilot.bench.checkpoint(output_dir/'report.partial.json',report)
    clock=[fixture['clock']]
    env=base.pilot.profile.environment() | {'DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS':'2048',
        'DUALBRAIN_ANSWER_REASONING_EFFORT':'provider_default','DUALBRAIN_ANSWER_TOKEN_BUDGET':'8192',
        'DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE':'auto','DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE':'off',
        'DUALBRAIN_MEMORY_OUTBOX_PATH':str(output_dir/'outbox.sqlite3')}
    with patch.dict(os.environ,env,clear=True),patch('time.time',lambda:clock[0]), \
            patch('core.unconscious_field.uuid',SimpleNamespace(uuid4=lambda:uuid.UUID(int=base.pilot.base.UUID_RNG.getrandbits(128)))):
        cases=[c for c in fixture['cases'] if c['partition']==phase]
        for index,case in enumerate(cases):
            order=ORDERS[index%len(ORDERS)] if phase=='main' else (ARMS[0],ARMS[2]) if index%2==0 else (ARMS[2],ARMS[0])
            for arm in order:
                clock[0]=fixture['clock']+fixture['cases'].index(case)
                with patch.dict(os.environ,{'DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE':'auto' if arm==ARMS[2] else 'off'}):
                    row=(await base.self_review(case,config,journal) if arm==ARMS[0] else
                         await base.dual(case,arm,config,journal,output_dir,clock))
                row['tier']=case['tier'];report['rows'].append(row);save()
                print(json.dumps({'phase':phase,'case':row['case'],'arm':arm,'grade':row['grade'],
                    'workflow_valid':row['workflow_valid'],'calls':row['usage']['calls']}),flush=True)
                if budget.blocked or study_cap_denied(journal):break
            if budget.blocked or study_cap_denied(journal):break
    journal.finish()
    report.update(complete=True,outcome='transport_blocked' if budget.blocked else
        'budget_incomplete' if study_cap_denied(journal) else 'comparisons_complete',
        live_provider_calls=journal.calls,usage=base.pilot.bench.usage(journal.events),
        campaign_admissions=budget.admissions,summary=summarize(report['rows']),
        phase_receipt={'requests':len(journal.events),'events':len(journal.data['transport_events']),
            'transport_events_sha256':journal.data['transport_events_sha256'],'denied':journal.data['denied_admissions']})
    report['canonical_sha256']=base.pilot.bench.digest(canonical(report));save()
    return report,frozen if frozen else journal.data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model');mode.add_argument('--journal',type=Path)
    parser.add_argument('--phase',choices=tuple(PHASE_LIMITS),required=True)
    parser.add_argument('--max-calls',type=int,choices=(GLOBAL_LIMIT,),required=True)
    parser.add_argument('--prior-report',type=Path);parser.add_argument('--confirmation-plan',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    try:
        fixture=base.pilot.bench.read_json(FIXTURE)
        frozen=base.pilot.bench.read_json(args.journal) if args.journal else None
        model=args.model or frozen['metadata']['config']['model']
        config=base.pilot.load_llm_config(default_model=model) if frozen is None else base.pilot.LLMConfig('openai',model,'REPLAY_ONLY')
        if not config or config.provider!='openai':raise ValueError('OpenAI-compatible configuration required')
        config=replace(config,model=model,max_output_tokens=4096,timeout_seconds=60,auto_continue=False,max_continuations=0)
        prior=base.pilot.bench.read_json(args.prior_report) if args.prior_report else None
        prior_hash=file_hash(args.prior_report) if prior else None
        plan=None
        if args.phase=='confirmation':
            if args.confirmation_plan is None:raise ValueError('Require a frozen confirmation plan')
            plan=base.pilot.bench.read_json(args.confirmation_plan)
            validate_prior(prior,fixture,config,plan)
            if plan['prior_report_sha256']!=prior_hash:raise ValueError('Original report bytes changed')
            relative=args.confirmation_plan.resolve().relative_to(base.pilot.base.ROOT.parent)
            subprocess.run(['git','ls-files','--error-unmatch',str(relative)],
                cwd=base.pilot.base.ROOT.parent,check=True,capture_output=True)
        elif prior is not None:raise ValueError('Main phase must not spend prior calibration output')
        if args.model and subprocess.check_output(['git','status','--porcelain'],cwd=base.pilot.base.ROOT.parent,text=True).strip():
            raise ValueError('Freeze source, fixture, protocol and graders before paid output')
        if frozen:
            with base.pilot.forbid_provider_network() as audit:
                report,journal=asyncio.run(run(args.output_dir,fixture,config,args.phase,prior=prior,prior_hash=prior_hash,plan=plan,frozen=frozen))
            report['unexpected_blocked_http_attempts']=audit.blocked_requests
        else:report,journal=asyncio.run(run(args.output_dir,fixture,config,args.phase,prior=prior,prior_hash=prior_hash,plan=plan))
        for name,data in (('journal.json',journal),('report.json',report),('summary.json',public(report))):
            base.pilot.bench.write_private(args.output_dir/name,data)
        print(json.dumps({'complete':True,'outcome':report['outcome'],'usage':report['usage'],
            'campaign_admissions':report['campaign_admissions'],'provider_calls':report['live_provider_calls']}))
    except BaseException as exc:
        base.pilot.bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Stress study incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__':main()
