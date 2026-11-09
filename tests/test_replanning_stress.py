"""Prospective task/reference, ablation, budget and real-duration replay contracts."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_replanning_stress as b
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import config,response
from test_benchmark_reasoning_frontier import wrong


def cfg():
    return replace(config(),model='gpt-6-luna',max_output_tokens=4096,
        timeout_seconds=60,auto_continue=False,max_continuations=0)


def test_fixture_reproduces_with_fresh_bounded_complete_goals_and_independent_oracles():
    fixture=b.base.pilot.bench.read_json(b.FIXTURE)
    assert fixture==b.tasks.build_fixture()
    b.validate(fixture)
    assert len({c['question'] for c in fixture['cases']})==12
    old=b.base.pilot.bench.read_json(b.base.FIXTURE)
    assert not ({c['question'] for c in fixture['cases']} & {c['question'] for c in old['cases']})
    for case in fixture['cases']:
        assert b.extract_task_plan(case['question']).goals[2].depends_on==('g1','g2')
        if case['domain']=='logic':
            data=case['data'];models=[]
            for values in itertools.product((False,True),repeat=len(data['variables'])):
                if all(any(values[abs(x)-1]==(x>0) for x in clause) for clause in data['clauses']):
                    models.append(values)
            expected={'status':{name:'entailed_true' if all(v[i] for v in models) else
                'entailed_false' if all(not v[i] for v in models) else 'undetermined'
                for i,name in enumerate(data['variables'])}}
            assert models and expected==case['expected']
        elif case['domain']=='sql':assert case['expected']==b.tasks.base.sql_reference(case['data'])


def test_evidence_stresses_conflict_equal_support_and_permanent_expiry_boundaries():
    fixture=b.tasks.build_fixture()
    for case in [c for c in fixture['cases'] if c['domain']=='evidence']:
        lookup={(q['topic'],q['time']):case['expected']['queries'][q['id']] for q in case['data']['queries']}
        assert lookup['field_0',2]['source_ids']==['s0a']
        assert lookup['field_0',4]=={'status':'conflict','value':None,'source_ids':['s0b','s0c']}
        assert lookup['field_1',4]['status']=='supported'
        assert lookup['field_1',4]['source_ids']==['s1b','s1c']
        for group in range(3):
            assert lookup[f'field_{group}',6]=={'status':'unknown','value':None,'source_ids':[]}
            if (f'field_{group}',5) in lookup:
                assert lookup[f'field_{group}',5]['source_ids']==[f's{group}'+('e' if case['level']>=6 else 'c')]


class Provider:
    def __init__(self,journal,fixture,failure=False):self.journal,self.fixture,self.failure=journal,fixture,failure
    async def _post_json(self,_url,payload,_headers,**kwargs):
        if self.failure:raise RuntimeError('PRIVATE_TRANSPORT_FAILURE')
        role=self.journal.events[-1]['role']
        identity,arm=self.journal.case.rsplit(':',1)
        case=next(c for c in self.fixture['cases'] if c['id']==identity)
        assert 'PRIVATE_ORACLE_CANARY' not in json.dumps(payload)
        if role in {'self_critique','critic'}:
            empty=False
            if role=='critic':
                draft=payload['messages'][-1]['content'].split('Draft answer (review for correctness):\n',1)[1].split('\n\nReturn JSON only.',1)[0]
                empty=not draft
            data={'verdict':'issues' if empty else 'ok','issues':['Missing complete candidate.'] if empty else [],
                  'fixes':['Reconstruct from the original task.'] if empty else []}
            for line in payload['messages'][0]['content'].splitlines():
                if line.startswith('{"version": "explicit_task_goals_v1"'):
                    data['goal_findings']=[{'goal_id':g['goal_id'],'status':'unresolved' if empty else 'uncertain',
                        'quote':'','reason':'Scripted mechanism observation.'} for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        if identity=='main:0:events' and role=='draft':return response('','length')
        erroneous=(identity=='main:1:sql' and role=='self_audit'
            or identity=='main:1:events' and arm=='dual_no_replan'
            or identity=='main:0:logic' and arm=='dual_replan')
        return response(json.dumps(wrong(case) if erroneous else case['expected']))


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',[False,True])
async def test_all_frozen_arms_or_transport_stop_replay_without_network_then_confirmation_keeps_counter(tmp_path,failure):
    fixture=b.tasks.build_fixture();fixture['oracle_canary']='PRIVATE_ORACLE_CANARY'
    live,repeat=tmp_path/'live',tmp_path/'replay';live.mkdir();repeat.mkdir()
    with forbid_provider_network() as audit:
        result,journal=await b.run(live,fixture,cfg(),'main',transport_factory=lambda j:Provider(j,fixture,failure))
        replay,_=await b.run(repeat,fixture,cfg(),'main',frozen=json.loads(json.dumps(journal)))
    assert b.canonical(result)==b.canonical(replay)
    assert replay['live_provider_calls']==audit.blocked_requests==0
    assert result['usage']['calls']<=96
    safe=json.dumps(b.public(result))
    assert all(s not in safe for s in ('"question":','"answer":','"initial_answer":','"quote":',
        'PRIVATE_ORACLE_CANARY','PRIVATE_TRANSPORT_FAILURE','FAKE_NO_CREDENTIAL'))
    if failure:
        assert result['outcome']=='transport_blocked' and result['usage']['calls']==1
        assert result['usage']['unknown_usage_calls']==1 and result['usage']['input_tokens'] is None
        with pytest.raises(ValueError):b.validate_prior(result,fixture,cfg(),{})
        return
    assert result['outcome']=='comparisons_complete' and len(result['rows'])==24
    assert {p['outcome'] for p in result['summary']['pairs']}=={'higher','same','lower'}
    for arm,planned in [('dual_no_replan',False),('dual_replan',True)]:
        row=next(r for r in result['rows'] if r['case']=='main:0:events' and r['arm']==arm)
        assert bool(row['task_goals']['replanning']) is planned
        assert row['grade']['exact'] and row['answer_complete'] and not row['workflow_valid']
        assert row['system2']['mode']=='on' and row['plan_state_cleared']
    path=tmp_path/'main.json';b.base.pilot.bench.write_private(path,result)
    prior_hash=b.file_hash(path)
    plan={'prior_canonical_sha256':result['canonical_sha256'],'prior_report_sha256':prior_hash,
        'prior_admissions':result['usage']['calls'],'candidate_code_sha256':b.base.pilot.code_digest()}
    confirm,confirm_repeat=tmp_path/'confirm',tmp_path/'confirm-replay';confirm.mkdir();confirm_repeat.mkdir()
    with forbid_provider_network() as audit:
        confirmed,tape=await b.run(confirm,fixture,cfg(),'confirmation',prior=result,prior_hash=prior_hash,plan=plan,
            transport_factory=lambda j:Provider(j,fixture))
        reproduced,_=await b.run(confirm_repeat,fixture,cfg(),'confirmation',prior=result,prior_hash=prior_hash,
            plan=plan,frozen=json.loads(json.dumps(tape)))
    assert b.canonical(confirmed)==b.canonical(reproduced)
    assert reproduced['live_provider_calls']==audit.blocked_requests==0
    assert confirmed['usage']['calls']<=32 and len(confirmed['rows'])==8
    assert confirmed['campaign_admissions']==result['usage']['calls']+confirmed['usage']['calls']<=128
    assert {r['case'].split(':')[0] for r in confirmed['rows']}=={'confirmation'}
    changed=deepcopy(result);changed['usage']['calls']=0
    with pytest.raises(ValueError):b.validate_prior(changed,fixture,cfg(),plan)


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel',[False,True])
async def test_duration_replay_preserves_elapsed_wait_and_cancellation_requires_a_real_owner(cancel):
    budget=b.base.CampaignBudget()
    journal=b.Journal(cfg(),{'test':'duration'},campaign=budget,max_calls=4,replay_wait_seconds=1.)
    class Transport:
        async def _post_json(self,*args,**kwargs):
            if cancel:await asyncio.Event().wait()
            await asyncio.sleep(.025)
            return response('ok')
    journal.transport=Transport();journal.begin_trial('timed')
    async def invoke(j):
        if cancel:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(j.post('critic','https://fake.invalid',{'max_completion_tokens':16},{}),.03)
        else:await j.post('draft','https://fake.invalid',{'max_completion_tokens':16},{})
    with forbid_provider_network() as audit:
        await invoke(journal);journal.finish()
        frozen=json.loads(json.dumps(journal.data))
        replay=b.Journal(cfg(),{'test':'duration'},campaign=b.base.CampaignBudget(),max_calls=4,
            replay_wait_seconds=1.,frozen=frozen)
        replay.begin_trial('timed')
        started=time.perf_counter();await invoke(replay);elapsed=time.perf_counter()-started
        replay.finish()
    assert replay.calls==audit.blocked_requests==0
    assert replay.data['transport_events']==journal.data['transport_events']
    assert elapsed>=frozen['occurrences'][0]['latency_ms']/1000.*.9


@pytest.mark.asyncio
async def test_final_campaign_admission_is_nonrefundable_and_closes_at_128():
    budget=b.base.CampaignBudget();budget.admissions=127
    journal=b.Journal(cfg(),{'test':'global'},campaign=budget,max_calls=4)
    class Transport:
        async def _post_json(self,*args,**kwargs):return response('ok')
    journal.transport=Transport();journal.begin_trial('last')
    with forbid_provider_network() as audit:
        await journal.post('draft','https://fake.invalid',{'max_completion_tokens':4096},{})
        with pytest.raises(b.base.pilot.AdmissionError):
            await journal.post('critic','https://fake.invalid',{'max_completion_tokens':2048},{})
    assert budget.admissions==128 and journal.calls==1 and journal.reserved==4096
    assert b.study_cap_denied(journal) and audit.blocked_requests==0


@pytest.mark.asyncio
async def test_duration_replay_reproduces_a_consumed_critic_window_without_injecting_failure(monkeypatch):
    from core.models import RightBrainModel
    monkeypatch.setenv('DUALBRAIN_CRITIC_JSON_REPAIR_CALL','1')
    config=replace(cfg(),timeout_seconds=.2)
    live=b.Journal(config,{'test':'critic-window'},campaign=b.base.CampaignBudget(),max_calls=4)
    class Transport:
        def __init__(self):self.calls=0
        async def _post_json(self,*args,**kwargs):
            self.calls+=1
            await asyncio.sleep(.04 if self.calls==1 else .2)
            return response('Unstructured review.' if self.calls==1 else
                '{"verdict":"ok","issues":[],"fixes":[]}')
    live.transport=Transport()
    async def invoke(journal):
        journal.begin_trial('review')
        model=RightBrainModel(config)
        async def post(url,payload,headers,**kwargs):
            return await journal.post('critic',url,payload,headers,**kwargs)
        model._llm_client._post_json=post
        return await model.criticise_reasoning('review','Compare the supplied records.','A provisional answer.',
            allow_micro_fallback=False)
    with forbid_provider_network() as audit:
        result=await invoke(live);live.finish()
        replay=b.Journal(config,{'test':'critic-window'},campaign=b.base.CampaignBudget(),max_calls=4,
            frozen=json.loads(json.dumps(live.data)))
        repeated=await invoke(replay);replay.finish()
    assert result==repeated and result['critic_status']=='deadline_exhausted'
    assert result['critic_external_review_valid'] is False
    assert len(live.events)==len(replay.events)==2 and live.calls==2
    assert replay.calls==audit.blocked_requests==0
