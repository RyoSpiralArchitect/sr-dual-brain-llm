"""Completion-aware plans use existing RPC stages without new calls or truth claims."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_contract_live as pilot
from core.answer_generation import AnswerGeneration, Generation, current_generation, record_completion
from core.completion_replanning import propose_replan, parse_work_product, work_product_cap
from core.task_goals import current_task, extract_task_plan
from core.thalamic_reticular_nucleus import (
    COMPLETION_REPLANNING as MODULE, GateConfig, GateFacts, GateMode, ThalamicReticularNucleus,
)
from engine_stdio import _dispatch_request
from offline_provider_guard import forbid_provider_network
from scheduled_http_journal import ScheduledJournal
from test_benchmark_contract_live import config, response
from test_dual_brain import AlwaysConsultPolicy, make_stage_controller

CASES = (
    ('A job is running with owner birch. Ownership transfers to coral. A finish from birch '
     'must be ignored because only the current owner may finish.\nRequirements:\n'
     '1. Apply the events in order and derive the final state and owner.\n'
     '2. Using step 1, return only a JSON object with state and owner.',
     '{"state":"running","owner":"coral"}'),
    ('For a SQL LEFT JOIN, the left table has one account and the right table has no rows. '
     'The select list is COUNT(e.id) AS n and MAX(e.id) AS latest.\nRequirements:\n'
     '1. Derive both aggregates using SQL NULL rules.\n'
     '2. Using step 1, return only a JSON object with n and latest.',
     '{"n":0,"latest":null}'),
    ('A primary record is valid from time 2 through time 5, with the upper endpoint exclusive. '
     'No other record exists. The query is at time 5.\nRequirements:\n'
     '1. Determine whether a current authoritative value is available.\n'
     '2. Using step 1, return only a JSON object with status and value; missing value is null.',
     '{"status":"unknown","value":null}'),
)


def generation(status='empty',finish='length',*,remaining=128):
    state = AnswerGeneration()
    state.total_cap,state.reserved = 128+remaining,128
    text = 'PRIVATE_PARTIAL_CANDIDATE' if status!='empty' else ''
    state.record('draft',Generation(text,status,finish),text,128)
    return state,text


@pytest.mark.parametrize('status,finish,remaining,expected', [
    ('empty','length',128,True),('partial','length',128,True),
    ('empty','max_tokens',128,True),('complete','stop',128,False),
    ('partial','unknown',128,False),('failed','unknown',128,False),
    ('empty','stop',128,False),('unverified','stop',128,False),
    ('empty','length',0,False),
])
def test_only_known_length_limited_candidates_with_real_remaining_allocation_can_replan(status,finish,remaining,expected):
    state,draft = generation(status,finish,remaining=remaining)
    plan = propose_replan(extract_task_plan(CASES[0][0]),state,draft)
    assert (plan is not None) is expected
    if plan:
        assert plan.remaining_output_tokens==128 and plan.goals[1].depends_on==('g1',)
        assert plan.matches(draft,state)
        assert not plan.matches(draft+'changed',state)
        assert 'PRIVATE_PARTIAL_CANDIDATE' not in json.dumps(plan.summary())
        assert not plan.summary()['verified_complete']


def test_replan_cannot_assume_goal_structure_or_reuse_a_changed_generation_snapshot():
    state,draft = generation()
    assert propose_replan(extract_task_plan('Unstructured request.'),state,draft) is None
    plan = propose_replan(extract_task_plan(CASES[0][0]),state,draft)
    state.record('integration',Generation('complete','complete','stop'),'complete',128)
    assert not plan.matches(draft,state)


@pytest.mark.parametrize('text,finish,valid', [
    ('{"goal_id":"g1","work":"A concrete derivation.","unresolved":[]}', 'stop', True),
    ('{"goal_id":"g2","work":"Wrong goal.","unresolved":[]}', 'stop', False),
    ('{"goal_id":"g1","work":"A","work":"B","unresolved":[]}', 'stop', False),
    ('{"goal_id":"g1","work":"A","unresolved":[]}', 'length', False),
    ('{"goal_id":"g1","work":"A","unresolved":[],"verified":true}', 'stop', False),
])
def test_internal_product_requires_bound_complete_strict_json(text,finish,valid):
    state,draft=generation()
    plan=propose_replan(extract_task_plan(CASES[0][0]),state,draft)
    result=Generation(text,'complete' if finish=='stop' else 'partial',finish)
    product,_=parse_work_product(result,plan)
    assert (product is not None) is valid
    if product:
        assert product.source_sha256==plan.snapshot_sha256
        assert 'unverified' in product.handoff()


@pytest.mark.asyncio
@pytest.mark.parametrize('product_text,expected', [
    ('{"goal_id":"g1","work":"PRIVATE_DERIVATION","unresolved":["Check the endpoint."]}',True),
    ('{"goal_id":"g1","work":"PRIVATE_DERIVATION","work":"duplicate","unresolved":[]}',False),
])
async def test_work_product_is_bounded_internal_and_handed_to_existing_integration(monkeypatch,product_text,expected):
    import core.answer_generation as ordinary
    import core.task_goals as goals
    from core.models import LeftBrainModel
    from core.task_goals import TaskState
    question=CASES[0][0]
    state,draft=generation(remaining=4096)
    state.total_cap,state.reserved=8192,4096
    state.records[0]['max_output_tokens']=4096
    plan=propose_replan(extract_task_plan(question),state,draft)
    assert work_product_cap(plan,state,draft,4096,60000)==1024
    assert work_product_cap(plan,state,draft,4096,10000)==0
    task=TaskState(extract_task_plan(question))
    task.accept_replan(plan)
    monkeypatch.setenv('DUALBRAIN_COMPLETION_WORK_PRODUCTS','1')
    model=LeftBrainModel(replace(config(),max_output_tokens=4096))
    calls=[]
    class Client:
        async def complete(self,prompt,**kwargs):
            calls.append((prompt,kwargs))
            value=product_text if len(calls)==1 else CASES[0][1]
            record_completion('openai',value,'stop')
            return value
    model._llm_client=Client()
    generation_token=ordinary._turn.set(state)
    task_token=goals._active.set(task)
    try:
        answer=await model.integrate_info_async(question=question,draft=draft,info='Complete the request.')
    finally:
        goals._active.reset(task_token)
        ordinary._turn.reset(generation_token)
    assert answer==CASES[0][1] and len(calls)==2
    assert [kwargs['max_output_tokens'] for _,kwargs in calls]==[1024,3072]
    assert ('PRIVATE_DERIVATION' in calls[1][0]) is expected
    assert 'PRIVATE_DERIVATION' not in calls[1][1]['system']
    assert state.summary(answer)['terminal_complete']
    assert [row['stage'] for row in state.records]==['draft','work_product','integration']
    receipt=task.summary(answer)['work_products'][0]
    assert receipt['handed_to_integration'] is expected and not receipt['verified_complete']
    assert 'PRIVATE_DERIVATION' not in json.dumps(task.summary(answer))


def test_gate_auto_uses_typed_completion_and_budget_facts_and_default_is_off():
    trn = ThalamicReticularNucleus()
    assert trn.configure(MODULE,{}).mode is GateMode.OFF
    facts = GateFacts(eligible=True,generation_incomplete=True,generation_length_limited=True,
        task_goal_count=2,remaining_output_tokens=128)
    auto = GateConfig(GateMode.AUTO,'explicit',True)
    assert trn.decide(MODULE,facts=facts,config=auto).enabled
    for changes in ({'generation_incomplete':False},{'generation_length_limited':False},
            {'remaining_output_tokens':0},{'remaining_output_tokens':True},{'task_goal_count':1}):
        assert not trn.decide(MODULE,facts=replace(facts,**changes),config=auto).enabled
    blocked = replace(facts,eligible=False,blocked_reason='primary_write_pending')
    assert not trn.decide(MODULE,facts=blocked,config=GateConfig(GateMode.ON,'explicit',True)).enabled


@pytest.mark.parametrize('veto',['off','invalid','primary'])
def test_previous_plan_history_cannot_bypass_a_new_admission_veto(monkeypatch,veto):
    import core.answer_generation as ordinary
    import core.task_goals as goals
    from core.task_goals import TaskState
    question=CASES[0][0]
    controller=make_stage_controller()
    state,draft=generation()
    task=TaskState(extract_task_plan(question),qid='veto',owner=controller.orchestrator)
    controller.orchestrator.task_plans['veto']=task
    generation_token=ordinary._turn.set(state)
    task_token=goals._active.set(task)
    monkeypatch.setenv('DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE','on')
    try:
        controller._integration_with_replanning(question,draft,'Existing bounded work.')
        assert 'Completion recovery work plan' in task.prompt('integration',draft=draft)
        if veto=='primary':
            controller.memory.conversation_context=True
            controller.memory.begin_turn(question,'primary',
                {'id':'scope','aliases':['Scope'],'remember':True})
        else:
            monkeypatch.setenv('DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE',veto)
        controller._integration_with_replanning(question,draft,'Existing bounded work.')
        assert len(task.replans)==1 and task.pending_replan is None
        assert 'Completion recovery work plan' not in task.prompt('integration',draft=draft)
    finally:
        goals._active.reset(task_token)
        ordinary._turn.reset(generation_token)
        controller.orchestrator.clear_task('veto')


class Provider:
    def __init__(self,journal,final,initial,revision):
        self.journal,self.final,self.initial,self.revision = journal,final,initial,revision
        self.inputs = []
        self.critic_owner_ms = []
        self.integration_started = asyncio.Event()

    async def _post_json(self,_url,payload,_headers,**kwargs):
        role = self.journal.events[-1]['role']
        self.inputs.append((role,payload))
        if role=='draft':
            return response(self.final if self.initial=='complete' else
                'PRIVATE_PARTIAL_CANDIDATE' if self.initial=='partial' else '',
                'stop' if self.initial=='complete' else 'length')
        if role=='critic':
            from core.callosum import remaining_consultation_ms
            self.critic_owner_ms.append(remaining_consultation_ms())
            if self.revision=='accept_truncated':
                return response(json.dumps({'verdict':'ok','issues':[],'fixes':[]}))
            reviewed = payload['messages'][-1]['content'].split('Draft answer (review for correctness):\n',1)[1].split('\n\nReturn JSON only.',1)[0]
            finished = reviewed==self.final
            # A first review deliberately exercises ordinary revision even for
            # the complete-draft retention test; these are scripted mechanisms.
            complete = finished and len([r for r,_ in self.inputs if r=='integration'])>0
            data = {'verdict':'ok' if complete else 'issues',
                'issues':[] if complete else ['The requested conditions need rechecking.'],
                'fixes':[] if complete else ['Reconstruct from the full original data.']}
            contract = next((json.loads(line) for line in payload['messages'][0]['content'].splitlines()
                if line.startswith('{"version": "explicit_task_goals_v1"')),None)
            if contract is not None:
                data['goal_findings'] = [{'goal_id':g['goal_id'],'status':'addressed' if complete else 'unresolved',
                    'quote':self.final if complete else '', 'reason':'Scripted stage observation.'}
                    for g in contract['goals']]
            return response(json.dumps(data))
        if role=='detail':
            return response('Use each stated condition when constructing the requested result.')
        if role=='integration':
            if payload['messages'][0]['content'].startswith('Prepare one internal, bounded work product'):
                if self.revision=='cancel_product':
                    self.integration_started.set()
                    await asyncio.Event().wait()
                return response(json.dumps({'goal_id':'g1',
                    'work':'PRIVATE_FINISHED_GOAL_WORK', 'unresolved':[]}))
            self.integration_started.set()
            if self.revision=='cancel': await asyncio.Event().wait()
            return response(self.final if self.revision=='complete' else 'PRIVATE_INCOMPLETE_REVISION',
                'stop' if self.revision=='complete' else 'length')
        raise AssertionError('Unexpected extra model stage: '+role)


async def run_turn(tmp_path,question,final,*,module='auto',initial='empty',revision='complete',budget=256,
                   primary=False,cancel=False,system2='on',candidate_aware=False,stage_cap=128,
                   work_products=False):
    cfg = replace(config(),max_output_tokens=stage_cap,auto_continue=False,max_continuations=0)
    journal = ScheduledJournal(cfg,{'test':'completion-replanning'},max_calls=12)
    provider = Provider(journal,final,initial,revision)
    journal.transport = provider
    env = pilot.profile.environment() | {
        'DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE':'auto',
        'DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE':module,
        'DUALBRAIN_ANSWER_TOKEN_BUDGET':str(budget),
        'DUALBRAIN_CRITIC_MICRO_FALLBACK':'0',
        'DUALBRAIN_MODULE_ACC_MONITOR_MODE':'off',
        'DUALBRAIN_MODULE_CEREBELLAR_CORRECTION_MODE':'off',
        'DUALBRAIN_MODULE_SYSTEM2_STALLED_FOLLOWUP_MODE':'off',
        'DUALBRAIN_CRITIC_COMPLETION_AWARE':'1' if candidate_aware else '0',
        'DUALBRAIN_COMPLETION_WORK_PRODUCTS':'1' if work_products else '0',
    }
    with patch.dict('os.environ',env,clear=True),forbid_provider_network() as audit:
        session,_ = await pilot.make_session('completion-replanning-rpc',None,cfg,journal,[1794009600.])
        session.controller.policy = AlwaysConsultPolicy()  # Positive stage exposure, not quality evidence.
        journal.case = 'replan'
        params = {'session_id':session.session_id,'question':question,'qid':'replan',
            'leading_brain':'left','system2_mode':system2,'executive_mode':'off',
            'executive_observer_mode':'off','return_telemetry':True}
        if primary: params['memory_context']={'id':'replan-scope','aliases':['Replan scope'],'remember':True}
        try:
            turn = asyncio.create_task(_dispatch_request({'id':'replan','method':'process','params':params},
                sessions={session.session_id:session},state_store=None,blob_pipe=None,
                blob_root=tmp_path,blob_index={},vision_cache={},vision_cache_limit=0,send=lambda *_:None))
            if cancel:
                await asyncio.wait_for(provider.integration_started.wait(),5.)
                turn.cancel()
                with pytest.raises(asyncio.CancelledError): await turn
                result = None
            else:
                result = await turn
            assert not session.controller.orchestrator.task_plans
            assert current_task() is current_generation() is None
        finally:
            await session.close()
    journal.finish()
    assert audit.blocked_requests==0
    return result,provider,journal


@pytest.mark.asyncio
@pytest.mark.parametrize('question,final',CASES)
@pytest.mark.parametrize('initial',['empty','partial'])
async def test_real_rpc_reconstructs_all_goals_in_existing_stages_without_extra_allocation(tmp_path,question,final,initial):
    result,provider,journal = await run_turn(tmp_path,question,final,initial=initial)
    assert result['ok'] and result['result']['answer']==final
    metrics = result['result']['metrics']
    assert metrics['answer_generation']['terminal_complete']
    assert metrics['answer_generation']['reserved_output_tokens']==256
    assert metrics['modules']['gates'][MODULE]['enabled']
    assert len(metrics['task_goals']['replanning'])==1 and not metrics['task_goals']['verified_complete']
    integrations = [p for role,p in provider.inputs if role=='integration']
    assert len(integrations)==1 and journal.calls<=4
    system = integrations[0]['messages'][0]['content']
    assert 'Completion recovery work plan' in system and 'reconstruct_by_goal_dependencies' in system
    assert '"depends_on": ["g1"]' in system
    assert all(g.text in system for g in extract_task_plan(question).goals)
    safe = json.dumps(metrics['task_goals'])
    assert not any(x in safe for x in ('PRIVATE_','goal text','Completion recovery work plan'))


@pytest.mark.asyncio
async def test_off_retains_existing_recovery_and_never_injects_a_new_plan(tmp_path):
    result,provider,_ = await run_turn(tmp_path,*CASES[0],module='off')
    assert result['ok'] and result['result']['answer']==CASES[0][1]
    assert not result['result']['metrics']['task_goals']['replanning']
    assert all('Completion recovery work plan' not in p['messages'][0]['content'] for _,p in provider.inputs)


@pytest.mark.asyncio
async def test_incomplete_candidate_is_finished_before_one_actual_critic_review(tmp_path):
    result,provider,journal=await run_turn(tmp_path,*CASES[0],candidate_aware=True)
    assert result['ok'] and result['result']['answer']==CASES[0][1]
    assert [role for role,_ in provider.inputs]==['draft','integration','critic']
    execution=result['result']['metrics']['system2']
    assert execution['critic_attempts']==1 and execution['attempted_rounds']==[2]
    assert execution['external_review_completed'] is True
    assert result['result']['metrics']['task_goals']['replanning']
    assert journal.calls==3


@pytest.mark.asyncio
@pytest.mark.parametrize('question,final',CASES)
async def test_real_rpc_carries_one_terminal_internal_product_across_domains(tmp_path,question,final):
    result,provider,journal=await run_turn(tmp_path,question,final,candidate_aware=True,
        stage_cap=1024,budget=2048,work_products=True)
    assert result['ok'] and result['result']['answer']==final
    assert [role for role,_ in provider.inputs]==['draft','integration','integration','critic']
    metrics=result['result']['metrics']
    assert metrics['task_goals']['work_products'][0]['status']=='available'
    assert metrics['task_goals']['work_products'][0]['handed_to_integration']
    assert [row['stage'] for row in metrics['answer_generation']['stages']]==[
        'draft','work_product','integration']
    assert [row['max_output_tokens'] for row in metrics['answer_generation']['stages']]==[
        1024,256,768]
    assert metrics['answer_generation']['reserved_output_tokens']==2048
    assert 'PRIVATE_FINISHED_GOAL_WORK' in provider.inputs[2][1]['messages'][-1]['content']
    assert 'PRIVATE_FINISHED_GOAL_WORK' not in provider.inputs[2][1]['messages'][0]['content']
    assert 'PRIVATE_FINISHED_GOAL_WORK' not in json.dumps(metrics)
    assert journal.calls==4


@pytest.mark.asyncio
async def test_unfinished_revision_is_never_sent_for_semantic_review(tmp_path):
    result,provider,_=await run_turn(tmp_path,*CASES[0],candidate_aware=True,revision='partial')
    assert result['ok']
    assert [role for role,_ in provider.inputs]==['draft','integration']
    assert not result['result']['metrics']['system2']['used']
    assert result['result']['metrics']['answer_generation']['terminal_complete'] is False


@pytest.mark.asyncio
async def test_complete_candidate_gets_a_review_owner_window_without_extra_calls(tmp_path):
    result,provider,_=await run_turn(tmp_path,*CASES[0],candidate_aware=True,initial='complete')
    assert result['ok'] and provider.critic_owner_ms
    assert provider.critic_owner_ms[0] > 20000
    assert [role for role,_ in provider.inputs].count('critic')==2


@pytest.mark.asyncio
async def test_no_turn_window_retains_complete_candidate_without_provider_review(tmp_path,monkeypatch):
    monkeypatch.setattr('core.dual_brain.remaining_turn_ms',lambda: 7000.)
    result,provider,_=await run_turn(tmp_path,*CASES[0],candidate_aware=True,initial='complete')
    assert result['ok'] and result['result']['answer']==CASES[0][1]
    assert [role for role,_ in provider.inputs]==['draft']
    system2=result['result']['metrics']['system2']
    assert not system2['used'] and system2['critic_status']=='turn_budget_unavailable'


@pytest.mark.asyncio
async def test_review_of_trimmed_complete_candidate_cannot_resolve_full_answer(tmp_path):
    final=('A complete but deliberately long candidate. ' + 'evidence boundary. '*170).strip()
    result,provider,_=await run_turn(tmp_path,CASES[0][0],final,candidate_aware=True,
        initial='complete',revision='accept_truncated')
    assert result['ok'] and result['result']['answer']==final
    assert [role for role,_ in provider.inputs]==['draft','critic']
    system2=result['result']['metrics']['system2']
    assert system2['external_review_completed'] is True
    assert not system2['resolved'] and system2['resolution_basis']=='review_snapshot_truncated'


@pytest.mark.asyncio
async def test_complete_draft_survives_a_failed_integration_even_when_replanning_forced_on(tmp_path):
    result,provider,_ = await run_turn(tmp_path,*CASES[0],module='on',initial='complete',revision='partial')
    assert result['ok'] and result['result']['answer']==CASES[0][1]
    metrics=result['result']['metrics']
    assert metrics['answer_generation']['terminal_complete']
    assert any(s['status']=='partial' for s in metrics['answer_generation']['stages'])
    assert not metrics['task_goals']['replanning']
    assert all('Completion recovery work plan' not in p['messages'][0]['content'] for _,p in provider.inputs)


@pytest.mark.asyncio
async def test_no_replan_or_extra_left_request_when_allocation_exhausted(tmp_path):
    result,provider,_ = await run_turn(tmp_path,*CASES[0],module='on',budget=128)
    assert result['ok']
    metrics=result['result']['metrics']
    assert not metrics['answer_generation']['terminal_complete']
    assert metrics['answer_generation']['reserved_output_tokens']==128
    assert not metrics['task_goals']['replanning']
    assert not any(role=='integration' for role,_ in provider.inputs)


@pytest.mark.asyncio
async def test_cancellation_clears_active_replan_and_preserves_owned_stage_cancellation(tmp_path):
    result,_,journal = await run_turn(tmp_path,*CASES[0],revision='cancel',cancel=True)
    assert result is None
    assert any(o.get('error_type')=='CancelledError' and o['request']['role']=='integration'
        for o in journal.data['occurrences'])


@pytest.mark.asyncio
async def test_cancelled_internal_product_cannot_be_handed_to_a_later_stage(tmp_path):
    result,provider,journal=await run_turn(tmp_path,*CASES[0],candidate_aware=True,
        stage_cap=1024,budget=2048,work_products=True,revision='cancel_product',cancel=True)
    assert result is None and [role for role,_ in provider.inputs]==['draft','integration']
    assert journal.data['occurrences'][-1].get('error_type')=='CancelledError'


@pytest.mark.asyncio
async def test_manual_system2_off_keeps_ordinary_detail_and_never_adds_a_critic(tmp_path):
    result,provider,_ = await run_turn(tmp_path,*CASES[0],system2='off')
    assert result['ok']
    assert not result['result']['metrics']['system2']['used']
    assert not any(role.startswith('critic') for role,_ in provider.inputs)
    assert len([role for role,_ in provider.inputs if role=='draft'])==1


@pytest.mark.asyncio
async def test_primary_source_write_cannot_acquire_a_recovery_plan_even_forced_on(tmp_path):
    result,provider,_ = await run_turn(tmp_path,*CASES[0],module='on',initial='complete',primary=True)
    assert result['ok']
    assert not any('Completion recovery work plan' in p['messages'][0]['content'] for _,p in provider.inputs)
    assert not any('Task goal assignments' in p['messages'][0]['content'] for _,p in provider.inputs)
