#!/usr/bin/env python3
"""Prospective semantic calibration and natural persistent memory trajectory."""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import inspect
import os
from pathlib import Path
import secrets
import subprocess
import time
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import answer_assessment_v4 as assessment
import benchmark_contract_live as pilot
from engine_stdio import _dispatch_request
from core.prefrontal_cortex import PrefrontalCortex

CALIBRATION = pilot.base.ROOT/'examples/clarification_calibration_v4.json'
FIXTURE = pilot.base.ROOT/'examples/natural_memory_trajectory.json'
CALL_LIMIT = 64
RUNTIME_LIMIT = 49  # Six calibration packets and at most nine read judgments reserve 15.
SESSION_ID = 'natural-memory:trajectory'


def metadata(calibration, fixture, config, judge):
    return pilot.metadata(fixture,config) | {
        'experiment':fixture['origin'],'call_limit':CALL_LIMIT,'runtime_call_limit':RUNTIME_LIMIT,
        'calibration_sha256':pilot.bench.digest(calibration),'judge_config':pilot.bench.public_config(judge),
        'semantic_grading':assessment.VERSION,'judge_instructions_sha256':pilot.bench.digest(assessment.INSTRUCTIONS),
        'policy':'unmodified_production_policy','drafts':'native_generation_no_substitution',
        'persistence':'isolated_postgresql_and_private_outbox',
        'replay_feedback':'record_measured_latency_only_verify_all_other_feedback_inputs',
        'output_caps':{'generation_ceiling':2048,'support':4096,'semantic_judge':2048,
                       'ordinary_detail_small':450,'ordinary_detail_large':900},
        'repeat_rule':'one_fixed_order_no_resampling_or_rescue'}


def validate(calibration, fixture):
    if (calibration.get('schema_version')!=1 or len(calibration.get('cases',[]))!=6
            or fixture.get('schema_version')!=1 or len(fixture.get('turns',[]))!=12):
        raise ValueError('Require the complete fixed calibration and trajectory')
    for c in calibration['cases']:
        if len(c['variants'])!=2: raise ValueError('Two authored contrasting answers required')
        for v in c['variants'].values():
            if set(v['expected'])!=set(assessment.SEMANTIC_CRITERIA): raise ValueError('Complete expectations required')
    if sum(not t['memory_context']['remember'] for t in fixture['turns'])!=9:
        raise ValueError('Nine read judgments are reserved')


def canonical(report):
    return {'calibration':report['calibration'], 'calibration_summary':report['calibration_summary'],
        'rows':[{k:v for k,v in r.items() if k not in {'elapsed_ms','choice_elapsed_ms','restart_elapsed_ms'}}
                for r in report['rows']], 'grades':report['grades'], 'usage':report['usage'],
        'served_models':report['served_models'],'feedback_sha256':report['feedback_sha256']}


class FeedbackTape:
    """Live calls are unchanged; replay restores only measured latency inputs.

    No action, reward or success is replayed as an oracle. Non-timing inputs and
    ordered coverage must match the new execution, even if a core guard catches
    an exception while delivering feedback.
    """
    def __init__(self, journal):
        self.journal=journal; self.rows=[]; self.position=0; self.errors=[]
        self.frozen=(journal.frozen or {}).get('feedback_tape')
        if journal.frozen and (not isinstance(self.frozen,list) or
                journal.frozen.get('feedback_sha256')!=pilot.bench.digest(self.frozen)):
            raise ValueError('Missing or corrupt feedback observations')

    def attach(self, owner, name, method_name):
        original=getattr(owner,method_name); signature=inspect.signature(original)
        def observed(*args, **kwargs):
            inputs=dict(signature.bind(*args,**kwargs).arguments)
            row={'case':self.journal.case,'owner':name,'inputs':inputs}
            if self.frozen is not None:
                if self.position>=len(self.frozen):
                    self.errors.append('extra_feedback'); raise ValueError('Extra feedback')
                expected=self.frozen[self.position];self.position+=1
                actual_control={k:v for k,v in inputs.items() if k!='latency_ms'}
                expected_control={k:v for k,v in expected['inputs'].items() if k!='latency_ms'}
                if row['case']!=expected['case'] or name!=expected['owner'] or actual_control!=expected_control:
                    self.errors.append('control_differs'); raise ValueError('Feedback control differs')
                inputs['latency_ms']=expected['inputs']['latency_ms']
                row['inputs']=inputs
            self.rows.append(deepcopy(row))
            self.journal.data['feedback_tape']=self.rows
            self.journal.data['feedback_sha256']=pilot.bench.digest(self.rows)
            self.journal.checkpoint()
            return original(**inputs)
        setattr(owner,method_name,observed)

    def finish(self):
        if self.errors or self.frozen is not None and (
                self.position!=len(self.frozen) or self.rows!=self.frozen):
            raise ValueError('Feedback replay coverage differs')
        self.journal.data['feedback_tape']=self.rows
        self.journal.data['feedback_sha256']=pilot.bench.digest(self.rows)
        return self.journal.data['feedback_sha256']


def calibration_summary(rows):
    observations=[]
    for row in rows:
        for name,expected in row['expected'].items():
            for criterion,verdict in expected.items():
                actual=((row.get('scores') or {}).get(name) or {}).get(criterion,{}).get('verdict','unknown')
                observations.append({'case':row['id'],'variant':name,'criterion':criterion,
                                     'expected':verdict,'actual':actual,'matches':verdict==actual})
    return {'comparisons':len(observations),'matches':sum(o['matches'] for o in observations),
            'disagreements':[o for o in observations if not o['matches']],
            'by_expected_verdict':{v:{'comparisons':sum(o['expected']==v for o in observations),
                'matches':sum(o['expected']==v and o['matches'] for o in observations)}
                for v in ('pass','fail','uncertain')},
            'failed_packets':sum('error_type' in r for r in rows),
            'authority':'authored_expectations_not_independent_human_adjudication'}


async def grade(case, labels, config, journal, role):
    before=len(journal.events)
    result={}
    try: result['scores']=await assessment.assess(case,labels,config,journal,role=role)
    except Exception as exc: result['error_type']=type(exc).__name__
    result['usage']=pilot.bench.usage(journal.events[before:])
    return result


async def open_session(store, config, journal, clock, tape):
    # seed=True in the older helper forces a left-only policy. Never use it here.
    session,_=await pilot.make_session(SESSION_ID,store,config,journal,clock,seed=False)
    pilot.track_stage(session.left,'generate_source_answer','source_write')
    tape.attach(session.controller.hypothalamus,'hypothalamus','update_feedback')
    tape.attach(session.controller.basal_ganglia,'basal_ganglia','integrate_feedback')
    return session


async def choose(session, declaration, options, journal):
    before=len(journal.events); started=time.perf_counter()
    reply=await _dispatch_request({'id':'choices','method':'record_choices','params':{
        'session_id':SESSION_ID,'scope':declaration['scope']}},sessions={SESSION_ID:session},
        state_store=session.state_store,blob_pipe=None,send=lambda *_:None,**options)
    if not reply.get('ok') or len(journal.events)!=before:
        raise ValueError('Choice inventory must succeed without model calls')
    matches=[c for c in reply['result']['choices'] if declaration['label'] in c['label'] and c['selectable']]
    if len(matches)!=1: raise ValueError('The prospectively named choice is unavailable or ambiguous')
    return matches[0]['record_ref'],(time.perf_counter()-started)*1000.


def generation_observations(occurrences):
    result=[]
    for o in occurrences:
        if o['request']['role']=='support': continue
        raw=o.get('response');choices=raw.get('choices') if isinstance(raw,dict) else None
        choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
        message=choice.get('message');text=message.get('content') if isinstance(message,dict) else None
        finish=choice.get('finish_reason')
        result.append({'stage':o['request']['role'],
            'finish_reason':finish if isinstance(finish,str) and finish in {'stop','length','tool_calls','content_filter'} else 'unknown',
            'nonempty_text':isinstance(text,str) and bool(text.strip()),'transport_error':o.get('error_type')})
    return result


def annotate_exposure(row, occurrences):
    """Only actually supplied blocks become semantic context; keep raw state separate."""
    block=row.pop('memory_text','')
    if not isinstance(row['available_context'],dict): return
    inputs=[{'stage':o['request']['role'],'messages':deepcopy(o['request']['payload']['messages'])}
            for o in occurrences if o['request']['role'] in assessment.STAGES]
    texts=[m['content'] for stage in inputs for m in stage['messages'] if isinstance(m.get('content'),str)]
    row['source_exposure_confirmed']=bool(block) and any(block in text for text in texts)
    row['retrieval_source_qids']=row['source_qids']
    if row['available_context']['approved_records'] and not row['source_exposure_confirmed']:
        row['available_context']['approved_records']=None
        row['available_context']['source_checks_by_qid']={}
        row['source_qids']=None
    condition_text=json.dumps(row['condition'],ensure_ascii=False,sort_keys=True) if row['condition'] else None
    row['condition_provided']=bool(condition_text) and any(condition_text in text for text in texts)
    if row['condition_provided']:row['available_context']['missing_evidence']=row['condition']
    reference=row['available_context']['reference']
    reference_provided=reference is not None and any(
        PrefrontalCortex.reference_context(reference,approved_qa_supplied=approved) in text
        for text in texts for approved in (True,False))
    if not reference_provided:row['available_context']['reference']=None
    row['available_context']['generation_messages']=inputs


async def execute_turn(session, turn, context, store, options, journal):
    before=len(journal.events); captures=[]; prepare=session.memory.prepare_retrieval; preparation_observed=False
    target=next((t for t in session.memory.past_qas if t.qid==turn.get('target_qid')),None)
    generators={name:getattr(session.left,name) for name in ('generate_answer','generate_source_answer')}
    def snapshot(question):
        return pilot.source_snapshot(session.memory,question) | {
            'binding_status':session.memory.replay_resolution(question).binding_status,
            'memory_text':session.memory.retrieve_related(question)}
    async def observed(question, **kwargs):
        nonlocal preparation_observed
        preparation_observed=True
        result=await prepare(question,**kwargs)
        captures.append(snapshot(question))
        return result
    session.memory.prepare_retrieval=observed
    for name,original in generators.items():
        async def generated(question, *args, _original=original, **kwargs):
            captures.append(snapshot(question))
            return await _original(question,*args,**kwargs)
        setattr(session.left,name,generated)
    started=time.perf_counter()
    row={'id':turn['id'],'question':turn['question'],'write':context['remember'],
         'status':'unavailable','answer':None,'policy_class':type(session.controller.policy).__name__}
    try:
        params={'session_id':SESSION_ID,'question':turn['question'],'qid':turn['id'],
            'memory_operation_id':'natural-'+turn['id'],'memory_context':context,
            'system2_mode':'auto','executive_mode':'off','executive_observer_mode':'off','return_telemetry':True}
        reply=await _dispatch_request({'id':turn['id'],'method':'process','params':params},
            sessions={SESSION_ID:session},state_store=store,blob_pipe=None,send=lambda *_:None,**options)
        row['elapsed_ms']=(time.perf_counter()-started)*1000.
        row['preparation_observed']=preparation_observed
        row.update(captures[-1] if captures else {'available_context':None,'source_qids':[], 'condition':None,
                                                'intent':None,'binding_status':'unknown'})
        if reply.get('ok'):
            result=reply['result']; row.update(status='ok',answer=result['answer'],
                system2=pilot.accounting(result['metrics'],result['telemetry']),
                detail=result['metrics'].get('detail'), route=result['metrics']['policy']['consultation_route'],
                policy_action=result['metrics']['policy']['action'],
                gates=result['metrics']['modules']['gates'],
                commit={k:(result.get('memory_commit') or {}).get(k) for k in ('state','operation_id')})
        else:
            row['error_type']=(reply.get('error') or {}).get('type','EngineError')
        active=await store.load_conversation_sources(SESSION_ID)
        row['active_primary_qids']=sorted(t.qid for t in active)
        if context['remember']:
            written=next((t for t in active if t.qid==turn['id']),None)
            row['write_complete']=(row['status']=='ok' and written is not None and
                written.answer==row['answer'] and not written.recalled_answer and
                not session.memory._answerability_source(written)['truncated'])
        occurrences=journal.data['occurrences'][before:]
        row['generation_receipts']=generation_observations(occurrences)
        annotate_exposure(row,occurrences)
        row['reference_private']=not context.get('record_ref') or all(context['record_ref'] not in
            json.dumps(o['request']['payload']) for o in occurrences)
        row['usage']=pilot.bench.usage(journal.events[before:])
        row['calls_by_role']=dict(pilot.Counter(e['role'] for e in journal.events[before:]))
        row['contracts']={'answer_available':row['status']=='ok',
            'primary_identities':row['active_primary_qids']==sorted(turn['expected_primary']),
            'committed':(row.get('commit') or {}).get('state')=='committed',
            'execution_accounting':(row.get('system2') or {}).get('consistent') is True,
            'reference_private':row['reference_private']}
        if context['remember']:
            row['contracts'].update(write_complete=row['write_complete'],
                requested_content=turn['required_text'] in (row['answer'] or ''))
        else:
            packet_case={'question':turn['question'],'requested_operation':turn['requested_operation'],
                         'available_context':row['available_context']}
            if target: packet_case['replay_target']={k:getattr(target,k) for k in ('qid','retrieval_scope','answer')}
            row['diagnostics_v3']=pilot.assessment.public_diagnostics(pilot.assessment.diagnostics(packet_case,row['answer']))
            row['contracts'].update(binding_status=row['binding_status']==turn['expected_state'],
                source_selection=row['source_qids']==turn['expected_sources'],
                expected_action=(row['condition'] or {}).get('action') in
                    turn.get('expected_actions',[turn.get('expected_action')]),
                exact_quote=row['diagnostics_v3']['exact_match']['verdict']=='pass' if target else None)
        return row
    finally:
        session.memory.prepare_retrieval=prepare
        for name,original in generators.items():setattr(session.left,name,original)


async def run(output_dir, calibration, fixture, config, judge, journal, dsn):
    validate(calibration,fixture); pilot.base.enforce_runtime()
    report={'schema_version':1,'metadata':metadata(calibration,fixture,config,judge),'complete':False,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.base.ROOT.parent,text=True).strip(),
        'calibration':[],'rows':[],'grades':[]}
    save=lambda:pilot.bench.checkpoint(output_dir/'report.partial.json',report)
    for index,case in enumerate(calibration['cases']):
        names=sorted(case['variants']); names=names if index%2==0 else names[::-1]
        labels={label:{'answer':case['variants'][name]['answer']} for label,name in zip(('A','B'),names)}
        journal.case='calibration:'+case['id']
        result=await grade(case,labels,judge,journal,'calibration_v4')
        if 'scores' in result: result['scores']={name:result['scores'][label] for label,name in zip(('A','B'),names)}
        report['calibration'].append({'id':case['id'],'order':names,
            'expected':{name:case['variants'][name]['expected'] for name in names},**result});save()
    report['calibration_summary']=calibration_summary(report['calibration'])
    # Reserve every planned read judgment before admitting runtime work.
    journal.max_calls=len(journal.events)+RUNTIME_LIMIT
    clock=[fixture['clock']]; store=pilot.PostgresStateStore(dsn,table_prefix='natural_memory_'+secrets.token_hex(6))
    options=dict(blob_root=output_dir,blob_index={},vision_cache={},vision_cache_limit=0)
    env=pilot.profile.environment();env.update(DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir/'outbox.sqlite3'),
                                              DUALBRAIN_PG_PERSIST_TELEMETRY='0')
    session=None; refs={}; schema_ready=False; episode_started=time.perf_counter();tape=FeedbackTape(journal)
    with patch.dict(os.environ,env,clear=True),patch('time.time',lambda:clock[0]),\
            patch('core.unconscious_field.uuid',SimpleNamespace(uuid4=lambda:uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        try:
            await store.ensure_schema();schema_ready=True
            session=await open_session(store,config,journal,clock,tape)
            for index,turn in enumerate(fixture['turns']):
                clock[0]=fixture['clock']+index; journal.case=turn['id']
                active=await store.load_conversation_sources(SESSION_ID)
                if not set(turn.get('requires',[]))<=set(t.qid for t in active):
                    report['rows'].append({'id':turn['id'],'status':'not_executed','reason':'dependency_unavailable',
                        'write':turn['memory_context']['remember'],'usage':pilot.bench.usage([]),
                        'contracts':{'turn_executed':False}});save();continue
                context=deepcopy(turn['memory_context']); choice_ms=0.;restart_ms=0.
                try:
                    choice=turn.get('choice_before_restart') or turn.get('choice')
                    if choice:
                        ref,choice_ms=await choose(session,{'scope':context['id'],**choice},options,journal)
                        refs[choice['save_as']]=ref
                    if turn.get('reopen'):
                        started=time.perf_counter();await session.close()
                        session=await open_session(store,config,journal,clock,tape)
                        restart_ms=(time.perf_counter()-started)*1000.
                    if turn.get('use_ref'): context['record_ref']=refs[turn['use_ref']]
                    elif turn.get('choice'): context['record_ref']=refs[turn['choice']['save_as']]
                except (KeyError,ValueError) as exc:
                    report['rows'].append({'id':turn['id'],'status':'not_executed','reason':'choice_unavailable',
                        'write':context['remember'],'usage':pilot.bench.usage([]),
                        'contracts':{'turn_executed':False}});save();continue
                row=await execute_turn(session,turn,context,store,options,journal)
                row.update(choice_elapsed_ms=choice_ms,restart_elapsed_ms=restart_ms)
                report['rows'].append(row);save()
                print(json.dumps({'id':row['id'],'status':row['status'],'condition':row.get('condition'),
                                  'route':row.get('route'),'usage':row['usage'],'contracts':row.get('contracts')}),flush=True)
            report['episode_elapsed_ms']=(time.perf_counter()-episode_started)*1000.
        finally:
            if session: await session.close()
            if schema_ready:
                with store._connect() as conn:
                    for name in pilot.base.TABLES: conn.execute(f'DROP TABLE IF EXISTS {store._table(name)} CASCADE')
    journal.max_calls=CALL_LIMIT
    for row in report['rows']:
        if row['write'] or row['status']!='ok':
            report['grades'].append({'id':row['id'],'status':'not_assessed','reason':'source_write_or_unavailable'});continue
        if not row['reference_private']:
            report['grades'].append({'id':row['id'],'status':'not_assessed','reason':'reference_exposure'});continue
        turn=next(t for t in fixture['turns'] if t['id']==row['id'])
        journal.case='grade:'+row['id']
        result=await grade({'question':row['question'],'requested_operation':turn['requested_operation'],
            'available_context':row['available_context']},{'A':{'answer':row['answer']}},judge,journal,'answer_judge_v4')
        if 'scores' in result: result['scores']=result['scores']['A']
        report['grades'].append({'id':row['id'],**result});save()
    report['feedback_sha256']=tape.finish()
    journal.finish()
    report.update(complete=True,live_provider_calls=journal.calls,usage=pilot.bench.usage(journal.events),
        calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
        served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}))
    report['contract_failures']=[{'case':r['id'],'contract':k} for r in report['rows']
        for k,v in r.get('contracts',{}).items() if v is False]
    report['trajectory_complete']=all(r['status']=='ok' for r in report['rows'])
    report['semantic_interpretation']=('all_authored_calibration_expectations_match' if
        report['calibration_summary']['matches']==24 and not report['calibration_summary']['failed_packets']
        else 'calibration_disagreement_unvalidated_model_observations')
    report['canonical_sha256']=pilot.bench.digest(canonical(report));save();return report


def public_summary(report):
    clean={k:deepcopy(v) for k,v in report.items() if k not in {'rows','calibration','grades'}}
    clean['rows']=[{k:deepcopy(v) for k,v in row.items() if k not in
        {'answer','question','available_context','intent','memory_text'}} | {
            'answer_sha256':pilot.bench.digest(row['answer']) if isinstance(row.get('answer'),str) else None}
        for row in report['rows']]
    for group in ('calibration','grades'):
        clean[group]=[{k:v for k,v in row.items() if k!='scores'} | {'verdicts':(
            {name:{c:s['verdict'] for c,s in scores.items()} for name,scores in row['scores'].items()}
            if group=='calibration' else {c:s['verdict'] for c,s in row['scores'].items()})}
            if 'scores' in row else deepcopy(row) for row in report[group]]
    return clean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--model');mode.add_argument('--journal',type=Path)
    parser.add_argument('--judge-model',default='gpt-6-sol');parser.add_argument('--max-calls',type=int,choices=(CALL_LIMIT,),required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        cal=pilot.bench.read_json(CALIBRATION);fixture=pilot.bench.read_json(FIXTURE);validate(cal,fixture)
        frozen=pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git','status','--porcelain'],cwd=pilot.base.ROOT.parent,text=True).strip():
            raise ValueError('Freeze implementation and study inputs before paid calls')
        model=args.model or frozen['metadata']['config']['model']
        config=pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig('openai',model,'REPLAY_ONLY')
        if not config or config.provider!='openai': raise ValueError('OpenAI-compatible configuration required')
        config=replace(config,model=model,max_output_tokens=2048,timeout_seconds=60,auto_continue=False,max_continuations=0)
        judge=replace(config,model=args.judge_model)
        dsn=os.environ.get('DUALBRAIN_TEST_PG_DSN')
        if not dsn: raise ValueError('Isolated test PostgreSQL DSN required')
        journal=pilot.RawJournal(config,metadata(cal,fixture,config,judge),frozen=frozen,max_calls=CALL_LIMIT,
            save=lambda data:pilot.bench.checkpoint(args.output_dir/'journal.partial.json',data))
        if frozen:
            with pilot.forbid_provider_network() as audit: report=asyncio.run(run(args.output_dir,cal,fixture,config,judge,journal,dsn))
            report['unexpected_blocked_http_attempts']=audit.blocked_requests
        else: report=asyncio.run(run(args.output_dir,cal,fixture,config,judge,journal,dsn))
        pilot.bench.write_private(args.output_dir/'report.json',report)
        pilot.bench.write_private(args.output_dir/'journal.json',frozen or journal.data)
        pilot.bench.write_private(args.output_dir/'summary.json',public_summary(report))
        print(json.dumps({'complete':True,'live_provider_calls':journal.calls,'usage':report['usage'],
                          'calibration':report['calibration_summary'],'failures':report['contract_failures']}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Natural memory confirmation incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__': main()
