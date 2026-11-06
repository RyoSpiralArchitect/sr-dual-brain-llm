#!/usr/bin/env python3
"""Shared natural clarification drafts, controlled consult pressure and exact raw replay."""
import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_contract_live as pilot
from core.memory_clarification import record_choice_clarifications
from core.prefrontal_cortex import PrefrontalCortex
from engine_stdio import _dispatch_request

FIXTURE = pilot.base.ROOT / 'examples/clarification_detail_confirmation.json'
CALL_LIMIT = 24
ARM_MODES = {'selective_off':('off','auto'), 'selective_auto':('auto','auto'),
             'forced_on':('auto','on'), 'forced_off':('auto','off')}


class ConsultPressure:
    def decide(self, _state): return 1


def metadata(fixture, config):
    return pilot.metadata(fixture,config) | {'experiment':fixture['origin'], 'call_limit':CALL_LIMIT,
        'arms':{arm:list(modes) for arm,modes in ARM_MODES.items()},
        'policy':'fixed_consult_pressure_for_controlled_exposure',
        'drafts':'one_natural_left_completion_per_case_shared_across_arms',
        'persistence':'in_memory_only', 'judging':'none'}


def canonical(report):
    return pilot.canonical(report) | {'shared_drafts':report['shared_drafts']}


async def session_for(case, arm, config, journal, clock):
    session,_ = await pilot.make_session('clarification-detail:'+case['id']+':'+arm,None,config,journal,clock)
    session.controller.policy = ConsultPressure()
    for record in case['records']: session.memory.store(pilot.MemoryTrace(**record))
    context = {'id':case['scope'],'aliases':case['aliases'],'remember':False}
    if case.get('change_qid'):
        before=len(journal.events)
        page=session.memory.record_choices(case['scope'])
        assert len(journal.events)==before
        context['record_ref']=next(c['record_ref'] for c in page['choices'] if 'backup' in c['label'])
        session.memory.past_qas=[replace(t,answer=case['changed_answer']) if t.qid==case['change_qid'] else t
                                for t in session.memory.past_qas]
    return session,context


async def run(output_dir, fixture, config, journal):
    if (fixture.get('schema_version')!=1 or tuple(c['id'] for c in fixture['cases'])
            !=('choose_english','reselect_japanese')):
        raise ValueError('Require the complete prospective fixture')
    pilot.base.enforce_runtime()
    report={'schema_version':1,'metadata':metadata(fixture,config),'complete':False,'shared_drafts':[], 'rows':[],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.base.ROOT.parent,text=True).strip()}
    save=lambda:pilot.bench.checkpoint(output_dir/'report.partial.json',report)
    clock=[fixture['clock']]; env=pilot.profile.environment()
    options=dict(blob_root=output_dir,blob_index={},vision_cache={},vision_cache_limit=0)
    with patch.dict(os.environ,env,clear=True),patch('time.time',lambda:clock[0]),\
            patch('core.unconscious_field.uuid',SimpleNamespace(uuid4=lambda:uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        for case in fixture['cases']:
            session,context=await session_for(case,'shared',config,journal,clock)
            journal.case=case['id']+':shared_draft'; before=len(journal.events)
            try:
                session.memory.begin_turn(case['question'],'shared',context)
                condition=session.memory.missing_evidence(case['question'])
                prompt=PrefrontalCortex.refresh_record_replay_context(session.memory,case['question'],'')
                draft=await session.left.generate_answer(case['question'],prompt)
                is_canonical=draft in record_choice_clarifications(condition)
                report['shared_drafts'].append({'id':case['id'],'draft_sha256':pilot.bench.digest(draft),
                    'canonical':is_canonical,'usage':pilot.bench.usage(journal.events[before:])})
                save()
            finally:
                await session.close()
            for arm in case['arms']:
                mode,system2=ARM_MODES[arm]
                os.environ['DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE']=mode
                session,context=await session_for(case,arm,config,journal,clock)
                journal.case=case['id']+':'+arm; before=len(journal.events)
                async def shared(*_args, **_kwargs): return draft
                session.left.generate_answer=shared
                try:
                    params={'session_id':session.session_id,'question':case['question'],
                        'qid':case['id']+':'+arm,'leading_brain':'left','system2_mode':system2,
                        'memory_context':context,'return_telemetry':True,'executive_mode':'off','executive_observer_mode':'off'}
                    reply=await _dispatch_request({'id':arm,'method':'process','params':params},
                        sessions={session.session_id:session},state_store=None,blob_pipe=None,send=lambda *_:None,**options)
                    if not reply.get('ok'): raise ValueError('Engine request unavailable')
                    result=reply['result']; metrics=result['metrics']; detail=metrics.get('detail')
                    row={'id':case['id']+':'+arm,'arm':arm,'question':case['question'],'answer':result['answer'],
                        'status':'ok','draft_sha256':pilot.bench.digest(draft),'canonical_draft':is_canonical,
                        'route':metrics['policy']['consultation_route'],'detail':detail,
                        'selection_state':session.memory.replay_resolution(case['question']).binding_status,
                        'system2':pilot.accounting(metrics,result['telemetry']), 'commit':{}}
                    row.update(pilot.source_snapshot(session.memory,case['question']));pilot.assess_row(case,row)
                    row['usage']=pilot.bench.usage(journal.events[before:])
                    row['calls_by_role']=dict(pilot.Counter(e['role'] for e in journal.events[before:]))
                    row['contracts'].update(selection_state=row['selection_state']==case['expected_state'],
                        auto_skip=(row['route'] or {}).get('reason')=='bounded_record_clarification' if arm=='selective_auto' else None,
                        protected_mode=(row['route'] is None and row['system2']['used'] is (system2=='on'))
                            if system2 in {'on','off'} else None,
                        zero_provider_after_skip=row['usage']['calls']==0 if arm=='selective_auto' and is_canonical else None,
                        draft_retained_without_detail=result['answer']==draft if detail and not detail['usable'] else None,
                        no_repeat_of_returned_detail=(len(detail['deliveries'])==1 if detail and
                            detail['deliveries'][0]['status']!='delivery_error' else None),
                        reference_private=all(context.get('record_ref','NO_REF_NOT_A_TOKEN') not in
                            json.dumps(o['request']['payload']) for o in journal.data['occurrences'][before:]))
                    report['rows'].append(row);save()
                    print(json.dumps({'id':row['id'],'route':row['route'],'detail':detail,'usage':row['usage'],
                                      'contracts':row['contracts']}),flush=True)
                finally:
                    await session.close()
        journal.finish()
        report.update(complete=True,live_provider_calls=journal.calls,usage=pilot.bench.usage(journal.events),
            calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
            served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}))
        report['contract_failures']=[{'case':r['id'],'contract':k} for r in report['rows'] for k,v in r['contracts'].items() if v is False]
        report['canonical_sha256']=pilot.bench.digest(canonical(report));save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--model');mode.add_argument('--journal',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--max-calls',type=int,choices=(CALL_LIMIT,),required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        fixture=pilot.bench.read_json(FIXTURE); frozen=pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git','status','--porcelain'],cwd=pilot.base.ROOT.parent,text=True).strip():
            raise ValueError('Freeze code and inputs before paid calls')
        model=args.model or frozen['metadata']['config']['model']
        config=pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig('openai',model,'REPLAY_ONLY')
        if not config or config.provider!='openai': raise ValueError('OpenAI-compatible configuration required')
        config=replace(config,model=model,max_output_tokens=2048,timeout_seconds=60,auto_continue=False,max_continuations=0)
        journal=pilot.RawJournal(config,metadata(fixture,config),frozen=frozen,max_calls=CALL_LIMIT,
            save=lambda data:pilot.bench.checkpoint(args.output_dir/'journal.partial.json',data))
        if frozen:
            with pilot.forbid_provider_network() as audit: report=asyncio.run(run(args.output_dir,fixture,config,journal))
            report['unexpected_blocked_http_attempts']=audit.blocked_requests
        else: report=asyncio.run(run(args.output_dir,fixture,config,journal))
        pilot.bench.write_private(args.output_dir/'report.json',report)
        pilot.bench.write_private(args.output_dir/'journal.json',frozen or journal.data)
        pilot.bench.write_private(args.output_dir/'summary.json',pilot.public_summary(report))
        print(json.dumps({'complete':True,'live_provider_calls':journal.calls,'usage':report['usage'],
                          'failures':report['contract_failures']}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Clarification/detail confirmation incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__': main()
