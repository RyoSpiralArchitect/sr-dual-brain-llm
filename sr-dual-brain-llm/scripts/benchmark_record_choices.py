#!/usr/bin/env python3
"""Prospective API choices -> selected quotation, including real PostgreSQL restart."""
import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import secrets
import subprocess
from types import SimpleNamespace
import uuid
from unittest.mock import patch

import benchmark_contract_live as pilot
from engine_stdio import _dispatch_request

FIXTURE = pilot.base.ROOT / 'examples/record_choices_confirmation.json'
CALL_LIMIT = 32
IDS = ('selected_english','selected_japanese','changed_selection','durable_selected','restart_rejected','durable_relisted')


def metadata(fixture, config):
    return pilot.metadata(fixture, config) | {'call_limit':CALL_LIMIT,'experiment':'read_only_record_choices',
        'case_ids':list(IDS),'choice_cost':'no_model_calls','token_provenance':'fresh_per_memory_instance_not_canonical'}


async def request_choices(session, scope, options, journal):
    before = len(journal.events)
    reply = await _dispatch_request({'id':'choices','method':'record_choices','params':{
        'session_id':session.session_id,'scope':scope}}, sessions={session.session_id:session},
        state_store=session.state_store, blob_pipe=None, send=lambda *_:None, **options)
    if not reply.get('ok') or len(journal.events)!=before:
        raise ValueError('Choices must succeed without provider calls')
    return reply['result']


async def turn(session, case, ref, journal, options):
    journal.case=case['id']; before=len(journal.events); captures=[]
    prepare=session.memory.prepare_retrieval
    async def observed(question, **kwargs):
        result=await prepare(question, **kwargs)
        captures.append(pilot.source_snapshot(session.memory, question))
        return result
    session.memory.prepare_retrieval=observed
    params={'session_id':session.session_id,'question':case['question'],'qid':case['id'],
        'system2_mode':case.get('system2_mode','auto'),'leading_brain':'left',
        'executive_mode':'off','executive_observer_mode':'off','return_telemetry':True,
        'memory_context':{'id':case['scope'],'aliases':case['aliases'],'remember':False,'record_ref':ref}}
    if session.state_store is not None:
        params['memory_operation_id']='choices-'+case['id']
    try:
        reply=await _dispatch_request({'id':case['id'],'method':'process','params':params},
            sessions={session.session_id:session},state_store=session.state_store,
            blob_pipe=None,send=lambda *_:None,**options)
        if not reply.get('ok'):
            raise ValueError('Engine request unavailable')
        result=reply['result']
        row={'id':case['id'],'question':case['question'],'answer':result['answer'],'status':'ok',
            'record_selection':result.get('record_selection'),'system2':pilot.accounting(result['metrics'],result['telemetry']),
            'commit':{k:(result.get('memory_commit') or {}).get(k) for k in ('state','operation_id')}}
        row.update(captures[-1] if captures else pilot.source_snapshot(session.memory,case['question']))
        target=next((t for t in session.memory.past_qas if t.qid==case.get('target_qid')),None)
        pilot.assess_row(case,row,target)
        row['contracts'].update(selection_state=(row['record_selection'] or {}).get('status')==case['expected_state'],
            expected_action=(row['condition'] or {}).get('action')==case.get('expected_action'),
            explicit_mode_honored=(row['system2']['used'] is (case['system2_mode']=='on'))
                if case.get('system2_mode') in {'on','off'} else None,
            reference_not_in_provider=all(ref not in json.dumps(r['request']['payload'])
                for r in journal.data['occurrences'][before:]))
        if session.state_store is not None:
            active=await session.state_store.load_conversation_sources(session.session_id)
            row['active_primary_qids']=sorted(t.qid for t in active)
            row['contracts']['read_only_primaries']=row['active_primary_qids']==sorted(case['primary_qids'])
            row['contracts']['committed']=row['commit']['state']=='committed'
        row['usage']=pilot.bench.usage(journal.events[before:])
        row['calls_by_role']=dict(pilot.Counter(e['role'] for e in journal.events[before:]))
        return row
    finally:
        session.memory.prepare_retrieval=prepare


async def run(output_dir, fixture, config, journal, dsn):
    if fixture.get('schema_version')!=1 or tuple(c['id'] for c in fixture['cases'])!=IDS:
        raise ValueError('Complete fixed choices fixture required')
    pilot.base.enforce_runtime()
    report={'schema_version':1,'metadata':metadata(fixture,config),'complete':False,'rows':[],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.base.ROOT.parent,text=True).strip()}
    save=lambda:pilot.bench.checkpoint(output_dir/'report.partial.json',report)
    clock=[fixture['clock']]; options=dict(blob_root=output_dir,blob_index={},vision_cache={},vision_cache_limit=0)
    store=pilot.PostgresStateStore(dsn,table_prefix='record_choices_'+secrets.token_hex(6))
    env=pilot.profile.environment();env.update(DUALBRAIN_MEMORY_OUTBOX_PATH=str(output_dir/'outbox.sqlite3'),
                                              DUALBRAIN_PG_PERSIST_TELEMETRY='0')
    session=None; saved_ref=None
    with patch.dict(os.environ,env,clear=True),patch('time.time',lambda:clock[0]),\
            patch('core.unconscious_field.uuid',SimpleNamespace(uuid4=lambda:uuid.UUID(int=pilot.base.UUID_RNG.getrandbits(128)))):
        try:
            await store.ensure_schema()
            for index,case in enumerate(fixture['cases']):
                clock[0]=fixture['clock']+index
                if case['id']!='durable_relisted':
                    if session: await session.close()
                    session,_=await pilot.make_session('choices:durable' if case.get('durable') else 'choices:'+case['id'],
                        store if case.get('durable') else None,config,journal,clock)
                if case.get('records'):
                    traces=[pilot.MemoryTrace(**r) for r in case['records']]
                    if case.get('durable'):
                        await store.append_memory_traces(session.session_id,traces)
                    else:
                        for trace in traces: session.memory.store(trace)
                if case['id']=='restart_rejected':
                    ref=saved_ref
                else:
                    page=await request_choices(session,case['scope'],options,journal)
                    ref=next(c['record_ref'] for c in page['choices'] if case['choose_label'] in c['label'])
                    if not ref: raise ValueError('Expected selectable choice')
                    if case['id']=='durable_selected': saved_ref=ref
                setup_before=len(journal.events)
                if case.get('changed_answer'):
                    session.memory.begin_turn(case['question'],case['id'],{'id':case['scope'],
                        'aliases':case['aliases'],'remember':False,'record_ref':ref})
                    journal.case=case['id']+':before-change'
                    await session.memory.prepare_retrieval(case['question'])
                    session.memory.past_qas=[replace(t,answer=case['changed_answer']) if t.qid==case['change_qid'] else t
                                            for t in session.memory.past_qas]
                setup_usage=pilot.bench.usage(journal.events[setup_before:])
                row=await turn(session,case,ref,journal,options)
                row['setup_usage']=setup_usage
                row['runtime_usage']=row['usage']
                row['runtime_calls_by_role']=row['calls_by_role']
                row['usage']=pilot.bench.usage(journal.events[setup_before:])
                row['calls_by_role']=dict(pilot.Counter(e['role'] for e in journal.events[setup_before:]))
                report['rows'].append(row);save()
                print(json.dumps({'id':case['id'],'selection':row['record_selection'],'contracts':row['contracts'],
                                  'usage':row['usage']}),flush=True)
            journal.finish()
            report.update(complete=True,live_provider_calls=journal.calls,usage=pilot.bench.usage(journal.events),
                calls_by_role=dict(pilot.Counter(e['role'] for e in journal.events)),
                served_models=sorted({e['served_model'] for e in journal.events if e.get('served_model')}))
            report['contract_failures']=[{'case':r['id'],'contract':k} for r in report['rows'] for k,v in r['contracts'].items() if v is False]
            report['canonical_sha256']=pilot.bench.digest(pilot.canonical(report));save();return report
        finally:
            if session: await session.close()
            with store._connect() as conn:
                for name in pilot.base.TABLES: conn.execute(f'DROP TABLE IF EXISTS {store._table(name)} CASCADE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--model');mode.add_argument('--journal',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--max-calls',type=int,choices=(CALL_LIMIT,),required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        fixture=pilot.bench.read_json(FIXTURE);frozen=pilot.bench.read_json(args.journal) if args.journal else None
        if args.model and subprocess.check_output(['git','status','--porcelain'],cwd=pilot.base.ROOT.parent,text=True).strip():
            raise ValueError('Freeze code/inputs before paid calls')
        dsn=os.environ.get('DUALBRAIN_TEST_PG_DSN')
        if not dsn: raise ValueError('An isolated test PostgreSQL DSN is required')
        model=args.model or frozen['metadata']['config']['model']
        config=pilot.load_llm_config(default_model=model) if not frozen else pilot.LLMConfig('openai',model,'REPLAY_ONLY')
        if not config or config.provider!='openai': raise ValueError('OpenAI-compatible configuration required')
        config=replace(config,model=model,max_output_tokens=2048,timeout_seconds=60,auto_continue=False,max_continuations=0)
        journal=pilot.RawJournal(config,metadata(fixture,config),frozen=frozen,max_calls=CALL_LIMIT,
            save=lambda data:pilot.bench.checkpoint(args.output_dir/'journal.partial.json',data))
        if frozen:
            with pilot.forbid_provider_network() as audit: report=asyncio.run(run(args.output_dir,fixture,config,journal,dsn))
            report['unexpected_blocked_http_attempts']=audit.blocked_requests
        else: report=asyncio.run(run(args.output_dir,fixture,config,journal,dsn))
        pilot.bench.write_private(args.output_dir/'report.json',report)
        pilot.bench.write_private(args.output_dir/'journal.json',frozen or journal.data)
        pilot.bench.write_private(args.output_dir/'summary.json',pilot.public_summary(report))
        print(json.dumps({'complete':True,'live_provider_calls':journal.calls,'usage':report['usage'],'failures':report['contract_failures']}))
    except BaseException as exc:
        pilot.bench.checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Record choices confirmation incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__': main()
