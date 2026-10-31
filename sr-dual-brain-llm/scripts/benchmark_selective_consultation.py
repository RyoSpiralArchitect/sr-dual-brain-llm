#!/usr/bin/env python3
"""Frozen-draft routing pilot with production sessions and exact transport replay."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import uuid
from unittest.mock import patch

import benchmark_persistent_answers as base
from benchmark_conversation_memory import usage_for
from benchmark_controlled_memory import checkpoint
from benchmark_frozen_drafts import digest, read_json, write_private
from core.consultation_router import VERSION
from core.llm_client import LLMConfig, load_llm_config
from core.policy import TORCH_AVAILABLE
from core.shared_memory import MemoryTrace

FIXTURE = base.ROOT/'examples/selective_consultation_benchmark.json'
ARMS = ('legacy_auto', 'selective_auto', 'forced_on')
ENV = {'DUALBRAIN_MEMORY_CONVERSATION_CONTEXT':'1', 'DUALBRAIN_SYSTEM2_PRIORITY':'balanced',
       'DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN':'0'}


def validate(fixture):
    cases = fixture['cases']
    if fixture.get('origin') != 'authored_selective_consultation_pilot' or fixture.get('schema_version') != 1:
        raise ValueError('Unexpected fixture')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case')
    for c in cases:
        if c['split'] not in {'development','evaluation'} or c['origin'] not in {'model','synthetic_clean','synthetic_faulty'}:
            raise ValueError('Invalid split or draft origin')
        if set(c['criteria']) != set(base.CRITERIA) or not c['records'] or not c['problem_id']:
            raise ValueError('Incomplete case')
        if (c['origin']=='model') != (c['draft'] is None):
            raise ValueError('Synthetic drafts must be frozen; model drafts must be generated')
    dev = {r['retrieval_scope'] for c in cases if c['split']=='development' for r in c['records']}
    evaluation = {r['retrieval_scope'] for c in cases if c['split']=='evaluation' for r in c['records']}
    if dev & evaluation:
        raise ValueError('Development and evaluation scopes overlap')


async def session(case, arm, config, journal, clock):
    if TORCH_AVAILABLE:
        import torch
        torch.manual_seed(int(digest(case['id'])[:8],16))
    engine, clients = await base.session_for(case['id']+':'+arm, None, config, journal,
                                            'left' if arm=='draft' else arm, clock)
    for r in case['records']:
        engine.memory.store(MemoryTrace(**r))
    return engine, clients


async def generate_draft(case, config, journal, clock, options):
    engine, clients = await session(case, 'draft', config, journal, clock)
    client = base.RecordedClient(config, journal, 'draft')
    clients[0], engine.left._llm_client = client, client
    captured = []
    original = engine.left.generate_answer
    async def capture(*args, **kwargs):
        answer = await original(*args, **kwargs)
        captured.append(answer)
        return answer
    engine.left.generate_answer = capture
    try:
        _, events, elapsed = await base.process(engine, clients,
            {'question':case['question'],'qid':case['id']+':draft','leading_brain':'left','system2_mode':'off'}, journal, options)
        if len(captured)!=1:
            raise ValueError('Require exactly one natural initial draft')
        return captured[0], events, elapsed
    finally:
        await engine.close()


async def run(args, fixture, config, journal):
    base.enforce_runtime()
    report = {'schema_version':1,'complete':False,'experiment':'selective_conversation_consultation',
              'phase':args.phase,'fixture_sha256':digest(fixture),'router_version':VERSION,
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT.parent,text=True).strip(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'router_sha256':hashlib.sha256((base.ROOT/'core/consultation_router.py').read_bytes()).hexdigest(),
              'drafts':{},'rows':[],'grades':[],
              'runtime':ENV,'torch_policy':TORCH_AVAILABLE,'call_limit':journal.max_calls}
    save = lambda: checkpoint(args.output_dir/'report.partial.json', report)
    save()
    cases = [c for c in fixture['cases'] if c['split']==args.phase]
    random.Random(20261101).shuffle(cases)
    clock = [1793150000.]
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, ENV), patch('time.time',lambda:clock[0]), \
         patch('uuid.uuid4',lambda:uuid.UUID(int=base.UUID_RNG.getrandbits(128))):
        for index, case in enumerate(cases):
            if case['origin']=='model':
                with patch.dict(os.environ, {'DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT':'0'}):
                    draft, events, elapsed = await generate_draft(case, config, journal, clock, options)
            else:
                draft, events, elapsed = case['draft'], [], 0.
            report['drafts'][case['id']] = {'origin':case['origin'],'text':draft,'usage':usage_for(events),'elapsed_ms':elapsed}
            report['rows'].append({'episode':case['id'],'id':'answer','arm':'no_revision','origin':case['origin'],
                'problem_id':case['problem_id'],'status':'ok','answer':draft,'elapsed_ms':0.,'usage':usage_for([]),
                'calls_by_role':{},'route':None})
            order = list(ARMS[index%3:]+ARMS[:index%3])
            if index%2:
                order.reverse()
            for arm in order:
                with patch.dict(os.environ, {'DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT':'1' if arm=='selective_auto' else '0'}):
                    engine, clients = await session(case, arm, config, journal, clock)
                    async def frozen(*_args, **_kwargs):
                        return draft
                    engine.left.generate_answer = frozen
                    row = {'episode':case['id'],'id':'answer','arm':arm,'origin':case['origin'],
                           'problem_id':case['problem_id'],'order':order,'status':'unavailable','answer':None,
                           'draft_sha256':digest(draft)}
                    report['rows'].append(row)
                    save()
                    try:
                        result, events, elapsed = await base.process(engine, clients,
                            {'question':case['question'],'qid':case['id']+':probe','leading_brain':'left',
                             'system2_mode':'on' if arm=='forced_on' else 'auto'}, journal, options)
                        row.update(status='ok', answer=result['answer'], elapsed_ms=elapsed, usage=usage_for(events),
                                   calls_by_role=dict(Counter(e['role'] for e in events)), metrics=result['metrics'],
                                   route=result['metrics']['policy']['consultation_route'])
                        print(json.dumps({'case':case['id'],'arm':arm,'calls':journal.calls,
                                          'route':row['route']['action'] if row['route'] else None}),flush=True)
                        save()
                    finally:
                        await engine.close()
        grading = {'episodes':[{'id':c['id'],'split':args.phase,'probes':[{'id':'answer',**{k:c[k] for k in ('question','reference','criteria')}}]} for c in cases]}
        def save_grades(grades):
            report['grades']=grades
            save()
        report['grades']=await base.grade(journal,replace(config,model=args.judge_model),grading,args.phase,report['rows'],save_grades)
    journal.finish()
    report.update(complete=True,provider_calls=journal.calls,input_tokens=journal.input_tokens,output_tokens=journal.output_tokens,
                  frozen_replay=journal.frozen is not None,grader_failures=sum('error_type' in g for g in report['grades']))
    report['summaries']={}
    for origin in ('all','model','synthetic_clean','synthetic_faulty'):
        summary={}
        for arm in ('no_revision',*ARMS):
            rows=[r for r in report['rows'] if r['arm']==arm and (origin=='all' or r['origin']==origin)]
            ids={r['episode'] for r in rows}
            grades=[g for g in report['grades'] if g.get('arm')==arm and g['episode'] in ids]
            summary[arm]={'trials':len(rows),'available':sum(r['status']=='ok' for r in rows),'graded':len(grades),
                'all_pass':sum(all(v['verdict']=='pass' for v in g['scores'].values()) for g in grades),
                'grounding_failures':sum(g['scores']['grounding']['verdict']=='fail' for g in grades),
                'calls':sum(r['usage']['calls'] for r in rows),
                'input_tokens':sum(r['usage']['input_tokens'] for r in rows),
                'output_tokens':sum(r['usage']['output_tokens'] for r in rows),
                'handler_ms_median':statistics.median(r['elapsed_ms'] for r in rows) if rows else None}
        report['summaries'][origin]=summary
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['development','evaluation'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model')
    parser.add_argument('--journal',type=Path)
    parser.add_argument('--judge-model',default='gpt-6-sol')
    parser.add_argument('--max-calls',type=int,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        base.enforce_runtime()
        if bool(args.model)==bool(args.journal) or args.max_calls>(64 if args.phase=='development' else 160):
            raise ValueError('Choose live or frozen mode within the prospectively allocated budget')
        fixture=read_json(FIXTURE)
        validate(fixture)
        frozen=read_json(args.journal) if args.journal else None
        model=args.model or frozen['model']
        config=load_llm_config(default_model=model)
        if config is None:
            if args.model:
                raise ValueError('Provider configuration required')
            config=LLMConfig('openai',model,'FROZEN_NO_KEY')
        config=replace(config,model=model,max_output_tokens=2048,auto_continue=False,max_continuations=0,timeout_seconds=60)
        if config.provider!='openai':
            raise ValueError('OpenAI-compatible protocol required')
        metadata={'fixture_sha256':digest(fixture),'phase':args.phase,'model':model,'judge_model':args.judge_model,
                  'judge_policy_sha256':digest(base.JUDGE),'support_policy_sha256':digest(base.POLICY),
                  'router_version':VERSION,'runtime':ENV,'arms':list(ARMS)}
        journal=base.Journal(config,metadata,frozen=frozen,max_calls=args.max_calls,
                             save=lambda d:checkpoint(args.output_dir/'journal.partial.json',d))
        report=asyncio.run(run(args,fixture,config,journal))
        write_private(args.output_dir/'report.json',report)
        print(json.dumps({k:report[k] for k in ('summaries','provider_calls','input_tokens','output_tokens','grader_failures')},indent=2))
    except BaseException as exc:
        checkpoint(args.output_dir/'failure.json',{'complete':False,'error_type':type(exc).__name__})
        print('Benchmark incomplete: '+type(exc).__name__,file=__import__('sys').stderr)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
