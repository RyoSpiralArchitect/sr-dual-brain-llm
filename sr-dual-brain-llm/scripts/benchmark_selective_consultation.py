#!/usr/bin/env python3
"""Frozen-draft routing pilot with production sessions and exact transport replay."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
import uuid
from unittest.mock import patch

import benchmark_persistent_answers as base
from benchmark_conversation_memory import usage_for
from benchmark_controlled_memory import checkpoint
from benchmark_frozen_drafts import digest, read_json, write_private, sum_usage
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


def generation_failure(row):
    """A returned but unusable completion is not a transport outage."""
    response = row.get('response') or {}
    choices = response.get('choices') or []
    if row.get('error_type') != 'ValueError' or row.get('http_status') or len(choices)!=1:
        return False
    choice = choices[0]
    message = choice.get('message') or {}
    return (choice.get('finish_reason') in {'length','max_tokens'}
            or (choice.get('finish_reason')=='stop' and isinstance(message.get('content'),str)
                and not message['content'].strip() and not message.get('refusal')))


def failed_events_are_completions(journal, events):
    failed=[e for e in events if e.get('error_type')]
    rows=(journal.frozen or journal.data)['responses']
    return bool(failed) and all(generation_failure(rows[e['key']][journal.counts[e['key']]-1]) for e in failed)


def production_unchanged(revision):
    return (re.fullmatch(r'[0-9a-f]{40}',revision or '') is not None
            and subprocess.run(['git','diff','--quiet',revision,'--',
                                'sr-dual-brain-llm/core','sr-dual-brain-llm/scripts/engine_stdio.py'],
                               cwd=base.ROOT.parent).returncode==0)


def resume_prefix(args, fixture, config, journal):
    path=getattr(args,'resume',None)
    if path is None:
        return None
    old=read_json(path/'report.partial.json')
    data=read_json(path/'journal.partial.json')
    if not production_unchanged(old.get('source_revision')):
        raise ValueError('Production runtime changed after the frozen prefix')
    if (data.get('complete') or old.get('complete') or old.get('grades')
            or old['fixture_sha256']!=digest(fixture) or old['phase']!=args.phase
            or old['call_limit']!=journal.max_calls
            or data.get('model')!=config.model or data.get('judge_model')!=args.judge_model
            or old['router_sha256']!=hashlib.sha256((base.ROOT/'core/consultation_router.py').read_bytes()).hexdigest()
            or any(data.get(k)!=v for k,v in journal.data.items() if k not in {'complete','responses'})):
        raise ValueError('Continuation differs from the frozen ungraded prefix')
    rows=[(key,r) for key,values in data['responses'].items() for r in values]
    errors=[r for _,r in rows if r.get('error_type')]
    if not errors or not all(generation_failure(r) for r in errors) or not data.get('transport_blocked'):
        raise ValueError('Only returned invalid completions permit continuation; transport failures stay blocked')
    planned={c['id'] for c in fixture['cases'] if c['split']==args.phase}
    pairs=[(r['episode'],r['arm']) for r in old['rows']]
    if (len(set(pairs))!=len(pairs) or any(c not in planned or a not in ('no_revision',*ARMS) for c,a in pairs)
            or old['rows'][-1]['status']!='unavailable' or old['rows'][-1].get('answer') is not None
            or any(r['status']!='ok' for r in old['rows'][:-1])):
        raise ValueError('Require an immutable completed prefix and one failed last cell')
    if len(rows)>=journal.max_calls:
        raise ValueError('Continuation has no remaining allocated calls')
    for cid,draft in old['drafts'].items():
        if cid not in planned or not any(r['episode']==cid and r['arm']=='no_revision'
                                        and r['answer']==draft['text'] for r in old['rows']):
            raise ValueError('Shared draft has no matching immutable baseline')
    for r in old['rows']:
        if r['arm']!='no_revision' and r['draft_sha256']!=digest(old['drafts'][r['episode']]['text']):
            raise ValueError('Frozen initial draft identity differs')
    events=[journal._event(r['request']['role'],False,key,r) for key,r in rows]
    totals=usage_for(events)
    used=[r['usage'] for r in old['rows'][:-1]]+[d['usage'] for d in old['drafts'].values()]
    # Older interrupted reports checkpointed a cell before its usage was known.
    # Recover its exact remaining token/call totals; never invent handler time.
    failed=old['rows'][-1]
    failed['usage']={k:(totals[k]-sum(u[k] for u in used)
                         if totals[k] is not None and all(u[k] is not None for u in used) else None)
                     for k in ('calls','input_tokens','output_tokens','records_sent')}
    shared_models=sum(d['origin']=='model' for d in old['drafts'].values())
    roles=Counter(e['role'] for e in events)
    roles.subtract(sum((Counter(r['calls_by_role']) for r in old['rows'][:-1]),Counter()))
    roles.subtract({'draft':shared_models,'support':sum(d['usage']['calls'] for d in old['drafts'].values())-shared_models})
    if any(v<0 for v in roles.values()) or sum(roles.values())!=failed['usage']['calls']:
        raise ValueError('Interrupted usage cannot be reconciled exactly')
    failed.update(calls_by_role={k:v for k,v in roles.items() if v},elapsed_ms=failed.get('elapsed_ms'),
                  route=failed.get('route'),failure={'kind':'invalid_completion','retained_from_prefix':True})
    lineage={'prefix_report_sha256':hashlib.sha256((path/'report.partial.json').read_bytes()).hexdigest(),
             'prefix_journal_sha256':hashlib.sha256((path/'journal.partial.json').read_bytes()).hexdigest(),
             'prefix_source_revision':old['source_revision'],'prefix_calls':len(rows)}
    journal.data=deepcopy(data)
    journal.data.update(complete=False,transport_blocked=False,continuation=lineage)
    journal.calls=len(rows)
    journal.events=events
    journal.counts=defaultdict(int,{key:len(values) for key,values in data['responses'].items()})
    journal.input_tokens,journal.output_tokens=totals['input_tokens'],totals['output_tokens']
    old['continuation']=lineage
    return old


async def run(args, fixture, config, journal):
    base.enforce_runtime()
    prefix=resume_prefix(args,fixture,config,journal)
    report = {'schema_version':1,'complete':False,'experiment':'selective_conversation_consultation',
              'phase':args.phase,'fixture_sha256':digest(fixture),'router_version':VERSION,
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT.parent,text=True).strip(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'router_sha256':hashlib.sha256((base.ROOT/'core/consultation_router.py').read_bytes()).hexdigest(),
              'drafts':{},'rows':[],'grades':[],
              'runtime':ENV,'torch_policy':TORCH_AVAILABLE,'call_limit':journal.max_calls}
    if prefix is not None:
        report.update(drafts=prefix['drafts'],rows=prefix['rows'],continuation=prefix['continuation'])
    elif journal.frozen and journal.frozen.get('continuation'):
        report['continuation']=journal.frozen['continuation']
    save = lambda: checkpoint(args.output_dir/'report.partial.json', report)
    save()
    cases = [c for c in fixture['cases'] if c['split']==args.phase]
    random.Random(20261101).shuffle(cases)
    clock = [1793150000.]
    options = dict(blob_root=args.output_dir, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, ENV), patch('time.time',lambda:clock[0]), \
         patch('uuid.uuid4',lambda:uuid.UUID(int=base.UUID_RNG.getrandbits(128))):
        for index, case in enumerate(cases):
            if case['id'] in report['drafts']:
                draft=report['drafts'][case['id']]['text']
            elif case['origin']=='model':
                with patch.dict(os.environ, {'DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT':'0'}):
                    draft, events, elapsed = await generate_draft(case, config, journal, clock, options)
            else:
                draft, events, elapsed = case['draft'], [], 0.
            if case['id'] not in report['drafts']:
                report['drafts'][case['id']] = {'origin':case['origin'],'text':draft,'usage':usage_for(events),'elapsed_ms':elapsed}
                report['rows'].append({'episode':case['id'],'id':'answer','arm':'no_revision','origin':case['origin'],
                    'problem_id':case['problem_id'],'status':'ok','answer':draft,'elapsed_ms':0.,'usage':usage_for([]),
                    'calls_by_role':{},'route':None})
            order = list(ARMS[index%3:]+ARMS[:index%3])
            if index%2:
                order.reverse()
            for arm in order:
                if any(r['episode']==case['id'] and r['arm']==arm for r in report['rows']):
                    continue  # Completed AND failed attempts are immutable; never resample them.
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
                    started,before=time.perf_counter(),len(journal.events)
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
                    except Exception:
                        events=journal.events[before:]
                        if not failed_events_are_completions(journal,events):
                            raise
                        row.update(elapsed_ms=(time.perf_counter()-started)*1000,usage=usage_for(events),
                            calls_by_role=dict(Counter(e['role'] for e in events)),route=None,
                            failure={'kind':'invalid_completion','retained_from_prefix':False})
                        journal.data['transport_blocked']=False
                        save()  # No fallback answer or retry; next cell gets a fresh session.
                    finally:
                        await engine.close()
        grading = {'episodes':[{'id':c['id'],'split':args.phase,'probes':[{'id':'answer',**{k:c[k] for k in ('question','reference','criteria')}}]} for c in cases]}
        def save_grades(grades):
            report['grades']=grades
            save()
        report['grades']=await base.grade(journal,replace(config,model=args.judge_model),grading,args.phase,
                                          [r for r in report['rows'] if r['status']=='ok'],save_grades)
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
                'input_tokens':sum_usage([r['usage'] for r in rows],'input_tokens'),
                'output_tokens':sum_usage([r['usage'] for r in rows],'output_tokens'),
                'handler_timed_trials':sum(r['elapsed_ms'] is not None for r in rows),
                'handler_ms_median':statistics.median(r['elapsed_ms'] for r in rows if r['elapsed_ms'] is not None)
                                    if any(r['elapsed_ms'] is not None for r in rows) else None}
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
    parser.add_argument('--resume',type=Path,help='Continue only unattempted cells from an ungraded invalid-completion prefix')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    try:
        base.enforce_runtime()
        if bool(args.model)==bool(args.journal) or args.max_calls>(64 if args.phase=='development' else 160):
            raise ValueError('Choose live or frozen mode within the prospectively allocated budget')
        fixture=read_json(FIXTURE)
        validate(fixture)
        frozen=read_json(args.journal) if args.journal else None
        if args.resume is not None and frozen is not None:
            raise ValueError('Continuation is live-only; replay the complete combined journal without --resume')
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
        if frozen and frozen.get('continuation'):
            metadata['continuation']=frozen['continuation']
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
