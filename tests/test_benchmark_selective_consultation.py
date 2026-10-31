import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_selective_consultation as bench
from core.llm_client import LLMConfig


def test_frozen_fixture_separates_development_and_preserves_counterfactual_provenance():
    fixture=bench.read_json(bench.FIXTURE)
    bench.validate(fixture)
    evaluation=[c for c in fixture['cases'] if c['split']=='evaluation']
    assert len(evaluation)==8 and len({c['problem_id'] for c in evaluation})==6
    assert sum(c['origin']=='model' for c in evaluation)==4
    assert {c['origin'] for c in evaluation}=={'model','synthetic_clean','synthetic_faulty'}


@pytest.mark.parametrize('missing_usage', [False, True])
def test_real_sessions_frozen_draft_identity_selective_telemetry_and_exact_replay(tmp_path, monkeypatch, missing_usage):
    monkeypatch.delenv('DUALBRAIN_MEMORY_EMBEDDING_MODEL',raising=False)
    monkeypatch.delenv('DUALBRAIN_MEMORY_ANSWERABILITY_MODEL',raising=False)
    fixture=bench.read_json(bench.FIXTURE)
    config=LLMConfig('openai','fake-model','NEVER_SAVE_THIS_KEY',auto_continue=False)
    args=SimpleNamespace(phase='development',output_dir=tmp_path/'live',judge_model='fake-judge')
    args.output_dir.mkdir()
    journal=bench.base.Journal(config,{},max_calls=64)
    async def post(url,payload,headers,**kwargs):
        fmt=payload.get('response_format',{})
        if fmt.get('type')=='json_schema':
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'scores':{label:{c:{'verdict':'pass','quote':answer} for c in bench.base.CRITERIA}
                                      for label,answer in packet['answers'].items()}})
        elif fmt:
            packet=json.loads(payload['messages'][-1]['content'])
            text=json.dumps({'decisions':[{'id':r['id'],'supported':True,'quote':r['answer']} for r in packet['records']]})
        elif 'rigorous reviewer' in payload['messages'][0]['content']:
            text=json.dumps({'verdict':'ok','issues':[],'fixes':[]})
        else:
            text='According to the record, retention is 11 days.'
        response={'model':payload['model'],'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        if not missing_usage:
            response['usage']={'prompt_tokens':1,'completion_tokens':1}
        return response
    journal.transport._post_json=post
    report=asyncio.run(bench.run(args,fixture,config,journal))
    assert report['complete'] and report['grader_failures']==0 and len(report['rows'])==8
    assert (report['input_tokens'] is None) is missing_usage
    assert (report['summaries']['all']['selective_auto']['input_tokens'] is None) is missing_usage
    for case in ('dev-lookup','dev-calculation'):
        rows=[r for r in report['rows'] if r['episode']==case and r['arm']!='no_revision']
        assert len({r['draft_sha256'] for r in rows})==1
    lookup=next(r for r in report['rows'] if r['episode']=='dev-lookup' and r['arm']=='selective_auto')
    assert lookup['route']['action']=='skip' and lookup['calls_by_role'].get('right',0)==0
    assert not lookup['metrics']['system2']['enabled']
    assert 'NEVER_SAVE_THIS_KEY' not in json.dumps(journal.data)
    args.output_dir=tmp_path/'replay'
    args.output_dir.mkdir()
    replay=bench.base.Journal(config,{},frozen=journal.data,max_calls=64)
    async def forbidden(*args,**kwargs):
        pytest.fail('Replay attempted a provider call')
    replay.transport._post_json=forbidden
    repeated=asyncio.run(bench.run(args,fixture,config,replay))
    assert repeated['provider_calls']==0 and repeated['grades']==report['grades']
    assert [r['answer'] for r in repeated['rows']]==[r['answer'] for r in report['rows']]
