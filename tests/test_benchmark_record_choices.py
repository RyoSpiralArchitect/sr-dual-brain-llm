"""Actual API choices/selected quotes and PostgreSQL restart must replay exactly."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_record_choices as b
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network


class Provider:
    async def _post_json(self,_url,payload,_headers,**kwargs):
        messages=payload['messages']
        try: packet=json.loads(messages[-1]['content'])
        except ValueError: packet={}
        if 'records' in packet:
            text=json.dumps({'decisions':[{'id':r['id'],'supported':True,'quote':r['answer']} for r in packet['records']]})
        elif 'critic' in messages[0]['content'].lower():
            text='{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.9}'
        else:
            text='Please choose a current record again.'
            for line in messages[0]['content'].splitlines():
                if line.startswith('{"source_question":'):
                    text=json.loads(line)['recorded_answer'];break
        return response(text)


@pytest.mark.asyncio
async def test_fixed_choices_api_and_durable_restart_replay(tmp_path):
    dsn=os.environ.get('DUALBRAIN_TEST_PG_DSN')
    if not dsn: pytest.skip('isolated PostgreSQL required')
    fixture=b.pilot.bench.read_json(b.FIXTURE)
    live,replay=tmp_path/'live',tmp_path/'replay';live.mkdir();replay.mkdir()
    journal=b.pilot.RawJournal(config(),b.metadata(fixture,config()),max_calls=b.CALL_LIMIT)
    journal.transport=Provider()
    with forbid_provider_network(): a=await b.run(live,fixture,config(),journal,dsn)
    assert a['complete'] and len(a['rows'])==6 and not a['contract_failures']
    frozen=b.pilot.RawJournal(config(),b.metadata(fixture,config()),frozen=deepcopy(journal.data),max_calls=b.CALL_LIMIT)
    with forbid_provider_network(): c=await b.run(replay,fixture,config(),frozen,dsn)
    assert frozen.transport is None and frozen.calls==0
    assert b.pilot.canonical(a)==b.pilot.canonical(c)
    assert a['usage']['calls']==sum(r['usage']['calls'] for r in a['rows'])
    assert all(r['usage']['calls']==sum(r['calls_by_role'].values()) for r in a['rows'])
    public=json.dumps(b.pilot.public_summary(a))
    assert all(t not in public for t in ['r1.','FAKE_NO_CREDENTIAL','"answer"','"record_ref"'])
