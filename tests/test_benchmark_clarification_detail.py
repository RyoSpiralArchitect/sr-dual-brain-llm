"""Shared-draft mode controls retain empty raw envelopes and replay without HTTP."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import benchmark_clarification_detail as b
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network


class Provider:
    def __init__(self, fault): self.fault=fault
    async def _post_json(self,_url,payload,_headers,**_kwargs):
        system=payload['messages'][0]['content']
        if 'rigorous reviewer (critic)' in system:
            return response('{"verdict":"ok","issues":[],"fixes":[],"confidence_r":0.9}')
        if self.fault and 'When given a draft, return only additional content' in system:
            return response('', 'length')
        text=('現在の候補から、再掲する記録をもう一度選んでください。'
              if '再掲' in payload['messages'][-1]['content'] else
              'Please choose which complete record to quote from the current record choices.')
        return response(text)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', [False, True])
async def test_full_shared_draft_mode_comparison_and_fault_replay(tmp_path, fault):
    fixture=b.pilot.bench.read_json(b.FIXTURE)
    live,replay=tmp_path/'live',tmp_path/'replay';live.mkdir();replay.mkdir()
    journal=b.pilot.RawJournal(config(),b.metadata(fixture,config()),max_calls=b.CALL_LIMIT)
    journal.transport=Provider(fault)
    with forbid_provider_network() as audit: a=await b.run(live,fixture,config(),journal)
    assert a['complete'] and len(a['rows'])==6 and not a['contract_failures']
    assert audit.blocked_requests==0
    for r in a['rows']:
        if r['arm']=='selective_auto': assert r['usage']['calls']==0 and r['detail'] is None
        if r['detail'] and fault:
            assert not r['detail']['usable'] and len(r['detail']['deliveries'])==1
            assert r['detail']['deliveries'][0]['status']=='empty_response'
    frozen=b.pilot.RawJournal(config(),b.metadata(fixture,config()),
        frozen=json.loads(json.dumps(journal.data)),max_calls=b.CALL_LIMIT)
    with forbid_provider_network(): c=await b.run(replay,fixture,config(),frozen)
    assert frozen.transport is None and frozen.calls==0 and b.canonical(a)==b.canonical(c)
    assert a['usage']['calls']==sum(r['usage']['calls'] for r in a['rows'])+sum(r['usage']['calls'] for r in a['shared_drafts'])
    public=json.dumps(b.pilot.public_summary(a))
    assert all(t not in public for t in ('r1.','"record_ref"','"answer"','"question"','FAKE_NO_CREDENTIAL'))
