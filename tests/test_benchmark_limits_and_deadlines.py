"""Real serialized caps and typed local failures cannot poison provider health."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_benchmark_contract_live import pilot,config,response,Transport
from offline_provider_guard import forbid_provider_network
from core.review_budget import ReviewBudgetExhausted


@pytest.mark.asyncio
async def test_stage_clients_forward4096_while_historical_adapter_keeps2048():
    cfg=replace(config(),max_output_tokens=4096)
    journal=pilot.RawJournal(cfg,{'test':'caps'},max_calls=6)
    seen=[]
    class Captured:
        async def _post_json(self,_url,payload,_headers,**kwargs):
            seen.append(payload);return response('Complete answer.')
    journal.transport=Captured();journal.case='caps'
    with forbid_provider_network() as audit:
        session,_=await pilot.make_session('caps',None,cfg,journal,[1794182400.])
        try:
            draft=await session.left.generate_answer('Give a complete answer.','')
            await session.left.integrate_info_async(question='Give a complete answer.',draft=draft,info='Check all constraints.')
        finally:await session.close()
        legacy=pilot.base.RecordedClient(cfg,journal,'legacy')
        await legacy.complete('Legacy bounded request.',max_output_tokens=4096,require_complete=True)
        modern=pilot.StageClient(cfg,journal,'modern')
        await modern.complete('Configured ceiling fallback.',require_complete=True)
    assert [p.get('max_completion_tokens',p.get('max_tokens')) for p in seen]==[4096,4096,2048,4096]
    assert [r['request']['role'] for r in journal.data['occurrences']][:2]==['draft','integration']
    assert audit.blocked_requests==0


@pytest.mark.asyncio
@pytest.mark.parametrize('cap',[0,-1,True,'4096'])
async def test_modern_invalid_cap_does_not_open_transport(cap):
    journal=pilot.RawJournal(config(),{'test':'invalid-cap'})
    client=pilot.StageClient(config(),journal,'modern')
    with forbid_provider_network() as audit:
        with pytest.raises(ValueError):await client.complete('Request.',max_output_tokens=cap)
    assert journal.calls==audit.blocked_requests==0


@pytest.mark.asyncio
@pytest.mark.parametrize('attempted',[True,False])
async def test_typed_review_window_expiry_keeps_reservation_but_not_global_fault_and_replays(attempted):
    import benchmark_replanning_stress as stress
    budget=stress.base.CampaignBudget()
    cfg=replace(config(),max_output_tokens=4096)
    journal=stress.Journal(cfg,{'test':'local-window'},campaign=budget,max_calls=4)
    journal.begin_trial('first')
    journal.transport=Transport([ReviewBudgetExhausted('Local window elapsed',http_attempted=attempted),response('next')])
    with forbid_provider_network() as audit:
        with pytest.raises(ReviewBudgetExhausted):
            await journal.post('critic','fake',{'max_completion_tokens':2048},{})
        assert not journal.blocked and not budget.blocked
        assert journal.reserved==2048 and budget.admissions==1
        journal.begin_trial('next')
        await journal.post('draft','fake',{'max_completion_tokens':4096},{})
        journal.finish()
        replay=stress.Journal(cfg,{'test':'local-window'},campaign=stress.base.CampaignBudget(),
            max_calls=4,frozen=deepcopy(journal.data))
        replay.begin_trial('first')
        with pytest.raises(ReviewBudgetExhausted):
            await replay.post('critic','fake',{'max_completion_tokens':2048},{})
        replay.begin_trial('next');await replay.post('draft','fake',{'max_completion_tokens':4096},{})
        replay.finish()
    assert replay.events==journal.events and replay.calls==audit.blocked_requests==0
    assert journal.calls==1+attempted
    assert journal.data['occurrences'][0]['http_attempted'] is attempted
    assert journal.events[0]['input_tokens']==(None if attempted else 0)
    assert journal.data['usage']['unknown_usage_calls']==int(attempted)
    assert budget.admissions==2  # No refund from the hard conservative campaign bound.


@pytest.mark.asyncio
@pytest.mark.parametrize('error',[TimeoutError('Unclassified timeout'),OSError('Private network error'),
    RuntimeError('openai returned429: private provider data')])
async def test_genuine_or_unclassified_transport_fault_still_closes_all_later_admission(error):
    import benchmark_replanning_stress as stress
    budget=stress.base.CampaignBudget()
    journal=stress.Journal(config(),{'test':'real-fault'},campaign=budget,max_calls=4)
    journal.transport=Transport([error,response('must not execute')]);journal.begin_trial('fault')
    with forbid_provider_network() as audit:
        with pytest.raises(type(error)):await journal.post('critic','fake',{'max_completion_tokens':2048},{})
        with pytest.raises(pilot.AdmissionError):await journal.post('draft','fake',{'max_completion_tokens':4096},{})
    assert journal.blocked and budget.blocked and journal.calls==1
    assert audit.blocked_requests==0 and 'private provider data' not in json.dumps(journal.data)
