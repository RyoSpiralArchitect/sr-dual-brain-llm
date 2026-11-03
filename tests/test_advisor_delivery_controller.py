from dataclasses import replace
import json

import pytest

from core.advisor_timing import AdvisorTiming, model_profile
from core.director_reasoner import DirectorReasonerModel
from core.executive_reasoner import ExecutiveReasonerModel
from core.llm_client import LLMConfig
from core.thalamic_reticular_nucleus import DIRECTOR_ADVICE, EXECUTIVE_ADVICE, EXECUTIVE_METRICS_OBSERVATION
from test_advisor_timing import setup
from test_dual_brain import make_stage_controller


@pytest.mark.asyncio
async def test_auto_selects_actual_compact_request_and_keeps_full_history_and_manual_modes(monkeypatch):
    c = make_stage_controller()
    model = DirectorReasonerModel(LLMConfig('openai','gpt-6-luna','OFFLINE',max_output_tokens=2048))
    c.director_model = model
    ledger = c.hypothalamus.advisor_timing = AdvisorTiming()
    full = model_profile(model)
    ledger.record(DIRECTOR_ADVICE,full,2000.,censored=True)
    requests = []

    async def post(url,payload,headers,**kwargs):
        requests.append(payload)
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'memo':'Keep the requested qualification.','control':{'consult':'force'},'confidence':.8})}}]}

    model._llm_client._post_json = post
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','auto')
    compact = c._advisor_model(DIRECTOR_ADVICE,model)
    key = model_profile(compact)
    assert key != full and ledger.estimate(DIRECTOR_ADVICE,key).ms is None
    stage = setup(c)
    result = await c._wait_advisor(stage.director_task,timeout_ms=1600.)
    assert result.source=='llm' and result.control['consult']=='force'
    assert requests[0]['max_completion_tokens']==256
    assert ledger.estimate(DIRECTOR_ADVICE,key).completed_samples==1
    assert ledger.estimate(DIRECTOR_ADVICE,full).ms==2000.
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','on')
    assert c._advisor_model(DIRECTOR_ADVICE,model) is model
    stage = setup(c)
    await c._wait_advisor(stage.director_task,timeout_ms=1600.)
    assert requests[-1]['max_completion_tokens']==2048
    monkeypatch.setenv('DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE','off')
    assert setup(c).director_task is None and len(requests)==2


@pytest.mark.asyncio
async def test_metrics_view_does_not_mutate_shared_executive_client_or_accept_late_profile_change(monkeypatch):
    c = make_stage_controller()
    model = ExecutiveReasonerModel(LLMConfig('openai','gpt-6-luna','OFFLINE',max_output_tokens=1024))
    c.executive_model = model
    monkeypatch.setenv('DUALBRAIN_MODULE_EXECUTIVE_METRICS_OBSERVATION_MODE','auto')
    view = c._advisor_model(EXECUTIVE_METRICS_OBSERVATION,model)
    assert c._advisor_model(EXECUTIVE_ADVICE,model) is model
    assert model._llm_client.config.auto_continue and view.llm_config.auto_continue is False
    original = model_profile(view)
    age = [0.]
    ledger = c.hypothalamus.advisor_timing = AdvisorTiming(clock=lambda:age[0])
    ledger.record(EXECUTIVE_METRICS_OBSERVATION,original,3000.,censored=True)
    age[0] = 16.
    calls = []

    async def advise(**kwargs):
        calls.append(kwargs)

    task = c._start_advisor(EXECUTIVE_METRICS_OBSERVATION,view,qid='profile-change',factory=advise,
        recovery_probe=True,phase_budget_ms=2500.)
    model._llm_client.config = replace(model._llm_client.config,max_output_tokens=128)
    assert model_profile(view)!=original and view.llm_config.max_output_tokens==128
    await task
    assert not calls and ledger.recovery_status(EXECUTIVE_METRICS_OBSERVATION,original,2500.).allowed
    assert model._llm_client.config.max_output_tokens==128 and model._llm_client.config.auto_continue
    assert task.done()
