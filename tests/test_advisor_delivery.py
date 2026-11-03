import asyncio
from dataclasses import replace
import json

import pytest

from core.advisor_delivery import compact_model, parse_compact_advice
from core.director_reasoner import DirectorReasonerModel
from core.executive_reasoner import ExecutiveReasonerModel
from core.llm_client import LLMConfig
from core.thalamic_reticular_nucleus import DIRECTOR_ADVICE, EXECUTIVE_ADVICE, EXECUTIVE_METRICS_OBSERVATION


class RecordingClient:
    def __init__(self, raw, *, limit=1024, error=None):
        self.config = LLMConfig('openai', 'test-advisor', 'OFFLINE_ONLY', max_output_tokens=limit)
        self.raw, self.error, self.calls = raw, error, []

    async def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error:
            raise self.error
        return self.raw


def model_with_client(model_type, raw, **kwargs):
    client = RecordingClient(raw, **kwargs)
    model = model_type(client.config)
    model._llm_client = client
    return model, client


@pytest.mark.asyncio
async def test_sparse_director_retains_every_operational_control_and_full_tail_instruction():
    control = {'consult': 'force', 'temperature': .2, 'max_chars': 600,
               'memory': {'working_memory': 'keep', 'long_term': 'drop'},
               'append_clarifying_question': 'Which target environment?'}
    model, client = model_with_client(DirectorReasonerModel, json.dumps({
        'memo': 'Keep relevant recent context, exclude older context and verify the missing environment.',
        'control': control, 'confidence': .8}))
    question = 'x' * 1800 + 'Important tail: do not reuse the old project.'
    result = await model.advise(question=question, context='y' * 2000, delivery_mode='compact')
    assert result.source == 'llm' and result.control == control
    assert question in client.calls[0][0] and 'y' * 1601 not in client.calls[0][0]
    kwargs = client.calls[0][1]
    assert kwargs['max_output_tokens'] == 256 and kwargs['allow_continuation'] is False and kwargs['require_complete'] is True


@pytest.mark.asyncio
async def test_metrics_compact_is_partial_post_answer_observation_with_no_steering_or_mixin():
    model, client = model_with_client(ExecutiveReasonerModel, json.dumps({
        'memo': 'The supplied partial answer snippet shows a retry limit; the unseen tail is not assessed.',
        'confidence': .7}), limit=128)
    result = await model.advise(question='Assess the supplied answer.', context='Observer report (internal; snippet only).',
                                delivery_mode='metrics_compact')
    assert result.source == 'llm' and result.directives == {} and result.mix_in == ''
    kwargs = client.calls[0][1]
    assert kwargs['max_output_tokens'] == 128 and kwargs['allow_continuation'] is False and kwargs['require_complete'] is True
    assert 'partial snippets' in kwargs['system'] and 'after the answer' in kwargs['system']


@pytest.mark.parametrize('raw', [
    '```json\n{"memo":"x","control":{},"confidence":0.5}\n```',
    '{"memo":"x","memo":"y","control":{},"confidence":0.5}',
    '{"memo":"x","control":{},"confidence":true}',
    '{"memo":"x","control":{},"confidence":NaN}',
    '{"memo":"x","control":{},"confidence":0.5,"extra":1}',
    '{"memo":"x","control":{"consult":true},"confidence":0.5}',
    '{"memo":"x","control":{"temperature":true},"confidence":0.5}',
    '{"memo":"x","control":{"max_chars":80.5},"confidence":0.5}',
    '{"memo":"x","control":{"memory":{"long_term":"invent"}},"confidence":0.5}',
    '{"memo":"x","control":{"memory":{"unknown":"drop"}},"confidence":0.5}',
    '{"memo":"x","control":{"append_clarifying_question":[]},"confidence":0.5}',
    '{"memo":"x","control":{"consult":"force"}',
    json.dumps({'memo': 'x' * 241, 'control': {}, 'confidence': .5}),
])
def test_compact_director_rejects_wrappers_truncation_duplicates_and_untyped_or_extra_controls(raw):
    with pytest.raises(ValueError):
        parse_compact_advice(raw, director=True)


@pytest.mark.parametrize('raw', [
    '{"memo":"x","confidence":0.5,"directives":{}}',
    '{"memo":"x","confidence":0.5,"mix_in":"new answer"}',
    '{"memo":"","confidence":0.5}',
])
def test_metrics_cannot_return_pre_answer_controls_or_empty_success(raw):
    with pytest.raises(ValueError):
        parse_compact_advice(raw, director=False)


@pytest.mark.parametrize('model_type,mode', [(DirectorReasonerModel, 'compact'), (ExecutiveReasonerModel, 'metrics_compact')])
@pytest.mark.asyncio
async def test_bad_output_and_provider_errors_are_unusable_without_invented_controls(model_type, mode):
    for raw, error in [('{"memo":"partial', None), ('', RuntimeError('provider failed'))]:
        model, _ = model_with_client(model_type, raw, error=error)
        result = await model.advise(question='Task', context='', delivery_mode=mode)
        assert result.source == 'error' and result.confidence == 0.
        assert getattr(result, 'control', getattr(result, 'directives', None)) == {}
    model, _ = model_with_client(model_type, '', error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await model.advise(question='Task', context='', delivery_mode=mode)


@pytest.mark.asyncio
async def test_dynamic_compact_views_follow_effective_config_without_mutating_or_cloning_clients():
    model, client = model_with_client(DirectorReasonerModel, '{"memo":"Use current context.","control":{},"confidence":0.8}')
    view = compact_model(model, DIRECTOR_ADVICE)
    assert view is not model and view._llm_client is None and view.advisor_contract == 'director_compact_v1'
    assert view.llm_config.max_output_tokens == 256 and not view.llm_config.auto_continue and view.llm_config.max_continuations == 0
    assert client.config.max_output_tokens == 1024 and client.config.auto_continue
    client.config = replace(client.config, max_output_tokens=64, model='new-role-model')
    assert view.llm_config.max_output_tokens == 64 and view.llm_config.model == 'new-role-model'
    result = await view.advise(question='Task', context='')
    assert result.source == 'llm' and client.calls[-1][1]['max_output_tokens'] == 64
    assert model._llm_client is client and model.llm_config.max_output_tokens == 1024
    executive, _ = model_with_client(ExecutiveReasonerModel, '{"memo":"Partial snippet observed.","confidence":0.8}')
    assert compact_model(executive, EXECUTIVE_ADVICE) is executive
    assert compact_model(executive, EXECUTIVE_METRICS_OBSERVATION).advisor_contract == 'metrics_compact_v1'


def test_custom_models_are_not_silently_opted_into_a_new_api():
    class CustomDirector(DirectorReasonerModel):
        pass
    config = LLMConfig('openai', 'custom', 'OFFLINE_ONLY')
    custom = CustomDirector(config)
    assert compact_model(custom, DIRECTOR_ADVICE) is custom


@pytest.mark.asyncio
async def test_default_full_delivery_preserves_existing_request_and_response_contract():
    model, client = model_with_client(ExecutiveReasonerModel, json.dumps({
        'memo': 'Full memo.', 'directives': {'tone': 'friendly'}, 'mix_in': 'Hello.', 'confidence': .8}))
    result = await model.advise(question='Task', context='')
    assert result.source == 'llm' and result.mix_in == 'Hello.' and result.directives == {'tone': 'friendly'}
    assert set(client.calls[0][1]) == {'system', 'temperature'}
