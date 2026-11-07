"""Ordinary completion provenance, output allocations and cancellation ownership."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))

from core.answer_generation import current_generation, guard_answer_generation
from core.llm_client import LLMClient, LLMConfig
from core.models import LeftBrainModel, RightBrainModel
from core.task_goals import TaskState, extract_task_plan, goal_review
from offline_provider_guard import forbid_provider_network


class Harness:
    def __init__(self):
        self.events = []
        self.telemetry = SimpleNamespace(log=lambda event, **data: self.events.append((event, data)))

    @guard_answer_generation
    async def run(self, action):
        state = current_generation()
        value = await action(state)
        return value, state


def model_for(responses, *, stream=False, provider='openai', model='gpt-6-luna'):
    config = LLMConfig(provider, model, 'FAKE_NO_CREDENTIAL', max_output_tokens=32,
                       auto_continue=True, max_continuations=2)
    client = LLMClient(config)
    calls = []

    async def post(_url, payload, _headers, **kwargs):
        calls.append(payload)
        value = responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    client._post_json = post
    left = LeftBrainModel(config)
    left._llm_client = client if stream else SimpleNamespace(complete=client.complete)
    return left, client, calls


def envelope(text='A concrete answer.', reason='stop'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': reason}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('text,reason,status', [
    ('A concrete answer.', 'stop', 'complete'), ('Visible partial', 'length', 'partial'),
    ('', 'length', 'empty'), ('', 'stop', 'empty'), ('Visible partial', None, 'partial'),
    ('PRIVATE_BAD_OUTPUT', 'content_filter', 'invalid'),
])
async def test_actual_envelope_cannot_gain_false_completion(text, reason, status):
    left, _, calls = model_for([envelope(text, reason)])
    async def action(state):
        answer = await left.generate_answer('Explain the schedule.', '')
        assert state.status(answer) == status
        assert left.estimate_confidence(answer) == (0.9 if status == 'complete' else 0.)
        summary = state.summary(answer)
        assert summary['terminal_complete'] == (status == 'complete')
        assert summary['semantic_correctness'] == 'not_assessed'
        assert 'PRIVATE_BAD_OUTPUT' not in json.dumps(summary)
        return answer
    with forbid_provider_network() as audit:
        (answer, state) = await Harness().run(action)
    assert audit.blocked_requests == 0 and len(calls) == 1
    assert answer == (text if status in {'complete', 'partial'} else '')
    assert current_generation() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('failure,status', [
    (envelope('An unfinished revision', 'length'), 'partial'),
    (envelope('', 'length'), 'empty'),
    ({'choices': [{'message': {'content': 'PRIVATE_REFUSAL', 'refusal': 'blocked'}, 'finish_reason': 'stop'}]}, 'invalid'),
    (TimeoutError('PRIVATE_PROVIDER_ERROR'), 'failed'),
])
async def test_bad_integration_retains_complete_draft_and_buffers_rejected_deltas(failure, status):
    left, _, calls = model_for([envelope(), failure])
    visible = []
    async def action(state):
        draft = await left.generate_answer('Explain.', '')
        result = await left.integrate_info_async(question='Explain.', draft=draft,
            info='PRIVATE_INTERNAL_NOTES', on_delta=visible.append)
        assert result == draft and state.status(result) == 'complete'
        assert state.records[-1]['status'] == status
        assert 'PRIVATE_' not in json.dumps(state.summary(result))
        return result
    with forbid_provider_network():
        await Harness().run(action)
    assert visible == ['A concrete answer.'] and len(calls) == 2


@pytest.mark.asyncio
async def test_shared_ceiling_is_reserved_once_without_continuation_or_automatic_escalation(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_ANSWER_TOKEN_BUDGET', '50')
    left, _, calls = model_for([envelope('', 'length'), envelope('', 'length')])
    async def action(state):
        draft = await left.generate_answer('Explain.', '')
        for _ in range(3):
            await left.integrate_info_async(question='Explain.', draft=draft, info='Concrete repair')
        assert state.reserved == state.total_cap == 50
        assert [r['status'] for r in state.records] == ['empty', 'empty', 'budget_exhausted', 'budget_exhausted']
        return draft
    with forbid_provider_network():
        await Harness().run(action)
    assert [p['max_completion_tokens'] for p in calls] == [32, 18]


@pytest.mark.asyncio
@pytest.mark.parametrize('budget', ['-1', 'not_a_number', '0'])
async def test_invalid_or_exhausted_budget_makes_no_request(monkeypatch, budget):
    monkeypatch.setenv('DUALBRAIN_ANSWER_TOKEN_BUDGET', budget)
    left, _, calls = model_for([])
    async def action(state):
        assert await left.generate_answer('Explain.', '') == ''
        assert state.records[-1]['status'] == 'budget_exhausted'
    with forbid_provider_network():
        await Harness().run(action)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,model,expected', [
    ('openai', 'gpt-6-luna', 'low'), ('openai', 'gpt-4o', None), ('mistral', 'mistral-large', None),
])
async def test_opt_in_effort_enters_supported_request_without_increasing_cap(monkeypatch, provider, model, expected):
    monkeypatch.setenv('DUALBRAIN_ANSWER_REASONING_EFFORT', 'low')
    left, _, calls = model_for([envelope()], provider=provider, model=model)
    async def action(state):
        answer = await left.generate_answer('Explain.', '')
        assert state.records[-1]['request_reasoning_effort'] == expected
        return answer
    with forbid_provider_network():
        await Harness().run(action)
    assert calls[0].get('reasoning_effort') == expected
    assert calls[0].get('max_completion_tokens', calls[0].get('max_tokens')) == 32


@pytest.mark.asyncio
async def test_unavailable_and_legacy_clients_are_not_successful_external_generation():
    left, _, _ = model_for([])
    async def action(state):
        left._llm_client = None
        assert await left.generate_answer('説明して。', '') == ''
        assert state.status('') == 'unavailable'
        assert '回答を生成できませんでした' in state.present('', '説明して。')
        async def legacy(*args, **kwargs): return 'A custom answer without a provider receipt.'
        left._llm_client = SimpleNamespace(complete=legacy)
        answer = await left.generate_answer('Explain.', '')
        assert state.status(answer) == 'unverified' and not state.summary(answer)['terminal_complete']
        return answer
    with forbid_provider_network():
        await Harness().run(action)


@pytest.mark.asyncio
@pytest.mark.parametrize('ending,status', [('length', 'partial'), ('stop', 'complete'), ('disconnect', 'partial'), ('refusal', 'invalid')])
async def test_stream_terminal_and_partial_transport_status_are_preserved(ending, status):
    left, client, calls = model_for([], stream=True)
    visible = []
    async def stream_call(messages, *, on_delta, **kwargs):
        calls.append(client._openai_style_payload(messages, temperature=.4,
            max_output_tokens=kwargs['max_output_tokens'], stream=True))
        async def events():
            yield json.dumps({'choices': [{'delta': {'content': 'Visible text'}, 'finish_reason': None}]})
            if ending == 'disconnect': raise OSError('PRIVATE_STREAM_ERROR')
            if ending == 'refusal':
                yield json.dumps({'choices': [{'delta': {'refusal': 'blocked'}, 'finish_reason': 'stop'}]})
            else:
                yield json.dumps({'choices': [{'delta': {}, 'finish_reason': ending}]})
        return await client._consume_openai_style_stream(events(), on_delta=on_delta)
    client._openai_style_stream_call = stream_call
    async def action(state):
        answer = await left.generate_answer('Explain.', '', on_delta=visible.append)
        assert state.status(answer) == status
        assert state.records[0]['status'] == status
        return answer
    with forbid_provider_network():
        (answer, _) = await Harness().run(action)
    assert len(calls) == 1 and visible == ['Visible text']
    assert answer == ('' if status == 'invalid' else 'Visible text')


@pytest.mark.asyncio
async def test_cancellation_remains_owned_and_context_does_not_leak():
    left, client, calls = model_for([])
    entered = asyncio.Event()
    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    client._post_json = blocked
    owner = Harness()
    async def action(state):
        await left.generate_answer('Explain.', '')
    with forbid_provider_network():
        task = asyncio.create_task(owner.run(action))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    assert current_generation() is None
    assert owner.events[-1][1]['turn_status'] == 'cancelled'
    assert owner.events[-1][1]['stages'][-1]['status'] == 'cancelled'


def test_goal_reports_cannot_discharge_an_unconfirmed_generation():
    question = 'Requirements:\n1. Explain the cause.\n2. Give a test.'
    plan = extract_task_plan(question)
    state = TaskState(plan)
    draft = 'A cause and test.'
    findings = [{'goal_id': g.goal_id, 'status': 'addressed', 'quote': 'cause', 'reason': 'Provisional.'} for g in plan.goals]
    state.observe({'goal_review': goal_review(plan, draft, findings, external_valid=True, verdict='ok'),
        'critic_external_review_valid': True, 'verdict': 'ok'}, draft)
    summary = state.summary(draft, generation_complete=False)
    assert all(g['status'] == 'unconfirmed_generation' for g in summary['goals'])
    assert all(g['last_report'] == 'reported_addressed' for g in summary['goals'])
    assert not summary['verified_complete']


@pytest.mark.asyncio
@pytest.mark.parametrize('first_reason,next_reason,expected', [('stop', 'length', 'complete'), ('length', 'stop', 'complete')])
async def test_identical_visible_text_does_not_erase_or_invent_a_terminal_receipt(first_reason, next_reason, expected):
    left, _, calls = model_for([envelope('The same text', first_reason), envelope('The same text', next_reason)])
    async def action(state):
        draft = await left.generate_answer('Explain.', '')
        result = await left.integrate_info_async(question='Explain.', draft=draft, info='Recheck it.')
        assert result == draft and state.status(result) == expected
        assert [r['status'] for r in state.records] == [
            'complete' if first_reason == 'stop' else 'partial',
            'complete' if next_reason == 'stop' else 'partial']
        return result
    with forbid_provider_network():
        await Harness().run(action)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_client_dropping_strict_validation_cannot_claim_external_completion():
    left, client, calls = model_for([envelope()])
    async def drops_contract(prompt, **kwargs):
        return await client.complete(prompt, require_complete=False, allow_continuation=False)
    left._llm_client = SimpleNamespace(complete=drops_contract)
    async def action(state):
        text = await left.generate_answer('Explain.', '')
        assert state.status(text) == 'unverified' and left.estimate_confidence(text) == 0.
        return text
    with forbid_provider_network():
        await Harness().run(action)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unobserved_control_draft_keeps_legacy_confidence_without_fabricating_a_receipt():
    left, _, calls = model_for([])
    async def action(state):
        assert left.estimate_confidence('A frozen control draft.') == .9
        assert not state.observed('A frozen control draft.')
        assert state.summary('A frozen control draft.')['answer_status'] == 'not_observed'
    with forbid_provider_network():
        await Harness().run(action)
    assert not calls


@pytest.mark.asyncio
async def test_provider_default_and_source_generation_do_not_inherit_opt_in_effort(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_ANSWER_REASONING_EFFORT', 'low')
    left, _, calls = model_for([envelope(), envelope()])
    async def action(state):
        assert await left.generate_source_answer('Write the actual source.', '')
        assert not state.records and 'reasoning_effort' not in calls[0]
        monkeypatch.delenv('DUALBRAIN_ANSWER_REASONING_EFFORT')
        text = await left.generate_answer('Explain.', '')
        assert state.status(text) == 'complete' and 'reasoning_effort' not in calls[1]
        return text
    with forbid_provider_network():
        await Harness().run(action)


@pytest.mark.asyncio
async def test_request_timeout_uses_the_remaining_rpc_deadline():
    from core.turn_lifecycle import TurnDispatcher
    left, client, calls = model_for([])
    timeouts = []
    async def post(_url, payload, _headers, **kwargs):
        timeouts.append(kwargs['timeout_seconds'])
        return envelope()
    client._post_json = post
    replies = []
    done = asyncio.Event()
    async def handler(_envelope):
        async def action(state): return await left.generate_answer('Explain.', '')
        value, _ = await Harness().run(action)
        return {'id': 'bounded', 'ok': True, 'result': value}
    def send(reply):
        replies.append(reply)
        done.set()
    with forbid_provider_network():
        dispatcher = TurnDispatcher(handler, send, timeout_seconds=.5)
        dispatcher.submit({'id': 'bounded'})
        await asyncio.wait_for(done.wait(), 2.)
        await dispatcher.close()
    assert replies[0]['ok'] and 0 < timeouts[0] <= .5


@pytest.mark.asyncio
async def test_low_effort_is_scoped_to_ordinary_detail_and_leaves_critic_alone(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_ANSWER_REASONING_EFFORT', 'low')
    left, client, calls = model_for([envelope('A concrete alternative.'),
        envelope(json.dumps({'verdict': 'ok', 'issues': [], 'fixes': []}))])
    right = RightBrainModel(left.llm_config)
    right._llm_client = SimpleNamespace(complete=client.complete)
    with forbid_provider_network():
        detail = await right.deepen('detail', 'Explain the cause.', 'A draft.', None)
        critique = await right.criticise_reasoning('critic', 'Explain the cause.', 'A draft.')
    assert detail['detail_complete'] and critique['critic_external_review_valid']
    assert calls[0]['reasoning_effort'] == 'low' and calls[0]['max_completion_tokens'] == 450
    assert 'reasoning_effort' not in calls[1]


@pytest.mark.asyncio
async def test_overlapping_turns_keep_receipts_and_effort_separate(monkeypatch):
    left, client, calls = model_for([])
    entered, release = asyncio.Event(), asyncio.Event()
    async def post(_url, payload, _headers, **kwargs):
        calls.append(payload)
        if payload['messages'][-1]['content'] == 'first':
            entered.set()
            await release.wait()
            return envelope('Same visible bytes', 'length')
        return envelope('Same visible bytes', 'stop')
    client._post_json = post
    async def action(question, state):
        answer = await left.generate_answer(question, '')
        return state.summary(answer)
    monkeypatch.setenv('DUALBRAIN_ANSWER_REASONING_EFFORT', 'low')
    with forbid_provider_network():
        first = asyncio.create_task(Harness().run(lambda s: action('first', s)))
        await entered.wait()
        monkeypatch.delenv('DUALBRAIN_ANSWER_REASONING_EFFORT')
        second, _ = await Harness().run(lambda s: action('second', s))
        release.set()
        (one, _), = await asyncio.gather(first)
    assert one['answer_status'] == 'partial' and second['answer_status'] == 'complete'
    assert one['stages'][0]['request_reasoning_effort'] == 'low'
    assert second['stages'][0]['request_reasoning_effort'] is None
    assert current_generation() is None and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [{'PRIVATE_PROVIDER_DATA': 'must not be serialized'}, None, ''])
async def test_custom_client_shapes_are_withheld_without_fabricating_a_completion(value):
    left, _, calls = model_for([])
    async def legacy(*args, **kwargs): return value
    left._llm_client = SimpleNamespace(complete=legacy)
    async def action(state):
        answer = await left.generate_answer('Explain.', '')
        assert answer == ''
        assert state.status(answer) == ('empty' if value == '' else 'invalid')
        assert 'PRIVATE_PROVIDER_DATA' not in json.dumps(state.summary(answer))
        return answer
    with forbid_provider_network():
        await Harness().run(action)
    assert not calls
