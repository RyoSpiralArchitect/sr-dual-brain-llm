import asyncio

import pytest

from core.llm_client import LLMClient, LLMConfig


@pytest.mark.parametrize('provider,model', [('openai', 'gpt-6-luna'), ('anthropic', 'claude-fixture')])
@pytest.mark.asyncio
async def test_bounded_request_does_not_continue_or_change_concurrent_full_request(monkeypatch, provider, model):
    config = LLMConfig(provider, model, 'OFFLINE_ONLY', max_output_tokens=1024,
                       auto_continue=True, max_continuations=2)
    client = LLMClient(config)
    entered = set()
    both_entered = asyncio.Event()
    calls = {'full': [], 'bounded': []}

    async def post(_url, payload, _headers, **_kwargs):
        role = payload['messages'][0]['content']
        calls[role].append(payload)
        entered.add(role)
        if len(entered) == 2:
            both_entered.set()
        await both_entered.wait()
        first = len(calls[role]) == 1
        text = 'Partial' if first else ' complete.'
        if provider == 'anthropic':
            return {'content': [{'type': 'text', 'text': text}],
                    'stop_reason': 'max_tokens' if first else 'end_turn'}
        return {'choices': [{'message': {'content': text},
                             'finish_reason': 'length' if first else 'stop'}]}

    monkeypatch.setattr(client, '_post_json', post)
    bounded, full = await asyncio.wait_for(asyncio.gather(
        client.complete('bounded', max_output_tokens=256, allow_continuation=False, require_complete=True),
        client.complete('full', require_complete=True),
        return_exceptions=True,
    ), timeout=1.)

    assert isinstance(bounded, RuntimeError)
    assert 'successful terminal completion' in str(bounded)
    assert full == 'Partial complete.'
    assert len(calls['bounded']) == 1 and len(calls['full']) == 2
    cap = 'max_completion_tokens' if provider == 'openai' else 'max_tokens'
    assert calls['bounded'][0][cap] == 256
    assert all(p[cap] == 1024 for p in calls['full'])
    if provider == 'openai':
        assert all('max_tokens' not in p and 'temperature' not in p for p in calls['bounded'] + calls['full'])
    assert (config.max_output_tokens, config.auto_continue, config.max_continuations) == (1024, True, 2)


@pytest.mark.parametrize('provider,model', [('openai', 'gpt-6-luna'), ('anthropic', 'claude-fixture')])
@pytest.mark.asyncio
async def test_bounded_request_accepts_a_complete_response_in_one_attempt(monkeypatch, provider, model):
    client = LLMClient(LLMConfig(provider, model, 'OFFLINE_ONLY', auto_continue=True))
    calls = []

    async def post(_url, payload, _headers, **_kwargs):
        calls.append(payload)
        if provider == 'anthropic':
            return {'content': [{'type': 'text', 'text': 'Useful guidance.'}], 'stop_reason': 'end_turn'}
        return {'choices': [{'message': {'content': 'Useful guidance.'}, 'finish_reason': 'stop'}]}

    monkeypatch.setattr(client, '_post_json', post)
    assert await client.complete('question', max_output_tokens=256, allow_continuation=False,
                                 require_complete=True) == 'Useful guidance.'
    assert len(calls) == 1
