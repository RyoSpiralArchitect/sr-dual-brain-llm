import asyncio
import http.client
from pathlib import Path
import sys
from urllib import request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
from offline_provider_guard import ProviderNetworkBlocked, forbid_provider_network
from core.llm_client import LLMClient, LLMConfig


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_guard_blocks_production_json_and_stream_before_sending_even_if_caught(stream):
    config = LLMConfig('openai', 'offline-test', 'PRIVATE_SENTINEL',
        api_base='https://fake.invalid/v1', auto_continue=False)
    client = LLMClient(config)
    with pytest.raises(ProviderNetworkBlocked, match='blocked 1') as exc:
        with forbid_provider_network() as audit:
            try:
                await (client.complete_stream if stream else client.complete)('PRIVATE_PROMPT')
            except ProviderNetworkBlocked:
                pass  # Production fallback must not conceal an attempted HTTP request.
            assert audit.blocked_requests == 1
    assert 'PRIVATE' not in str(exc.value) and 'fake.invalid' not in str(exc.value)


@pytest.mark.parametrize('transport', ['urllib', 'http'])
def test_common_sync_http_entry_is_also_blocked(transport):
    with pytest.raises(ProviderNetworkBlocked, match='blocked 1'):
        with forbid_provider_network():
            if transport == 'urllib':
                request.urlopen('https://fake.invalid/PRIVATE')
            else:
                http.client.HTTPSConnection('fake.invalid').request('GET', '/PRIVATE')


@pytest.mark.asyncio
async def test_fake_transport_and_drained_late_request_have_distinct_outcomes():
    client = LLMClient(LLMConfig('openai', 'fake', 'NOT_A_KEY'))
    async def fake(*_args, **_kwargs):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'scripted'}}]}
    client._post_json = fake
    with forbid_provider_network() as audit:
        assert await client.complete('question', require_complete=True) == 'scripted'
    assert audit.blocked_requests == 0
    with pytest.raises(ProviderNetworkBlocked, match='blocked 1'):
        with forbid_provider_network():
            late = asyncio.create_task(LLMClient(client.config).complete_stream('late'))
            await asyncio.gather(late, return_exceptions=True)


def test_guard_restores_transports_and_preserves_unrelated_errors():
    original = request.urlopen
    with pytest.raises(ValueError, match='fixture failure'):
        with forbid_provider_network():
            raise ValueError('fixture failure')
    assert request.urlopen is original
