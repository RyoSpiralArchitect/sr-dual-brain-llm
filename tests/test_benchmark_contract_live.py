"""Measurement failures must not be laundered by the live acceptance harness."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_contract_live as pilot
from core.llm_client import CompletionValidationError, LLMConfig
from offline_provider_guard import forbid_provider_network


def config():
    return LLMConfig('openai', 'fake-contract-live', 'FAKE_NO_CREDENTIAL',
                     api_base='https://fake.invalid/v1', max_output_tokens=2048,
                     auto_continue=False, max_continuations=0, timeout_seconds=60)


def response(text='Complete response', finish='stop', usage=True):
    result = {'choices': [{'message': {'content': text}, 'finish_reason': finish}], 'model': config().model}
    if usage:
        result['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5}
    return result


class Transport:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), 0

    async def _post_json(self, *_args, **_kwargs):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)


@pytest.mark.asyncio
@pytest.mark.parametrize('text,finish', [('', 'stop'), ('Partial', 'length'), ('Refused', 'content_filter')])
async def test_invalid_returned_envelope_reaches_production_validation(text, finish):
    journal = pilot.RawJournal(config(), {'test': 'raw'})
    envelope = response(text, finish)
    journal.transport = Transport([envelope])
    client = pilot.base.RecordedClient(config(), journal, 'right')
    with forbid_provider_network(), pytest.raises(CompletionValidationError):
        await client.complete('Check this draft.', require_complete=True)
    journal.finish()
    assert journal.data['occurrences'][0]['response'] == envelope
    assert 'error_type' not in journal.data['occurrences'][0]
    assert client.failure_types == ['CompletionValidationError']
    assert pilot.bench.usage(journal.events)['known_input_tokens'] == 10
    assert not journal.blocked


@pytest.mark.asyncio
async def test_duplicate_seed_attempts_are_costed_and_replayed_in_exact_order():
    journal = pilot.RawJournal(config(), {'test': 'occurrences'}, max_calls=2)
    journal.transport = Transport([response('first'), response('second', usage=False)])
    with forbid_provider_network():
        for _ in range(2):
            await journal.post('seed', 'https://fake.invalid', {'same': True}, {'Authorization': 'PRIVATE_KEY'})
        with pytest.raises(pilot.AdmissionError):
            await journal.post('seed', 'https://fake.invalid', {'same': True}, {})
    journal.finish()
    assert journal.calls == 2 and len(journal.events) == 2
    usage = pilot.bench.usage(journal.events)
    assert usage['unknown_usage_calls'] == 1 and usage['input_tokens'] is None
    assert 'PRIVATE_KEY' not in json.dumps(journal.data)
    # The replay cap remains the same and must deny exactly the same third request.
    replay = pilot.RawJournal(config(), {'test': 'occurrences'}, frozen=deepcopy(journal.data), max_calls=2)
    assert replay.transport is None
    with forbid_provider_network():
        answers = [await replay.post('seed', 'https://fake.invalid', {'same': True}, {}) for _ in range(2)]
        with pytest.raises(pilot.AdmissionError):
            await replay.post('seed', 'https://fake.invalid', {'same': True}, {})
    replay.finish()
    assert [a['choices'][0]['message']['content'] for a in answers] == ['first', 'second']
    assert replay.calls == 0 and pilot.bench.usage(replay.events) == usage


@pytest.mark.asyncio
async def test_transport_failure_closes_paid_admission_and_usage_stays_unknown():
    journal = pilot.RawJournal(config(), {'test': 'failure'})
    transport = Transport([RuntimeError('PRIVATE_ERROR')])
    journal.transport = transport
    with forbid_provider_network():
        with pytest.raises(RuntimeError):
            await journal.post('right', 'https://fake.invalid', {}, {})
        for _ in range(3):
            with pytest.raises(pilot.AdmissionError):
                await journal.post('right', 'https://fake.invalid', {}, {})
    journal.finish()
    assert transport.calls == journal.calls == 1
    assert len(journal.data['denied_admissions']) == 3
    assert pilot.bench.usage(journal.events)['unknown_usage_calls'] == 1
    assert 'PRIVATE_ERROR' not in json.dumps(journal.data)


@pytest.mark.asyncio
async def test_recorded_aiohttp_oserror_replays_as_sticky_transport_failure():
    import aiohttp
    journal = pilot.RawJournal(config(), {'test': 'aiohttp_oserror'})
    journal.transport = Transport([aiohttp.ClientOSError(1, 'PRIVATE_NETWORK_ERROR')])
    with forbid_provider_network():
        with pytest.raises(aiohttp.ClientOSError):
            await journal.post('draft', 'https://fake.invalid', {'stage': 'draft'}, {})
    journal.finish()
    assert journal.data['occurrences'][0]['error_type'] == 'ClientOSError'
    replay = pilot.RawJournal(config(), {'test': 'aiohttp_oserror'}, frozen=deepcopy(journal.data))
    with forbid_provider_network() as audit:
        with pytest.raises(OSError):
            await replay.post('draft', 'https://fake.invalid', {'stage': 'draft'}, {})
    replay.finish()
    assert audit.blocked_requests == 0
    assert replay.data['occurrences'][0]['error_type'] == 'ClientOSError'
    assert replay.data['occurrences'][0]['sha256'] == journal.data['occurrences'][0]['sha256']
    divergent = pilot.RawJournal(config(), {'test': 'aiohttp_oserror'}, frozen=deepcopy(journal.data))
    with forbid_provider_network():
        with pytest.raises(OSError):
            await divergent.post('draft', 'https://fake.invalid', {'stage': 'draft'}, {})
        with pytest.raises(pilot.AdmissionError):
            await divergent.post('draft', 'https://fake.invalid', {'stage': 'draft'}, {})
    # The second request was not present on the live tape, so exact replay
    # correctly rejects its extra denial instead of claiming coverage.
    with pytest.raises(ValueError, match='coverage'):
        divergent.finish()


@pytest.mark.asyncio
async def test_replay_rejects_tampering_order_changes_and_unused_occurrences():
    journal = pilot.RawJournal(config(), {'test': 'integrity'})
    journal.transport = Transport([response('first'), response('second')])
    for key in (1, 2):
        await journal.post('left', 'https://fake.invalid', {'key': key}, {})
    journal.finish()
    frozen = deepcopy(journal.data)
    frozen['occurrences'][0]['response'] = response('tampered')
    with pytest.raises(ValueError, match='Corrupt'):
        pilot.RawJournal(config(), {'test': 'integrity'}, frozen=frozen)
    replay = pilot.RawJournal(config(), {'test': 'integrity'}, frozen=deepcopy(journal.data))
    with pytest.raises(pilot.AdmissionError):
        await replay.post('left', 'https://fake.invalid', {'key': 2}, {})
    with pytest.raises(ValueError, match='coverage'):
        replay.finish()
    replay = pilot.RawJournal(config(), {'test': 'integrity'}, frozen=deepcopy(journal.data))
    await replay.post('left', 'https://fake.invalid', {'key': 1}, {})
    with pytest.raises(ValueError, match='coverage'):
        replay.finish()


@pytest.mark.asyncio
async def test_overlapping_http_is_journaled_by_admission_not_completion():
    release, entered = asyncio.Event(), asyncio.Event()
    class Overlap:
        async def _post_json(self, _url, payload, _headers):
            if payload['key'] == 1:
                entered.set()
                await release.wait()
            else:
                release.set()
            return response(str(payload['key']))
    journal = pilot.RawJournal(config(), {'test': 'overlap'})
    journal.transport = Overlap()
    first = asyncio.create_task(journal.post('left', 'fake', {'key': 1}, {}))
    await entered.wait()
    second = asyncio.create_task(journal.post('right', 'fake', {'key': 2}, {}))
    await asyncio.gather(first, second)
    journal.finish()
    assert [r['request']['payload']['key'] for r in journal.data['occurrences']] == [1, 2]


class EngineTransport:
    """Test-only Fake; no real-model quality/cost claim is made from this."""
    async def _post_json(self, _url, payload, _headers, **_kwargs):
        messages = payload['messages']
        try:
            packet = json.loads(messages[-1]['content'])
        except ValueError:
            packet = {}
        if 'records' in packet:
            ambiguous = 'Project Larch' in packet.get('question', '')
            content = json.dumps({'decisions': [{'id': r['id'], 'supported': not ambiguous,
                'quote': '' if ambiguous else r['answer']} for r in packet['records']]})
        elif 'critic' in messages[0]['content'].lower():
            content = json.dumps({'verdict': 'ok', 'issues': [], 'fixes': [], 'confidence_r': .9})
        else:
            content = 'Please supply the current source or clarify the requested measure.'
            for line in messages[0]['content'].splitlines():
                if line.startswith('{"source_question":'):
                    content = json.loads(line)['recorded_answer']
                    break
            if 'record its data retention period as 11 days' in messages[-1]['content']:
                content = 'Data retention period: 11 days.'
            elif 'period is now 7 days' in messages[-1]['content']:
                content = 'Data retention period: 7 days.'
        return response(content)


@pytest.mark.asyncio
async def test_full_engine_pilot_and_exact_replay_use_isolated_real_store(tmp_path):
    dsn = os.environ.get('DUALBRAIN_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Requires an isolated PostgreSQL test DSN')
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    first, second = tmp_path / 'first', tmp_path / 'replay'
    first.mkdir()
    second.mkdir()
    with patch.dict(os.environ, {'DUALBRAIN_MEMORY_EMBEDDING_MODEL': '', 'DUALBRAIN_MEMORY_ANSWERABILITY_MODEL': ''}):
        journal = pilot.RawJournal(config(), pilot.metadata(fixture, config()))
        journal.transport = EngineTransport()
        with forbid_provider_network():
            result = await pilot.run(first, fixture, config(), journal, dsn=dsn)
        assert result['complete'] and len(result['rows']) == 15
        assert all(r['system2']['consistent'] for r in result['rows'])
        assert all(r['contracts']['durable_commit'] for r in result['rows'][:5])
        assert all(r['diagnostics']['exact_match']['verdict'] == 'pass' for r in result['rows'][:5] if 'quote' in r['id'])
        replay_journal = pilot.RawJournal(config(), pilot.metadata(fixture, config()), frozen=deepcopy(journal.data))
        with forbid_provider_network():
            replay = await pilot.run(second, fixture, config(), replay_journal, dsn=dsn)
        assert replay_journal.calls == 0 and replay_journal.transport is None
        assert pilot.canonical(replay) == pilot.canonical(result)
        public = json.dumps(pilot.public_summary(result))
        assert all(value not in public for value in ('FAKE_NO_CREDENTIAL', 'NONEXPOSED_CURRENT', '"answer"', '"available_context"'))


@pytest.mark.asyncio
async def test_invalid_critic_is_separate_from_available_final_answer(tmp_path):
    class EmptyCritic(EngineTransport):
        async def _post_json(self, url, payload, headers, **kwargs):
            if 'critic' in payload['messages'][0]['content'].lower():
                return response('')
            return await super()._post_json(url, payload, headers, **kwargs)
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    case = next(c for c in fixture['cases'] if c['id'] == 'critic_on')
    journal = pilot.RawJournal(config(), {'test': 'stage_availability'})
    journal.transport = EmptyCritic()
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, pilot.profile.environment(), clear=True), forbid_provider_network():
        session, clients = await pilot.make_session('invalid-critic', None, config(), journal, [fixture['clock']])
        try:
            row = await pilot.turn(session, clients, case, journal, options)
        finally:
            await session.close()
    assert row['status'] == 'ok' and row['answer']
    assert not row['stages_available']
    assert row['system2']['consistent']
    assert row['system2']['external_review_completed'] is False
    assert row['system2']['attempts'][0]['validation']['initial']['status'] == 'empty_response'
    assert any('CompletionValidationError' in entry['types'] for entry in row['client_failures'])
