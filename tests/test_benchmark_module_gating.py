import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_module_gating as bench
from core.llm_client import LLMConfig


def test_profiles_clear_inherited_controls_and_keep_caller_modes():
    case = bench.read_json(bench.FIXTURE)['cases'][0]
    inherited = {'DUALBRAIN_MODULE_EXECUTIVE_ADVICE_MODE': 'off', 'DUALBRAIN_PG_DSN': 'PRIVATE',
                 'DUALBRAIN_ACC_OVERRIDE_CONSULT': '1', 'LLM_API_KEY': 'PRIVATE_KEY'}
    baseline = bench.environment(case, 'legacy_default', inherited)
    automatic = bench.environment(case, 'all_auto', inherited)
    assert 'DUALBRAIN_PG_DSN' not in automatic
    assert 'DUALBRAIN_ACC_OVERRIDE_CONSULT' not in baseline
    assert all(spec.mode_env not in baseline for spec in bench.DEFAULT_REGISTRY.values())
    assert all(automatic[spec.mode_env] == 'auto' for spec in bench.DEFAULT_REGISTRY.values())
    assert case['executive_mode'] == 'observe' and case['executive_observer_mode'] == 'both'


def test_fixture_is_complete_and_separates_injected_controls():
    fixture = bench.read_json(bench.FIXTURE)
    bench.validate(fixture)
    assert len({c['problem_id'] for c in fixture['cases']}) == 5
    fixture['cases'].pop()
    with pytest.raises(ValueError):
        bench.validate(fixture)


@pytest.mark.asyncio
async def test_cancelled_attempt_is_retained_unknown_and_replays_without_transport():
    config = LLMConfig('openai', 'test', 'PRIVATE_KEY')
    journal = bench.Journal(config, {}, max_calls=2)

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    journal.transport._post_json = cancelled
    with pytest.raises(asyncio.CancelledError):
        await journal.post('director', 'https://example.invalid', {}, {})
    assert not journal.data.get('transport_blocked')
    assert bench.usage(journal.events)['unknown_usage_calls'] == 1
    assert journal.events[0]['error_type'] == 'CancelledError'
    journal.finish()
    replay = bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=2)
    replay.transport = None
    with pytest.raises(asyncio.CancelledError):
        await replay.post('director', 'https://example.invalid', {}, {})
    replay.finish()
    assert replay.calls == 0 and replay.missing == []


@pytest.mark.asyncio
async def test_transport_failure_blocks_new_paid_requests():
    journal = bench.Journal(LLMConfig('openai', 'test', 'PRIVATE_KEY'), {}, max_calls=2)

    async def broken(*args, **kwargs):
        raise RuntimeError('PRIVATE_ERROR')

    journal.transport._post_json = broken
    with pytest.raises(RuntimeError):
        await journal.post('left', 'https://example.invalid', {}, {})
    with pytest.raises(RuntimeError):
        await journal.post('right', 'https://example.invalid', {}, {})
    assert journal.calls == 1 and len(journal.events) == 1
    assert 'PRIVATE_ERROR' not in json.dumps(journal.data)


async def fake_response(url, payload, headers, **kwargs):
    messages = payload['messages']
    try:
        packet = json.loads(messages[-1]['content'])
    except (ValueError, TypeError):
        packet = {}
    system = messages[0]['content']
    if 'answers' in packet:
        content = {'scores': {label: {criterion: {'verdict': 'pass', 'quote': answer[:1]}
                                    for criterion in bench.base.CRITERIA}
                              for label, answer in packet['answers'].items()}}
    elif 'records' in packet:
        content = {'decisions': [{'id': r['id'], 'supported': True, 'quote': r['answer']}
                                 for r in packet['records']]}
    elif 'Executive Reasoning' in system:
        content = {'memo': 'Advice.', 'directives': {}, 'mix_in': '', 'confidence': .9}
    elif 'Director' in system:
        content = {'memo': 'Direct.', 'control': {'consult': 'auto'}, 'confidence': .9}
    elif 'critic' in system.lower():
        content = {'verdict': 'ok', 'issues': [], 'fixes': [], 'critic_sum': 'No issues.', 'confidence_r': .9}
    else:
        content = 'Hello.'
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': content if isinstance(content, str) else json.dumps(content)}}],
            'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}


@pytest.mark.asyncio
async def test_full_production_pilot_and_exact_replay_are_network_free_with_fake_transport(tmp_path):
    fixture = bench.read_json(bench.FIXTURE)
    config = LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False, max_continuations=0)
    journal = bench.Journal(config, {}, max_calls=96)
    journal.transport._post_json = fake_response
    live = await bench.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), fixture, config, journal)
    assert live['complete'] and live['summaries']['grader_failures'] == 0
    assert live['summaries']['jointly_graded_pairs'] == 8
    replay_journal = bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=96)
    replay_journal.transport = None
    replay = await bench.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), fixture, config, replay_journal)
    assert replay['complete'] and replay['provider_calls'] == 0
    assert [(r['case'], r['arm'], r['status'], r['answer']) for r in live['rows']] == [
        (r['case'], r['arm'], r['status'], r['answer']) for r in replay['rows']]
    assert live['grades'] == replay['grades']
    public = json.dumps(bench.public_summary(live))
    assert 'PRIVATE_KEY' not in public and 'User message:' not in public
