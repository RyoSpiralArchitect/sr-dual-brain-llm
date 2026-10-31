"""Rubric integrity and failure-boundary checks without provider calls."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_persistent_answers as bench


def good_grades():
    return {'scores': {label: {criterion: {'verdict': 'pass', 'quote': answer}
                             for criterion in bench.CRITERIA} for label, answer in {'A': '記録なし', 'B': '7日'}.items()}}


@pytest.mark.parametrize('fault', ['missing-label', 'extra-field', 'nested-label', 'wrong-quote', 'duplicate'])
def test_entire_grader_packet_rejected_on_invalid_structure_or_evidence(fault):
    packet = good_grades()
    if fault == 'missing-label': del packet['scores']['B']
    if fault == 'extra-field': packet['scores']['A']['task']['other'] = True
    if fault == 'nested-label': packet['scores']['A']['B'] = packet['scores'].pop('B')
    if fault == 'wrong-quote': packet['scores']['B']['task']['quote'] = 'invented quote'
    raw = json.dumps(packet, ensure_ascii=False)
    if fault == 'duplicate': raw = raw.replace('"verdict": "pass"', '"verdict": "pass", "verdict": "fail"', 1)
    with pytest.raises(ValueError):
        bench.validate_grades(raw, {'A': '記録なし', 'B': '7日'})


def test_schema_and_parser_require_all_labels_and_distinguish_criteria():
    schema = bench.grade_schema(['A', 'B'])['json_schema']
    assert schema['strict']
    assert schema['schema']['properties']['scores']['required'] == ['A', 'B']
    packet = good_grades()
    packet['scores']['A']['clarification']['verdict'] = 'fail'
    result = bench.validate_grades(json.dumps(packet), {'A': '記録なし', 'B': '7日'})
    assert result['A']['task']['verdict'] == 'pass' and result['A']['clarification']['verdict'] == 'fail'


@pytest.mark.asyncio
async def test_hidden_provider_fallback_aborts_before_scoring_or_a_later_turn(monkeypatch):
    client = SimpleNamespace(failed=0)
    async def fallback(*args, **kwargs):
        client.failed += 1
        return {'answer': 'heuristic fallback'}
    monkeypatch.setattr(bench, '_handle_process', fallback)
    with pytest.raises(RuntimeError, match='episode aborted'):
        await bench.process(None, [client], {}, SimpleNamespace(events=[]), {})


def test_prospective_fixture_contains_disjoint_scopes_and_both_reference_cases():
    fixture = bench.read_json(bench.FIXTURE)
    bench.validate_fixture(fixture)
    scopes = {}
    for ep in fixture['episodes']:
        scopes.setdefault(ep['split'], set()).update(w['memory_context']['id'] for w in ep['writes'])
        assert {p['id'] for p in ep['probes']} >= {'named-unknown', 'ambiguous', 'cold-conditions'}
        assert ep['probes'][-1]['restart_before']
    assert scopes['development'].isdisjoint(scopes['evaluation'])


@pytest.mark.parametrize('phase,selected', [
    ('evaluation', ['conditions']), ('development', ['unknown-probe']),
    ('development', ['conditions', 'conditions']), ('development', []),
])
def test_diagnostic_selection_cannot_filter_evaluation_or_silently_lose_cases(phase, selected):
    fixture = bench.read_json(bench.FIXTURE)
    with pytest.raises(ValueError):
        bench.development_probe_selection(SimpleNamespace(phase=phase, development_probes=selected), fixture)


@pytest.mark.asyncio
async def test_budget_rejection_cannot_corrupt_a_prior_identical_response():
    from core.llm_client import LLMConfig
    journal = bench.Journal(LLMConfig('openai', 'test', 'PRIVATE_KEY'), {}, max_calls=1)
    async def post(*args, **kwargs):
        return {'model': 'test', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
    journal.transport._post_json = post
    await journal.post('left', 'https://example.invalid', {}, {})
    original = deepcopy(journal.data)
    with pytest.raises(ValueError):
        await journal.post('left', 'https://example.invalid', {}, {})
    assert journal.data == original and journal.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('cancelled', [False, True])
async def test_transport_failure_records_only_safe_status_and_unknown_usage(cancelled):
    from core.llm_client import LLMConfig
    journal = bench.Journal(LLMConfig('openai', 'test', 'PRIVATE_KEY'), {})
    async def post(*args, **kwargs):
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("openai returned 429: {'error': {'type': 'insufficient_quota', 'message': 'PRIVATE_KEY'}}")
    journal.transport._post_json = post
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        await journal.post('left', 'https://example.invalid', {}, {})
    row = next(iter(journal.data['responses'].values()))[0]
    assert row['error_type'] == ('CancelledError' if cancelled else 'RuntimeError')
    if not cancelled:
        assert row['http_status'] == 429 and row['provider_error_category'] == 'insufficient_quota'
    assert journal.input_tokens is None and journal.output_tokens is None
    assert 'PRIVATE_KEY' not in json.dumps(journal.data)


@pytest.mark.asyncio
async def test_first_provider_failure_blocks_further_posts_in_the_run():
    from core.llm_client import LLMConfig
    journal = bench.Journal(LLMConfig('openai', 'test', 'PRIVATE_KEY'), {})
    calls = []
    async def post(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("openai returned 429: {'error': {'type': 'insufficient_quota'}}")
    journal.transport._post_json = post
    with pytest.raises(RuntimeError):
        await journal.post('support', 'https://example.invalid', {'messages': [{'content': '{"records": []}'}]}, {})
    with pytest.raises(RuntimeError, match='blocked'):
        await journal.post('left', 'https://example.invalid', {'different': True}, {})
    assert len(calls) == journal.calls == 1


@pytest.mark.asyncio
async def test_frozen_failure_blocks_its_turn_even_if_same_key_later_succeeds():
    from core.llm_client import LLMConfig
    config=LLMConfig('openai','test','PRIVATE_KEY')
    request={'role':'right','endpoint_sha256':__import__('hashlib').sha256(b'https://example.invalid').hexdigest(),'payload':{}}
    error={'request':request,'error_type':'ValueError','response':{'choices':[{'finish_reason':'stop','message':{'content':''}}]}}
    success={'request':request,'response':{'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}}
    journal=bench.Journal(config,{},frozen={'complete':True,'responses':{bench.digest(request):[error,success]}})
    with pytest.raises(RuntimeError):
        await journal.post('right','https://example.invalid',{}, {})
    assert journal.data['transport_blocked']
    journal.data['transport_blocked']=False  # A new, independently planned cell.
    assert (await journal.post('right','https://example.invalid',{}, {}))['choices'][0]['message']['content']=='ok'
    journal.finish()
    assert journal.calls==0


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', bench.EXTERNAL_BACKEND_FLAGS)
@pytest.mark.parametrize('seed', [False, True])
async def test_external_backends_are_rejected_before_any_seed_or_probe_session(flag, seed, monkeypatch):
    monkeypatch.setenv(flag, 'untracked-backend')
    called = []
    async def create(*args, **kwargs):
        called.append(True)
        pytest.fail('Session factory reached an unwrapped backend')
    monkeypatch.setattr(bench.EngineSession, 'create', create)
    with pytest.raises(ValueError, match=flag):
        await bench.session_for('guarded', None, None, None, 'left', [0.], seed=seed)
    assert not called


@pytest.mark.asyncio
async def test_frozen_run_rejects_embedding_selector_before_loading_or_replaying(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MEMORY_EMBEDDING_MODEL', 'untracked-backend')
    with pytest.raises(ValueError, match='DUALBRAIN_MEMORY_EMBEDDING_MODEL'):
        await bench.run(None, None, None, None)
