import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_memory_review as pilot
from test_benchmark_module_gating import fake_response


def fixtures():
    return pilot.bench.read_json(pilot.CALIBRATION), pilot.bench.read_json(pilot.DIALOGUE)


def test_forward_only_fixture_and_registered_controls():
    cal, dialogue = fixtures()
    pilot.validate(cal, dialogue)
    old = pilot.bench.read_json(pilot.boundaries.FIXTURE)
    assert not {c['question'] for c in cal['cases']} & {c['question'] for c in old['cases']}
    env = pilot.environment_contract()
    assert all(env[spec.mode_env] == ('off' if key in {
        'director_advice', 'executive_advice', 'executive_metrics_observation'} else 'auto')
        for key, spec in pilot.DEFAULT_REGISTRY.items())
    assert not any(k.endswith('PG_DSN') for k in env)
    broken = deepcopy(cal)
    broken['cases'][0]['variants']['omission']['expected_obligations'].pop('check')
    with pytest.raises(ValueError):
        pilot.validate(broken, dialogue)


def response_scores(output, expected, obligations, modern):
    scores = {c: {'verdict': expected[c], 'field_path': '/memo', 'quote': output['fields']['memo'],
        'interpretation': 'other', 'basis': pilot.v2.BASIS[c][0]} for c in pilot.v3.CRITERIA}
    if modern:
        scores['critical_coverage']['obligation_checks'] = obligations
    return scores


def test_obligation_audit_validates_coverage_without_overwriting_semantic_verdicts():
    cal, _ = fixtures()
    case = cal['cases'][0]
    variant = case['variants']['omission']
    labels = {'A': {'fields': variant['fields'], 'role': case['role']}}
    scores = response_scores(labels['A'], variant['expected'], variant['expected_obligations'], True)
    parsed = pilot.v3.validate_scores(json.dumps({'scores': {'A': scores}}), case, labels)
    assert parsed['A']['critical_coverage']['verdict'] == 'fail'
    assert parsed['A']['critical_coverage']['context'] == variant['fields']['memo']
    scores['critical_coverage']['verdict'] = 'pass'
    with pytest.raises(ValueError):
        pilot.v3.validate_scores(json.dumps({'scores': {'A': scores}}), case, labels)


async def transport(cal, *, failure=False):
    expected = {json.dumps(v['fields'], sort_keys=True): v for c in cal['cases'] for v in c['variants'].values()}
    calls = []
    async def post(url, payload, headers, **kwargs):
        calls.append(payload)
        if failure:
            raise OSError('PRIVATE_FAILURE')
        packet = json.loads(payload['messages'][-1]['content']) if payload['messages'][-1]['content'].startswith('{') else {}
        if 'outputs' not in packet:
            return await fake_response(url, payload, headers, **kwargs)
        modern = payload['messages'][0]['content'] == pilot.v3.INSTRUCTIONS
        assert not any(k in packet for k in ('expected', 'variants', 'arm', 'order'))
        scores = {}
        for label, output in packet['outputs'].items():
            v = expected[json.dumps(output['fields'], sort_keys=True)]
            scores[label] = response_scores(output, v['expected'], v['expected_obligations'], modern)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'scores': scores})}}],
                'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    return post, calls


def comparable(report):
    return {'calibration': report['calibration'], 'calibration_summary': report['calibration_summary'],
        'rows': [{k: v for k, v in r.items() if k != 'elapsed_ms'} for r in report['rows']],
        'grades': report['grades'], 'usage': report['usage']}


@pytest.mark.asyncio
async def test_retained_production_dialogue_has_exact_zero_transport_replay(tmp_path):
    cal, dialogue = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False, max_continuations=0)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, calls = await transport(cal)
    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, cal, dialogue, config, journal)
    assert live['complete'] and len(live['calibration']) == 16 and len(live['rows']) == 18
    assert all(r['status'] == 'ok' for r in live['rows'])
    assert len(calls) == live['usage']['calls'] <= pilot.CALL_LIMIT
    indexed = {(r['case'], r['arm']): r for r in live['rows']}
    assert indexed['competing', 'memory_bridge']['route']['competition_groups'] == 1
    for arm in pilot.ARMS:
        assert indexed['corrected', arm]['source_qids'] == ['cedar-current']
        assert indexed['expired', arm]['source_qids'] == ['moss-owner']  # fake checker deliberately approves every record
        for turn in dialogue['turns']:
            assert indexed[turn['id'], arm]['draft_sha256'] == pilot.bench.digest(indexed[turn['id'], 'draft']['answer'])
    replay_journal = pilot.bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=pilot.CALL_LIMIT)
    replay_journal.transport = None
    replay = await pilot.run(args, cal, dialogue, config, replay_journal)
    assert replay['live_provider_calls'] == 0 and replay_journal.counts == journal.counts
    assert comparable(replay) == comparable(live)
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and '"quote"' not in public and '"context"' not in public
    assert all(c['question'] not in public for c in cal['cases'])


@pytest.mark.asyncio
async def test_transport_failure_retains_all_planned_missing_cells_without_retry(tmp_path):
    cal, dialogue = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, calls = await transport(cal, failure=True)
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), cal, dialogue, config, journal)
    assert len(calls) == report['usage']['calls'] == 1
    assert len(report['calibration']) == 16 and len(report['rows']) == 18
    assert report['usage']['unknown_usage_calls'] == 1
    assert all(r['status'] == 'unavailable' for r in report['rows'])
    assert 'PRIVATE_FAILURE' not in json.dumps(pilot.public_summary(report))
