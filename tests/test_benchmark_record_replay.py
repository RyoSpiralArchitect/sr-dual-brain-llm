from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import os

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_record_replay as pilot
from test_benchmark_module_gating import fake_response


def fixtures():
    return pilot.bench.read_json(pilot.CALIBRATION), pilot.bench.read_json(pilot.FIXTURE)


def test_frozen_fixture_and_blind_packets_keep_expected_values_out():
    cal, fixture = fixtures()
    pilot.validate(cal, fixture)
    for case in cal['cases']:
        a, na = pilot.labels_for(case, 0)
        _, nb = pilot.labels_for(case, 1)
        assert list(na.values()) == list(nb.values())[::-1]
        packet = pilot.assessment.packet(case, a)
        assert set(packet) == {'question', 'reference', 'criteria', 'requested_operation', 'available_context', 'answers'}
        assert 'expected' not in json.dumps(packet) and 'variants' not in packet
    broken = deepcopy(fixture)
    broken['support_repeats'] = 4
    with pytest.raises(ValueError):
        pilot.validate(cal, broken)


async def make_transport(cal, *, failure=False, deny_first=False, copy_replay=False):
    calls = []
    expected = {v['answer']: v['expected'] for c in cal['cases'] for v in c['variants'].values()}
    seen = {}
    async def post(url, payload, headers, **kwargs):
        calls.append(payload)
        if failure:
            raise OSError('PRIVATE_FAILURE')
        try:
            packet = json.loads(payload['messages'][-1]['content'])
        except ValueError:
            packet = {}
        if 'records' in packet:
            key = json.dumps(packet, sort_keys=True)
            seen[key] = seen.get(key, 0) + 1
            deny = deny_first and seen[key] == 1
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                'decisions': [{'id': r['id'], 'supported': not deny,
                    'quote': r['answer'] if not deny else ''} for r in packet['records']]})}}],
                'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        if 'answers' in packet:
            scores = {label: {c: {'verdict': expected.get(answer, {}).get(c, 'pass'), 'quote': answer[:1]}
                for c in pilot.bench.base.CRITERIA} for label, answer in packet['answers'].items()}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'scores': scores})}}],
                'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        if copy_replay:
            for line in payload['messages'][0]['content'].splitlines():
                if line.startswith('{"source_question":'):
                    recorded = json.loads(line)['recorded_answer']
                    return {'choices': [{'finish_reason': 'stop', 'message': {'content': recorded}}],
                        'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        return await fake_response(url, payload, headers, **kwargs)
    return post, calls


def comparable(r):
    return {key: [{k: v for k, v in row.items() if k != 'elapsed_ms'} for row in r[key]]
            for key in ('calibration', 'support', 'rows', 'grades')} | {
        'calibration_summary': r['calibration_summary'], 'usage': r['usage']}


@pytest.mark.asyncio
async def test_actual_sessions_reuse_exact_repeat_zero_receipts_and_replay_without_network(tmp_path):
    cal, fixture = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False, max_continuations=0)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, calls = await make_transport(cal)
    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, cal, fixture, config, journal)
    assert live['complete'] and len(live['calibration']) == 16 and len(live['support']) == 36
    assert len(live['rows']) == 8 and all(row['status'] == 'ok' for row in live['rows'])
    for row in live['rows']:
        assert not any(role.startswith('support') for role in row['calls_by_role'])
        assert len(row['receipt_keys']) == 1
        if row['case'] == 'corrected_followup':
            assert row['source_qids'] == ['alder-current']
    indexed = {(r['case'], r['arm'], r['repeat']): r for r in live['support']}
    assert indexed['corrected_followup', 'replay_contract', 0]['intent']
    assert not indexed['extra_clause', 'replay_contract', 0]['intent']
    assert not indexed['incomplete', 'replay_contract', 0]['intent']
    assert indexed['incomplete', 'replay_contract', 0]['usage']['calls'] == 0
    replay_journal = pilot.bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=pilot.CALL_LIMIT)
    replay_journal.transport = None
    replay = await pilot.run(args, cal, fixture, config, replay_journal)
    assert replay['live_provider_calls'] == 0 and replay_journal.counts == journal.counts
    assert comparable(replay) == comparable(live)
    assert len(calls) == live['usage']['calls'] <= pilot.CALL_LIMIT
    public = json.dumps(pilot.public_summary(live))
    assert not any(key in public for key in ('PRIVATE_KEY', '"quote"', '"answer"', '"receipt"', '"cache"'))


@pytest.mark.asyncio
async def test_repeat_zero_denial_is_not_replaced_by_later_approval(tmp_path):
    cal, fixture = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, _ = await make_transport(cal, deny_first=True)
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), cal, fixture, config, journal)
    for row in report['rows']:
        first = next(s for s in report['support'] if s['case'] == row['case']
                     and s['arm'] == row['arm'] and s['repeat'] == 0)
        assert row['source_qids'] == first['source_qids']
        if row['case'] in {'explicit_quote', 'corrected_followup', 'scope_switch'}:
            assert not row['source_qids']
    assert any(row['source_qids'] for row in report['support'] if row['repeat'] > 0)


@pytest.mark.asyncio
async def test_transport_failure_preserves_all_planned_cells_and_unknown_usage(tmp_path):
    cal, fixture = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, calls = await make_transport(cal, failure=True)
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), cal, fixture, config, journal)
    assert len(calls) == report['usage']['calls'] == 1
    assert report['usage']['unknown_usage_calls'] == 1
    assert len(report['support']) == 36 and len(report['rows']) == 8
    assert all(row['status'] == 'unavailable' for row in report['rows'])
    assert 'PRIVATE_FAILURE' not in json.dumps(pilot.public_summary(report))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['auto', 'on', 'off'])
async def test_real_process_literal_generation_keeps_manual_system2_authority(tmp_path, mode):
    cal, fixture = fixtures()
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, _ = await make_transport(cal, copy_replay=True)
    clock = [fixture['clock']]
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(os.environ, pilot.previous.environment(), clear=True), patch('time.time', lambda: clock[0]):
        case = fixture['cases'][0]
        measured = await pilot.support_cell(case, pilot.ARMS[1], 0, config, journal, clock)
        row = await pilot.runtime_cell(case, pilot.ARMS[1], measured['receipt'], config, journal, clock,
                                       options, system2_mode=mode)
    assert row['status'] == 'ok' and row['source_qids'] == ['juniper']
    if mode == 'auto':
        assert row['exact_approved_answer'] is True
        assert row['route']['selective']['reason'] == 'supported_literal_replay'
        assert row['calls_by_role'] == {'left': 1}
    else:
        assert row['route']['selective'] is None
        assert row['route']['system2'] is (mode == 'on')
        if mode == 'on':
            assert row['calls_by_role']['right'] >= 1
