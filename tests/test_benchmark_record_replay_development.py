from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_record_replay_development as pilot


def setup_journal(*, frozen=None):
    fixture, config = pilot.bench.read_json(pilot.FIXTURE), pilot.configuration()
    journal = pilot.bench.Journal(config, pilot.metadata(fixture, config), frozen=frozen,
                                  max_calls=pilot.CALL_LIMIT)
    provider = None if frozen else pilot.FakeProvider()
    journal.transport = provider
    return fixture, config, journal, provider


@pytest.mark.asyncio
async def test_production_controls_keep_three_support_occurrences_and_replay_exactly(tmp_path):
    fixture, config, journal, provider = setup_journal()
    report = await pilot.run(tmp_path, fixture, config, journal, provider)
    assert report['complete'] and report['expectation_failures'] == []
    assert len(report['support']) == 66 and len(report['rows']) == len(report['assessments']) == 22
    assert provider.calls == report['transport_attempts'] == report['usage']['calls'] <= pilot.CALL_LIMIT
    assert report['provider_network_calls'] == 0
    assert report['calls_by_role']['answer_judge_v3'] == 22
    for row in report['rows']:
        assert not any(role.startswith('support') for role in row['calls_by_role'])
    support = {(row['case'], row['arm'], row['repeat']): row for row in report['support']}
    runtime = {(row['case'], row['arm']): row for row in report['rows']}
    for arm in pilot.ARMS:
        assert support['repeat_zero_denial', arm, 0]['source_qids'] == []
        assert support['repeat_zero_denial', arm, 1]['source_qids'] == ['silver-moss-current']
        assert runtime['repeat_zero_denial', arm]['source_qids'] == []
        assert not support['extra_clause', arm, 0]['intent']
        assert not support['semantic_question', arm, 0]['intent']
        for case in ('expired_source', 'superseded_source', 'incomplete_source'):
            assert support[case, arm, 0]['usage']['calls'] == 0
            assert runtime[case, arm]['source_qids'] == []
    partial = support['partial_approval', 'replay_contract', 0]
    assert partial['result']['status'] == 'withheld' and partial['result']['error_type'] == 'ValueError'
    assert runtime['partial_approval', 'replay_contract']['source_qids'] == []
    for case in ('multiword_quote', 'corrected_japanese', 'corrected_normalized_pronoun', 'cross_project'):
        assert runtime[case, 'replay_contract']['exact_stored_answer'] is True
    assessments = {(row['case'], row['arm']): row['diagnostics'] for row in report['assessments']}
    typed = assessments['multiword_quote', 'replay_contract']
    assert typed['exact_match']['verdict'] == typed['source_validity']['verdict'] == 'pass'
    assert typed['legacy_model_grade']['task']['verdict'] == 'fail'
    assert typed['equality_task_disagreement']['status'] == 'disagreement'
    assert all(row['legacy_grade_origin'] == 'scripted_development_control' for row in report['assessments'])

    _, _, replay_journal, no_provider = setup_journal(frozen=deepcopy(journal.data))
    assert replay_journal.transport is None
    replay = await pilot.run(tmp_path, fixture, config, replay_journal, no_provider)
    assert replay['transport_attempts'] == 0 and replay['provider_network_calls'] == 0
    assert replay_journal.counts == journal.counts and replay_journal.missing == []
    assert pilot.comparable(replay) == pilot.comparable(report)
    public = json.dumps(pilot.public_summary(report))
    assert not any(token in public for token in ('"quote"', '"answer"', '"receipt"', '"cache"', 'FAKE_NO_CREDENTIAL'))
    assert 'equality_task_disagreement' in public and 'legacy_model_verdicts' in public


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['query', 'source', 'policy'])
async def test_changed_binding_cannot_import_a_frozen_approval(tmp_path, change):
    fixture, config, journal, provider = setup_journal()
    case = deepcopy(fixture['cases'][0])
    clock = [fixture['clock']]
    options = dict(blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0)
    with patch.dict(pilot.os.environ, pilot.previous.previous.environment(), clear=True), \
            patch('time.time', lambda: clock[0]):
        provider.select(case, 'replay_contract')
        first = await pilot.previous.support_cell(case, 'replay_contract', 0, config, journal, clock)
        arm = 'replay_contract'
        if change == 'query':
            case['question'] = 'Please ' + case['question'][0].lower() + case['question'][1:]
        elif change == 'source':
            case['records'][0]['answer'] += ' Changed.'
        else:
            arm = 'generic_contract'
        provider.select(case, arm)
        row = await pilot.previous.runtime_cell(case, arm, first['receipt'], config, journal, clock, options)
    assert row['status'] == 'ok' and row['source_qids'] == [] and row['receipt_keys'] == []
    assert not any(role.startswith('support') for role in row['calls_by_role'])


def test_equality_oracle_keeps_original_bytes_when_source_text_is_sanitized():
    fixture, config, _, _ = setup_journal()
    case = deepcopy(fixture['cases'][0])
    original = '  Keep the hatch closed.\n[telemetry (raw)] secret  '
    case['records'][0]['answer'] = original
    row = {'arm': 'replay_contract', 'available_context': {'reference': 'resolved', 'approved_records': []}}
    clock = [fixture['clock']]
    with patch('time.time', lambda: clock[0]):
        observed = pilot.observed_case(case, row, config, clock)
    assert observed['replay_target']['answer'] == original
    assert observed['available_context']['source_checks']['complete'] is False
    exact = pilot.assessment.diagnostics(observed, original)
    altered = pilot.assessment.diagnostics(observed, 'Keep the hatch closed.')
    assert exact['exact_match']['verdict'] == 'pass' and altered['exact_match']['verdict'] == 'fail'
    assert exact['source_validity']['verdict'] == 'fail'


def test_journal_rejects_changed_fixture_or_assessment_binding():
    fixture, config, journal, _ = setup_journal()
    journal.finish()
    for field in ('fixture_sha256', 'assessment_contract_sha256', 'code_sha256'):
        metadata = pilot.metadata(fixture, config)
        metadata[field] = 'changed'
        with pytest.raises(ValueError):
            pilot.bench.Journal(config, metadata, frozen=deepcopy(journal.data), max_calls=pilot.CALL_LIMIT)


@pytest.mark.asyncio
async def test_failed_transport_keeps_unavailable_axes_and_every_planned_cell(tmp_path):
    fixture, config, journal, provider = setup_journal()
    async def broken(*_args, **_kwargs):
        raise OSError('Do not publish provider text')
    provider._post_json = broken
    report = await pilot.run(tmp_path, fixture, config, journal, provider)
    assert report['transport_attempts'] == report['usage']['calls'] == 1
    assert report['usage']['unknown_usage_calls'] == 1
    assert report['usage']['input_tokens'] is None and report['usage']['output_tokens'] is None
    assert len(report['support']) == 66 and len(report['rows']) == len(report['assessments']) == 22
    assert all(row['status'] == 'unavailable' for row in report['rows'])
    assert all(row['diagnostics']['attribution']['verdict'] == 'unknown' for row in report['assessments'])
    assert all(row['diagnostics']['information_sufficiency']['verdict'] == 'unknown' for row in report['assessments'])
    assert report['expectation_failures']
    assert 'Do not publish' not in json.dumps(pilot.public_summary(report))
    assert report['support'][0]['local_gate_error_type'] == 'OSError'
    assert report['support'][0]['result']['error_type'] == 'OSError'
    assert report['support'][0]['result']['error_type_basis'] == 'provider_receipt'

    _, _, replay_journal, no_provider = setup_journal(frozen=deepcopy(journal.data))
    replay = await pilot.run(tmp_path, fixture, config, replay_journal, no_provider)
    assert replay['transport_attempts'] == 0 and replay_journal.missing == []
    assert replay['support'][0]['local_gate_error_type'] == 'RuntimeError'
    assert replay['support'][0]['result']['error_type'] == 'OSError'
    assert replay['usage']['unknown_usage_calls'] == 1
    assert pilot.comparable(report) == pilot.comparable(replay)
