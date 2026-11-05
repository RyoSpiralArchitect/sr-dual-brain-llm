"""Independent availability/receipts through production failure boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_critic_failures as pilot


@pytest.mark.asyncio
async def test_fixed_failure_matrix_and_zero_transport_replay(tmp_path):
    report, journal = await pilot.run(tmp_path)
    assert report['complete'] and report['expectation_failures'] == []
    assert len(report['rows']) == len(pilot.CASES) == 18
    assert report['provider_network_calls'] == 0
    assert report['blocked_provider_requests'] == 0
    assert report['transport_attempts'] == len(journal['responses']) == report['usage']['calls']
    assert report['transport_attempts'] <= pilot.CALL_LIMIT
    rows = {row['case']: row for row in report['rows']}
    for name in ('empty', 'length', 'filtered', 'missing_finish', 'missing_fields', 'scalar_issue',
                 'transport_error', 'repair_empty', 'repair_filtered', 'repair_schema',
                 'verify_empty', 'verify_schema'):
        assert rows[name]['answer_available']
        assert rows[name]['last_external_review_valid'] is False
        assert rows[name]['source_qids'] == ['cedar-current']
        assert rows[name]['answer_completion_confirmed']
        assert rows[name]['source_context']['source_approved']
    assert rows['clean']['answer_origin'] == 'draft'
    assert rows['clean']['last_external_review_valid'] is True
    assert rows['integration_error']['answer_origin'] == 'local_fallback'
    assert rows['integration_error']['answer_completion_confirmed'] is False
    assert rows['repair_clean']['reviews'][-1]['critic_kind'] == 'external_repaired'
    assert rows['repair_clean']['last_external_review_valid'] is True
    assert [r['role'] for r in rows['scalar_issue']['receipts']].count('critic_repair') == 0
    for name in ('transport_error', 'unknown_usage', 'integration_error'):
        assert rows[name]['usage']['unknown_usage_calls'] == 1
        assert rows[name]['usage']['input_tokens'] is None
        assert rows[name]['usage']['output_tokens'] is None
    for name in ('cancel_critic', 'cancel_integration'):
        assert rows[name]['cancelled']
        assert not rows[name]['answer_available']
        assert not rows[name]['trace_published'] and not rows[name]['stored_probe']
        assert rows[name]['usage']['cancelled_calls'] == 1
        assert rows[name]['usage']['unknown_usage_calls'] == 1
        assert rows[name]['next_turn_reusable'] is True
        assert rows[name]['recovery_usage']['calls'] > 0
        assert rows[name]['recovery_usage']['unknown_usage_calls'] == 0
    assert all(row['occupied_after_turn'] == 0 for row in report['rows'])
    # Unknown receipts are retained, not converted into a zero-cost success.
    assert report['usage']['input_tokens'] is None and report['usage']['output_tokens'] is None
    assert report['usage']['known_input_tokens'] > 0
    replay, replay_journal = await pilot.run(tmp_path, frozen=deepcopy(journal))
    assert replay['transport_attempts'] == 0
    assert replay_journal == journal
    assert pilot.comparable(replay) == pilot.comparable(report)


def test_changed_contract_or_incomplete_journal_is_rejected():
    journal = {'metadata': pilot.metadata(), 'complete': True, 'responses': []}
    for key in ('code_sha256', 'fixture_sha256', 'call_limit'):
        changed = deepcopy(journal)
        changed['metadata'][key] = 'changed'
        with pytest.raises(ValueError, match='differs'):
            pilot.RawJournal(frozen=changed)
    journal['complete'] = False
    with pytest.raises(ValueError, match='incomplete'):
        pilot.RawJournal(frozen=journal)


@pytest.mark.asyncio
async def test_replay_occurrence_mismatch_stays_fatal_when_caller_catches_error():
    journal = pilot.RawJournal(frozen={'metadata': pilot.metadata(), 'complete': True, 'responses': [],
                                     'responses_sha256': pilot.digest([])})
    with pytest.raises(ValueError, match='occurrence'):
        await journal.post(pilot.CASES[0], 'support', 'https://fake.invalid', {}, {})
    with pytest.raises(ValueError, match='Unconsumed'):
        journal.finish()


@pytest.mark.parametrize('row', [
    {'request': {}, 'error_type': 'UnsupportedError'},
    {'request': {}, 'response': {}, 'error_type': 'OSError'},
    {'request': {}},
])
def test_malformed_frozen_receipts_fail_before_any_production_fallback(row):
    responses = [row]
    frozen = {'metadata': pilot.metadata(), 'complete': True, 'responses': responses,
              'responses_sha256': pilot.digest(responses)}
    with pytest.raises(ValueError, match='receipt envelope'):
        pilot.RawJournal(frozen=frozen)


def test_response_checksum_is_checked_independently_of_request_metadata():
    frozen = {'metadata': pilot.metadata(), 'complete': True,
              'responses': [{'request': {}, 'response': {'changed': True}}],
              'responses_sha256': pilot.digest([])}
    with pytest.raises(ValueError, match='integrity'):
        pilot.RawJournal(frozen=frozen)


@pytest.mark.asyncio
async def test_unreplayable_recording_error_stays_fatal_with_partial_receipt():
    saved = []
    journal = pilot.RawJournal(save=lambda data: saved.append(deepcopy(data)))
    with pytest.raises(json.JSONDecodeError):
        await journal.post(pilot.CASES[0], 'support', 'https://fake.invalid',
                           {'messages': [{'content': 'invalid JSON'}]}, {})
    assert saved[-1]['complete'] is False
    assert saved[-1]['responses'][0]['error_type'] == 'JSONDecodeError'
    with pytest.raises(ValueError, match='Unconsumed'):
        journal.finish()


@pytest.mark.asyncio
async def test_partial_receipts_survive_a_fatal_later_case(tmp_path, monkeypatch):
    original = pilot.run_cell
    async def fail_later(case, journal, output_dir):
        if case[0] == pilot.CASES[1][0]:
            raise RuntimeError('Deliberate later failure')
        return await original(case, journal, output_dir)
    monkeypatch.setattr(pilot, 'run_cell', fail_later)
    with pytest.raises(RuntimeError, match='Deliberate'):
        await pilot.run(tmp_path)
    partial = pilot.read_json(tmp_path / 'journal.partial.json')
    assert partial['complete'] is False and len(partial['responses']) == 3
    assert partial['responses_sha256'] == pilot.digest(partial['responses'])
    assert len(pilot.read_json(tmp_path / 'report.partial.json')['rows']) == 1
