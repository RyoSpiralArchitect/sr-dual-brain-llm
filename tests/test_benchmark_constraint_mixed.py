"""Native ablation/replay mechanics with scripted transport, never quality data."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_constraint_mixed as study
import core.micro_critic as micro
from core.constraint_checks import check_constraints
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import config, response


class Provider:
    def __init__(self, journal, failure):
        self.journal, self.failure, self.judgments = journal, failure, 0

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        role = self.journal.events[-1]['role']
        if self.failure == 'transport' and self.journal.calls == 1:
            raise RuntimeError('PRIVATE_PROVIDER_FAILURE')
        text = payload['messages'][-1]['content']
        if role == 'task_goal_judge':
            self.judgments += 1
            if self.failure == 'judge' and self.judgments == 1:
                return response('{}')
            packet = json.loads(text)
            # Deliberately shape-only test scores; not semantic ground truth.
            return response(json.dumps({'scores': {label: {c: {'verdict': 'pass', 'quote': answer[:1]}
                for c in study.study.CRITERIA} for label, answer in packet['answers'].items()}}))
        assert 'comparison_reference' not in text and '"judge_labels"' not in text
        await asyncio.sleep(0)
        if role == 'critic':
            data = {'verdict': 'ok', 'issues': [], 'fixes': []}
            system = payload['messages'][0]['content']
            for line in system.splitlines():
                if line.startswith('{"version": "explicit_task_goals_v1"'):
                    data['goal_findings'] = [{'goal_id': g['goal_id'], 'status': 'uncertain',
                        'quote': '', 'reason': 'Mechanism-only observation.'} for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        if 'p=3a-b+c' in text:
            return response('For p=11,q=7,r=4, (a,b,c)=' + ('(13/3,3,1).' if role == 'integration' else '(4,3,1).'))
        if 'job-coral' in text:
            answer = {'job_id': 'job-coral', 'state': 'cancellation unconfirmed'}
            if role == 'integration':
                answer['next_action'] = 'check acknowledgement'
            return response(json.dumps(answer))
        if self.failure == 'partial' and role == 'draft' and self.journal.case.startswith('dedup_call_boundary:'):
            return response('An incomplete draft', 'length')
        return response('A mechanism-only answer for the supplied task.')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none', 'transport', 'judge', 'partial'])
async def test_mixed_native_conditions_exact_replay_and_failure_retention(tmp_path, failure):
    fixture = study.pilot.bench.read_json(study.FIXTURE)
    cfg = replace(config(), model='gpt-6-luna', max_output_tokens=2048,
                  timeout_seconds=60, auto_continue=False, max_continuations=0)
    judge = replace(cfg, model='gpt-6-sol')
    live, repeat = tmp_path/'live', tmp_path/'replay'
    live.mkdir()
    repeat.mkdir()
    journal = study.ScheduledJournal(cfg, study.metadata(fixture, cfg, judge), max_calls=96, replay_wait_seconds=30.)
    journal.transport = Provider(journal, failure)
    with forbid_provider_network() as audit:
        result = await study.run(live, fixture, cfg, judge, journal)
        frozen = study.ScheduledJournal(cfg, result['metadata'], max_calls=96,
            frozen=json.loads(json.dumps(journal.data)), replay_wait_seconds=30.)
        replay = await study.run(repeat, fixture, cfg, judge, frozen)
    assert result['complete'] and len(result['rows']) == 20 and len(result['grades']) == 10
    assert study.canonical(result) == study.canonical(replay)
    assert journal.calls <= 96 and frozen.calls == audit.blocked_requests == 0
    assert frozen.position == len(journal.events) and frozen.schedule_position == len(journal.data['transport_events'])
    assert micro.check_constraints is check_constraints
    assert all(r['constraint_state_cleared'] for r in result['rows'])
    if failure == 'transport':
        assert journal.calls == 1 and journal.blocked and result['usage']['unknown_usage_calls'] == 1
        assert result['usage']['input_tokens'] is None
        assert all(not g['runtime_qualified'] for g in result['grades'])
    else:
        assert all(r['policy_class'] == 'RightBrainPolicy' and r['plan_state_cleared'] for r in result['rows'])
        assert all(g['first_requests_matched'] for g in result['grades'])
        assert sum(g['status'] == 'graded' for g in result['grades']) == (9 if failure == 'judge' else 10)
        assert journal.transport.judgments == 10
        for item in journal.data['occurrences']:
            assert 'reasoning_effort' not in item['request']['payload']
            assert item['request']['payload']['max_completion_tokens'] <= 2048
        for row in result['rows']:
            if row['arm'] == 'ablated':
                assert row['constraint_check'] is None
                assert not any(o['contradictions'] for o in row['detector_observations'])
        exercised = [r for r in result['rows'] if r['arm'] == 'witnesses'
                     and r['case'] in {'three_sensor_scope', 'status_object'}]
        assert any(any(o['contradictions'] for o in r['detector_observations']) for r in exercised)
        if failure == 'partial':
            pair = next(g for g in result['grades'] if g['case'] == 'dedup_call_boundary')
            assert not pair['runtime_qualified']
    public = json.dumps(study.public_summary(result))
    assert all(s not in public for s in ('"question":', '"answer":', '"quote":', '"expected":',
        'comparison_reference', 'PRIVATE_PROVIDER_FAILURE', 'FAKE_NO_CREDENTIAL'))


def test_ablation_keeps_legacy_checks_and_restores_on_cancellation():
    observed = []
    with pytest.raises(asyncio.CancelledError):
        with study.detector_condition('ablated', observed):
            result = micro.micro_criticise_reasoning('Compute 2+2.', '5')
            assert result.verdict == 'issues' and result.domain == 'arithmetic'
            raise asyncio.CancelledError
    assert micro.check_constraints is check_constraints and len(observed) == 1


def test_exact_probe_requires_values_and_rejects_duplicate_keys_and_byte_changes():
    fixture = study.pilot.bench.read_json(study.FIXTURE)
    obj = next(c for c in fixture['cases'] if c['id'] == 'status_object')
    correct = obj['exact_probe']['expected']
    assert study.exact_probe(obj, json.dumps(correct))['status'] == 'pass'
    wrong = correct | {'state': 'cancelled'}
    assert check_constraints(obj['question'], json.dumps(wrong)).status == 'consistent'
    assert study.exact_probe(obj, json.dumps(wrong))['status'] == 'fail'
    assert study.exact_probe(obj, '{"job_id":"a","job_id":"b"}')['status'] == 'fail'
    literal = fixture['cases'][-1]
    expected = literal['exact_probe']['expected']
    assert study.exact_probe(literal, expected)['status'] == 'pass'
    assert study.exact_probe(literal, expected+'\n')['status'] == 'fail'


@pytest.mark.parametrize('change', ['order', 'labels', 'domain', 'negative', 'probe', 'goals'])
def test_incomplete_or_unbalanced_fixture_is_rejected_before_calls(change):
    fixture = deepcopy(study.pilot.bench.read_json(study.FIXTURE))
    if change == 'order': fixture['cases'].reverse()
    elif change == 'labels': fixture['cases'][0]['judge_labels'] = {'A': 'ablated', 'B': 'witnesses'}
    elif change == 'domain': fixture['cases'][0]['domain'] = 'code'
    elif change == 'negative': fixture['cases'][0]['negative_control'] = True
    elif change == 'probe': del fixture['cases'][2]['exact_probe']
    else: fixture['cases'][0]['expected_goal_count'] = 2
    with pytest.raises(ValueError):
        study.validate(fixture)
