"""Native paired flow, blind grading and exact replay without provider access."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_task_goals as study
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network


class Provider:
    def __init__(self, journal, failure):
        self.journal, self.failure, self.judgments = journal, failure, 0

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        role = self.journal.events[-1]['role']
        text = payload['messages'][-1]['content']
        system = payload['messages'][0]['content']
        if self.failure == 'transport' and len(self.journal.events) == 1:
            raise RuntimeError('PRIVATE_PROVIDER_FAILURE')
        if role == 'task_goal_judge':
            self.judgments += 1
            if self.failure == 'judge' and self.judgments == 1:
                return response('{"scores":{}}')
            packet = json.loads(text)
            assert set(packet) == {'question', 'comparison_reference', 'answers'}
            assert set(packet['answers']) == {'A', 'B'}
            assert all(s not in packet for s in ('arm', 'gates', 'task_goals', 'expected_goal_count'))
            scores = {label: {c: {'verdict': 'pass', 'quote': answer[:1]} for c in study.CRITERIA}
                      for label, answer in packet['answers'].items()}
            return response(json.dumps({'scores': scores}))
        assert 'comparison_reference' not in text and 'PRIVATE_TASK_GOAL_REFERENCE_ONLY' not in text
        await asyncio.sleep(0)
        if role == 'critic':
            data = {'verdict': 'ok', 'issues': [], 'fixes': []}
            for line in system.splitlines():
                if line.startswith('{"version": "explicit_task_goals_v1"'):
                    goals = json.loads(line)['goals']
                    data['goal_findings'] = [{'goal_id': g['goal_id'], 'status': 'uncertain',
                        'quote': '', 'reason': 'Mechanism-only Fake observation.'} for g in goals]
            return response(json.dumps(data))
        if '17 × 6' in text:
            return response('102')
        if role == 'detail':
            return response('A concrete mechanism-only alternative for this test.')
        return response('Mechanism-only draft for the supplied task.')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none', 'judge', 'transport'])
async def test_native_pairs_private_reference_boundary_and_scheduled_replay(tmp_path, failure):
    fixture = study.pilot.bench.read_json(study.FIXTURE)
    cfg = config()
    judge = replace(cfg, model='fake-goal-judge')
    live, replay = tmp_path / 'live', tmp_path / 'replay'
    live.mkdir()
    replay.mkdir()
    journal = study.ScheduledJournal(cfg, study.metadata(fixture, cfg, judge), max_calls=study.CALL_LIMIT,
                                     replay_wait_seconds=30.)
    journal.transport = Provider(journal, failure)
    with forbid_provider_network() as audit:
        result = await study.run(live, fixture, cfg, judge, journal)
        frozen = study.ScheduledJournal(cfg, study.metadata(fixture, cfg, judge), max_calls=study.CALL_LIMIT,
            frozen=json.loads(json.dumps(journal.data)), replay_wait_seconds=30.)
        repeated = await study.run(replay, fixture, cfg, judge, frozen)
    assert result['complete'] and len(result['rows']) == 10 and len(result['grades']) == 5
    assert audit.blocked_requests == 0 and frozen.calls == 0 and frozen.transport is None
    assert study.canonical(result) == study.canonical(repeated)
    if failure == 'transport':
        assert journal.calls == 1 and journal.blocked
        assert result['usage']['input_tokens'] is None and result['usage']['unknown_usage_calls'] == 1
        assert any(r['provider_degraded'] for r in result['rows'])
    else:
        assert all(r['status'] == 'ok' and r['policy_class'] == 'RightBrainPolicy' for r in result['rows'])
        for row in result['rows']:
            assert row['plan_state_cleared']
            if row['arm'] == 'auto' and row['case'] != 'simple':
                assert row['task_goals']['goal_count'] == 3 and row['assignment_request_counts'].get('draft') == 1
                assert not row['task_goals']['verified_complete']
            else:
                assert row['task_goals'] is None and not row['assignment_request_counts']
        assert sum(g['status'] == 'graded' for g in result['grades']) == (4 if failure == 'judge' else 5)
        assert journal.transport.judgments == 5
    assert result['usage']['calls'] <= study.CALL_LIMIT
    public = json.dumps(study.public_summary(result))
    assert all(s not in public for s in ('"answer":', '"question":', '"quote":', '"goal_review":',
        'comparison_reference', 'PRIVATE_PROVIDER_FAILURE', 'FAKE_NO_CREDENTIAL'))


@pytest.mark.asyncio
@pytest.mark.parametrize('error', ['empty', 'length', 'duplicate', 'labels', 'criteria', 'quote', 'verdict'])
async def test_invalid_judgment_is_retained_without_a_repair_request(error):
    case = study.pilot.bench.read_json(study.FIXTURE)['cases'][0]
    rows = {'off': {'answer': 'Original response.'}, 'auto': {'answer': 'Alternate response.'}}
    data = {'scores': {label: {c: {'verdict': 'pass', 'quote': answer[:1]} for c in study.CRITERIA}
                      for label, answer in [('A', 'Alternate response.'), ('B', 'Original response.')]}}
    if error == 'labels': data['scores'].pop('B')
    if error == 'criteria': data['scores']['A'].pop(study.CRITERIA[0])
    if error == 'quote': data['scores']['A'][study.CRITERIA[0]]['quote'] = 'INVENTED'
    if error == 'verdict': data['scores']['A'][study.CRITERIA[0]]['verdict'] = ['pass']
    text = '' if error == 'empty' else '{"scores":{},"scores":{}}' if error == 'duplicate' else json.dumps(data)
    journal = study.ScheduledJournal(config(), {'test': error}, max_calls=3)
    class Fake:
        async def _post_json(self, *_args, **_kwargs):
            return response(text, 'length' if error == 'length' else 'stop')
    journal.transport = Fake()
    with forbid_provider_network(), pytest.raises(ValueError):
        await study.assess(case, rows, config(), journal)
    assert journal.calls == 1
    journal.finish()


def test_comparison_preserves_uncertain_ungraded_and_runtime_qualification():
    def scores(value):
        return {c: {'verdict': value, 'quote': ''} for c in study.CRITERIA}
    rows = [
        {'case': 'unknown', 'runtime_qualified': True, 'scores': {'off': scores('pass'), 'auto': scores('uncertain')}},
        {'case': 'degraded', 'runtime_qualified': False, 'scores': {'off': scores('fail'), 'auto': scores('pass')}},
        {'case': 'ungraded', 'runtime_qualified': True},
    ]
    result = study.comparison(rows)
    for c in study.CRITERIA:
        assert result['totals']['all'][c]['uncertain'] == 1
        assert result['totals']['all'][c]['auto_only_pass'] == 1
        assert result['totals']['runtime_qualified'][c]['auto_only_pass'] == 0
        assert result['totals']['all'][c]['ungraded'] == 1
