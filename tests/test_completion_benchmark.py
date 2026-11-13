"""Fresh fixture, bounded four-arm routing and exact offline replay."""
import json

import pytest

from test_replanning_stress import cfg
from test_benchmark_contract_live import response
from offline_provider_guard import forbid_provider_network
import benchmark_completion_mechanisms as study


class Provider:
    def __init__(self, journal, fixture):
        self.journal, self.fixture = journal, fixture

    async def _post_json(self, _url, payload, _headers, **kwargs):
        role = self.journal.events[-1]['role']
        identity, _arm = self.journal.case.rsplit(':', 1)
        case = next(c for c in self.fixture['cases'] if c['id'] == identity)
        assert json.dumps(case['expected']) not in json.dumps(payload)
        if role == 'draft' and identity == 'logic_a':
            return response('', 'length')
        if role == 'work_product':
            return response(json.dumps({'goal_id': 'g1', 'work': 'Enumerate assignments.',
                                        'unresolved': []}))
        if role in {'critic', 'critic_repair'}:
            data = {'verdict': 'ok', 'issues': [], 'fixes': []}
            for line in payload['messages'][0]['content'].splitlines():
                if line.startswith('{"version": "explicit_task_goals_v1"'):
                    data['goal_findings'] = [{'goal_id': g['goal_id'], 'status': 'uncertain',
                        'quote': '', 'reason': 'Scripted mechanism observation.'}
                        for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        return response(json.dumps(case['expected']))


@pytest.mark.asyncio
async def test_four_arms_and_exact_replay_without_provider(tmp_path):
    fixture = study.base.pilot.bench.read_json(study.FIXTURE)
    manifest = study.base.pilot.bench.read_json(study.MANIFEST)
    study.validate_fixture(fixture)
    assert manifest == study.expected_manifest(fixture)
    old = study.base.pilot.bench.read_json(study.stress.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in old['cases']}
    live_dir, replay_dir = tmp_path/'live', tmp_path/'replay'
    live_dir.mkdir(); replay_dir.mkdir()
    with forbid_provider_network() as audit:
        live, tape = await study.run(live_dir, fixture, manifest, cfg(),
                                     transport_factory=lambda j: Provider(j, fixture))
        replay, _ = await study.run(replay_dir, fixture, manifest, cfg(),
                                    frozen=json.loads(json.dumps(tape)))
    assert audit.blocked_requests == 0
    assert live['outcome'] == 'comparisons_complete' and len(live['rows']) == 20
    assert study.canonical(live) == study.canonical(replay)
    assert live['usage']['calls'] <= study.CALL_LIMIT
    assert all(r['output_reserved'] <= study.base.OUTPUT_ALLOWANCE for r in live['rows'])
    assert any(r['role'] == 'work_product' and r['cap'] == 1024 for r in live['request_caps'])
    assert '"answer":' not in json.dumps(study.public(live))
