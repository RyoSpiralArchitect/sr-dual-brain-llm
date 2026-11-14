"""Frozen frontload controls, request caps and network-free exact replay."""
from dataclasses import replace
import json

import pytest

from test_replanning_stress import cfg
from test_benchmark_contract_live import response
from offline_provider_guard import forbid_provider_network
import benchmark_frontload_development as study


class Provider:
    def __init__(self, journal, fixture):
        self.journal, self.fixture = journal, fixture

    async def _post_json(self, _url, payload, _headers, **kwargs):
        role = self.journal.events[-1]['role']
        identity, arm = self.journal.case.rsplit(':', 1)
        case = next(c for c in self.fixture['cases'] if c['id'] == identity)
        if role == 'draft' and identity == 'logic_a':
            return response('', 'length')
        if role in {'critic', 'critic_repair', 'self_critique'}:
            empty = role == 'critic' and identity == 'logic_a'
            data = {'verdict': 'issues' if empty else 'ok',
                    'issues': ['No complete candidate'] if empty else [],
                    'fixes': ['Reconstruct from the task'] if empty else []}
            if role != 'self_critique':
                for line in payload['messages'][0]['content'].splitlines():
                    if line.startswith('{"version": "explicit_task_goals_v1"'):
                        data['goal_findings'] = [{'goal_id': g['goal_id'], 'status': 'unresolved' if empty else 'uncertain',
                            'quote': '', 'reason': 'Scripted route observation'}
                            for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        return response(json.dumps(case['expected']))


@pytest.mark.asyncio
async def test_four_arms_caps_budget_and_exact_replay(tmp_path):
    fixture = study.base.pilot.bench.read_json(study.FIXTURE)
    frozen_manifest = study.base.pilot.bench.read_json(study.MANIFEST)
    study.validate_fixture(fixture)
    assert frozen_manifest['fixture_sha256'] == study.base.pilot.bench.digest(fixture)
    # The stopped paid study stays pinned to its original source commit. This
    # synthetic check exercises the same runner against the current code hash.
    manifest = study.expected_manifest(fixture)
    for old_path in ('completion_mechanisms.json', 'completion_mechanisms_certified.json',
                     'replanning_corrected.json'):
        old = study.base.pilot.bench.read_json(study.ROOT/'sr-dual-brain-llm/examples'/old_path)
        assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in old['cases']}
    live_dir, replay_dir = tmp_path/'live', tmp_path/'replay'
    live_dir.mkdir(); replay_dir.mkdir()
    config = replace(cfg(), timeout_seconds=120)
    with forbid_provider_network() as audit:
        live, tape = await study.run(live_dir, fixture, manifest, config,
            transport_factory=lambda j: Provider(j, fixture))
        replay, _ = await study.run(replay_dir, fixture, manifest, config,
            frozen=json.loads(json.dumps(tape)))
    assert audit.blocked_requests == 0
    assert live['outcome'] == 'comparisons_complete' and len(live['rows']) == 20
    assert study.canonical(live) == study.canonical(replay)
    assert live['usage']['calls'] <= study.CALL_LIMIT
    assert all(r['output_reserved'] <= study.base.OUTPUT_ALLOWANCE for r in live['rows'])
    assert {r['cap'] for r in live['request_caps'] if r['role'] == 'draft'} == {4096, 8192}
    assert not any(r['role'] == 'integration' and r['arm'].startswith('frontload')
                   for r in live['request_caps'])
    assert '"answer":' not in json.dumps(study.stress.public(live))
