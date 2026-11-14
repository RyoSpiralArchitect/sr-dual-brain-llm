"""Separate profile arms, serialized efforts and exact offline replay."""
from dataclasses import replace
import json

import pytest

from test_replanning_stress import cfg
from test_benchmark_contract_live import response
from offline_provider_guard import forbid_provider_network
import benchmark_reasoning_profile_development as profile


class Provider:
    def __init__(self, journal, fixture):
        self.journal, self.fixture = journal, fixture

    async def _post_json(self, _url, payload, _headers, **kwargs):
        role = self.journal.events[-1]['role']
        identity, arm = self.journal.case.rsplit(':', 1)
        case = next(c for c in self.fixture['cases'] if c['id'] == identity)
        if role == 'draft' and identity == 'logic_a' and arm.startswith('default'):
            return response('', 'length')
        if role in {'critic', 'critic_repair', 'self_critique'}:
            empty = role == 'critic' and identity == 'logic_a' and arm == 'default_off'
            data = {'verdict': 'issues' if empty else 'ok',
                    'issues': ['No complete candidate'] if empty else [],
                    'fixes': ['Reconstruct from task'] if empty else []}
            if role != 'self_critique':
                for line in payload['messages'][0]['content'].splitlines():
                    if line.startswith('{"version": "explicit_task_goals_v1"'):
                        data['goal_findings'] = [{'goal_id': g['goal_id'],
                            'status': 'unresolved' if empty else 'uncertain',
                            'quote': '', 'reason': 'Scripted route observation'}
                            for g in json.loads(line)['goals']]
            return response(json.dumps(data))
        return response(json.dumps(case['expected']))


@pytest.mark.asyncio
async def test_profile_arms_and_exact_replay(tmp_path):
    study = profile.study
    names = ('tasks', 'FIXTURE', 'PROTOCOL', 'MANIFEST', 'ARMS', 'ORDERS',
             'CAPS', 'CALL_LIMIT', 'Journal', 'expected_manifest')
    original = {name: getattr(study, name) for name in names}
    original_dual = study.base.dual
    profile.configure()
    try:
        fixture = study.base.pilot.bench.read_json(study.FIXTURE)
        frozen_manifest = study.base.pilot.bench.read_json(study.MANIFEST)
        study.validate_fixture(fixture)
        # This historical paid screen remains pinned to its original source.
        # A new benchmark script changes the all-script code digest, so run
        # the synthetic check against a manifest for the current source.
        manifest = study.expected_manifest(fixture)
        assert {k: v for k, v in frozen_manifest.items() if k != 'candidate_code_sha256'} == {
            k: v for k, v in manifest.items() if k != 'candidate_code_sha256'}
        for old_path in ('completion_mechanisms_certified.json', 'frontload_development.json'):
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
        assert live['outcome'] == 'comparisons_complete' and len(live['rows']) == 25
        assert study.canonical(live) == study.canonical(replay)
        assert live['usage']['calls'] <= study.CALL_LIMIT
        assert all(r['output_reserved'] <= study.base.OUTPUT_ALLOWANCE for r in live['rows'])
        drafts = [o['request'] for o in tape['occurrences'] if o['request']['role'] == 'draft']
        for request in drafts:
            arm = request['case'].rsplit(':', 1)[-1]
            expected = profile.PROFILES[arm]['effort']
            assert request['payload'].get('reasoning_effort') == (None if expected == 'provider_default' else expected)
    finally:
        for name, value in original.items():
            setattr(study, name, value)
        study.base.dual = original_dual
