"""Untouched holdout arms, labels, serialized knobs and exact replay."""
from dataclasses import replace
import json

import pytest

from test_replanning_stress import cfg
from test_critic_cap_development import Provider
from offline_provider_guard import forbid_provider_network
import benchmark_critic_cap_holdout as holdout


@pytest.mark.asyncio
async def test_holdout_arms_and_exact_replay(tmp_path):
    study = holdout.study
    names = ('tasks', 'FIXTURE', 'PROTOCOL', 'MANIFEST', 'ARMS', 'ORDERS',
             'CAPS', 'CALL_LIMIT', 'PHASE', 'Journal', 'expected_manifest')
    original = {name: getattr(study, name) for name in names}
    original_dual = study.base.dual
    holdout.configure()
    try:
        fixture = study.base.pilot.bench.read_json(study.FIXTURE)
        manifest = study.base.pilot.bench.read_json(study.MANIFEST)
        study.validate_fixture(fixture)
        assert manifest == study.expected_manifest(fixture)
        for old_path in ('completion_mechanisms_certified.json', 'frontload_development.json',
                         'reasoning_profile_development.json', 'critic_cap_development.json'):
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
        assert live['outcome'] == 'comparisons_complete' and len(live['rows']) == 24
        assert live['metadata']['phase'] == 'holdout'
        assert study.canonical(live) == study.canonical(replay)
        assert live['usage']['calls'] <= study.CALL_LIMIT
        for occurrence in tape['occurrences']:
            request = occurrence['request']
            arm, role = request['case'].rsplit(':', 1)[-1], request['role']
            payload = request['payload']
            expected_effort = holdout.PROFILES[arm]['effort'] if role in {'draft', 'integration'} else 'provider_default'
            assert payload.get('reasoning_effort') == (None if expected_effort == 'provider_default' else expected_effort)
            assert payload.get('max_completion_tokens') == holdout.CAPS[arm][role]
    finally:
        for name, value in original.items():
            setattr(study, name, value)
        study.base.dual = original_dual
