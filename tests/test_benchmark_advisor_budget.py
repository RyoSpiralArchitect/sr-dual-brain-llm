from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_advisor_budget as pilot
from core.prefrontal_cortex import PrefrontalCortex
from test_benchmark_module_gating import fake_response


def test_fixture_and_profiles_isolate_only_advisor_timing():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    for case in fixture['cases']:
        assert not PrefrontalCortex.is_trivial_chat_turn(case['question'])
        baseline = pilot.environment(case, pilot.ARMS[0], {'DUALBRAIN_PG_DSN': 'PRIVATE'})
        timed = pilot.environment(case, pilot.ARMS[1], {'DUALBRAIN_PG_DSN': 'PRIVATE'})
        assert 'DUALBRAIN_PG_DSN' not in baseline
        differences = {k for k in baseline if baseline[k] != timed[k]}
        assert differences == {pilot.bench.DEFAULT_REGISTRY[m].mode_env for m in pilot.ADVISORS}
    fixture['cases'].pop()
    with pytest.raises(ValueError):
        pilot.validate(fixture)


@pytest.mark.asyncio
async def test_warm_censored_inputs_inhibit_work_and_exact_replay_restores_original_history(tmp_path, monkeypatch):
    import asyncio
    clock = [0.]
    monkeypatch.setattr('core.dual_brain.time.perf_counter', lambda: clock[0])
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False, max_continuations=0)
    journal = pilot.bench.Journal(config, {}, max_calls=96)

    async def transport(url, payload, headers, **kwargs):
        system = payload['messages'][0]['content']
        if 'Director' in system or any('Observer report (internal;' in str(m.get('content', ''))
                                      for m in payload['messages']):
            clock[0] += 3.
            raise asyncio.CancelledError
        return await fake_response(url, payload, headers, **kwargs)

    journal.transport._post_json = transport
    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, fixture, config, journal)
    assert live['complete'] and live['summaries']['jointly_graded_pairs'] == 3
    for case in fixture['cases']:
        seed = live['warm_history'][case['id']]
        assert seed['director_advice'] == {'ms': 3000., 'kind': 'censored_bound'}
        assert seed['executive_metrics_observation'] == {'ms': 3000., 'kind': 'censored_bound'}
    baseline, timed = [live['summaries']['by_arm'][arm] for arm in pilot.ARMS]
    assert baseline['calls_by_role']['director'] == 3 and 'director' not in timed['calls_by_role']
    assert baseline['calls_by_role']['executive_metrics'] == 3 and 'executive_metrics' not in timed['calls_by_role']
    assert timed['usage']['calls'] == baseline['usage']['calls'] - 6
    assert baseline['usage']['unknown_usage_calls'] == 6
    frozen = deepcopy(journal.data)
    replay_journal = pilot.bench.Journal(config, {}, frozen=frozen, max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args, fixture, config, replay_journal)
    assert replay['provider_calls'] == 0 and replay['warm_history'] == live['warm_history']
    assert [(r['case'], r['arm'], r['status'], r['answer'], r.get('history_sha256'),
             r.get('metrics', {}).get('modules', {}).get('gates')) for r in live['rows']] == [
           (r['case'], r['arm'], r['status'], r['answer'], r.get('history_sha256'),
             r.get('metrics', {}).get('modules', {}).get('gates')) for r in replay['rows']]
    assert replay['grades'] == live['grades']
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and 'User message:' not in public


@pytest.mark.parametrize('seed', [[], {'private_role': {'ms': 1, 'kind': 'completed_max'}},
    {'director_advice': {'ms': float('nan'), 'kind': 'completed_max'}},
    {'director_advice': {'ms': 1, 'kind': 'private_reason'}}])
def test_replay_rejects_invalid_observed_history(seed):
    with pytest.raises(ValueError):
        pilot.seed_history(SimpleNamespace(), seed)
