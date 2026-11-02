from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_advisor_sessions as pilot
from core.advisor_timing import AdvisorTiming
from test_benchmark_module_gating import fake_response


def test_complete_sequence_has_matched_profiles_and_distinct_prospective_inputs():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    earlier = pilot.bench.read_json(pilot.budget.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in earlier['cases']}
    fixture['cases'].pop()
    with pytest.raises(ValueError):
        pilot.validate(fixture)


def test_timing_recorder_keeps_real_history_and_replay_requires_exact_read_identity():
    now = [0.]
    ledger = AdvisorTiming(clock=lambda: now[0])
    recorder = pilot.ObservedTiming(ledger)
    profile, module = 'a' * 64, 'director_advice'
    recorder.begin()
    assert recorder.estimate(module, profile).ms is None
    recorder.record(module, profile, 1600., censored=True)
    assert recorder.estimate(module, profile).kind == 'censored_bound'
    now[0] = 61.
    assert recorder.estimate(module, profile).ms is None
    frozen = deepcopy(recorder.reads)
    recorder.begin(frozen)
    assert recorder.estimate(module, profile).ms is None
    assert recorder.estimate(module, profile).ms == 1600.
    assert recorder.estimate(module, profile).ms is None
    recorder.finish()
    recorder.begin(frozen)
    with pytest.raises(ValueError):
        recorder.estimate(module, 'b' * 64)
    with pytest.raises(ValueError):
        recorder.finish()
    # The production budget producer catches ledger errors as unknown. An
    # empty/corrupt replay must still fail even if all those reads were caught.
    recorder.begin([])
    try:
        recorder.estimate(module, profile)
    except ValueError:
        pass
    with pytest.raises(ValueError):
        recorder.finish()


@pytest.mark.asyncio
async def test_natural_same_session_history_and_rpc_timing_interventions_are_network_free(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'controlled-advisor', 'OFFLINE_ONLY', auto_continue=False, max_continuations=0)
    journal = pilot.SessionJournal(config, {}, max_calls=96)
    transport = pilot.ControlledTransport()
    journal.transport._post_json = transport.post
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path, judge_model='test'), fixture, config, journal, transport=transport)
    assert report['complete'] and report['evidence'] == 'controlled' and report['live_provider_calls'] == 0
    rows = {(r['case'],r['arm']):r for r in report['rows']}
    assert len(rows) == 12 and all(r['pending_advisors'] == 0 for r in rows.values())
    for arm in pilot.ARMS:
        assert [rows[(phase,arm)]['session_turn'] for phase in pilot.PHASES] == list(range(1,7))
    for module in ('director_advice','executive_metrics_observation'):
        timing = lambda phase: rows[(phase,'timed_advisors')]['metrics']['modules']['advisor_timing'][module]
        assert timing('cold')['outcome'] == 'timed_out'
        assert timing('warm')['outcome'] == 'not_admitted'
        # The provider is now fast, but the recent slow bound still suppresses it.
        assert timing('warm_again')['outcome'] == 'not_admitted'
        assert timing('expired')['estimate_kind'] == 'unknown'
        assert timing('expired')['outcome'] == 'completed' and timing('expired')['usable_output']
        assert timing('recovered_warm')['estimate_kind'] == 'completed_max'
        assert timing('recovered_warm')['outcome'] == 'completed'
    overlap = rows[('recovered_warm','timed_advisors')]
    assert overlap['status'] == 'ok' and overlap['rpc_seconds'] == 3.
    executive = overlap['metrics']['modules']['advisor_timing']['executive_advice']
    assert executive['budget_source'] == 'turn_deadline' and executive['outcome'] == 'completed'
    assert executive['usable_output'] and executive['estimated_latency_ms'] > 1000
    assert overlap['controlled_completed_roles'].index('executive') < overlap['controlled_completed_roles'].index('primary')
    tight = rows[('bounded_rpc','timed_advisors')]
    assert tight['status'] == 'ok'
    assert tight['metrics']['modules']['gates']['executive_advice']['reason'] == 'insufficient_time_budget'
    assert rows[('bounded_rpc','untimed_advisors')]['error_type'] == 'DeadlineExceeded'
    assert not rows[('bounded_rpc','untimed_advisors')]['may_have_committed']
    for phase in pilot.PHASES[:-1]:
        assert rows[(phase,pilot.ARMS[0])]['answer'] == rows[(phase,pilot.ARMS[1])]['answer'] == 'Controlled answer.'
        assert rows[(phase,pilot.ARMS[0])]['calls_by_role'].get('right',0) == rows[(phase,pilot.ARMS[1])]['calls_by_role'].get('right',0) == 0
    assert [rows[(phase,'timed_advisors')]['memory_sources'] for phase in pilot.PHASES] == list(range(1,7))
    assert rows[('bounded_rpc','untimed_advisors')]['memory_sources'] == 5
    assert report['idle_intervals'][0]['simulated']


@pytest.mark.asyncio
async def test_full_sequence_with_recorded_history_reads_has_exact_zero_transport_replay(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', auto_continue=False, max_continuations=0)
    journal = pilot.SessionJournal(config, {}, max_calls=96)
    journal.transport._post_json = fake_response
    intervals = []

    async def no_wait(seconds):
        intervals.append(seconds)

    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, fixture, config, journal, idle=no_wait)
    assert intervals == [61] and live['summaries']['jointly_graded_pairs'] == 6
    frozen = deepcopy(journal.data)
    replay_journal = pilot.SessionJournal(config, {}, frozen=frozen, max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args, fixture, config, replay_journal)
    assert replay['complete'] and replay['live_provider_calls'] == 0
    assert replay_journal.counts == journal.counts and not replay_journal.missing
    assert [(r['case'],r['arm'],r['status'],r['answer'],r['draft'],r['timing_reads'],r['memory_sources'],
        r['metrics']['modules']['gates']) for r in live['rows']] == [
        (r['case'],r['arm'],r['status'],r['answer'],r['draft'],r['timing_reads'],r['memory_sources'],
        r['metrics']['modules']['gates']) for r in replay['rows']]
    assert live['grades'] == replay['grades']
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and 'User message:' not in public and 'Controlled answer.' not in public
