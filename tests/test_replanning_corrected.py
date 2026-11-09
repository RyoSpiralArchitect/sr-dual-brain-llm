"""Never-sent selection, frozen failed lineage, actual caps and full exact replay."""
from copy import deepcopy
import json

import pytest

from test_replanning_stress import b, cfg, Provider
import benchmark_replanning_corrected as corrected
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import response


async def parent_study(tmp_path, pool):
    directory = tmp_path/'parent'
    directory.mkdir()
    class TimedOut:
        async def _post_json(self, *args, **kwargs):
            raise TimeoutError('PRIVATE_UNCLASSIFIED_ERROR')
    parent, journal = await b.run(directory, pool, cfg(), 'main', transport_factory=lambda _: TimedOut())
    fixture = corrected.build_fixture(pool, journal)
    plan = corrected.make_plan(fixture, parent, journal, 'original_report_bytes', 'original_journal_bytes')
    return parent, journal, fixture, plan


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_selection_is_exposure_only_and_study_or_transport_stop_replays_real_native_arms(tmp_path, failure):
    pool = b.tasks.build_fixture()
    with forbid_provider_network() as audit:
        parent, journal, fixture, plan = await parent_study(tmp_path, pool)
        assert parent['outcome'] == 'transport_blocked'
        assert fixture['excluded_case_ids'] == ['main:0:logic']
        assert len(fixture['cases']) == 8
        assert all(case in pool['cases'] for case in fixture['cases'])
        assert not set(corrected.exposed_ids(journal)) & {c['id'] for c in fixture['cases']}
        changed = deepcopy(journal)
        changed['occurrences'][0]['response'] = {'PRIVATE_CHANGED_OUTPUT': 'ignored for selection'}
        assert corrected.build_fixture(pool, changed) == fixture
        directories = [tmp_path/'corrected', tmp_path/'replay']
        for path in directories:
            path.mkdir()
        result, tape = await corrected.run(directories[0], fixture, pool, cfg(), parent, journal, plan,
            transport_factory=lambda j: Provider(j, fixture, failure))
        replay, _ = await corrected.run(directories[1], fixture, pool, cfg(), parent, journal, plan,
            frozen=json.loads(json.dumps(tape)))
    assert corrected.canonical(result) == corrected.canonical(replay)
    assert result['campaign_admissions'] == parent['usage']['calls'] + result['usage']['calls'] <= 128
    assert result['usage']['calls'] <= 96 and replay['live_provider_calls'] == audit.blocked_requests == 0
    if failure:
        assert result['outcome'] == 'transport_blocked' and len(result['rows']) == 1
        assert result['usage']['calls'] == result['usage']['unknown_usage_calls'] == 1
        assert result['usage']['input_tokens'] is result['usage']['output_tokens'] is None
        return
    assert result['outcome'] == 'comparisons_complete' and len(result['rows']) == 24
    assert {r['cap'] for r in result['request_caps'] if r['role'] in {'draft', 'integration'}} == {4096}
    for arm, planned in [('dual_no_replan', False), ('dual_replan', True)]:
        row = next(r for r in result['rows'] if r['case'] == 'main:0:events' and r['arm'] == arm)
        assert bool(row['task_goals']['replanning']) is planned
        assert row['answer_complete'] and row['grade']['exact'] and row['plan_state_cleared']
    safe = json.dumps(b.public(result))
    assert all(s not in safe for s in ('"answer":', '"initial_answer":', '"question":', '"quote":',
                                     'PRIVATE_UNCLASSIFIED_ERROR', 'FAKE_NO_CREDENTIAL'))


@pytest.mark.asyncio
async def test_changed_lineage_manifest_tasks_or_grader_close_admission(tmp_path):
    pool = b.tasks.build_fixture()
    with forbid_provider_network():
        parent, journal, fixture, plan = await parent_study(tmp_path, pool)
    corrected.validate(fixture, pool, parent, journal, plan, cfg())
    for key in ('parent_admissions', 'candidate_code_sha256', 'grader_sha256', 'excluded_case_ids',
                'parent_fixture_sha256', 'parent_canonical_sha256', 'selected_case_ids', 'phase_call_limit'):
        modified = deepcopy(plan)
        modified[key] = 'CHANGED'
        with pytest.raises(ValueError):
            corrected.validate(fixture, pool, parent, journal, modified, cfg())
    modified = deepcopy(fixture)
    modified['cases'][0]['question'] += '\nChanged input.'
    with pytest.raises(ValueError):
        corrected.validate(modified, pool, parent, journal, plan, cfg())
    modified = deepcopy(journal)
    modified['occurrences'][0]['error_type'] = 'RuntimeError'
    modified['occurrences'][0]['http_status'] = 429
    with pytest.raises(ValueError):
        corrected.validate(fixture, pool, parent, modified, plan, cfg())


@pytest.mark.asyncio
async def test_serialized_cap_mismatch_stops_before_http_without_shrinking_or_refund():
    budget = b.base.CampaignBudget()
    budget.admissions = 12
    journal = corrected.Journal(cfg(), {'test': 'actual-caps'}, campaign=budget, max_calls=96)
    seen = []
    class Transport:
        async def _post_json(self, url, payload, headers, **kwargs):
            seen.append(payload)
            return response('Complete answer.')
    journal.transport = Transport()
    journal.begin_trial('actual-cap')
    with forbid_provider_network() as audit:
        await journal.post('draft', 'fake', {'max_completion_tokens': 4096}, {})
        with pytest.raises(b.base.pilot.AdmissionError):
            await journal.post('integration', 'fake', {'max_completion_tokens': 2048}, {})
        with pytest.raises(b.base.pilot.AdmissionError):
            await journal.post('critic', 'fake', {'max_completion_tokens': 2048}, {})
    assert len(seen) == 1 and seen[0]['max_completion_tokens'] == 4096
    assert journal.configuration_fault and budget.admissions == 13 and journal.reserved == 4096
    assert audit.blocked_requests == 0
