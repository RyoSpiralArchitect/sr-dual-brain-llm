"""Fixed profile exposure, failure preservation and exact replay; no quality claim."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_reasoning_effort as study
from offline_provider_guard import forbid_provider_network
from test_benchmark_contract_live import config, response
from test_benchmark_task_goals import Provider as GoalProvider


class Provider(GoalProvider):
    async def _post_json(self, url, payload, headers, **kwargs):
        if self.journal.events[-1]['role'] != 'task_goal_judge' and '19 + 23' in payload['messages'][-1]['content']:
            if self.failure == 'transport' and len(self.journal.events) == 1:
                return await super()._post_json(url, payload, headers, **kwargs)
            return response('42')
        return await super()._post_json(url, payload, headers, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['none', 'transport', 'judge'])
async def test_fresh_pairs_expose_only_expected_effort_and_replay_all_attempts(tmp_path, failure):
    fixture = study.pilot.bench.read_json(study.FIXTURE)
    cfg = replace(config(), model='gpt-6-luna', max_output_tokens=2048, timeout_seconds=60,
                  auto_continue=False, max_continuations=0)
    judge = replace(cfg, model='gpt-6-sol')
    live, repeat = tmp_path/'live', tmp_path/'replay'
    live.mkdir();repeat.mkdir()
    journal = study.ScheduledJournal(cfg, study.metadata(fixture, cfg, judge), max_calls=64)
    journal.transport = Provider(journal, failure)
    with forbid_provider_network() as audit:
        report = await study.run(live, fixture, cfg, judge, journal)
        frozen = study.ScheduledJournal(cfg, report['metadata'], max_calls=64,
            frozen=json.loads(json.dumps(journal.data)))
        replay = await study.run(repeat, fixture, cfg, judge, frozen)
    assert study.canonical(report) == study.canonical(replay)
    assert report['complete'] and len(report['rows']) == 10 and len(report['grades']) == 5
    assert frozen.calls == audit.blocked_requests == 0
    assert journal.calls <= 64
    if failure == 'transport':
        assert journal.calls == 1 and journal.blocked and report['usage']['unknown_usage_calls'] == 1
    else:
        for row in report['rows']:
            assert row['policy_class'] == 'RightBrainPolicy' and row['plan_state_cleared']
            assert row['reasoning_profile'] in study.PROFILES
            assert row['task_goals'] is None if row['case'] == 'direct_control' else row['task_goals']['goal_count'] == 3
        for item in journal.data['occurrences']:
            request = item['request']; arm = request['case'].rsplit(':', 1)[-1]
            expected = 'low' if arm == 'low' and request['role'] in {'draft', 'integration', 'detail'} else None
            assert request['payload'].get('reasoning_effort') == expected
        assert sum(g['status'] == 'graded' for g in report['grades']) == (4 if failure == 'judge' else 5)
    public = json.dumps(study.public_summary(report))
    assert all(s not in public for s in ('"question":', '"answer":', '"quote":', 'FAKE_NO_CREDENTIAL'))
