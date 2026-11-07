"""Full real-store path with delayed previews and exact scheduled replay."""
import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_preview_admission as study
from test_benchmark_contract_live import config, response
from test_benchmark_natural_memory import Provider
from offline_provider_guard import forbid_provider_network


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['completed', 'cancelled', 'failed_write'])
async def test_delayed_natural_trajectory_and_transport_schedule_replay(tmp_path, mode):
    dsn = os.environ.get('DUALBRAIN_TEST_PG_DSN')
    if not dsn:
        pytest.skip('Requires isolated PostgreSQL')
    fixture = study.pilot.bench.read_json(study.FIXTURE)
    cfg = config()
    live, replay = tmp_path / 'live', tmp_path / 'replay'
    live.mkdir()
    replay.mkdir()
    journal = study.ScheduledJournal(cfg, study.metadata(fixture, cfg), max_calls=study.CALL_LIMIT)
    class DelayedProvider(Provider):
        async def _post_json(self, url, payload, headers, **kwargs):
            role = journal.events[-1]['role']
            if role == 'right':
                if mode == 'cancelled':
                    await asyncio.Event().wait()
                await asyncio.sleep(0)
                return response('Preview of the requested operation.')
            if role == 'draft':
                await asyncio.sleep(.002)
                if mode == 'failed_write' and '61 days' in payload['messages'][-1]['content']:
                    return response('')
            else:
                await asyncio.sleep(0)
            return await super()._post_json(url, payload, headers, **kwargs)
    journal.transport = DelayedProvider(study.pilot.bench.read_json(study.natural.CALIBRATION))
    with forbid_provider_network() as audit:
        result = await study.run(live, fixture, cfg, journal, dsn)
        frozen = study.ScheduledJournal(cfg, study.metadata(fixture, cfg), max_calls=study.CALL_LIMIT,
                                        frozen=json.loads(json.dumps(journal.data)))
        repeated = await study.run(replay, fixture, cfg, frozen, dsn)
    assert result['complete'] and len(result['rows']) == 12
    assert frozen.transport is None and frozen.calls == 0 and audit.blocked_requests == 0
    assert study.canonical(result) == study.canonical(repeated)
    assert result['transport_event_count'] == result['usage']['calls'] * 2
    assert all(g['status'] == 'not_assessed' for g in result['grades'])
    if mode == 'failed_write':
        assert not result['trajectory_complete'] and result['contract_failures']
        assert any(r['status'] == 'not_executed' for r in result['rows'])
    else:
        assert result['trajectory_complete'] and not result['contract_failures']
        assert sum(r['contracts'].get('exact_quote') is True for r in result['rows']) == 2
        assert any(r.get('preview', {}).get('started') for r in result['rows'])
        if mode == 'cancelled':
            assert result['usage']['cancelled_calls'] > 0 and result['usage']['input_tokens'] is None
            assert any(r.get('preview', {}).get('outcome') == 'cancelled_after_start' for r in result['rows'])
    public = json.dumps(study.public_summary(result))
    assert all(t not in public for t in ('"answer":', '"question":', '"messages":', '"record_ref":', 'r1.', 'FAKE_NO_CREDENTIAL'))
