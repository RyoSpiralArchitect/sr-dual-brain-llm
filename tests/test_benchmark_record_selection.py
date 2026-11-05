"""The fresh confirmation must retain support receipts and exact offline replay."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_record_selection as confirmation
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network


class PositiveSupportTransport:
    async def _post_json(self, _url, payload, _headers, **_kwargs):
        messages = payload['messages']
        try:
            packet = json.loads(messages[-1]['content'])
        except ValueError:
            packet = {}
        if 'records' in packet:
            text = json.dumps({'decisions': [{'id': r['id'], 'supported': True, 'quote': r['answer']}
                                            for r in packet['records']]})
        elif 'critic' in messages[0]['content'].lower():
            text = json.dumps({'verdict': 'ok', 'issues': [], 'fixes': [], 'confidence_r': .9})
        elif 'Conversation missing-evidence condition:' in messages[0]['content']:
            text = 'Which existing complete record do you want me to repeat?'
        else:
            text = 'A supported ordinary answer.'
            for line in messages[0]['content'].splitlines():
                if line.startswith('{"source_question":'):
                    text = json.loads(line)['recorded_answer']
                    break
        return response(text)


@pytest.mark.asyncio
async def test_positive_cache_cannot_bypass_selection_and_the_whole_study_replays(tmp_path):
    fixture = confirmation.pilot.bench.read_json(confirmation.FIXTURE)
    first, second = tmp_path / 'first', tmp_path / 'replay'
    first.mkdir()
    second.mkdir()
    metadata = confirmation.metadata(fixture, config())
    journal = confirmation.pilot.RawJournal(config(), metadata, max_calls=confirmation.CALL_LIMIT)
    journal.transport = PositiveSupportTransport()
    with patch.dict(os.environ, {'DUALBRAIN_MEMORY_EMBEDDING_MODEL': '', 'DUALBRAIN_MEMORY_ANSWERABILITY_MODEL': ''}):
        with forbid_provider_network():
            report = await confirmation.run(first, fixture, config(), journal)
        assert report['complete'] and len(report['rows']) == 9 and not report['contract_failures']
        for row in report['rows'][:5]:
            assert row['support_preparation']['decisions']
            assert all(value is True for value in row['support_preparation']['decisions'].values())
            assert row['source_qids'] == [] and not row['packet_leak_roles']
            assert row['condition']['action'] == 'clarify_record'
        assert report['rows'][0]['system2']['used'] is True
        assert report['rows'][3]['system2']['used'] is False
        assert all(r['diagnostics']['exact_match']['verdict'] == 'pass' for r in report['rows'][5:7])
        assert report['rows'][7]['contracts']['required_source_available'] is True
        assert report['rows'][8]['intent'] is None and report['rows'][8]['condition'] is None
        replay_journal = confirmation.pilot.RawJournal(config(), metadata,
            frozen=deepcopy(journal.data), max_calls=confirmation.CALL_LIMIT)
        with forbid_provider_network():
            replay = await confirmation.run(second, fixture, config(), replay_journal)
        assert replay_journal.transport is None and replay_journal.calls == 0
        assert confirmation.canonical(report) == confirmation.canonical(replay)
        assert report['usage']['calls'] == sum(r['usage']['calls'] for r in report['rows'])
        assert report['usage']['calls'] == sum(report['calls_by_role'].values())
        for row in report['rows']:
            assert row['usage']['calls'] == sum(row['calls_by_role'].values())
        public = json.dumps(confirmation.pilot.public_summary(report))
        assert not any(value in public for value in ('FAKE_NO_CREDENTIAL', 'WITHHELD_COBALT_RECORD', '"payload"', '"answer"'))
