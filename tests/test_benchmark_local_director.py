from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_local_director as pilot
from test_benchmark_module_gating import fake_response


def normalize(rows):
    return sorted(json.dumps(row, sort_keys=True) for row in rows)


def test_new_local_sequence_matches_explicit_caller_requirements_and_keeps_neutral_cases():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    old = pilot.bench.read_json(pilot.bench.base.ROOT / 'examples/advisor_delivery_benchmark.json')
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in old['cases']}
    assert tuple(c['system2_mode'] for c in fixture['cases']) == ('on', 'auto', 'on', 'auto', 'on')
    assert all('exactly' in c['question'].lower() for c in fixture['cases'])
    for case in fixture['cases']:
        if case['system2_mode'] == 'on':
            assert 'independently' in case['question'].lower()
    broken = deepcopy(fixture)
    broken['cases'][0]['system2_mode'] = 'off'
    with pytest.raises(ValueError):
        pilot.validate(broken)


@pytest.mark.asyncio
async def test_local_application_calibration_and_all_recorded_occurrences_replay_without_transport(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    calibration = pilot.bench.read_json(pilot.assessment.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', max_output_tokens=2048,
        auto_continue=False, max_continuations=0)
    journal = pilot.recovery.sessions.SessionJournal(config, {}, max_calls=96)
    packets, requests = [], []
    expected = {}
    for case in calibration['pairs']:
        for variant, fields in case['outputs'].items():
            expected[json.dumps(fields, sort_keys=True)] = case['expected'][variant]

    async def transport(url, payload, headers, **kwargs):
        system = payload['messages'][0]['content']
        try:
            packet = json.loads(payload['messages'][-1]['content'])
        except ValueError:
            packet = {}
        if isinstance(packet, dict) and 'outputs' in packet:
            packets.append(packet)
            scores = {}
            for label, output in packet['outputs'].items():
                leaf = next((path for path, text in output['evidence_fields'].items() if text.strip()), '/memo')
                text = output['evidence_fields'][leaf]
                verdicts = expected.get(json.dumps(output['fields'], sort_keys=True), {})
                scores[label] = {criterion: {'verdict': verdicts.get(criterion, 'pass'),
                    'field_path': leaf, 'quote': text[:1], 'interpretation': 'assertion'}
                    for criterion in pilot.assessment.CRITERIA}
            content = {'scores': scores}
        elif 'post-answer metrics observer' in system:
            requests.append('metrics')
            content = {'memo': 'Only the supplied snippet supports a limited monitoring observation.', 'confidence': .8}
        elif 'Director' in system:
            requests.append('director')
            content = {'memo': 'Preserve supplied task scope.', 'control': {'consult': 'auto'}, 'confidence': .8}
        else:
            if 'Executive Reasoning' in system:
                requests.append('executive')
            return await fake_response(url, payload, headers, **kwargs)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
                'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}

    async def idle(_seconds):
        pass

    journal.transport._post_json = transport
    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, fixture, config, journal, idle=idle)
    assert live['complete'] and live['summaries']['jointly_graded_pairs'] == 5
    assert live['calibration_usage']['calls'] == 6
    assert live['calibration']['summary']['graded_outputs'] == 12
    assert live['calibration']['summary']['grader_failures'] == 0
    assert all(row['agreements'] == row['planned_checks'] for row in live['calibration']['summary']['criteria'].values())
    assert len(packets) == 11 and len([e for e in journal.events if e['role'] == 'advice_calibration']) == 6
    assert requests.count('executive') == requests.count('metrics') == 10
    rows = {(row['case'], row['arm']): row for row in live['rows']}
    for case in fixture['cases']:
        row = rows[(case['id'], 'local_facts')]
        if case['system2_mode'] == 'on':
            local = [o for o in row['advisor_outputs'] if o['role'] == 'director' and o['source'] == 'local_facts']
            assert len(local) == 1 and local[0]['fields']['control']['consult'] == 'force'
            assert row['local_before_draft'] and row['phase_order'][0] == 'local_controls_applied'
            assert row['local_director_stage_ms'] < 1600.
            assert row['routing']['system2_active'] and row['routing']['action'] >= 1
            assert not row['metrics']['modules']['advisor_timing']['director_advice']['started']
        baseline = rows[(case['id'], 'compact_remote')]
        assert not any(o['source'] == 'local_facts' for o in baseline['advisor_outputs'])
    assert all(row['status'] == 'ok' and row['pending_advisors'] == 0 for row in live['rows'])
    for packet in packets:
        assert all(set(output) == {'role', 'fields', 'evidence_fields', 'input_facts', 'final_answer', 'obligations'}
                   for output in packet['outputs'].values())
    frozen = deepcopy(journal.data)
    replay_journal = pilot.recovery.sessions.SessionJournal(config, {}, frozen=frozen, max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args, fixture, config, replay_journal)
    assert replay['complete'] and replay['live_provider_calls'] == replay['transport_calls'] == 0
    assert replay_journal.counts == journal.counts
    for left, right in zip(live['rows'], replay['rows']):
        assert all(left[key] == right[key] for key in ('case', 'arm', 'status', 'answer', 'draft', 'advisor_outputs',
            'memory_sources', 'pending_advisors', 'phase_order', 'local_before_draft', 'routing'))
        assert left['metrics']['modules']['gates'] == right['metrics']['modules']['gates']
        assert normalize(left['timing_reads']) == normalize(right['timing_reads'])
    assert live['grades'] == replay['grades'] and live['advice_grades'] == replay['advice_grades']
    assert live['calibration'] == replay['calibration']
    public = pilot.public_summary(live)
    assert public['advice_summary']['local_facts']['director']['sources']['local_facts']['available'] >= 3
    text = json.dumps(public)
    assert 'PRIVATE_KEY' not in text and 'User message:' not in text and 'Observer report (internal;' not in text
    assert all(o['text'] not in text for row in live['rows'] for o in row['advisor_outputs'] if o['text'])
    assert all('context' not in score for row in public.get('calibration', {}).get('rows', []) for score in row.get('scores', {}).values())
