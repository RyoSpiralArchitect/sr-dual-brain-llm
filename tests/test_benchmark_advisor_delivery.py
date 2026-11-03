from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_advisor_delivery as pilot
from test_benchmark_module_gating import fake_response


def normalize(rows):
    return sorted(json.dumps(row, sort_keys=True) for row in rows)


def test_new_questions_and_role_obligations_are_complete_and_explicit():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    previous = pilot.bench.read_json(pilot.confidence.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in previous['cases']}
    assert all('exactly' in c['question'].lower() for c in fixture['cases'])
    assert sum('independently cross-check' in c['question'] for c in fixture['cases']) == 2
    broken = deepcopy(fixture)
    del broken['cases'][0]['advice_obligations']['metrics']
    with pytest.raises(ValueError):
        pilot.validate(broken)


def test_coverage_omission_and_grounding_have_distinct_validated_literal_evidence():
    outputs = {'A': {'text': 'memo: Read only. No expiry control.'}}
    scores = {c: {'verdict': 'pass', 'quote': 'Read only.'} for c in pilot.ADVICE_CRITERIA}
    scores['critical_coverage'] = {'verdict': 'fail', 'quote': 'No expiry control.'}
    parsed = pilot.validate_advice(json.dumps({'scores': {'A': scores}}), outputs)
    assert parsed['A']['grounding']['verdict'] == 'pass'
    assert parsed['A']['critical_coverage']['verdict'] == 'fail'
    scores['critical_coverage']['quote'] = 'imagined missing audit text'
    with pytest.raises(ValueError):
        pilot.validate_advice(json.dumps({'scores': {'A': scores}}), outputs)
    with pytest.raises(ValueError):
        pilot.validate_advice('{"scores":{},"scores":{}}', outputs)
    del scores['critical_coverage']
    with pytest.raises(ValueError):
        pilot.validate_advice(json.dumps({'scores': {'A': scores}}), outputs)


@pytest.mark.asyncio
async def test_retained_compact_requests_coverage_failures_and_profile_reads_replay_exactly(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'test', 'PRIVATE_KEY', max_output_tokens=2048,
                                  auto_continue=False, max_continuations=0)
    journal = pilot.recovery.sessions.SessionJournal(config, {}, max_calls=96)
    requests, packets = [], []

    async def transport(url, payload, headers, **kwargs):
        system = payload['messages'][0]['content']
        try:
            packet = json.loads(payload['messages'][-1]['content'])
        except ValueError:
            packet = {}
        if isinstance(packet, dict) and 'outputs' in packet:
            packets.append(packet)
            # This synthetic omission exercises accounting, not real utility.
            content = {'scores': {label: {criterion: {
                'verdict': 'fail' if criterion == 'critical_coverage' and output['role'] == 'director' else 'pass',
                'quote': output['text'][:1]} for criterion in pilot.ADVICE_CRITERIA}
                for label, output in packet['outputs'].items()}}
        elif 'post-answer metrics observer' in system:
            requests.append(('metrics_compact', payload))
            content = {'memo': 'Visible snippet supports only a bounded monitoring check.', 'confidence': .8}
        elif 'Director' in system and 'exactly memo, control, confidence' in system:
            requests.append(('director_compact', payload))
            content = {'memo': 'Keep task-specific scope.', 'control': {'consult': 'auto'}, 'confidence': .8}
        else:
            if 'Director' in system:
                requests.append(('director_full', payload))
            elif 'Executive Reasoning' in system:
                role = 'metrics_full' if 'Observer report (internal;' in payload['messages'][-1]['content'] else 'executive_full'
                requests.append((role, payload))
            return await fake_response(url, payload, headers, **kwargs)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
                'model': 'test', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}

    async def idle(_seconds):
        pass

    journal.transport._post_json = transport
    args = SimpleNamespace(output_dir=tmp_path, judge_model='test')
    live = await pilot.run(args, fixture, config, journal, idle=idle)
    assert live['complete'] and live['summaries']['jointly_graded_pairs'] == 5
    assert all(r['status'] == 'ok' and r['pending_advisors'] == 0 and r['memory_sources'] == r['session_turn'] for r in live['rows'])
    kinds = {kind for kind, _ in requests}
    assert kinds == {'director_compact', 'metrics_compact', 'director_full', 'metrics_full', 'executive_full'}
    assert len([1 for kind, _ in requests if kind == 'executive_full']) == 10
    for kind, payload in requests:
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        assert cap == (256 if kind.endswith('_compact') else 2048)
    rows = {(r['case'], r['arm']): r for r in live['rows']}
    for module in ('director_advice', 'executive_metrics_observation', 'executive_advice'):
        profiles = []
        for arm in pilot.ARMS:
            reads = rows[('initial', arm)]['timing_reads']
            profiles.append(next(r['profile'] for r in reads if r['module'] == module))
        assert (profiles[0] == profiles[1]) == (module == 'executive_advice')
    assert len(packets) == 5
    for packet in packets:
        assert all(set(o) == {'role', 'text', 'final_answer', 'input_facts', 'obligations'} for o in packet['outputs'].values())
        assert any(o['role'] == 'metrics' and 'observer_context' in o['input_facts'] for o in packet['outputs'].values())
        for output in packet['outputs'].values():
            facts = output['input_facts']
            if output['role'] == 'director':
                actual = 'Signals (JSON):\n' + facts['signals_json'] + '\n\nContext (may be empty):\n' + facts['context'] + '\n\n'
                assert any(actual in payload['messages'][-1]['content'] for kind, payload in requests if kind.startswith('director'))
            elif output['role'] == 'metrics':
                actual = 'Context (may be empty):\n' + facts['observer_context'] + '\n\nFocus keywords:'
                assert any(actual in payload['messages'][-1]['content'] for kind, payload in requests if kind.startswith('metrics'))
    frozen = deepcopy(journal.data)
    replay_journal = pilot.recovery.sessions.SessionJournal(config, {}, frozen=frozen, max_calls=96)
    replay_journal.transport = None
    replay = await pilot.run(args, fixture, config, replay_journal)
    assert replay['complete'] and replay['live_provider_calls'] == 0 and replay_journal.counts == journal.counts
    for left, right in zip(live['rows'], replay['rows']):
        assert all(left[k] == right[k] for k in ('case', 'arm', 'status', 'answer', 'draft', 'advisor_outputs', 'memory_sources', 'pending_advisors'))
        assert left['metrics']['modules']['gates'] == right['metrics']['modules']['gates']
        assert normalize(left['timing_reads']) == normalize(right['timing_reads'])
    assert live['grades'] == replay['grades'] and live['advice_grades'] == replay['advice_grades']
    public = pilot.public_summary(live)
    assert public['advice_summary']['compact_delivery']['director']['criterion_outcomes']['critical_coverage']['fail'] == 5
    assert public['advice_summary']['full_v4']['metrics']['missing_exposure'] == 0
    text = json.dumps(public)
    assert 'PRIVATE_KEY' not in text and 'User message:' not in text and 'Observer report (internal;' not in text
    assert all(o['text'] not in text for row in live['rows'] for o in row['advisor_outputs'])
