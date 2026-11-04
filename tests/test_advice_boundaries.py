from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import advisor_assessment as v1
import advisor_assessment_v2 as v2
import benchmark_advice_boundaries as pilot


def output(memo='Do not claim completion: it remains unknown.'):
    return {'memo': memo, 'control': {}, 'directives': {'do_not_say': ['Completion succeeded.']}, 'mix_in': ''}


def response_scores(fields, verdicts=None, *, modern=True):
    path, text = next((path, text) for path, text in v1.evidence_fields(fields).items() if text)
    return {c: {'verdict': (verdicts or {}).get(c, 'pass'), 'field_path': path,
        'quote': text[:12], 'interpretation': 'prohibition',
        **({'basis': v2.BASIS[c][0]} if modern else {})} for c in v1.CRITERIA}


def test_specialized_schema_keeps_label_and_criterion_basis_independent():
    modern = v2.schema(['A', 'B'])['json_schema']['schema']['properties']['scores']['properties']
    for label in modern.values():
        for criterion, score in label['properties'].items():
            assert score['properties']['basis']['enum'] == list(v2.BASIS[criterion])
            assert score['required'].count('basis') == 1
    assert 'basis' not in v1.schema(['A'])['json_schema']['schema']['properties']['scores']['properties']['A']['properties']['grounding']['properties']


@pytest.mark.parametrize('change', ['scope_as_grounding', 'untyped_basis', 'borrowed_quote', 'invented_context', 'duplicate_key'])
def test_basis_is_criterion_specific_and_complete_evidence_remains_validated(change):
    fields = output()
    scores = {'scores': {'A': response_scores(fields)}}
    score = scores['scores']['A']['grounding']
    if change == 'scope_as_grounding':
        score['basis'] = 'task_scope'
    elif change == 'untyped_basis':
        score['basis'] = []
    elif change == 'borrowed_quote':
        score['field_path'] = '/mix_in'
    elif change == 'invented_context':
        score['context'] = 'Completion succeeded.'
    else:
        with pytest.raises(ValueError):
            v2.validate_scores('{"scores":{},"scores":{}}', {'A': {'fields': fields}})
        return
    with pytest.raises(ValueError):
        v2.validate_scores(json.dumps(scores), {'A': {'fields': fields}})


def test_validation_retains_prohibition_context_without_automatically_passing_it():
    fields = output('Do not claim calibration is unknown, because calibration passed.')
    scores = response_scores(fields, {'grounding': 'fail'})
    scores['grounding']['basis'] = 'factual_support'
    normalized = v2.validate_scores(json.dumps({'scores': {'A': scores}}), {'A': {'fields': fields}})
    assert normalized['A']['grounding']['verdict'] == 'fail'
    assert normalized['A']['grounding']['context'] == fields['memo']
    assert normalized['A']['grounding']['interpretation'] == 'prohibition'
    assert 'context' not in scores['grounding']


def test_fixture_is_fresh_and_encodes_scope_authority_omissions_and_observer_access():
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    pilot.validate(fixture)
    old = pilot.bench.read_json(v1.FIXTURE)
    assert not {c['question'] for c in fixture['cases']} & {c['question'] for c in old['pairs']}
    indexed = {c['id']: c for c in fixture['cases']}
    for case_id, name in [('invoice_decoration', 'harmless_aside'), ('limiter_check_count', 'extra_check')]:
        expected = indexed[case_id]['variants'][name]['expected']
        assert expected['grounding'] == 'pass' and expected['relevance'] == 'fail'
    assert indexed['inventory_authority']['variants']['unauthorized_write']['expected']['grounding'] == 'fail'
    assert indexed['sensor_qualification']['variants']['omitted_units']['expected']['grounding'] == 'pass'
    assert indexed['sensor_qualification']['variants']['omitted_units']['expected']['critical_coverage'] == 'fail'
    assert indexed['sensor_qualification']['variants']['false_rationale']['expected']['grounding'] == 'fail'
    assert indexed['observer_access']['variants']['borrowed_full_answer']['expected']['grounding'] == 'fail'
    for case in fixture['cases']:
        a, names_a = pilot.labels_for(case, 0)
        b, names_b = pilot.labels_for(case, 1)
        assert set(names_a.values()) == set(names_b.values())
        assert list(names_a.values()) == list(names_b.values())[::-1]
        packet = v1.build_packet(case, a)
        assert all(set(o) == {'role', 'fields', 'evidence_fields', 'input_facts', 'final_answer', 'obligations'}
                   for o in packet['outputs'].values())
    broken = deepcopy(fixture)
    del broken['cases'][0]['variants']['unsupported']['expected']['grounding']
    with pytest.raises(ValueError):
        pilot.validate(broken)


async def fake_transport_for(fixture, *, disagree=False, malformed=False):
    expected = {json.dumps(v['fields'], sort_keys=True): v['expected']
                for case in fixture['cases'] for v in case['variants'].values()}
    calls = []

    async def post(_url, payload, _headers, **_kwargs):
        packet = json.loads(payload['messages'][-1]['content'])
        modern = payload['messages'][0]['content'] == v2.INSTRUCTIONS
        assert payload['max_completion_tokens'] == 4096
        assert not any(key in packet for key in ('expected', 'variants', 'arm', 'order'))
        for value in packet['outputs'].values():
            assert not any(key in value for key in ('expected', 'variant', 'arm', 'source', 'basis'))
        scores = {label: response_scores(o['fields'], expected[json.dumps(o['fields'], sort_keys=True)], modern=modern)
                  for label, o in packet['outputs'].items()}
        if disagree and not calls:
            scores['A']['grounding']['verdict'] = 'fail' if scores['A']['grounding']['verdict'] == 'pass' else 'pass'
        calls.append(payload)
        return {'choices': [{'finish_reason': 'length' if malformed and len(calls) == 1 else 'stop',
            'message': {'content': json.dumps({'scores': scores})}}], 'model': 'test',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}

    return post, calls


@pytest.mark.asyncio
async def test_paired_study_preserves_original_disagreement_counts_and_exact_http_replay(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'gpt-6-sol', 'PRIVATE_KEY', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    post, calls = await fake_transport_for(fixture, disagree=True)
    journal.transport._post_json = post
    args = SimpleNamespace(output_dir=tmp_path)
    before = deepcopy(fixture)
    live = await pilot.run(args, fixture, config, journal)
    assert fixture == before
    assert live['complete'] and len(live['rows']) == len(calls) == pilot.CALL_LIMIT == 24
    stats = live['summary']['by_arm']
    assert sum(s['criteria']['grounding']['disagreements'] for s in stats.values()) == 1
    assert sum(len(s['label_order_disagreements']) for s in stats.values()) == 1
    assert all(s['criteria']['grounding']['planned_checks'] == 36 for s in stats.values())
    replay_journal = pilot.bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=pilot.CALL_LIMIT)
    replay_journal.transport = None
    replay = await pilot.run(args, fixture, config, replay_journal)
    assert replay['complete'] and replay['live_provider_calls'] == 0
    assert replay_journal.counts == journal.counts and replay['rows'] == live['rows']
    assert replay['summary'] == live['summary']
    public = json.dumps(pilot.public_summary(live))
    assert 'PRIVATE_KEY' not in public and '"context"' not in public and '"quote"' not in public
    assert all(c['question'] not in public and all(v['fields']['memo'] not in public for v in c['variants'].values())
               for c in fixture['cases'])


@pytest.mark.asyncio
async def test_incomplete_completion_retains_invalid_packet_without_retry(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'gpt-6-sol', 'OFFLINE', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    journal.transport._post_json, calls = await fake_transport_for(fixture, malformed=True)
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path), fixture, config, journal)
    assert len(calls) == 24 and report['rows'][0]['error_type'] == 'ValueError'
    assert sum(s['valid_packets'] for s in report['summary']['by_arm'].values()) == 23
    assert sum(s['graded_outputs'] for s in report['summary']['by_arm'].values()) == 69
    assert report['usage']['calls'] == 24 and report['usage']['known_output_tokens'] == 480


@pytest.mark.asyncio
async def test_transport_failure_blocks_further_paid_calls_but_retains_planned_missing_cells(tmp_path):
    fixture = pilot.bench.read_json(pilot.FIXTURE)
    config = pilot.bench.LLMConfig('openai', 'gpt-6-sol', 'OFFLINE', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=pilot.CALL_LIMIT)
    calls = []

    async def failed(*a, **kw):
        calls.append(True)
        raise OSError('injected transport failure')

    journal.transport._post_json = failed
    report = await pilot.run(SimpleNamespace(output_dir=tmp_path), fixture, config, journal)
    assert len(calls) == report['usage']['calls'] == 1
    assert len(report['rows']) == 24 and all('error_type' in r for r in report['rows'])
    assert sum(s['unattempted_packets'] for s in report['summary']['by_arm'].values()) == 23
    assert report['usage']['unknown_usage_calls'] == 1 and report['usage']['output_tokens'] is None
    replay_journal = pilot.bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=pilot.CALL_LIMIT)
    replay_journal.transport = None
    replay = await pilot.run(SimpleNamespace(output_dir=tmp_path), fixture, config, replay_journal)
    assert replay['rows'] == report['rows'] and replay['live_provider_calls'] == 0


@pytest.mark.asyncio
async def test_v2_report_grading_is_explicit_and_does_not_change_v1_defaults():
    fields = output()
    case = {'id': 'one', 'question': 'A request.', 'reference': 'Unknown.',
            'advice_obligations': {'executive': 'Keep the outcome unknown.'}}
    config = pilot.bench.LLMConfig('openai', 'gpt-6-sol', 'OFFLINE', auto_continue=False)
    journal = pilot.bench.Journal(config, {}, max_calls=1)

    async def post(_url, payload, _headers, **kwargs):
        assert payload['messages'][0]['content'] == v2.INSTRUCTIONS
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'scores': {'A': response_scores(fields)}})}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}

    journal.transport._post_json = post
    report = {'rows': [{'case': 'one', 'arm': 'treatment', 'answer': 'Unknown.', 'advisor_outputs': [
        {'role': 'executive', 'fields': fields, 'source': 'llm'}]}]}
    await v2.grade_report(report, [case], config, journal)
    assert report['advice_grades'][0]['scores']['grounding']['basis'] == 'factual_support'
    assert v1.INSTRUCTIONS != v2.INSTRUCTIONS and v1.FIXTURE != pilot.FIXTURE
