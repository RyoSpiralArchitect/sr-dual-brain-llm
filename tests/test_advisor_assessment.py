from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import advisor_assessment as assessment


def fields(memo="Do not assert 'the archive is ready'; its status is unknown."):
    return {'memo': memo, 'control': {}, 'directives': {'do_not_say': ['the archive is ready']}, 'mix_in': ''}


def scores(*, path='/memo', quote='the archive is ready', verdict='pass', interpretation='prohibition'):
    return {'scores': {'A': {criterion: {'verdict': verdict, 'field_path': path, 'quote': quote,
        'interpretation': interpretation} for criterion in assessment.CRITERIA}}}


def test_selected_field_keeps_complete_negation_and_separate_prohibited_text():
    label = {'fields': fields()}
    raw = scores()
    parsed = assessment.validate_scores(json.dumps(raw), {'A': label})
    assert parsed['A']['grounding']['context'] == label['fields']['memo']
    assert parsed['A']['grounding']['quote'] == 'the archive is ready'
    assert parsed['A']['grounding']['interpretation'] == 'prohibition'
    # The same text in a prohibition list has an independent field provenance.
    raw['scores']['A']['relevance']['field_path'] = '/directives/do_not_say/0'
    parsed = assessment.validate_scores(json.dumps(raw), {'A': label})
    assert parsed['A']['relevance']['context'] == 'the archive is ready'
    assert parsed['A']['grounding']['context'].startswith('Do not assert')


@pytest.mark.parametrize('change', ['borrowed_field', 'invented_quote', 'truncated_context', 'unknown_interpretation', 'missing_criterion'])
def test_invalid_evidence_cannot_drop_negation_or_borrow_another_field(change):
    raw = scores()
    score = raw['scores']['A']['grounding']
    if change == 'borrowed_field':
        score['field_path'] = '/mix_in'
    elif change == 'invented_quote':
        score['quote'] = 'the export succeeded'
    elif change == 'truncated_context':
        score['context'] = 'the archive is ready'
    elif change == 'unknown_interpretation':
        score['interpretation'] = 'automatically_true'
    else:
        del raw['scores']['A']['critical_coverage']
    with pytest.raises(ValueError):
        assessment.validate_scores(json.dumps(raw), {'A': {'fields': fields()}})


def test_uncertain_empty_evidence_and_canonical_control_pointer_are_distinct():
    raw = scores(path='', quote='', verdict='uncertain', interpretation='uncertainty')
    parsed = assessment.validate_scores(json.dumps(raw), {'A': {'fields': fields()}})
    assert all(score['context'] == '' for score in parsed['A'].values())
    raw['scores']['A']['relevance']['verdict'] = 'pass'
    with pytest.raises(ValueError):
        assessment.validate_scores(json.dumps(raw), {'A': {'fields': fields()}})
    output = fields()
    output['control'] = {'a/b~c': 160, 'consult': 'force'}
    leaves = assessment.evidence_fields(output)
    assert leaves['/control/a~1b~0c'] == '160' and leaves['/control/consult'] == 'force'
    parsed = assessment.validate_scores(json.dumps(scores(path='/control/a~1b~0c', quote='160', interpretation='control')),
        {'A': {'fields': output}})
    assert parsed['A']['grounding']['context'] == '160'


def test_structured_packets_hide_policy_labels_but_preserve_operational_content():
    advice = SimpleNamespace(memo='Preserve scope.', control={'consult': 'force'}, confidence=.9,
        latency_ms=20, source='local_facts', directives={}, mix_in='')
    output = assessment.structured_output(advice)
    case = {'question': 'A scoped request.', 'reference': 'Scope remains explicit.',
            'advice_obligations': {'director': 'Keep requested consultation.'}}
    packet = assessment.build_packet(case, {'A': {'role': 'director', 'fields': output,
        'source': 'local_facts', 'arm': 'treatment', 'profile': 'PRIVATE_PROFILE', 'latency_ms': 20,
        'input_facts': {'caller_consultation': 'on'}, 'final_answer': 'A scoped answer.'}})
    actual = packet['outputs']['A']
    assert set(actual) == {'role', 'fields', 'evidence_fields', 'input_facts', 'final_answer', 'obligations'}
    assert actual['fields']['control']['consult'] == 'force'
    assert actual['input_facts'] == {'caller_consultation': 'on'}
    assert 'PRIVATE_PROFILE' not in json.dumps(packet) and 'local_facts' not in json.dumps(packet)
    with pytest.raises(ValueError):
        assessment.validate_fields(output | {'confidence': .9})
    with pytest.raises(ValueError):
        assessment.validate_fields(output | {'control': {'temperature': float('nan')}})
    with pytest.raises(ValueError):
        assessment.validate_scores('{"scores":{},"scores":{}}', {'A': {'fields': output}})


def test_calibration_is_independent_complete_and_keeps_adverse_scope_omissions():
    fixture = assessment.bench.read_json(assessment.FIXTURE)
    assessment.validate_calibration(fixture)
    assert len(fixture['pairs']) == 6
    assert all(case['expected']['asserted']['grounding'] == 'fail'
               and case['expected']['qualified']['grounding'] == 'pass' for case in fixture['pairs'])
    assert any(case['expected']['qualified'].get('relevance') == 'fail' for case in fixture['pairs'])
    assert any(case['expected']['qualified'].get('critical_coverage') == 'fail' for case in fixture['pairs'])
    broken = deepcopy(fixture)
    broken['pairs'].pop()
    with pytest.raises(ValueError):
        assessment.validate_calibration(broken)


def calibration_response(packet, fixture):
    case = next(case for case in fixture['pairs'] if case['question'] == packet['question'])
    result = {}
    for label, output in packet['outputs'].items():
        variant = next(key for key, value in case['outputs'].items() if value == output['fields'])
        path, text = next((key, text) for key, text in output['evidence_fields'].items() if text)
        result[label] = {criterion: {'verdict': case['expected'][variant].get(criterion, 'pass'),
            'field_path': path, 'quote': text[:10],
            'interpretation': 'prohibition' if variant == 'qualified' else 'assertion'} for criterion in assessment.CRITERIA}
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'scores': result})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}


@pytest.mark.asyncio
async def test_six_calibration_packets_count_attempts_preserve_disagreement_and_replay_without_transport():
    fixture = assessment.bench.read_json(assessment.FIXTURE)
    config = assessment.bench.LLMConfig('openai', 'gpt-6-sol', 'PRIVATE_KEY', auto_continue=False)
    journal = assessment.bench.Journal(config, {}, max_calls=96)
    packets = []

    async def transport(_url, payload, _headers, **_kwargs):
        packet = json.loads(payload['messages'][-1]['content'])
        assert payload['max_completion_tokens'] == 4096
        assert all(set(output) == {'role', 'fields', 'evidence_fields', 'input_facts', 'final_answer', 'obligations'}
                   for output in packet['outputs'].values())
        packets.append(packet)
        response = calibration_response(packet, fixture)
        if len(packets) == 1:
            # Deliberately disagree without rewriting the expected result.
            data = json.loads(response['choices'][0]['message']['content'])
            label = next(iter(data['scores']))
            original = data['scores'][label]['grounding']['verdict']
            data['scores'][label]['grounding']['verdict'] = 'fail' if original == 'pass' else 'pass'
            response['choices'][0]['message']['content'] = json.dumps(data)
        return response

    journal.transport._post_json = transport
    live = await assessment.calibrate(config, journal, fixture)
    journal.finish()
    assert journal.calls == len(packets) == 6
    assert {event['role'] for event in journal.events} == {'advice_calibration'}
    assert live['summary']['graded_outputs'] == 12 and live['summary']['grader_failures'] == 0
    assert live['summary']['criteria']['grounding']['disagreements'] == 1
    assert all('context' in score for row in live['rows'] for score in row['scores'].values())
    replay_journal = assessment.bench.Journal(config, {}, frozen=deepcopy(journal.data), max_calls=96)
    replay_journal.transport = None
    replay = await assessment.calibrate(config, replay_journal, fixture)
    replay_journal.finish()
    assert replay_journal.calls == 0 and replay == live
    public = json.dumps(live['summary'])
    assert 'PRIVATE_KEY' not in public and all(case['question'] not in public for case in fixture['pairs'])


@pytest.mark.asyncio
async def test_malformed_calibration_packet_keeps_failure_and_does_not_retry():
    fixture = assessment.bench.read_json(assessment.FIXTURE)
    config = assessment.bench.LLMConfig('openai', 'test', 'OFFLINE')
    journal = assessment.bench.Journal(config, {}, max_calls=96)
    attempts = [0]

    async def transport(_url, payload, _headers, **_kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            return {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}
        return calibration_response(json.loads(payload['messages'][-1]['content']), fixture)

    journal.transport._post_json = transport
    result = await assessment.calibrate(config, journal, fixture)
    assert attempts[0] == 6 and result['summary']['grader_failures'] == 1
    assert result['summary']['graded_outputs'] == 10
    assert result['summary']['criteria']['grounding']['planned_checks'] == 12
    assert result['summary']['criteria']['grounding']['available_checks'] == 10
    assert any(row.get('error_type') == 'ValueError' for row in result['rows'])
