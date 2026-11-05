from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import answer_assessment_v3 as assessment


def case():
    source = {'qid': 'record-current', 'retrieval_scope': 'maple',
              'question': 'Project Maple note?', 'answer': 'Café\n  4 days. ', 'status': 'unverified',
              'truncated': False, 'recorded_at': 100., 'valid_until': None, 'recalled_answer': False}
    return {'question': 'Repeat Project Maple recorded answer verbatim.',
            'reference': 'PRIVATE_EXPECTED_REFERENCE: never exposed to the answerer',
            'requested_operation': 'recorded_answer_replay',
            'replay_target': {key: source[key] for key in ('qid', 'retrieval_scope', 'answer')},
            'available_context': {'reference': 'resolved', 'approved_records': [source], 'evaluation_time': 200.,
                                  'source_checks': {key: True for key in assessment.SOURCE_CHECKS}}}


def scores(answer, verdict='pass'):
    return {key: {'verdict': verdict, 'quote': answer[:1] if answer else ''}
            for key in assessment.SEMANTIC_CRITERIA}


def test_malformed_expiry_is_not_represented_as_no_expiry_in_semantic_packet():
    fixture = case()
    record = fixture['available_context']['approved_records'][0]
    labels = {'A': {'answer': 'Please provide a current source.'}}
    for malformed in ('unknown', True, {'hidden_target': 'PRIVATE_VALUE'}):
        record['valid_until'] = malformed
        submitted = assessment.packet(fixture, labels)
        assert 'valid_until' not in submitted['available_context']['approved_records'][0]
        assert 'PRIVATE_VALUE' not in json.dumps(submitted)
    record['valid_until'] = None
    assert assessment.packet(fixture, labels)['available_context']['approved_records'][0]['valid_until'] is None


def test_per_source_packet_cannot_leak_guards_for_an_unexposed_identity():
    fixture = case()
    fixture['available_context']['source_checks_by_qid'] = {
        'record-current': {key: True for key in assessment.SOURCE_CHECKS},
        'PRIVATE_UNEXPOSED_ORACLE': {'current': True},
    }
    submitted = assessment.packet(fixture, {'A': {'answer': 'Recorded text.'}})
    assert set(submitted['available_context']['source_checks_by_qid']) == {'record-current'}
    assert 'PRIVATE_UNEXPOSED_ORACLE' not in json.dumps(submitted)


def legacy(answer, verdict='fail'):
    return {key: {'verdict': verdict, 'quote': answer[:1]} for key in assessment.bench.base.CRITERIA}


@pytest.mark.parametrize('answer', ['Cafe\u0301\n  4 days. ', 'Café\n  4 days.', 'Café\r\n  4 days. ',
                                   'Recorded answer: Café\n  4 days. '])
def test_equality_is_exact_bytes_without_unicode_or_whitespace_normalization(answer):
    fixture = case()
    result = assessment.diagnostics(fixture, answer, model_scores=scores(answer))
    assert result['exact_match']['verdict'] == 'fail'
    assert result['source_validity']['verdict'] == 'pass'
    assert result['attribution']['verdict'] == 'pass'
    assert result['information_sufficiency']['verdict'] == 'pass'
    exact = assessment.exact_match(fixture, fixture['replay_target']['answer'])
    assert exact['verdict'] == 'pass' and exact['answer_sha256'] == exact['target_sha256']


def test_exact_bytes_do_not_establish_source_validity_semantics_or_judge_error():
    fixture = case()
    fixture['available_context']['source_checks']['current'] = False
    fixture['available_context']['approved_records'][0]['status'] = 'superseded'
    answer = fixture['replay_target']['answer']
    old = legacy(answer)
    result = assessment.diagnostics(fixture, answer, legacy_model_grade=old)
    assert result['exact_match']['verdict'] == 'pass'
    assert result['source_validity']['verdict'] == 'fail'
    assert result['source_validity']['verifies_recorded_claims'] is False
    assert result['attribution']['verdict'] == result['information_sufficiency']['verdict'] == 'unknown'
    assert result['legacy_model_grade'] == old
    assert result['equality_task_disagreement'] == {'status': 'disagreement', 'exact_match': 'pass', 'model_task': 'fail'}
    result['legacy_model_grade']['task']['verdict'] = 'pass'
    assert old['task']['verdict'] == 'fail'


@pytest.mark.parametrize('key', assessment.SOURCE_CHECKS)
def test_each_declared_guard_is_independent_and_never_defaults_to_success(key):
    fixture = case()
    fixture['available_context']['source_checks'][key] = False
    assert assessment.source_validity(fixture)['verdict'] == 'fail'
    for malformed in (None, 1, 'true'):
        fixture['available_context']['source_checks'][key] = malformed
        result = assessment.source_validity(fixture)
        assert result['verdict'] == 'unknown' and result['checks'][key] is None


@pytest.mark.parametrize('changed', ['qid', 'retrieval_scope', 'answer'])
def test_exposed_source_must_match_target_identity_and_text(changed):
    fixture = case()
    fixture['available_context']['approved_records'][0][changed] = 'different'
    result = assessment.source_validity(fixture)
    assert result['verdict'] == 'fail' and result['checks']['target_binding'] is False


def test_absent_exposure_unknown_context_and_invalid_target_remain_distinct():
    fixture = case()
    answer = fixture['replay_target']['answer']
    fixture['available_context']['approved_records'] = []
    unavailable = assessment.diagnostics(fixture, answer)
    assert unavailable['exact_match']['verdict'] == 'pass'
    assert unavailable['source_validity']['verdict'] == 'fail'
    assert 'no_exposed_record' in unavailable['source_validity']['reasons']
    fixture['available_context'].pop('approved_records')
    assert assessment.source_validity(fixture)['verdict'] == 'unknown'
    fixture.pop('available_context')
    assert assessment.source_validity(fixture)['verdict'] == 'unknown'
    fixture['replay_target'].pop('answer')
    assert assessment.exact_match(fixture, answer)['verdict'] == 'unknown'
    fixture['requested_operation'] = 'fact_lookup'
    assert assessment.exact_match(fixture, answer)['verdict'] == 'not_applicable'


def test_semantic_packet_does_not_leak_unexposed_oracle_reference_or_old_grade():
    fixture = case()
    fixture['replay_target']['answer'] = 'PRIVATE_HIDDEN_TARGET'
    fixture['available_context']['approved_records'] = []
    fixture['available_context']['replay_target'] = fixture['replay_target']
    label = {'A': {'answer': 'Please supply the current source.', 'legacy_model_grade': {'PRIVATE_OLD': 'fail'}}}
    packet = assessment.packet(fixture, label)
    encoded = json.dumps(packet)
    assert set(packet) == {'question', 'requested_operation', 'available_context', 'answers'}
    assert not any(private in encoded for private in ('PRIVATE_HIDDEN_TARGET', 'PRIVATE_EXPECTED_REFERENCE', 'PRIVATE_OLD'))
    assert packet['available_context']['approved_records'] == []
    packet['available_context']['source_checks']['scope'] = False
    assert fixture['available_context']['source_checks']['scope'] is True


def test_nested_source_and_guard_metadata_cannot_leak_fixture_expectations():
    fixture = case()
    fixture['available_context']['approved_records'][0]['replay_target'] = 'PRIVATE_NESTED_TARGET'
    fixture['available_context']['source_checks']['expected_attribution'] = 'PRIVATE_EXPECTED_PASS'
    fixture['available_context']['source_checks_by_qid'] = {
        'record-current': {**fixture['available_context']['source_checks'], 'reference': 'PRIVATE_NESTED_REFERENCE'}}
    encoded = json.dumps(assessment.packet(fixture, {'A': {'answer': 'Visible answer.'}}))
    assert not any(value in encoded for value in ('PRIVATE_NESTED_TARGET', 'PRIVATE_EXPECTED_PASS', 'PRIVATE_NESTED_REFERENCE'))


@pytest.mark.parametrize('field', ['qid', 'recorded_at', 'truncated', 'status'])
def test_malformed_metadata_values_cannot_carry_nested_oracles(field):
    fixture = case()
    fixture['available_context']['approved_records'][0][field] = {'replay_target': 'PRIVATE_NESTED_ORACLE'}
    packet = assessment.packet(fixture, {'A': {'answer': 'Visible answer.'}})
    assert packet['available_context']['approved_records'][0][field] is None
    assert 'PRIVATE_NESTED_ORACLE' not in json.dumps(packet)
    assert assessment.source_validity(fixture)['verdict'] == 'unknown'


@pytest.mark.parametrize('field,value,check', [('status', 'superseded', 'current_status'),
                                            ('truncated', True, 'complete_text'),
                                            ('recalled_answer', True, 'primary_source'),
                                            ('answer', '', 'record_shape'),
                                            ('valid_until', 200., 'not_expired')])
def test_source_metadata_cannot_contradict_successful_declared_guards(field, value, check):
    fixture = case()
    fixture['available_context']['approved_records'][0][field] = value
    if field == 'answer':
        fixture['replay_target']['answer'] = value
    result = assessment.source_validity(fixture)
    assert all(result['declared_checks'].values())
    assert result['checks'][check] is False and result['verdict'] == 'fail'


@pytest.mark.parametrize('field', ['question', 'answer', 'qid', 'status', 'recorded_at', 'truncated', 'recalled_answer'])
def test_missing_required_source_metadata_is_unknown(field):
    fixture = case()
    fixture['available_context']['approved_records'][0].pop(field)
    assert assessment.source_validity(fixture)['verdict'] == 'unknown'


def test_expiry_requires_explicit_observation_time_and_reference_cannot_be_ambiguous():
    fixture = case()
    fixture['available_context']['approved_records'][0]['valid_until'] = 250.
    assert assessment.source_validity(fixture)['verdict'] == 'pass'
    fixture['available_context'].pop('evaluation_time')
    assert assessment.source_validity(fixture)['verdict'] == 'unknown'
    fixture['available_context']['evaluation_time'] = 250.
    assert assessment.source_validity(fixture)['verdict'] == 'fail'
    fixture['available_context']['approved_records'][0]['valid_until'] = None
    fixture['available_context']['reference'] = 'ambiguous'
    result = assessment.source_validity(fixture)
    assert result['verdict'] == 'fail' and result['checks']['reference_resolved'] is False


def test_multiple_fact_sources_need_per_record_guards_but_not_unique_source():
    fixture = case()
    fixture['requested_operation'] = 'fact_lookup'
    fixture['available_context']['reference'] = 'multiple_explicit'
    second = deepcopy(fixture['available_context']['approved_records'][0])
    second.update(qid='record-other', retrieval_scope='birch')
    fixture['available_context']['approved_records'].append(second)
    result = assessment.source_validity(fixture)
    assert result['verdict'] == 'unknown' and 'unique_source' not in result['checks']
    fixture['available_context']['source_checks_by_qid'] = {
        record['qid']: dict(fixture['available_context']['source_checks'])
        for record in fixture['available_context']['approved_records']}
    result = assessment.source_validity(fixture)
    assert result['verdict'] == 'pass' and 'target_binding' not in result['checks']
    fixture['available_context']['source_checks_by_qid']['record-other']['source_approved'] = False
    assert assessment.source_validity(fixture)['verdict'] == 'fail'


@pytest.mark.parametrize('guard', assessment.SOURCE_CHECKS)
def test_per_record_success_cannot_erase_known_aggregate_guard_failure(guard):
    fixture = case()
    fixture['available_context']['source_checks_by_qid'] = {
        'record-current': {key: True for key in assessment.SOURCE_CHECKS}}
    fixture['available_context']['source_checks'][guard] = False
    result = assessment.source_validity(fixture)
    assert result['verdict'] == 'fail' and result['checks'][guard] is False
    fixture['available_context'].pop('source_checks')
    assert assessment.source_validity(fixture)['verdict'] == 'pass'


@pytest.mark.parametrize('change', ['missing_label', 'extra_label', 'missing_criterion', 'extra_criterion',
                                  'nonliteral_quote', 'uncertain_nonliteral_quote', 'duplicate',
                                  'wrong_verdict', 'empty_pass_quote'])
def test_semantic_schema_fails_closed_on_malformed_or_unanchored_judgments(change):
    answer = 'The saved source says four days.'
    payload = {'scores': {'A': scores(answer)}}
    values = payload['scores']['A']
    if change == 'missing_label':
        payload['scores'] = {}
    elif change == 'extra_label':
        payload['scores']['B'] = deepcopy(values)
    elif change == 'missing_criterion':
        values.pop('attribution')
    elif change == 'extra_criterion':
        values['task'] = deepcopy(values['attribution'])
    elif change == 'nonliteral_quote':
        values['attribution']['quote'] = 'invented evidence'
    elif change == 'uncertain_nonliteral_quote':
        values['attribution'] = {'verdict': 'uncertain', 'quote': 'invented evidence'}
    elif change == 'wrong_verdict':
        values['attribution']['verdict'] = True
    elif change == 'empty_pass_quote':
        values['attribution']['quote'] = ''
    raw = json.dumps(payload)
    if change == 'duplicate':
        raw = raw.replace('"attribution":', '"attribution": {}, "attribution":', 1)
    with pytest.raises(ValueError):
        assessment.validate_scores(raw, {'A': {'answer': answer}})


def test_uncertainty_and_unavailable_assessment_are_not_failed_grades():
    fixture = case()
    answer = fixture['replay_target']['answer']
    uncertain = {key: {'verdict': 'uncertain', 'quote': ''} for key in assessment.SEMANTIC_CRITERIA}
    result = assessment.diagnostics(fixture, answer, model_scores=uncertain, legacy_model_grade=legacy(answer, 'uncertain'))
    assert result['attribution']['verdict'] == 'uncertain'
    assert result['equality_task_disagreement']['status'] == 'unknown'
    result = assessment.diagnostics(fixture, answer, model_error='ValueError', legacy_model_grade={'error_type': 'ValueError'})
    assert result['attribution']['verdict'] == result['information_sufficiency']['verdict'] == 'unknown'
    assert result['assessment_error'] == 'ValueError' and result['exact_match']['verdict'] == 'pass'
    result = assessment.diagnostics(fixture, answer, model_scores={})
    assert result['assessment_error'] == 'InvalidAssessment'
    assert result['attribution']['verdict'] == 'unknown'


def test_public_diagnostics_preserve_verdicts_without_source_text_or_quotes():
    fixture = case()
    answer = fixture['replay_target']['answer']
    result = assessment.diagnostics(fixture, answer, model_scores=scores(answer), legacy_model_grade=legacy(answer))
    public = assessment.public_diagnostics(result)
    assert public['legacy_model_verdicts'] == {key: 'fail' for key in assessment.bench.base.CRITERIA}
    assert public['equality_task_disagreement']['status'] == 'disagreement'
    assert 'quote' not in json.dumps(public) and 'Caf' not in json.dumps(public)
    assert 'replay_target' not in public and 'legacy_model_grade' not in public


@pytest.mark.asyncio
async def test_fake_assessment_keeps_four_axes_and_old_grade_independent():
    fixture = case()
    answer = fixture['replay_target']['answer']
    labels = {'A': {'answer': answer, 'legacy_model_grade': legacy(answer)}}
    captured = []

    class FakeJournal:
        async def post(self, role, url, payload, headers, **kwargs):
            captured.append(payload)
            return {'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'scores': {'A': scores(answer)}})}}]}

    config = assessment.bench.LLMConfig('openai', 'fake', 'PRIVATE_KEY')
    result = (await assessment.assess(fixture, labels, config, FakeJournal()))['A']
    assert result['exact_match']['verdict'] == result['source_validity']['verdict'] == 'pass'
    assert result['attribution']['verdict'] == result['information_sufficiency']['verdict'] == 'pass'
    assert result['legacy_model_grade']['task']['verdict'] == 'fail'
    assert result['equality_task_disagreement']['status'] == 'disagreement'
    submitted = json.loads(captured[0]['messages'][1]['content'])
    assert 'replay_target' not in submitted and 'reference' not in submitted
    fields = captured[0]['response_format']['json_schema']['schema']['properties']['scores']['properties']['A']
    assert fields['required'] == list(assessment.SEMANTIC_CRITERIA)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['extra_choice', 'refusal', 'truncated'])
async def test_fake_assessment_rejects_ambiguous_or_incomplete_provider_completions(fault):
    fixture = case()
    answer = fixture['replay_target']['answer']
    choice = {'finish_reason': 'stop', 'message': {'content': json.dumps({'scores': {'A': scores(answer)}})}}
    if fault == 'refusal':
        choice['message']['refusal'] = 'Cannot assess.'
    if fault == 'truncated':
        choice['finish_reason'] = 'length'

    class FakeJournal:
        async def post(self, *args, **kwargs):
            return {'choices': [choice, deepcopy(choice)] if fault == 'extra_choice' else [choice]}

    config = assessment.bench.LLMConfig('openai', 'fake', 'PRIVATE_KEY')
    with pytest.raises(ValueError):
        await assessment.assess(fixture, {'A': {'answer': answer}}, config, FakeJournal())
