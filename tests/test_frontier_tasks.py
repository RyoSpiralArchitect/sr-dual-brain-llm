"""Independent oracle cross-checks, state ownership and strict objective grading."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/scripts'))
import frontier_tasks as f
from core.task_goals import extract_task_plan

FIXTURE = Path(__file__).resolve().parents[1]/'sr-dual-brain-llm/examples/reasoning_frontier.json'


@pytest.mark.parametrize('level', [1, 2, 3])
def test_bitmask_entailment_matches_independent_boolean_enumeration(level):
    data, _ = f.make_logic(random.Random(51+level), level)
    models = []
    for values in itertools.product((False, True), repeat=len(data['variables'])):
        if all(any(values[abs(x)-1] if x > 0 else not values[-x-1] for x in clause) for clause in data['clauses']):
            models.append(sum(1 << i for i, value in enumerate(values) if value))
    assert sorted(models) == f.logic_models(data)
    assert models


def test_sql_bag_null_tie_and_exclusive_ban_boundary_against_hand_result():
    data = {'cutoff': 5, 'accounts': [['A'], ['B'], ['C']],
        'events': [[1,'B',4,'credit',3], [2,'B',5,'void',None], [3,'B',6,'credit',7]],
        'tags': [['A','blue'], ['B','active'], ['B','active'], ['B','blue']],
        'bans': [['B',2,5,'hold'], ['C',0,None,'hold']]}
    expected = {'rows': [
        {'account':'A','credit_total':0,'joined_events':0,'tag_kinds':1,'latest_event':None,'latest_kind':None},
        {'account':'B','credit_total':6,'joined_events':6,'tag_kinds':2,'latest_event':1,'latest_kind':'credit'}]}
    assert f.sql_answer(data) == f.sql_reference(data) == expected


@pytest.mark.parametrize('level', range(1, 8))
def test_generated_sql_has_independent_relational_agreement(level):
    for seed in range(3):
        data, _ = f.make_sql(random.Random(70+10*level+seed), level)
        assert f.sql_answer(data) == f.sql_reference(data)


def test_cancellation_ack_finish_epoch_and_owner_transitions():
    prefix = [['submit','J',1],['claim','J',1,'A'],['cancel','J',1]]
    def result(events): return f.events_answer({'jobs':['J'],'events':events})['jobs']['J']
    pending = result(prefix)
    assert pending == {'epoch':1,'state':'cancel_requested','owner':'A','result':None}
    assert result(prefix+[['finish','J',1,'A','x'],['ack_cancel','J',1,'A']]) == {
        'epoch':1,'state':'committed','owner':None,'result':'x'}
    assert result(prefix+[['ack_cancel','J',1,'A'],['finish','J',1,'A','x']]) == {
        'epoch':1,'state':'cancelled','owner':None,'result':None}
    assert result(prefix+[['transfer','J',1,'A','B'],['ack_cancel','J',1,'B']]) == pending
    assert result(prefix+[['submit','J',2],['claim','J',2,'B'],['submit','J',2],
        ['finish','J',1,'A','old'],['ack_cancel','J',2,'B']]) == {
            'epoch':2,'state':'running','owner':'B','result':None}
    assert result([['submit','J',1],['claim','J',1,'A'],['transfer','J',1,'A','B'],
        ['finish','J',1,'A','wrong'],['finish','J',1,'B','correct']])['result'] == 'correct'


def test_independent_job_interleavings_commute_and_terminal_duplicates_are_idempotent():
    data, _ = f.make_events(random.Random(891), 5)
    grouped = sorted(data['events'], key=lambda e: e[1])  # Stable per-job order.
    assert f.events_answer(data) == f.events_answer(data | {'events':grouped})
    terminal = {'jobs':['J'],'events':[['submit','J',1],['claim','J',1,'A'],['finish','J',1,'A','x']]}
    assert f.events_answer(terminal) == f.events_answer(terminal | {'events':terminal['events']*2})


def source(identity, **changes):
    return {'id':identity,'scope':'S','topic':'T','kind':'primary','value':'red','issued_at':1,
        'valid_from':1,'valid_to':None,'expires_at':None,'revoked_at':None,'supersedes':[],**changes}


def test_expired_replacement_does_not_resurrect_old_or_gain_cross_scope_authority():
    data = {'sources':[source('a'),source('b',issued_at=2,expires_at=3,supersedes=['a']),
        source('d',kind='derived',value='fake',supersedes=['b']),
        source('foreign',scope='other',supersedes=['a','b'])],
        'queries':[{'id':'q','scope':'S','topic':'T','time':4}]}
    assert f.evidence_answer(data) == {'queries':{'q':{'status':'unknown','value':None,'source_ids':[]}}}


def test_conflicting_current_sources_and_exclusive_validity_revocation_boundaries():
    data = {'sources':[source('a'),source('b',value='blue'),source('c',value='blue',valid_to=4),
        source('d',revoked_at=4),source('future',issued_at=5)],
        'queries':[{'id':'q','scope':'S','topic':'T','time':4}]}
    assert f.evidence_answer(data)['queries']['q'] == {'status':'conflict','value':None,'source_ids':['a','b']}
    data['sources'][1]['value'] = 'red'
    assert f.evidence_answer(data)['queries']['q'] == {'status':'supported','value':'red','source_ids':['a','b']}


def test_materialized_fixture_is_reproducible_bounded_and_has_two_goals_per_task():
    fixture = json.loads(FIXTURE.read_text())
    assert f.build_fixture() == fixture and len(fixture['cases']) == 52
    for case in fixture['cases']:
        assert case['expected'] == f.ORACLES[case['domain']](case['data'])
        assert f.grade(case,json.dumps(case['expected']))['exact']
        assert len(case['question']) <= 10000
        plan = extract_task_plan(case['question'])
        assert plan is not None and len(plan.goals) == 2,case['id']


@pytest.mark.parametrize('answer', [None, '', '{}', '[]', '```json\n{}\n```', '{"status":NaN}',
    '{"status":{},"status":{}}', '['*1000])
def test_malformed_and_incomplete_answers_cannot_pass(answer):
    case = f.make_case('holdout', 1, 'logic')
    assert not f.grade(case, answer)['exact']


def test_grading_separates_valid_semantic_error_from_wrong_types_and_source_set_order():
    case = f.make_case('holdout',1,'events')
    answer = deepcopy(case['expected'])
    key = next(iter(answer['jobs']))
    answer['jobs'][key]['epoch'] += 1
    grade = f.grade(case,json.dumps(answer))
    assert grade['schema_valid'] and not grade['exact'] and 0 < grade['item_accuracy'] < 1
    answer['jobs'][key]['epoch'] = True
    assert not f.grade(case,json.dumps(answer))['schema_valid']
    case = {'expected':{'queries':{'q':{'status':'supported','value':'x','source_ids':['a','b']}}}}
    answer = {'queries':{'q':{'status':'supported','value':'x','source_ids':['b','a']}}}
    assert f.grade(case,json.dumps(answer))['exact']
    answer['queries']['q']['source_ids'] = ['a','a','b']
    assert not f.grade(case,json.dumps(answer))['schema_valid']
