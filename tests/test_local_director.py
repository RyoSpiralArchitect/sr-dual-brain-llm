from copy import deepcopy

import pytest

from core.director_controls import build_local_director_plan, LOCAL_DIRECTOR_REASONS
from core.hypothalamus import Hypothalamus


@pytest.mark.parametrize('facts', [
    {},
    {'system2_mode': 'auto', 'system2_capable': True, 'system2_active': True},
    {'system2_mode': 'off', 'system2_capable': True, 'system2_active': True},
    {'system2_mode': 'ON', 'system2_capable': True, 'system2_active': True},
    {'system2_mode': ' on ', 'system2_capable': True, 'system2_active': True},
    {'system2_mode': None, 'system2_capable': True, 'system2_active': True},
    {'system2_mode': 'on', 'system2_capable': False, 'system2_active': True},
    {'system2_mode': 'on', 'system2_capable': True, 'system2_active': False},
    {'system2_mode': 'on', 'system2_capable': 1, 'system2_active': True},
    {'system2_mode': 'on', 'system2_capable': True, 'system2_active': 'true'},
    {'long_term_restricted': 1},
    {'long_term_restricted': 'true'},
    {'long_term_restricted': None},
])
def test_unknown_facts_are_neutral_without_forbidding_ordinary_consultation(facts):
    plan = build_local_director_plan(**facts)
    assert plan.controls == {} and plan.reasons == () and not plan.nonneutral
    assert plan.source == 'local_facts'
    remote = {'consult': 'force', 'max_chars': 600}
    assert plan.merge(remote) == remote


def test_authoritative_facts_preserve_unrelated_remote_controls_and_nested_working_memory():
    plan = build_local_director_plan(system2_mode='on', system2_capable=True,
        system2_active=True, long_term_restricted=True)
    remote = {'consult': 'skip', 'memory': {'working_memory': 'keep', 'long_term': 'keep'},
              'temperature': .2, 'max_chars': 800, 'append_clarifying_question': 'Which region?'}
    original = deepcopy(remote)
    merged = plan.merge(remote)
    assert merged == {**remote, 'consult': 'force',
                      'memory': {'working_memory': 'keep', 'long_term': 'drop'}}
    assert remote == original
    merged['memory']['working_memory'] = 'drop'
    assert remote == original
    assert set(plan.reasons) <= LOCAL_DIRECTOR_REASONS and len(plan.reasons) == 2


def test_controls_and_payload_are_fresh_and_neutral_plans_do_not_share_state():
    plan = build_local_director_plan(long_term_restricted=True)
    controls = plan.controls
    controls['memory']['long_term'] = 'keep'
    payload = plan.to_payload()
    payload['controls']['memory']['long_term'] = 'keep'
    payload['reasons'].append('invented')
    assert plan.controls == {'memory': {'long_term': 'drop'}}
    assert plan.reasons == ('existing_long_term_restriction',)
    assert plan.merge({'memory': None}) == {'memory': {'long_term': 'drop'}}
    assert build_local_director_plan().controls == {}


def test_hypothalamus_fact_projection_never_teaches_external_history_or_feedback():
    owner = Hypothalamus()
    before = (owner._reward_avg, owner._latency_avg_ms)
    plan = owner.local_director_plan(system2_mode='on', system2_capable=True, system2_active=True)
    assert plan.controls == {'consult': 'force'}
    assert plan.nonneutral and plan.source == 'local_facts'
    assert owner.advisor_timing._history == {}
    assert (owner._reward_avg, owner._latency_avg_ms) == before


def test_no_question_memory_presence_or_write_intent_enters_the_local_fact_api():
    for unsupported in ('question', 'has_working_memory', 'has_long_term', 'remember'):
        with pytest.raises(TypeError):
            build_local_director_plan(**{unsupported: 'infer something'})
