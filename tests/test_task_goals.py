"""User anchors, provisional review state, and turn-local ownership."""
import asyncio
from copy import deepcopy
import json

import pytest

from core.orchestrator import Orchestrator
from core.task_goals import (
    TaskState, bind_task, current_task, extract_task_plan, goal_review,
    guard_task_turn, plan_from_payload, worker_task,
)

QUESTION = ('Compare two queue designs.\nRequirements:\n'
            '1. Explain a bounded queue strategy.\n'
            '2. Give a concrete counterexample.\n'
            '3. Using step 1, describe an executable test.')
DRAFT = 'A bounded queue blocks a producer. A producer holding the consumer lock is a counterexample.'


def findings():
    return [
        {'goal_id': 'g1', 'status': 'addressed', 'quote': 'A bounded queue', 'reason': 'A strategy is proposed.'},
        {'goal_id': 'g2', 'status': 'addressed', 'quote': 'consumer lock', 'reason': 'A specific counterexample is given.'},
        {'goal_id': 'g3', 'status': 'unresolved', 'quote': '', 'reason': 'The requested executable test is absent.'},
    ]


@pytest.mark.parametrize('question', [QUESTION,
    '要件：\n- 制約付きの計画を説明する。\n- 成り立たない反例も示す。',
    '```python\nprint("data")\n```\nGoals:\n1. Explain the behavior.\n2. Give a test.'])
def test_plan_has_exact_user_anchors_and_bounded_dependencies(question):
    plan = extract_task_plan(question)
    assert plan is not None and 2 <= len(plan.goals) <= 4
    assert all(question[g.start:g.end] == g.text for g in plan.goals)
    assert plan_from_payload(question, plan.payload()) == plan
    if question == QUESTION:
        assert plan.goals[2].depends_on == ('g1',)


@pytest.mark.parametrize('question', [
    'What is 2 + 2?', 'Requirements:\n1. Say hello.',
    'Requirements:\n1. First.\n3. Third.',
    'Requirements:\n1. First.\n2. Second.\nThis extra requirement must not be dropped.',
    'Requirements:\n1. Using step 2, do something.\n2. Second.',
    'Requirements:\n- First.\n- Using step 1, do something.',
    'Requirements:\n1. First.\n2. Second.\n3. Third.\n4. Fourth.\n5. Fifth.',
    'Requirements:\n1. ' + 'x' * 401 + '\n2. Second.',
    '> Requirements:\n> 1. First.\n> 2. Second.',
    '```\nRequirements:\n1. First.\n2. Second.\n```',
    '~~~\n```\nRequirements:\n1. First.\n2. Second.',
    'Requirements:\n1. First.\n2. Second.\nGoals:\n1. Other.\n2. Other.',
    'Quote this text exactly:\nRequirements:\n1. First.\n2. Second.',
])
def test_unsupported_partial_quoted_or_literal_tasks_stay_direct(question):
    assert extract_task_plan(question) is None


@pytest.mark.parametrize('change', ['anchor', 'text', 'boolean', 'dependency', 'extra', 'other_question'])
def test_worker_rejects_modified_or_unanchored_plan(change):
    payload = deepcopy(extract_task_plan(QUESTION).payload())
    question = QUESTION
    if change == 'anchor': payload['goals'][0]['start'] += 1
    elif change == 'text': payload['goals'][0]['text'] = 'Invented hidden target'
    elif change == 'boolean': payload['goals'][0]['start'] = True
    elif change == 'dependency': payload['goals'][0]['depends_on'] = ['g3']
    elif change == 'extra': payload['expected'] = 'PRIVATE_ORACLE'
    else: question += ' Changed'
    assert plan_from_payload(question, payload) is None


@pytest.mark.parametrize('change', ['unknown_id', 'duplicate', 'missing', 'invented_quote', 'empty_positive', 'bad_status', 'empty_reason'])
def test_invalid_review_cannot_claim_goal_coverage(change):
    value = findings()
    if change == 'unknown_id': value[0]['goal_id'] = 'g99'
    elif change == 'duplicate': value[1]['goal_id'] = 'g1'
    elif change == 'missing': value.pop()
    elif change == 'invented_quote': value[0]['quote'] = 'PRIVATE_UNSEEN_TEXT'
    elif change == 'empty_positive': value[0]['quote'] = ''
    elif change == 'bad_status': value[0]['status'] = 'verified_complete'
    else: value[0]['reason'] = ''
    result = goal_review(extract_task_plan(QUESTION), DRAFT, value, external_valid=True, verdict='issues')
    assert result['status'] == 'invalid' and result['findings'] == []


def test_reports_never_prove_completion_and_revisions_require_rechecking():
    plan = extract_task_plan(QUESTION)
    state = TaskState(plan)
    review = goal_review(plan, DRAFT, findings(), external_valid=True, verdict='issues')
    state.observe({'goal_review': review, 'critic_external_review_valid': True, 'verdict': 'issues'}, DRAFT)
    assert state.needs_revision(DRAFT)
    before = state.summary(DRAFT)
    assert not before['verified_complete'] and before['goals'][0]['status'] == 'reported_addressed'
    assert before['goals'][2]['status'] == 'reported_unresolved'
    revised = DRAFT + ' A new test may or may not resolve the problem.'
    assert not state.needs_revision(revised)
    after = state.summary(revised)
    assert not after['review_snapshot_matches_answer']
    assert all(g['status'] == 'unreviewed_answer' for g in after['goals'])
    state.observe({'critic_external_review_valid': True}, revised)
    assert state.latest['status'] == 'missing' and not state.latest['findings']
    assert 'counterexample' not in json.dumps(after) and 'consumer lock' not in json.dumps(after)


def test_ok_verdict_with_unresolved_goal_is_not_a_valid_goal_review():
    result = goal_review(extract_task_plan(QUESTION), DRAFT, findings(), external_valid=True, verdict='ok')
    assert result['status'] == 'invalid'


def test_worker_context_does_not_inherit_an_unrelated_parent_plan():
    plan = extract_task_plan(QUESTION)
    with worker_task(QUESTION, plan.payload()):
        parent = current_task()
        assert parent is not None
        with worker_task(QUESTION, None):
            assert current_task() is None
        assert current_task() is parent
    assert current_task() is None


@pytest.mark.asyncio
async def test_concurrent_turns_and_abandonment_clear_owned_plan_state():
    owner = Orchestrator()
    entered = asyncio.Event()
    class Processor:
        @guard_task_turn
        async def run(self, qid, wait=False):
            assert current_task() is None
            bind_task(owner.begin_task(qid, owner.propose_task(QUESTION)))
            if wait:
                entered.set()
                await asyncio.Event().wait()
            await asyncio.sleep(0)
            assert current_task().qid == qid
            return current_task().plan
    processor = Processor()
    waiting = asyncio.create_task(processor.run('waiting', True))
    await entered.wait()
    result = await processor.run('ordinary')
    assert result.question == QUESTION and set(owner.task_plans) == {'waiting'}
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert not owner.task_plans and current_task() is None


@pytest.mark.asyncio
async def test_plan_cleanup_cannot_reset_an_existing_roundtrip_limit():
    owner = Orchestrator(max_roundtrips=1)
    assert owner.register_request('limited')
    assert not owner.register_request('limited')
    class Processor:
        @guard_task_turn
        async def run(self):
            bind_task(owner.begin_task('limited', owner.propose_task(QUESTION)))
    await Processor().run()
    assert not owner.task_plans and owner.active_roundtrips['limited'] == 2


def test_task_goal_admission_is_off_by_default_and_rejects_unknown_counts():
    from core.thalamic_reticular_nucleus import (
        TASK_GOAL_ORCHESTRATION, GateMode, GateConfig, GateFacts, ThalamicReticularNucleus,
    )
    gate = ThalamicReticularNucleus()
    assert gate.configure(TASK_GOAL_ORCHESTRATION, {}).mode == GateMode.OFF
    for count in (0, 1, 5, True, None, '3', 3.0):
        assert not gate.decide(TASK_GOAL_ORCHESTRATION,
            config=GateConfig(GateMode.AUTO, 'explicit', True),
            facts=GateFacts(eligible=True, task_goal_count=count)).enabled
    for count in (2, 3, 4):
        assert gate.decide(TASK_GOAL_ORCHESTRATION,
            config=GateConfig(GateMode.AUTO, 'explicit', True),
            facts=GateFacts(eligible=True, task_goal_count=count)).enabled
