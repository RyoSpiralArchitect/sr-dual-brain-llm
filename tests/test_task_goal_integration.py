"""Goal-specific inputs and revision flow through actual models/worker/controller."""
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_contract_live as pilot
from scheduled_http_journal import ScheduledJournal
from engine_stdio import _dispatch_request
from core.models import LeftBrainModel
from core.right_worker import _respond
from core.task_goals import current_task, extract_task_plan, worker_task
from core.thalamic_reticular_nucleus import TASK_GOAL_ORCHESTRATION as GOALS
from test_benchmark_contract_live import config, response
from test_dual_brain import AlwaysConsultPolicy, make_stage_controller
from test_task_goals import QUESTION, DRAFT, findings
from offline_provider_guard import forbid_provider_network

FINAL = DRAFT + ' Test with a full one-slot queue and a lock-holding producer; assert progress after releasing the lock.'


class Provider:
    def __init__(self, journal, contract='valid'):
        self.journal, self.contract, self.inputs = journal, contract, []

    async def _post_json(self, _url, payload, _headers, **_kwargs):
        role = self.journal.events[-1]['role']
        self.inputs.append((role, payload))
        if role == 'critic':
            reviewed = payload['messages'][-1]['content'].split('Draft answer (review for correctness):\n', 1)[1].split('\n\nReturn JSON only.', 1)[0]
            complete = 'Test with a full one-slot queue' in reviewed
            data = {'verdict': 'ok' if complete or self.contract == 'missing' else 'issues',
                    'issues': [] if complete or self.contract == 'missing' else ['The requested executable test is missing.'],
                    'fixes': [] if complete or self.contract == 'missing' else ['Supply a concrete test of producer progress.']}
            if self.contract != 'missing':
                data['goal_findings'] = findings()
                if complete:
                    data['goal_findings'][2] = {'goal_id': 'g3', 'status': 'addressed',
                        'quote': 'Test with a full one-slot queue', 'reason': 'A concrete test is supplied.'}
                if self.contract == 'invalid':
                    data['goal_findings'][0]['quote'] = 'PRIVATE_INVENTED_EVIDENCE'
            return response(json.dumps(data))
        if role == 'integration':
            return response(FINAL)
        if role == 'detail':
            return response('A producer holding the consumer lock while blocking on a full queue can prevent progress; a nonblocking offer with explicit retry is an alternative.')
        return response(DRAFT)


async def run_turn(tmp_path, *, mode='on', contract='valid', direct_fallback=False):
    journal = ScheduledJournal(config(), {'test': [mode, contract, direct_fallback]}, max_calls=20)
    provider = Provider(journal, contract)
    journal.transport = provider
    env = pilot.profile.environment()
    env['DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE'] = 'auto'
    with patch.dict(os.environ, env, clear=True), forbid_provider_network() as audit:
        session, _ = await pilot.make_session('goal-integration', None, config(), journal, [1794009600.])
        session.controller.policy = AlwaysConsultPolicy()  # Test-only positive stage exposure.
        if direct_fallback:
            async def unavailable(*_args, **_kwargs):
                return {'error': 'TestDeliveryFailure'}
            session.callosum.ask_detail = unavailable
        try:
            result = await _dispatch_request({'id': 'goal-test', 'method': 'process', 'params': {
                'session_id': session.session_id, 'question': QUESTION, 'qid': 'goal-test',
                'leading_brain': 'left', 'system2_mode': mode, 'executive_mode': 'off',
                'executive_observer_mode': 'off', 'return_telemetry': True}},
                sessions={session.session_id: session}, state_store=None, blob_pipe=None,
                blob_root=tmp_path, blob_index={}, vision_cache={}, vision_cache_limit=0, send=lambda *_: None)
            assert not session.controller.orchestrator.task_plans and current_task() is None
        finally:
            await session.close()
    journal.finish()
    assert audit.blocked_requests == 0 and result['ok']
    return result['result'], provider, journal


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['on', 'off'])
async def test_user_goals_reach_real_generation_consultation_and_integration(tmp_path, mode):
    result, provider, journal = await run_turn(tmp_path, mode=mode)
    metrics = result['metrics']
    assert metrics['modules']['gates'][GOALS]['enabled']
    assert 'Test with a full one-slot queue' in result['answer']
    wanted = {'draft', 'critic', 'integration'} if mode == 'on' else {'draft', 'detail', 'integration'}
    assert wanted <= {role for role, _ in provider.inputs}
    for role, payload in provider.inputs:
        if role in wanted:
            system = payload['messages'][0]['content']
            assert 'Task goal assignments' in system
            assert all(goal.text in system for goal in extract_task_plan(QUESTION).goals)
    assert bool(metrics['system2']['used']) is (mode == 'on')
    assert not metrics['task_goals']['verified_complete']
    assert journal.calls <= 8  # Existing bounded rounds; no per-goal planner loop.
    public = json.dumps(metrics['task_goals'])
    assert all(t not in public for t in ('consumer lock', 'counterexample', 'PRIVATE_', '"quote"', '"reason"'))


@pytest.mark.asyncio
@pytest.mark.parametrize('contract', ['missing', 'invalid'])
async def test_healthy_legacy_or_invalid_goal_review_cannot_become_goal_completion(tmp_path, contract):
    result, provider, _ = await run_turn(tmp_path, contract=contract)
    task = result['metrics']['task_goals']
    assert task['review_status'] == contract and not task['verified_complete']
    assert all(g['status'] != 'reported_addressed' for g in task['goals'])
    assert all('Your previous response' not in p['messages'][-1]['content'] for _, p in provider.inputs)


@pytest.mark.asyncio
async def test_direct_critic_fallback_keeps_the_goal_contract(tmp_path):
    result, provider, _ = await run_turn(tmp_path, direct_fallback=True)
    assert any(role == 'critic' for role, _ in provider.inputs)
    assert result['metrics']['task_goals']['review_status'] == 'valid'
    assert not result['metrics']['task_goals']['verified_complete']


def test_primary_write_vetoes_even_forced_goal_orchestration(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE', 'on')
    controller = make_stage_controller()
    controller.memory.conversation_context = True
    controller.memory.begin_turn(QUESTION, 'primary', {'id': 'task-scope', 'aliases': ['Task scope'], 'remember': True})
    gate = controller._prepare_task_goals(question=QUESTION, qid='primary', is_trivial_chat=False, vision_images=None)
    assert not gate.enabled and gate.reason == 'primary_write_pending' and current_task() is None


@pytest.mark.asyncio
async def test_source_generator_omits_task_assignments_even_inside_a_plan_context():
    captured = []
    class Client:
        async def complete(self, *args, **kwargs):
            captured.append(kwargs['system'])
            return 'Confirmed complete source text.'
    model = LeftBrainModel(config())
    model._llm_client = Client()
    with worker_task(QUESTION, extract_task_plan(QUESTION).payload()):
        await model.generate_source_answer(QUESTION, '')
    assert 'Task goal assignments' not in captured[0]


@pytest.mark.asyncio
async def test_worker_requires_canonical_question_anchors_for_role_instructions():
    seen = []
    class Model:
        async def deepen(self, *_args, **_kwargs):
            seen.append(current_task())
            return {'notes_sum': 'notes'}
    from core.shared_memory import SharedMemory
    payload = extract_task_plan(QUESTION).payload()
    request = {'qid': 'worker', 'type': 'ASK_DETAIL', 'question': QUESTION, 'task_plan': payload}
    await _respond(request, SharedMemory(), Model())
    request['task_plan'] = {**payload, 'hidden_goal': 'PRIVATE_TARGET'}
    await _respond(request, SharedMemory(), Model())
    assert seen[0] is not None and seen[1] is None and current_task() is None
    memory = SharedMemory(conversation_context=True)
    memory.begin_turn(QUESTION, 'source', {'id': 'task-scope', 'aliases': ['Task scope'], 'remember': True})
    request['task_plan'] = payload
    await _respond(request, memory, Model())
    assert seen[-1] is None
