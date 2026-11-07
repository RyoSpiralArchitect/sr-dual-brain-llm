"""Pre-draft inhibition never reads an eventual answer or policy action."""
import json
import asyncio
from pathlib import Path
import sys

import pytest

from core.memory_answerability import AnswerabilityGate
from core.shared_memory import SharedMemory, MemoryTrace
from core.thalamic_reticular_nucleus import (
    DEFAULT_REGISTRY, GateFacts, GateConfig, GateMode, ThalamicReticularNucleus,
    SPECULATIVE_PREVIEW_SUPPRESSION as PREVIEW,
)
from test_dual_brain import make_stage_controller
from test_memory_answerability import Checks


QUESTION = 'Repeat the recorded answer verbatim.'


def controller():
    memory = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Checks()))
    for qid, question in [('period', 'Project Flint Reach data-retention period'),
                           ('backup', 'Project Flint Reach backup location')]:
        memory.store(MemoryTrace(question, 'PRIVATE_SOURCE_' + qid, qid=qid,
            retrieval_scope='flint-reach', scope_aliases=('Flint Reach',)))
    memory.begin_turn(QUESTION, 'private-qid', {'id': 'flint-reach', 'aliases': ['Flint Reach'], 'remember': False})
    return make_stage_controller(memory=memory)


def gate(c, **changes):
    args = dict(question=QUESTION, qid='private-qid', leading='right', collaborative_lead=False,
        requested_leading='', system2_mode='auto', system2_active=False,
        precision_priority=False, director_control=None)
    args.update(changes)
    return c._preview_suppression_gate(**args)


@pytest.mark.parametrize('mode,enabled', [('off', False), ('on', True), ('auto', True)])
def test_pending_choice_is_known_without_generation_or_source_check(monkeypatch, mode, enabled):
    monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, mode)
    c = controller()
    def forbidden(*_args, **_kwargs):
        pytest.fail('A draft/policy decision must not be requested at this boundary')
    c.left_model.generate_answer = forbidden
    c.policy.decide = forbidden
    observed = gate(c)
    assert observed.enabled is enabled
    assert 'PRIVATE' not in json.dumps(observed.to_payload()) and 'Flint' not in json.dumps(observed.to_payload())
    if mode == 'auto': assert observed.reason == 'record_choice_pending'


@pytest.mark.parametrize('change,reason', [
    ({'requested_leading': 'right'}, 'right_lead'),
    ({'system2_mode': 'on'}, 'explicit_system2_mode'),
    ({'system2_mode': 'off'}, 'explicit_system2_mode'),
    ({'precision_priority': True}, 'precision_priority'),
    ({'director_control': {'consult': 'force'}}, 'director_override'),
    ({'leading': 'left', 'collaborative_lead': False}, 'outside_scope'),
    ({'system2_active': True}, 'outside_scope'),
])
@pytest.mark.parametrize('mode', ['on', 'auto'])
def test_protected_requests_veto_suppression(monkeypatch, change, reason, mode):
    monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, mode)
    observed = gate(controller(), **change)
    assert not observed.enabled and observed.reason == reason


@pytest.mark.parametrize('mode', ['on', 'auto'])
def test_pending_primary_write_is_protected(monkeypatch, mode):
    monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, mode)
    c = controller()
    c.memory.begin_turn(QUESTION, 'write', {'id': 'flint-reach', 'aliases': ['Flint Reach'], 'remember': True})
    observed = gate(c)
    assert not observed.enabled and observed.reason == 'primary_write_pending'


@pytest.mark.parametrize('question', ['Project Flint Reach retention?', 'What is Project Flint Reach retention rate?',
    'Summarize the recorded answer.', 'Repeat the recorded answer verbatim and verify it.'])
def test_auto_retains_preview_for_other_conditions_and_extra_operations(monkeypatch, question):
    monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, 'auto')
    c = controller()
    c.memory.begin_turn(question, 'probe', {'id': 'flint-reach', 'aliases': ['Flint Reach'], 'remember': False})
    observed = gate(c, question=question)
    assert not observed.enabled and observed.reason == 'preview_context_needed'


def test_default_is_off_and_auto_accepts_only_a_true_finite_fact():
    nucleus = ThalamicReticularNucleus()
    assert nucleus.configure(PREVIEW, {}).mode == GateMode.OFF
    for value in (False, None, 1, 'true', {}):
        observed = nucleus.decide(PREVIEW, config=GateConfig(GateMode.AUTO, 'explicit', True),
            facts=GateFacts(eligible=True, record_choice_pending=value))
        assert not observed.enabled


def test_unavailable_condition_preserves_auto_preview(monkeypatch):
    monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, 'auto')
    c = controller()
    def unavailable(*_args):
        raise ValueError('PRIVATE_SOURCE_FAILURE')
    c.memory.missing_evidence = unavailable
    observed = gate(c)
    assert not observed.enabled and observed.reason == 'preview_context_needed'
    assert 'PRIVATE' not in json.dumps(observed.to_payload())


@pytest.mark.asyncio
async def test_production_draft_stage_omits_one_fake_http_preview_without_replacing_the_draft(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
    from benchmark_contract_live import StageClient
    from scheduled_http_journal import ScheduledJournal
    from test_benchmark_contract_live import config, response
    from offline_provider_guard import forbid_provider_network
    observations = []
    for mode in ('off', 'auto'):
        monkeypatch.setenv(DEFAULT_REGISTRY[PREVIEW].mode_env, mode)
        c = controller()
        journal = ScheduledJournal(config(), {'test': mode})
        class Provider:
            async def _post_json(self, *_args, **_kwargs):
                return response('Optional preview notes.')
        journal.transport = Provider()
        c.right_model._llm_client = StageClient(config(), journal, 'right')
        c.right_model.uses_external_llm = True
        draft = 'Please choose which complete record to quote from the current record choices.'
        async def left(*_args, **_kwargs):
            # Synthetic delay provides positive exposure in this offline control.
            await asyncio.sleep(.005)
            return draft
        c.left_model.generate_answer = left
        decision = gate(c)
        with forbid_provider_network() as audit:
            stage = await c._run_left_draft_stage(question=QUESTION, context='', focus=None,
                leading='right', collaborative_lead=False, system2_active=False, vision_images=None,
                delta_cb=None, stream_final_only=False, emitted_any=False, focus_keywords=[],
                focus_metric=0., hemisphere_mode='balanced', hemisphere_bias=0., phase_add=lambda *_: None,
                suppress_preview=decision.enabled, qid='control')
        journal.finish()
        assert audit.blocked_requests == 0 and stage.draft == draft
        assert c.memory.missing_evidence(QUESTION).action == 'clarify_record'
        observations.append((journal.calls, stage.preview_observation['outcome']))
    assert observations == [(1, 'completed'), (0, 'suppressed')]
