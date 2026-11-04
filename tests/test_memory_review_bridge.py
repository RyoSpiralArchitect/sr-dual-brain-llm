import asyncio
from dataclasses import replace

import pytest

from core.anterior_cingulate import AnteriorCingulateCortex
from core.memory_answerability import AnswerabilityGate
from core.micro_critic import MicroCriticResult
from core.shared_memory import MemoryTrace, SharedMemory
from test_memory_answerability import Checks
from test_dual_brain import make_stage_controller, make_decision


def records():
    return [MemoryTrace('Project Cedar retention?', answer, qid=qid,
                        retrieval_scope='cedar', scope_aliases=('Cedar',))
            for qid, answer in [('old', 'Retention is 13 days.'), ('other', 'Retention is 6 days.')]]


async def prepared(traces=None):
    memory = SharedMemory(conversation_context=True, answerability_gate=AnswerabilityGate(Checks()))
    for trace in records() if traces is None else traces:
        memory.store(trace)
    memory.begin_turn('What is Project Cedar retention?', 'probe')
    await memory.prepare_retrieval('What is Project Cedar retention?')
    return memory


async def routing(monkeypatch, *, mode='auto', context=True, capable=True, promotion='auto'):
    monkeypatch.setenv('DUALBRAIN_MODULE_ACC_CONSULT_PROMOTION_MODE', promotion)
    monkeypatch.setenv('DUALBRAIN_MODULE_ACC_SYSTEM2_PROMOTION_MODE', promotion)
    memory = await prepared()
    controller = make_stage_controller(memory=memory)
    controller.anterior_cingulate_cortex = AnteriorCingulateCortex()
    question = 'What is Project Cedar retention?'
    parts = {'memory': memory.retrieve_related(question) if context else ''}
    frame = controller._build_context_frame(question=question, context_parts=parts)
    result = await controller._run_routing_stage(question=question, decision=make_decision(),
        draft='Retention is 6 days.', confidence=.99, affect={'risk': 0.}, salience_signal=None,
        context_parts=parts, context_frame=frame, system2_mode=mode, system2_priority='balanced',
        system2_active=False, system2_capable=capable, system2_reason='test', precision_priority=False,
        system2_low_signal_filter=True, director_task=None, director_payload=None, director_control=None,
        explicit_leading=True, force_right_lead=False, is_trivial_chat=False,
        phase_add=lambda *_: None, context_signal_len=len(parts['memory']), leading_brain='left')
    return result, frame, memory


def test_competing_supported_memory_reaches_acc_and_system2_without_arithmetic(monkeypatch):
    result, frame, memory = asyncio.run(routing(monkeypatch))
    assert result.system2_active and result.decision.action == 1
    assert 'memory' in result.acc_signal.sources
    assert result.decision.state['acc_memory_competition'] == 1
    assert 'Do not choose by recency' in frame.context
    assert all(trace.evidence_status == 'unverified' for trace in memory.past_qas)


@pytest.mark.parametrize('mode,context,capable', [('off', True, True), ('auto', False, True), ('auto', True, False)])
def test_memory_review_preserves_explicit_off_missing_context_and_capability(monkeypatch, mode, context, capable):
    result, _, _ = asyncio.run(routing(monkeypatch, mode=mode, context=context, capable=capable))
    assert not result.system2_active
    assert result.decision.action == 0


@pytest.mark.parametrize('change', ['same_answer', 'different_question', 'different_scope', 'expired', 'superseded', 'derived', 'denied', 'duplicate_identity'])
def test_memory_competition_requires_distinct_current_approved_sources(change):
    traces = records()
    second = traces[1]
    changes = {
        'same_answer': {'answer': traces[0].answer}, 'different_question': {'question': 'Project Cedar delay?'},
        'different_scope': {'retrieval_scope': 'birch', 'scope_aliases': ('Birch',)},
        'expired': {'valid_until': 0}, 'superseded': {'evidence_status': 'superseded'},
        'derived': {'recalled_answer': True}, 'denied': {'answer': 'DENY'},
        'duplicate_identity': {'qid': traces[0].qid},
    }
    traces[1] = replace(second, **changes[change])
    memory = asyncio.run(prepared(traces))
    assert not memory.competing_answer_groups('What is Project Cedar retention?')


def test_correction_and_expiry_invalidate_competition_and_cached_support():
    async def run():
        memory = await prepared()
        question = 'What is Project Cedar retention?'
        assert memory.competing_answer_groups(question) == (('old', 'other'),)
        memory.begin_turn('Project Cedar retention is 4 days.', 'new', {
            'id': 'cedar', 'aliases': ['Cedar'], 'remember': True, 'supersedes': ['old', 'other']})
        memory.store({'Q': 'Project Cedar retention is 4 days.', 'A': 'Retention is 4 days.'}, qid='new')
        memory.begin_turn(question, 'after')
        await memory.prepare_retrieval(question)
        assert not memory.competing_answer_groups(question)
        assert [t.qid for t in memory.retrieve_related_traces(question)] == ['new']
    asyncio.run(run())


def test_memory_cannot_bypass_promotion_off_through_temperature_or_basal(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_ACC_TEMPERATURE_DROP', '.3')
    result, _, _ = asyncio.run(routing(monkeypatch, promotion='off'))
    assert result.decision.action == 0 and not result.system2_active
    assert 'acc_temperature' not in result.decision.state


def test_configured_conflict_threshold_cannot_be_bypassed_by_basal(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_ACC_CONFLICT_THRESHOLD', '1.0')
    result, _, _ = asyncio.run(routing(monkeypatch))
    assert result.decision.action == 0 and not result.system2_active


@pytest.mark.parametrize('threshold,adjusts', [('.89', False), ('.75', True)])
def test_memory_signal_cannot_raise_weak_micro_evidence_into_temperature_authority(monkeypatch, threshold, adjusts):
    import core.dual_brain as dual
    monkeypatch.setenv('DUALBRAIN_ACC_CONFLICT_THRESHOLD', threshold)
    monkeypatch.setenv('DUALBRAIN_ACC_TEMPERATURE_DROP', '.1')
    micro = MicroCriticResult('issues', ['one issue'], [], 'issue', .9, 'arithmetic')
    monkeypatch.setattr(dual, 'micro_criticise_reasoning', lambda *_: micro)
    result, _, _ = asyncio.run(routing(monkeypatch))
    assert result.system2_active  # the independently eligible memory review still works
    assert ('acc_temperature' in result.decision.state) is adjusts
    if adjusts:
        independent = AnteriorCingulateCortex().monitor(question='', draft='', left_confidence=.99, micro=micro)
        assert result.decision.state['acc_temperature']['conflict'] == pytest.approx(independent.conflict_level)
