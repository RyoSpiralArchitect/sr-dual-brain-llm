"""Final routing, actual requests and usable reviews are distinct observations."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from core.anterior_cingulate import ConflictSignal
from core.director_reasoner import DirectorAdvice
from core.dual_brain import System2Status
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing
from core.thalamic_reticular_nucleus import (
    DEFAULT_REGISTRY, ACC_CONSULT_PROMOTION, ACC_SYSTEM2_PROMOTION,
    ACC_TEMPERATURE_ADJUSTMENT, CEREBELLAR_CORRECTION, DIRECTOR_ADVICE,
)
from test_dual_brain import AlwaysConsultPolicy, CaptureLeftModel, make_stage_controller

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
from engine_stdio import _extract_metrics


CLEAN = {
    'verdict': 'ok', 'issues': [], 'fixes': [], 'critic_sum': 'No issues detected.',
    'confidence_r': .9, 'critic_kind': 'external', 'critic_status': 'ok',
    'critic_review_valid': True, 'critic_external_review_valid': True,
    'critic_validation': {'initial': {'status': 'valid', 'error_type': None}, 'repair': None},
}
ISSUES = {**CLEAN, 'verdict': 'issues', 'issues': ['The draft omits a stated condition.'],
          'fixes': ['Include that condition.'], 'critic_sum': 'A stated condition is missing.'}
INVALID = {
    **CLEAN, 'verdict': 'issues', 'critic_status': 'empty_response',
    'issues': ['(fallback) External critic returned an empty response; unable to verify reasoning.'],
    'critic_sum': 'External critic response empty.', 'confidence_r': 0.,
    'critic_review_valid': False, 'critic_external_review_valid': False,
    'critic_validation': {'initial': {'status': 'empty_response', 'error_type': None}, 'repair': None},
}


class SnapshotTelemetry:
    """Catch accidental reliance on mutating a previously emitted policy event."""
    def __init__(self):
        self.events = []

    def log(self, event, **payload):
        self.events.append({'event': event, **deepcopy(payload)})


class ScriptedCallosum:
    slot_ms = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    async def ask_detail(self, payload, *, timeout_ms):
        self.payloads.append(deepcopy(payload))
        assert self.responses, 'Unexpected additional consultation'
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


class SkipDirector:
    async def advise(self, **kwargs):
        return DirectorAdvice(memo='Fixed control', control={'consult': 'skip'},
                              confidence=.9, latency_ms=0., source='test')


@pytest.fixture(autouse=True)
def offline_fixed_settings(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Provider transport is forbidden in this regression')
    monkeypatch.setattr('core.llm_client.aiohttp.ClientSession', forbidden)
    for module in (ACC_CONSULT_PROMOTION, ACC_SYSTEM2_PROMOTION, ACC_TEMPERATURE_ADJUSTMENT):
        monkeypatch.setenv(DEFAULT_REGISTRY[module].mode_env, 'off')
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_PRIORITY_AUTO', 'balanced')
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_PRIORITY_ON', 'balanced')
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN', '0')
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER', '0')


def controller(monkeypatch, responses, *, draft='The supplied records differ.', auto_active=False):
    telemetry = SnapshotTelemetry()
    c = make_stage_controller(left_model=CaptureLeftModel(draft=draft, confidence=.95), telemetry=telemetry)
    c.callosum = ScriptedCallosum(responses)
    c.right_model.uses_external_llm = True
    c.hippocampus = TemporalHippocampalIndexing()
    observed = {'basal': [], 'hippocampal': [], 'fallback_calls': []}
    original_decide = c.decide
    original_resolve = c._resolve_system2_activation
    original_basal = c.basal_ganglia.integrate_feedback
    original_hippocampal = c.hippocampus.consolidate_feedback

    def decide(*args, **kwargs):
        result = original_decide(*args, **kwargs)
        observed['decision'] = result
        return result

    def resolve(question, *, perception, settings):
        if settings.mode == 'auto':
            return System2Status(capable=True, active=auto_active,
                                 reason='memory' if auto_active else 'initial_auto_skip')
        return original_resolve(question, perception=perception, settings=settings)

    def basal(**kwargs):
        observed['basal'].append(kwargs)
        return original_basal(**kwargs)

    def hippocampal(**kwargs):
        observed['hippocampal'].append(kwargs)
        return original_hippocampal(**kwargs)

    async def unexpected_fallback(*args, **kwargs):
        observed['fallback_calls'].append(kwargs)
        raise AssertionError('Unexpected direct critic fallback')

    monkeypatch.setattr(c, 'decide', decide)
    monkeypatch.setattr(c, '_resolve_system2_activation', resolve)
    monkeypatch.setattr(c.basal_ganglia, 'integrate_feedback', basal)
    monkeypatch.setattr(c.hippocampus, 'consolidate_feedback', hippocampal)
    monkeypatch.setattr(c.right_model, 'criticise_reasoning', unexpected_fallback)
    return c, observed


def assert_persisted_use(c, observed, expected):
    assert observed['decision'].state['system2_used'] is expected
    assert observed['basal'] and observed['basal'][-1]['system2_used'] is expected
    assert observed['hippocampal'][-1]['system2_used'] is expected
    assert c.hippocampus.episodes[-1].annotations['system2_used'] is expected


@pytest.mark.asyncio
async def test_acc_final_route_reaches_metrics_after_initial_auto_skip(monkeypatch):
    for module in (ACC_CONSULT_PROMOTION, ACC_SYSTEM2_PROMOTION):
        monkeypatch.setenv(DEFAULT_REGISTRY[module].mode_env, 'auto')
    c, observed = controller(monkeypatch, [CLEAN], draft='2+2=5')
    c.anterior_cingulate_cortex.monitor = lambda **_: ConflictSignal(.95)
    await c.process('Compute 2+2', leading_brain='left', system2_mode='auto')
    snapshots = [e for e in c.telemetry.events if e['event'] == 'system2_mode']
    assert [(e['phase'], e['enabled']) for e in snapshots] == [('initial', False), ('routing', True)]
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['enabled'] is True and metrics['reason'] == 'acc_bump'
    assert metrics['round_target'] >= 1 and metrics['rounds'] == 1
    assert metrics['used'] is True and metrics['critic_attempts'] == metrics['critic_responses'] == 1
    assert metrics['review_completed'] is metrics['external_review_completed'] is True
    assert metrics['critic_validation'] == CLEAN['critic_validation']
    assert [p['type'] for p in c.callosum.payloads] == ['ASK_CRITIC']
    assert not observed['fallback_calls']
    assert_persisted_use(c, observed, True)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['auto', 'on', 'off'])
async def test_director_skip_records_final_admission_and_actual_use(monkeypatch, mode):
    monkeypatch.setenv(DEFAULT_REGISTRY[DIRECTOR_ADVICE].mode_env, 'on')
    c, observed = controller(monkeypatch, [CLEAN] if mode == 'on' else [], auto_active=True)
    c.director_model = SkipDirector()
    answer = await c.process('Explain the supplied project records.', leading_brain='left',
                             system2_mode=mode, executive_observer_mode='director')
    expected = mode == 'on'
    state = observed['decision'].state
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert state['system2_enabled'] is metrics['enabled'] is expected
    assert metrics['used'] is expected
    assert metrics['critic_attempts'] == len(c.callosum.payloads) == int(expected)
    assert metrics['rounds'] == int(expected)
    assert metrics['attempted_rounds'] == ([1] if expected else [])
    assert metrics['review_completed'] is expected
    if expected:
        assert state['director_consult_blocked'] == 'system2_on'
    else:
        assert state['director_consult'] == 'skip'
    if mode == 'auto':
        assert metrics['reason'] == 'director_skip' and state['director_system2_suppressed']
    if not expected:
        assert metrics['round_target'] == 0
    assert answer == 'The supplied records differ.' and not c.left_model.integrations
    assert not observed['fallback_calls']
    assert_persisted_use(c, observed, expected)


@pytest.mark.asyncio
async def test_director_suppression_does_not_create_a_replacement_correction(monkeypatch):
    monkeypatch.setenv(DEFAULT_REGISTRY[DIRECTOR_ADVICE].mode_env, 'on')
    monkeypatch.setenv(DEFAULT_REGISTRY[CEREBELLAR_CORRECTION].mode_env, 'on')
    c, observed = controller(monkeypatch, [], draft='2+2=5', auto_active=True)
    c.director_model = SkipDirector()
    answer = await c.process('Compute 2+2', leading_brain='left', system2_mode='auto',
                             executive_observer_mode='director')
    assert answer == '2+2=5'
    assert not c.callosum.payloads and not c.left_model.integrations
    assert observed['decision'].state['module_gates'][CEREBELLAR_CORRECTION]['reason'] == 'director_override'
    assert_persisted_use(c, observed, False)


@pytest.mark.asyncio
async def test_ordinary_consultation_under_off_is_not_recorded_as_system2_use(monkeypatch):
    c, observed = controller(monkeypatch, [{'notes_sum': 'The records have different conditions.', 'confidence_r': .9}])
    c.policy = AlwaysConsultPolicy()
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='off')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert [p['type'] for p in c.callosum.payloads] == ['ASK_DETAIL']
    assert metrics['enabled'] is metrics['used'] is False
    assert metrics['critic_attempts'] == metrics['critic_responses'] == metrics['rounds'] == 0
    assert not any(event['event'] == 'system2_refinement' for event in c.telemetry.events)
    assert_persisted_use(c, observed, False)


@pytest.mark.asyncio
async def test_timeout_is_an_attempt_without_completed_review(monkeypatch):
    c, observed = controller(monkeypatch, [asyncio.TimeoutError()])
    answer = await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['enabled'] is metrics['used'] is True
    assert metrics['critic_attempts'] == 1 and metrics['critic_responses'] == 0
    assert metrics['attempted_rounds'] == [1]
    assert metrics['review_completed'] is metrics['external_review_completed'] is False
    assert metrics['attempts'][0]['outcome'] == 'failed'
    assert metrics['attempts'][0]['error_type'] == 'TimeoutError'
    assert metrics['rounds'] == 1  # Existing rounds count attempted rounds, including timeout.
    assert metrics['local_recovery'] is None
    assert answer == 'The supplied records differ.' and not c.left_model.integrations
    assert_persisted_use(c, observed, True)


@pytest.mark.asyncio
async def test_local_timeout_recovery_does_not_certify_the_external_request(monkeypatch):
    c, _ = controller(monkeypatch, [asyncio.TimeoutError()], draft='2+2=4')
    await c.process('Compute 2+2', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['critic_attempts'] == 1 and metrics['critic_responses'] == 0
    assert metrics['review_completed'] is metrics['external_review_completed'] is False
    assert metrics['local_recovery']['source'].startswith('micro:')
    assert metrics['local_recovery']['resolved'] is True


@pytest.mark.asyncio
async def test_returned_invalid_review_is_separate_from_successful_transport(monkeypatch):
    c, _ = controller(monkeypatch, [INVALID])
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['critic_attempts'] == metrics['critic_responses'] == 1
    assert metrics['review_completed'] is metrics['external_review_completed'] is False
    assert metrics['critic_review_valid'] is metrics['critic_external_review_valid'] is False
    assert metrics['critic_validation'] == INVALID['critic_validation']
    assert metrics['resolved'] is False and not c.left_model.integrations


@pytest.mark.asyncio
async def test_direct_fallback_has_its_own_attempt_and_review_provenance(monkeypatch):
    c, observed = controller(monkeypatch, [{'error': 'unavailable'}])

    async def fallback(*args, **kwargs):
        observed['fallback_calls'].append(kwargs)
        return deepcopy(CLEAN)

    monkeypatch.setattr(c.right_model, 'criticise_reasoning', fallback)
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert len(c.callosum.payloads) == len(observed['fallback_calls']) == 1
    assert metrics['critic_attempts'] == metrics['critic_responses'] == 2
    assert metrics['attempted_rounds'] == [1]
    assert [a['source'] for a in metrics['attempts']] == ['callosum', 'direct_fallback']
    assert metrics['review_completed'] is metrics['external_review_completed'] is True
    assert_persisted_use(c, observed, True)


@pytest.mark.asyncio
@pytest.mark.parametrize('verification', ['valid', 'invalid', 'failed'])
async def test_latest_verification_controls_completion_without_erasing_first_review(monkeypatch, verification):
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN', '2')
    result = CLEAN if verification == 'valid' else INVALID if verification == 'invalid' else asyncio.TimeoutError()
    c, observed = controller(monkeypatch, [ISSUES, result])
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['attempted_rounds'] == [1, 2]
    assert metrics['attempts'][0]['review_valid'] is True
    assert metrics['critic_review_valid'] is True
    assert metrics['review_completed'] is metrics['external_review_completed'] is (verification == 'valid')
    assert metrics['critic_attempts'] == (3 if verification == 'failed' else 2)
    assert metrics['critic_responses'] == (1 if verification == 'failed' else 2)
    assert len(observed['fallback_calls']) == int(verification == 'failed')
    if verification == 'invalid':
        assert metrics['verify_critic_validation'] == INVALID['critic_validation']
    if verification == 'failed':
        assert metrics['attempts'][-1]['source'] == 'direct_fallback'
        assert metrics['attempts'][-1]['outcome'] == 'failed'
    assert_persisted_use(c, observed, True)


@pytest.mark.asyncio
async def test_third_round_preserves_each_review_and_reports_latest_invalid_completion(monkeypatch):
    monkeypatch.setenv('DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN', '3')
    initial = {**ISSUES, 'issues': ['A stated condition is missing.', 'The boundary assumption is missing.']}
    verify = {**ISSUES, 'issues': ['A stated condition is missing.']}
    c, _ = controller(monkeypatch, [initial, verify, INVALID])
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['attempted_rounds'] == [1, 2, 3]
    assert [a['review_valid'] for a in metrics['attempts']] == [True, True, False]
    assert metrics['review_completed'] is metrics['external_review_completed'] is False
    assert metrics['critic_review_valid'] is metrics['verify_critic_review_valid'] is True
    assert metrics['round3_critic_review_valid'] is False
    assert metrics['round3_critic_validation'] == INVALID['critic_validation']


@pytest.mark.asyncio
async def test_cancellation_records_attempt_and_drains_without_persistence(monkeypatch):
    c, observed = controller(monkeypatch, [])
    started, drained = asyncio.Event(), asyncio.Event()

    async def pending(payload, *, timeout_ms):
        started.set()
        try:
            await asyncio.Future()
        finally:
            drained.set()

    monkeypatch.setattr(c.callosum, 'ask_detail', pending)
    task = asyncio.create_task(c.process('Compare the supplied project records.',
                                        leading_brain='left', system2_mode='on'))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert drained.is_set() and not c.hippocampus.episodes and not observed['basal']
    assert metrics['used'] is True and metrics['critic_attempts'] == 1
    assert metrics['critic_responses'] == 0 and metrics['review_completed'] is False
    assert metrics['attempts'][0]['outcome'] == 'cancelled'


@pytest.mark.asyncio
async def test_adapter_without_validity_metadata_remains_unknown(monkeypatch):
    legacy = {key: value for key, value in CLEAN.items()
              if key not in {'critic_review_valid', 'critic_external_review_valid', 'critic_validation'}}
    c, _ = controller(monkeypatch, [legacy])
    await c.process('Compare the supplied project records.', leading_brain='left', system2_mode='on')
    metrics = _extract_metrics(c.telemetry.events)['system2']
    assert metrics['used'] is True and metrics['critic_responses'] == 1
    assert metrics['review_completed'] is metrics['external_review_completed'] is None


def test_old_events_do_not_invent_execution_or_review_availability():
    metrics = _extract_metrics([{'event': 'system2_mode', 'enabled': True, 'mode': 'on'}])['system2']
    assert metrics['enabled'] is True
    assert metrics['used'] is metrics['critic_attempts'] is metrics['review_completed'] is None
