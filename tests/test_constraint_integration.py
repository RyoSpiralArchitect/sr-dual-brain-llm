"""Witnesses through actual critic, worker, bounded revision and turn cleanup."""
import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.constraint_checks import check_constraints, current_constraints
from core.models import RightBrainModel
from core.right_worker import _respond
from core.source_completion import guard_source_completions
from core.micro_critic import micro_criticise_reasoning
from test_constraint_checks import QUESTION, BAD, GOOD, JSON_QUESTION
from test_dual_brain import AlwaysConsultPolicy, make_stage_controller, make_decision

CLEAN = '{"verdict":"ok","issues":[],"fixes":[]}'
JSON_BAD = '{"name":"cat","debug":true}'
JSON_GOOD = '{"name":"cat","state":"awake"}'


@pytest.fixture(autouse=True)
def no_network():
    with forbid_provider_network() as audit:
        yield
    assert audit.blocked_requests == 0


class CriticClient:
    def __init__(self, result=CLEAN):
        self.result, self.calls = result, 0

    async def complete(self, *_args, **_kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize('question,draft', [(QUESTION, BAD), (JSON_QUESTION, JSON_BAD)])
def test_witness_overrules_clean_external_critic_without_extra_model_call(monkeypatch, question, draft):
    monkeypatch.setenv('DUALBRAIN_CRITIC_MICRO_SANITY_OVERRIDE', '0')
    model = RightBrainModel()
    model._llm_client = client = CriticClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)
    result = asyncio.run(_respond({'qid': 'witness', 'type': 'ASK_CRITIC',
        'question': question, 'draft': draft}, SimpleNamespace(), model))
    assert result['verdict'] == 'issues' and result['critic_kind'] == 'micro_sanity_override'
    assert result['critic_external_review_valid'] is True
    assert result['constraint_check']['status'] == 'contradiction' and client.calls == 1
    assert 'claim' not in result['constraint_check'] and question not in json.dumps(result['constraint_check'])


@pytest.mark.parametrize('completion,status,valid', [
    ('', 'empty_response', False), (RuntimeError('failure'), 'provider_error', True),
    ('{"verdict":"issues","issues":["Another problem"],"fixes":["Review it"]}', 'ok', True),
])
def test_failure_provenance_and_other_issues_cannot_discard_witness(completion, status, valid):
    model = RightBrainModel()
    model._llm_client = CriticClient(completion)
    model.llm_config = SimpleNamespace(timeout_seconds=40)
    result = asyncio.run(model.criticise_reasoning('witness', QUESTION, BAD))
    assert result['critic_status'] == status and result['critic_review_valid'] is valid
    assert result['constraint_check']['status'] == 'contradiction'
    assert any('residual -1' in issue for issue in result['issues'])


def test_opted_out_micro_and_delayed_primary_worker_retain_authority():
    model = RightBrainModel()
    model._llm_client = CriticClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)
    direct = asyncio.run(model.criticise_reasoning('off', QUESTION, BAD, allow_micro_fallback=False))
    delayed = asyncio.run(_respond({'qid': 'primary', 'type': 'ASK_CRITIC',
        'question': QUESTION, 'draft': BAD}, SimpleNamespace(primary_write_pending=True), model))
    assert direct['verdict'] == delayed['verdict'] == 'ok'
    assert 'constraint_check' not in direct and 'constraint_check' not in delayed


def test_primary_context_and_byte_exact_quote_do_not_get_local_rewrite():
    class Source:
        memory = SimpleNamespace(conversation_context=True)

        @guard_source_completions
        async def run(self, *, memory_context):
            assert micro_criticise_reasoning(QUESTION, BAD) is None

    asyncio.run(Source().run(memory_context={'remember': True}))
    assert micro_criticise_reasoning('Reproduce verbatim: Compute 2+2', '2+2=5') is None


def test_passing_json_shape_does_not_hide_existing_arithmetic_contradiction():
    result = micro_criticise_reasoning(
        'Compute 2+2.\nReturn a JSON object with exactly these keys: result.', '{"result":5}')
    assert result is not None and result.verdict == 'issues' and result.domain == 'arithmetic'


@pytest.mark.parametrize('question,draft', [(QUESTION, GOOD), (JSON_QUESTION, JSON_GOOD)])
def test_passing_local_check_does_not_supply_routing_confidence_or_overall_approval(question, draft):
    assert check_constraints(question, draft).status == 'consistent'
    assert micro_criticise_reasoning(question, draft) is None


@pytest.mark.parametrize('question,bad,good', [(QUESTION, BAD, GOOD), (JSON_QUESTION, JSON_BAD, JSON_GOOD)])
@pytest.mark.parametrize('repair', ['correct', 'unchanged', 'unparseable', 'failure'])
def test_cerebellum_uses_existing_single_revision_and_preserves_unresolved(question, bad, good, repair):
    class Left:
        uses_external_llm = True
        calls = 0

        async def integrate_info_async(self, **kwargs):
            self.calls += 1
            assert 'stated assumptions' in kwargs['info']
            if repair == 'failure':
                raise RuntimeError('failed integration')
            return {'correct': good, 'unchanged': bad, 'unparseable': 'I fixed it.'}[repair]

    left = Left()
    controller, decision = make_stage_controller(left_model=left), make_decision()

    async def noop():
        pass

    result = asyncio.run(controller._run_conflict_regulation_stage(
        question=question, decision=decision, final_answer=bad, confidence=.95, detail_notes=None,
        system2_active=False, system2_mode='auto', is_trivial_chat=False, emit_reset_if_needed=noop,
        delta_cb=None, stream_final_only=False, phase_add=lambda *_: None))
    assert left.calls == 1
    assert decision.state['cerebellum_micro']['resolved'] is (repair == 'correct')
    assert decision.state['cerebellum_micro']['final_issues'] == (0 if repair == 'correct' else 1)
    assert result.final_answer == (good if repair == 'correct' else 'I fixed it.' if repair == 'unparseable' else bad)


@pytest.mark.parametrize('mode,active', [('off', False), ('on', False), ('auto', True)])
def test_existing_manual_modes_and_active_system2_do_not_gain_cerebellar_calls(mode, active):
    class Left:
        uses_external_llm = True

        async def integrate_info_async(self, **kwargs):
            pytest.fail('No extra correction in this mode')

    async def noop():
        pass

    controller = make_stage_controller(left_model=Left())
    result = asyncio.run(controller._run_conflict_regulation_stage(
        question=QUESTION, decision=make_decision(), final_answer=BAD, confidence=.95, detail_notes=None,
        system2_active=active, system2_mode=mode, is_trivial_chat=False, emit_reset_if_needed=noop,
        delta_cb=None, stream_final_only=False, phase_add=lambda *_: None))
    assert result.final_answer == BAD


# Actual LLM collectors, callosum worker, controller and public metrics, using
# scripted transport. No paid quality claim follows from these controlled tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sr-dual-brain-llm/scripts'))
import benchmark_contract_live as pilot
from scheduled_http_journal import ScheduledJournal
from test_benchmark_contract_live import config, response
from offline_provider_guard import forbid_provider_network
from engine_stdio import _extract_metrics


@pytest.mark.parametrize('question,draft,repair', [
    (QUESTION, BAD, GOOD), (JSON_QUESTION, JSON_BAD, JSON_GOOD),
    (QUESTION, BAD, 'The solution has been fixed.'),
    (JSON_QUESTION, JSON_BAD, 'The object is fixed.'),
])
def test_real_turn_keeps_unknown_repair_unresolved_and_cleans_state(question, draft, repair):
    async def run():
        journal = ScheduledJournal(config(), {'test': 'constraint-integration'}, max_calls=12)

        class Provider:
            async def _post_json(self, _url, payload, _headers, **_kwargs):
                role = journal.events[-1]['role']
                return response(CLEAN if role == 'critic' else repair if role == 'integration' else draft)

        journal.transport = Provider()
        env = pilot.profile.environment()
        with patch.dict(os.environ, env, clear=True), forbid_provider_network() as audit:
            session, _ = await pilot.make_session('constraint-integration', None, config(), journal, [1794081600.])
            session.controller.policy = AlwaysConsultPolicy()
            try:
                answer = await session.controller.process(question, qid='constraint-integration',
                    leading_brain='left', system2_mode='on')
                events = session.controller.telemetry.events
                observed = [event for event in events if event['event'] == 'constraint_consistency']
                assert observed
                expected = 'consistent' if repair in (GOOD, JSON_GOOD) else 'unknown'
                assert observed[-1]['status'] == expected
                system2 = [event for event in events if event['event'] == 'system2_refinement']
                assert system2 and system2[-1]['resolved'] is (expected == 'consistent')
                metrics = _extract_metrics(events)
                assert metrics['constraint_check']['status'] == expected
                assert metrics['system2']['resolved'] is (expected == 'consistent')
                assert current_constraints() is None
                assert answer and len(journal.events) <= 24 and audit.blocked_requests == 0
            finally:
                await session.close()
    asyncio.run(run())


def test_final_presentation_unknown_vetoes_prior_system2_resolution():
    metrics = _extract_metrics([
        {'event': 'system2_mode', 'mode': 'on', 'enabled': True},
        {'event': 'system2_refinement', 'resolved': True, 'final_issues': 0},
        {'event': 'constraint_consistency', 'status': 'unknown', 'unresolved_obligations': 1},
    ])
    assert metrics['system2']['resolved'] is False
    assert metrics['system2']['resolution_basis'] == 'local_constraint_unresolved'
