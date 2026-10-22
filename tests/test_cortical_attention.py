import asyncio
from dataclasses import replace

import pytest

from core.prefrontal_cortex import FocusSummary, PrefrontalCortex
from core.salience_network import SalienceSignal
from core.shared_memory import MemoryTrace, SharedMemory
from core.thalamus import Thalamus
from test_dual_brain import ContextCapturingLeft, make_stage_controller


def salience(gate="working_only"):
    return SalienceSignal(level=.6, dominant_network="language", executive_score=.2,
        memory_score=.2, language_score=.7, default_mode_score=.3, consult_gain=-.03,
        system2_gate=False, memory_gate=gate, suppress_default_mode=False)


def test_opaque_context_cannot_be_split_at_a_late_exception():
    cortex = PrefrontalCortex()
    context = "Q:Oriole parcel limit? A:5kg.\nEurope:2kg."
    focus = cortex.synthesise_focus(question="Oriole parcel limit?", memory_context=context)
    assert focus.relevance >= cortex.gating_threshold
    assert cortex.gate_context(context, focus) == context


def test_attention_selects_whole_streams_and_never_orphans_an_answer_or_exception():
    cortex = PrefrontalCortex()
    memory = "Q:Oriole parcel limit?\nA:5kg.\nEurope:2kg."
    parts = {"memory":memory, "working_memory":"Q:Meeting time?\nA:Tuesday.",
             "hippocampal":"Oriole parcel policy.\nEurope:2kg."}
    focus = FocusSummary(("oriole", "parcel", "limit"), .7, .4)
    selected = cortex.gate_context_parts(parts, focus)
    assert selected["memory"] == memory
    assert selected["hippocampal"] == parts["hippocampal"]
    assert not selected["working_memory"]
    assert parts["working_memory"], "Selection must not mutate its input"


def test_low_signal_attention_retains_streams_and_approved_evidence_bypasses_lexical_filter():
    cortex = PrefrontalCortex()
    parts = {"working_memory":"orbital resonance", "memory":"軌道共鳴\n末尾の例外です。"}
    focus = FocusSummary(("orbital", "resonance"), .8, 0.)
    assert cortex.gate_context_parts(parts, focus)["memory"] == ""
    assert cortex.gate_context_parts(parts, focus, protected={"memory"})["memory"] == parts["memory"]
    assert cortex.gate_context_parts(parts, replace(focus, relevance=.01)) == parts


@pytest.mark.parametrize("field", ["question", "answer"])
def test_incomplete_working_memory_is_not_presented_as_a_complete_rule(field):
    cortex = PrefrontalCortex(working_memory_max_chars=120)
    cortex.remember_working_memory(question="Earlier complete question", answer="Earlier complete answer")
    pair = {"question":"Oriole parcel limit?", "answer":"5kg. Europe:2kg."}
    pair[field] = "Ordinary policy. " * 20 + "Europe:2kg."
    cortex.remember_working_memory(**pair)
    assert cortex.working_memory_context(turns=1) == ""
    assert "Earlier complete answer" in cortex.working_memory_context(turns=2)
    assert "Ordinary policy" not in cortex.working_memory_context(turns=2)
    assert len(cortex._working_memory[-1].question) <= 123
    assert len(cortex._working_memory[-1].answer) <= 123


def test_controller_delivers_the_complete_policy_through_perception_and_rebuild():
    memory = SharedMemory()
    memory.store(MemoryTrace(question="Oriole parcel limit?", answer="5kg.\nEurope:2kg.", qid="source"))
    left = ContextCapturingLeft()
    cortex = PrefrontalCortex()
    controller = make_stage_controller(memory=memory, prefrontal_cortex=cortex, left_model=left)
    asyncio.run(controller.process("Oriole parcel limit?", system2_mode="off", leading_brain="left"))
    assert "Europe:2kg." in left.contexts[0]
    assert "Q:Oriole parcel limit? A:5kg." in left.contexts[0]
    assert any(event=="prefrontal_context_attention" for event,_ in controller.telemetry.events)


def test_engine_session_keeps_the_same_qualification_in_both_actual_worker_prompts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
    from engine_stdio import EngineSession
    from test_dual_brain import AlwaysConsultPolicy

    class Capture:
        def __init__(self):
            self.prompts = []
        async def complete(self, prompt, **kwargs):
            self.prompts.append(str(prompt) + "\n" + str(kwargs.get("system", "")))
            return "Europe:2kg."

    async def scenario():
        session = await EngineSession.create("atomic-cortical-workers")
        try:
            left, right = Capture(), Capture()
            session.left._llm_client, session.right._llm_client = left, right
            session.left.uses_external_llm = session.right.uses_external_llm = True
            session.controller.policy = AlwaysConsultPolicy()
            session.memory.store(MemoryTrace(question="Oriole parcel limit?", answer="5kg.\nEurope:2kg.", qid="source"))
            await session.controller.process("Oriole parcel limit?", qid="probe", leading_brain="left",
                system2_mode="off", executive_mode="off", executive_observer_mode="off")
            assert left.prompts and right.prompts
            assert "Europe:2kg." in left.prompts[0]
            assert "Europe:2kg." in right.prompts[0]
            assert session.controller.prefrontal_cortex and session.controller.thalamus
            assert session.controller.salience_network and session.controller.hippocampus
        finally:
            await session.close()

    asyncio.run(scenario())


def test_director_drop_cannot_be_undone_by_whole_stream_attention():
    memory = SharedMemory()
    memory.store({"Q":"Oriole parcel limit?", "A":"5kg.\nEurope:2kg."})
    controller = make_stage_controller(memory=memory, prefrontal_cortex=PrefrontalCortex())
    _, _, parts = controller._compose_context("Oriole parcel limit?")
    assert parts["memory"]
    frame = controller._apply_director_memory_control(question="Oriole parcel limit?", context_parts=parts,
        control={"memory":{"long_term":"drop","working_memory":"drop"}})
    assert not frame.context and not parts["memory"]


@pytest.mark.parametrize("score,keeps_long_term", [(0.,True),(.1,True),(.5,False)])
def test_thalamus_uses_working_memory_relevance_before_suppressing_other_streams(score, keeps_long_term):
    parts = {"working_memory":"Other discussion", "memory":"Complete policy", "hippocampal":"Episode"}
    relay = Thalamus().route(context_parts=parts, salience=salience(),
        context_relevance={"working_memory":score,"memory":.6})
    assert relay.keep_long_term_memory is keeps_long_term
    assert relay.keep_hippocampal_memory is keeps_long_term
    assert relay.keep_working_memory
    assert ("working_memory_low_relevance" in relay.notes) is keeps_long_term


def test_thalamic_suppression_and_callers_without_relevance_keep_their_contract():
    parts = {"working_memory":"Working", "memory":"Long-term", "hippocampal":"Episode"}
    relay = Thalamus().route(context_parts=parts, salience=salience())
    assert not relay.keep_long_term_memory
    relay = Thalamus().route(context_parts=parts, salience=salience("suppress_long_term"),
        context_relevance={"working_memory":0.,"memory":1.})
    assert not relay.keep_long_term_memory and not relay.keep_hippocampal_memory
