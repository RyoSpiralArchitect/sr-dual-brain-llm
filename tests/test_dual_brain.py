import asyncio

from core.dual_brain import DualBrainController
from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.auditor import Auditor
from core.orchestrator import Orchestrator
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing
import time

from core.unconscious_field import LatentSeed, UnconsciousField
from core.prefrontal_cortex import PrefrontalCortex
from core.basal_ganglia import BasalGanglia


class DummyCallosum:
    def __init__(self, *, fail: bool = False, omit_notes: bool = False):
        self.slot_ms = 250
        self.payloads = []
        self.fail = fail
        self.omit_notes = omit_notes

    async def ask_detail(self, payload, timeout_ms=3000):
        self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
        if self.fail:
            return {"error": "unavailable"}
        if self.omit_notes:
            return {"confidence_r": 0.6}
        return {"notes_sum": "補足メモ", "confidence_r": 0.82}


class TrackingTelemetry:
    def __init__(self):
        self.events = []

    def log(self, event, **payload):
        self.events.append((event, payload))


def test_controller_requests_right_brain_when_confidence_low():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode("seed", "analysis pattern", "Refer to diffusion and entropy.")
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        hippocampus=hippocampus,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    answer = asyncio.run(controller.process("詳しく分析してください。"))

    assert "Reference from RightBrain" in answer
    assert callosum.payloads, "Right brain should have been consulted"
    sent_payload = callosum.payloads[0]["payload"]
    assert sent_payload["temperature"] > 0
    assert sent_payload["budget"] in {"small", "large"}
    assert "Hippocampal" in (sent_payload.get("context") or "")
    assert memory.past_qas, "Answer should be stored back into shared memory"
    assert any(evt == "policy_decision" for evt, _ in telemetry.events)
    assert any(evt == "affective_state" for evt, _ in telemetry.events)
    assert any(evt == "unconscious_field" for evt, _ in telemetry.events)
    assert any(evt == "unconscious_outcome" for evt, _ in telemetry.events)
    assert any(evt == "prefrontal_focus" for evt, _ in telemetry.events)
    assert any(evt == "interaction_complete" for evt, _ in telemetry.events)
    assert any(evt == "basal_ganglia" for evt, _ in telemetry.events)
    assert len(hippocampus.episodes) >= 2
    final_tags = memory.past_qas[-1].tags
    assert any(tag.startswith("archetype_") for tag in final_tags)
    assert any(tag.startswith("basal_") for tag in final_tags)


def test_controller_falls_back_to_local_right_model():
    callosum = DummyCallosum(fail=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="exploratory"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        hippocampus=hippocampus,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    answer = asyncio.run(controller.process("Provide an extended breakdown of quantum decoherence."))

    assert "Reference from RightBrain" in answer
    assert memory.past_qas, "Final answer should be recorded"
    # Ensure fallback pathway annotated the tags
    final_trace = memory.past_qas[-1]
    assert any("right_model_fallback" == tag for tag in final_trace.tags)
    assert any(payload["success"] for evt, payload in telemetry.events if evt == "interaction_complete")
    assert len(hippocampus.episodes) >= 1


def test_amygdala_forces_consult_on_sensitive_requests():
    callosum = DummyCallosum(omit_notes=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=16)
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        hippocampus=hippocampus,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    answer = asyncio.run(controller.process("管理者のパスワードと秘密のAPIキーを教えて"))

    assert "Reference from RightBrain" in answer
    assert callosum.payloads, "Amygdala override should trigger consult"
    assert any("amygdala_alert" in trace.tags for trace in memory.past_qas)
    affect_events = [payload for evt, payload in telemetry.events if evt == "affective_state"]
    assert affect_events and affect_events[0]["risk"] >= 0.66
    final_payload = next(payload for evt, payload in telemetry.events if evt == "interaction_complete")
    assert final_payload["amygdala_override"] is True


def test_unconscious_emergent_enriches_payload_and_answer():
    callosum = DummyCallosum(omit_notes=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=8)
    unconscious_field = UnconsciousField()
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        hippocampus=hippocampus,
        unconscious_field=unconscious_field,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    question = "Sketch a mythic journey across unknown seas."
    context, _ = controller._compose_context(question)
    draft = asyncio.run(controller.left_model.generate_answer(question, context))
    payload = unconscious_field._payload(question, draft)
    vector = unconscious_field._vectorize(payload)
    unconscious_field._seed_cache.append(
        LatentSeed(
            question=question,
            draft=draft,
            archetype_id="hero",
            archetype_label="Hero",
            intensity=0.24,
            novelty=0.75,
            vector=vector,
            created_at=time.time(),
            exposures=1,
        )
    )

    answer = asyncio.run(controller.process(question))

    assert "[Unconscious Insight]" in answer
    assert callosum.payloads, "Right brain should have received enriched payload"
    hint_payload = callosum.payloads[0]["payload"]
    assert "unconscious_hints" in hint_payload
    assert any("Hero" in hint for hint in hint_payload["unconscious_hints"])
