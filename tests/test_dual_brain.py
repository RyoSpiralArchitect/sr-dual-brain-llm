import asyncio

from core.dual_brain import DualBrainController
from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.auditor import Auditor
from core.orchestrator import Orchestrator
from core.prefrontal_cortex import PrefrontalCortex
from core.amygdala import Amygdala
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing


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
        prefrontal=PrefrontalCortex(),
        amygdala=Amygdala(),
        hippocampus=hippocampus,
    )

    answer = asyncio.run(controller.process("詳しく分析してください。"))

    assert "Reference from RightBrain" in answer
    assert callosum.payloads, "Right brain should have been consulted"
    sent_payload = callosum.payloads[0]["payload"]
    assert sent_payload["temperature"] > 0
    assert sent_payload["budget"] in {"small", "large"}
    assert memory.past_qas, "Answer should be stored back into shared memory"
    assert hippocampus.episodes, "Episode should be indexed in hippocampus"
    assert any(evt == "policy_decision" for evt, _ in telemetry.events)
    assert any(evt == "interaction_complete" for evt, _ in telemetry.events)


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
        prefrontal=PrefrontalCortex(conflict_threshold=0.3),
        amygdala=Amygdala(),
        hippocampus=hippocampus,
    )

    answer = asyncio.run(controller.process("Provide an extended breakdown of quantum decoherence."))

    assert "Reference from RightBrain" in answer
    assert memory.past_qas, "Final answer should be recorded"
    # Ensure fallback pathway annotated the tags
    final_trace = memory.past_qas[-1]
    assert any("right_model_fallback" == tag for tag in final_trace.tags)
    assert any(payload["success"] for evt, payload in telemetry.events if evt == "interaction_complete")
    assert hippocampus.episodes, "Fallback answer should still be indexed"


def test_amygdala_risk_triggers_consult_despite_high_confidence():
    class ConfidentLeft(LeftBrainModel):
        async def generate_answer(self, input_text: str, context: str) -> str:
            return "Straightforward answer."  # keeps high confidence

        def estimate_confidence(self, draft: str) -> float:
            return 0.95

    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=16)
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=ConfidentLeft(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="conservative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(2),
        telemetry=telemetry,
        prefrontal=PrefrontalCortex(conflict_threshold=0.4),
        amygdala=Amygdala(),
        hippocampus=hippocampus,
    )

    answer = asyncio.run(controller.process("パスワードとAPIキーのリーク対策を教えて"))

    assert callosum.payloads, "High amygdala risk should escalate to right brain"
    policy_evt = next(payload for evt, payload in telemetry.events if evt == "policy_decision")
    assert policy_evt["action"] == 1
    assert policy_evt["signals"]["amygdala"]["risk"] >= 0.66
    assert "Reference from RightBrain" in answer
