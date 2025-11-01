import asyncio

from core.dual_brain import DualBrainController
from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.auditor import Auditor
from core.orchestrator import Orchestrator


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
    )

    answer = asyncio.run(controller.process("詳しく分析してください。"))

    assert "Reference from RightBrain" in answer
    assert callosum.payloads, "Right brain should have been consulted"
    sent_payload = callosum.payloads[0]["payload"]
    assert sent_payload["temperature"] > 0
    assert sent_payload["budget"] in {"small", "large"}
    assert memory.past_qas, "Answer should be stored back into shared memory"
    assert any(evt == "policy_decision" for evt, _ in telemetry.events)
    assert any(evt == "interaction_complete" for evt, _ in telemetry.events)


def test_controller_falls_back_to_local_right_model():
    callosum = DummyCallosum(fail=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
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
    )

    answer = asyncio.run(controller.process("Provide an extended breakdown of quantum decoherence."))

    assert "Reference from RightBrain" in answer
    assert memory.past_qas, "Final answer should be recorded"
    # Ensure fallback pathway annotated the tags
    final_trace = memory.past_qas[-1]
    assert any("right_model_fallback" == tag for tag in final_trace.tags)
    assert any(payload["success"] for evt, payload in telemetry.events if evt == "interaction_complete")
