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
    def __init__(self):
        self.slot_ms = 250
        self.payloads = []

    async def ask_detail(self, payload, timeout_ms=3000):
        self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
        return {"notes_sum": "補足メモ", "confidence_r": 0.82}


def test_controller_requests_right_brain_when_confidence_low():
    callosum = DummyCallosum()
    memory = SharedMemory()
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
    )

    answer = asyncio.run(controller.process("詳しく分析してください。"))

    assert "Reference from RightBrain" in answer
    assert callosum.payloads, "Right brain should have been consulted"
    sent_payload = callosum.payloads[0]["payload"]
    assert sent_payload["temperature"] > 0
    assert sent_payload["budget"] in {"small", "large"}
    assert memory.past_qas, "Answer should be stored back into shared memory"
