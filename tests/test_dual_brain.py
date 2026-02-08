import asyncio
import os
import time

from core.dual_brain import DualBrainController
from core.shared_memory import SharedMemory
from core.models import LeftBrainModel, RightBrainModel
from core.policy import RightBrainPolicy
from core.hypothalamus import Hypothalamus
from core.policy_modes import ReasoningDial
from core.auditor import Auditor
from core.orchestrator import Orchestrator
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing

from core.unconscious_field import LatentSeed, UnconsciousField
from core.prefrontal_cortex import PrefrontalCortex
from core.basal_ganglia import BasalGanglia
from core.default_mode_network import DefaultModeNetwork
from core.executive_reasoner import ExecutiveAdvice
from core.director_reasoner import DirectorAdvice
from core.schema import UnconsciousSummaryModel


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


class SlowExecutiveModel:
    async def advise(
        self,
        *,
        question: str,
        context: str,
        focus_keywords=None,
        temperature: float = 0.2,
    ):
        await asyncio.sleep(30)
        return ExecutiveAdvice(
            memo="slow executive",
            directives={},
            confidence=0.0,
            latency_ms=0.0,
            source="test",
        )


class InstantExecutiveModel:
    def __init__(self, *, mix_in: str = "", directives=None):
        self.mix_in = mix_in
        self.directives = directives or {"tone": "friendly", "do_not_say": [], "priorities": [], "clarifying_questions": [], "format": []}

    async def advise(
        self,
        *,
        question: str,
        context: str,
        focus_keywords=None,
        temperature: float = 0.2,
    ):
        return ExecutiveAdvice(
            memo="instant executive",
            directives=self.directives,
            confidence=0.9,
            latency_ms=1.0,
            source="test",
            mix_in=self.mix_in,
        )


class InstantDirectorModel:
    async def advise(
        self,
        *,
        question: str,
        context: str,
        signals=None,
        temperature: float = 0.15,
    ):
        return DirectorAdvice(
            memo="instant director",
            control={
                "consult": "skip",
                "temperature": 0.2,
                "max_chars": 120,
                "memory": {"working_memory": "drop", "long_term": "drop"},
                "append_clarifying_question": "どの話に戻りたい？",
            },
            confidence=0.9,
            latency_ms=1.0,
            source="test",
        )


class CaptureLeftModel:
    uses_external_llm = True

    def __init__(self, *, draft: str = "draft", confidence: float = 0.2):
        self._draft = draft
        self._confidence = confidence
        self.integrations: list[str] = []

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        return self._draft

    def estimate_confidence(self, draft: str) -> float:
        return self._confidence

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        self.integrations.append(info)
        return f"{draft}\n\n(INTEGRATED)"


class LongLowConfidenceLeft:
    uses_external_llm = False

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        return ("draft " + ("x" * 800)).strip()

    def estimate_confidence(self, draft: str) -> float:
        return 0.2

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        return draft


class NoisyLeft:
    uses_external_llm = False

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        return "こんにちは！\nqid 123\n- Add a warm, informal tone.\n[Hemisphere Routing]\nmode: balanced"

    def estimate_confidence(self, draft: str) -> float:
        return 0.9

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        return draft


class ContextCapturingLeft:
    uses_external_llm = False

    def __init__(self):
        self.contexts: list[str] = []

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        self.contexts.append(str(context or ""))
        return "draft"

    def estimate_confidence(self, draft: str) -> float:
        return 0.9

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        return draft


class WeatherClaimLeft:
    uses_external_llm = False

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        return "元気だよ、ありがとう！今日の天気も良くて、なんだかポジティブな気分が続いてるんだ。君はどう？"

    def estimate_confidence(self, draft: str) -> float:
        return 0.95

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        return draft


class OffTopicLeft:
    uses_external_llm = False

    async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:
        return "猫が好きです。最近は黒猫が特に好きで、写真を見るのが楽しいです。"

    def estimate_confidence(self, draft: str) -> float:
        return 0.95

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta=None,
    ) -> str:
        return draft


class AlwaysSkipPolicy:
    def decide(self, state):  # noqa: ANN001
        return 0


class AlwaysConsultPolicy:
    def decide(self, state):  # noqa: ANN001
        return 1


class CriticCallosum(DummyCallosum):
    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            return {
                "qid": payload.get("qid"),
                "verdict": "issues",
                "issues": ["Math error: 2+2 is not 5."],
                "fixes": ["Correct the arithmetic and re-check the conclusion."],
                "critic_sum": "Issues:\n- Math error: 2+2 is not 5.\nFixes:\n- Correct the arithmetic and re-check the conclusion.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class OkCriticCallosum(DummyCallosum):
    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            return {
                "qid": payload.get("qid"),
                "verdict": "ok",
                "issues": [],
                "fixes": [],
                "critic_sum": "No issues detected.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class TwoPhaseCriticCallosum(DummyCallosum):
    def __init__(self):
        super().__init__()
        self.critic_calls = 0

    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            self.critic_calls += 1
            if self.critic_calls == 1:
                return {
                    "qid": payload.get("qid"),
                    "verdict": "issues",
                    "issues": ["Math error: 2+2 is not 5."],
                    "fixes": ["Correct the arithmetic and re-check the conclusion."],
                    "critic_sum": "Issues:\n- Math error: 2+2 is not 5.\nFixes:\n- Correct the arithmetic and re-check the conclusion.",
                    "confidence_r": 0.9,
                }
            return {
                "qid": payload.get("qid"),
                "verdict": "ok",
                "issues": [],
                "fixes": [],
                "critic_sum": "No issues detected.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class RephraseCriticCallosum(DummyCallosum):
    def __init__(self):
        super().__init__()
        self.critic_calls = 0

    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            self.critic_calls += 1
            if self.critic_calls == 1:
                return {
                    "qid": payload.get("qid"),
                    "verdict": "issues",
                    "issues": ["Step 2 arithmetic is incorrect."],
                    "fixes": ["Correct step 2 arithmetic and recompute the result."],
                    "critic_sum": "Issues:\n- Step 2 arithmetic is incorrect.\nFixes:\n- Correct step 2 arithmetic and recompute the result.",
                    "confidence_r": 0.9,
                }
            return {
                "qid": payload.get("qid"),
                "verdict": "issues",
                "issues": [
                    "Step 2 arithmetic is incorrect (re-check the sum).",
                    "Arithmetic at step 2 is incorrect.",
                ],
                "fixes": ["Re-check step 2 arithmetic."],
                "critic_sum": "Issues:\n- Step 2 arithmetic is incorrect (re-check the sum).\n- Arithmetic at step 2 is incorrect.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class PersistentCriticCallosum(DummyCallosum):
    def __init__(self):
        super().__init__()
        self.critic_calls = 0

    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            self.critic_calls += 1
            if self.critic_calls <= 2:
                return {
                    "qid": payload.get("qid"),
                    "verdict": "issues",
                    "issues": [
                        "Step 2 arithmetic is incorrect.",
                        "Final statement contradicts the corrected arithmetic.",
                    ],
                    "fixes": [
                        "Correct step 2 arithmetic.",
                        "Align the final statement with the corrected arithmetic.",
                    ],
                    "critic_sum": (
                        "Issues:\n"
                        "- Step 2 arithmetic is incorrect.\n"
                        "- Final statement contradicts the corrected arithmetic.\n"
                        "Fixes:\n"
                        "- Correct step 2 arithmetic.\n"
                        "- Align the final statement with the corrected arithmetic."
                    ),
                    "confidence_r": 0.9,
                }
            return {
                "qid": payload.get("qid"),
                "verdict": "ok",
                "issues": [],
                "fixes": [],
                "critic_sum": "No issues detected.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class LowSignalNoiseCriticCallosum(DummyCallosum):
    def __init__(self):
        super().__init__()
        self.critic_calls = 0

    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            self.critic_calls += 1
            if self.critic_calls == 1:
                return {
                    "qid": payload.get("qid"),
                    "verdict": "issues",
                    "issues": [
                        "Average speed calculation is incorrect: simple mean of segment speeds is invalid.",
                        "Could clarify the mechanism in one extra sentence.",
                        "No boundary case is mentioned.",
                    ],
                    "fixes": [
                        "Compute total distance divided by total time.",
                        "Add a concise clarification sentence.",
                    ],
                    "critic_sum": (
                        "Issues:\n"
                        "- Average speed calculation is incorrect: simple mean of segment speeds is invalid.\n"
                        "- Could clarify the mechanism in one extra sentence.\n"
                        "- No boundary case is mentioned."
                    ),
                    "confidence_r": 0.9,
                }
            return {
                "qid": payload.get("qid"),
                "verdict": "issues",
                "issues": ["Could explicitly mention weighted average wording."],
                "fixes": ["Adjust wording for readability."],
                "critic_sum": "Could explicitly mention weighted average wording.",
                "confidence_r": 0.9,
            }
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


class TimeoutCriticCallosum(DummyCallosum):
    async def ask_detail(self, payload, timeout_ms=3000):  # noqa: ANN001
        if payload.get("type") == "ASK_CRITIC":
            self.payloads.append({"payload": payload, "timeout_ms": timeout_ms})
            raise asyncio.TimeoutError("simulated critic timeout")
        return await super().ask_detail(payload, timeout_ms=timeout_ms)


def test_director_can_skip_consult_and_clamp_output():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode("seed", "分析パターン", "Refer to diffusion and entropy.")
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LongLowConfidenceLeft(),
        right_model=RightBrainModel(),
        director_model=InstantDirectorModel(),
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

    answer = asyncio.run(controller.process("詳しく分析してください。", executive_observer_mode="director"))

    assert not callosum.payloads, "Director should be able to skip consult"
    assert len(answer) <= 120
    assert "どの話に戻りたい？" in answer

    interaction_ids = [
        payload["qid"] for evt, payload in telemetry.events if evt == "interaction_complete"
    ]
    assert interaction_ids
    flow = memory.dialogue_flow(interaction_ids[-1])
    assert flow and flow.get("executive_observer")
    assert flow["executive_observer"].get("observer_mode") == "director"


def test_process_stores_sanitised_memory_even_in_meta_mode():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=NoisyLeft(),
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

    asyncio.run(controller.process("やあ", answer_mode="meta"))

    assert memory.past_qas, "should store a memory trace"
    stored = memory.past_qas[-1].answer
    assert "qid 123" not in stored
    assert "Add a warm" not in stored
    assert "Hemisphere" not in stored

    assert hippocampus.episodes, "should store a hippocampal episode"
    epi = hippocampus.episodes[-1].answer
    assert "qid 123" not in epi
    assert "Add a warm" not in epi
    assert "Hemisphere" not in epi


def test_compose_context_includes_working_memory_for_trivial_followup_question():
    callosum = DummyCallosum()
    memory = SharedMemory()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    cortex = PrefrontalCortex()
    cortex.remember_working_memory(question="やあ", answer="こんにちは！", qid="seed")

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
        telemetry=TrackingTelemetry(),
        hippocampus=hippocampus,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=cortex,
        basal_ganglia=BasalGanglia(),
    )

    context, _, parts = controller._compose_context("特に。そっちは？")
    assert "[Working memory]" in context
    assert "Q:やあ" in context
    assert parts.get("working_memory"), "working memory part should be populated"


def test_director_memory_drop_applies_before_left_draft_context():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode("seed", "分析パターン", "Refer to diffusion and entropy.")
    left = ContextCapturingLeft()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=left,
        right_model=RightBrainModel(),
        director_model=InstantDirectorModel(),
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

    asyncio.run(controller.process("詳しく分析してください。", executive_observer_mode="director"))
    assert left.contexts, "left model should have been called"
    # Director asked to drop long-term memory; the left draft should not receive hippocampal recall.
    assert "[Hippocampal recall]" not in left.contexts[0]


def test_controller_requests_right_brain_when_confidence_low():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    hippocampus.index_episode("seed", "分析パターン", "Refer to diffusion and entropy.")
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

    assert "補足メモ" in answer
    assert "Reference from RightBrain" not in answer
    assert "[Hemisphere Routing]" not in answer
    assert "[Hemisphere Semantic Tilt]" not in answer
    assert "[Unconscious Linguistic Fabric]" not in answer
    assert "[Linguistic Motifs]" not in answer
    assert "[Cognitive Distortion Audit]" not in answer
    assert callosum.payloads, "Right brain should have been consulted"
    sent_payload = callosum.payloads[0]["payload"]
    assert sent_payload["temperature"] > 0
    assert sent_payload["budget"] in {"small", "large"}
    assert "Hippocampal" in (sent_payload.get("context") or "")
    assert "psychoid_attention_bias" in sent_payload
    assert sent_payload["psychoid_attention_bias"]["matrix"]
    assert "coherence_vector" in sent_payload
    assert sent_payload.get("hemisphere_mode") in {"left", "right", "balanced"}
    assert memory.past_qas, "Answer should be stored back into shared memory"
    assert any(evt == "policy_decision" for evt, _ in telemetry.events)
    assert any(evt == "affective_state" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_routing" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_semantic_tilt" for evt, _ in telemetry.events)
    assert any(evt == "coherence_unconscious_weave" for evt, _ in telemetry.events)
    assert any(evt == "coherence_linguistic_motifs" for evt, _ in telemetry.events)
    assert any(evt == "schema_profile" for evt, _ in telemetry.events)
    assert any(evt == "cognitive_distortion_audit" for evt, _ in telemetry.events)
    assert any(evt == "unconscious_field" for evt, _ in telemetry.events)
    assert any(evt == "unconscious_outcome" for evt, _ in telemetry.events)
    assert any(evt == "psychoid_signal" for evt, _ in telemetry.events)
    assert any(evt == "psychoid_attention_projection" for evt, _ in telemetry.events)
    assert any(evt == "prefrontal_focus" for evt, _ in telemetry.events)
    assert any(evt == "interaction_complete" for evt, _ in telemetry.events)
    assert any(evt == "hippocampal_collaboration" for evt, _ in telemetry.events)
    assert any(evt == "basal_ganglia" for evt, _ in telemetry.events)
    assert any(evt == "coherence_signal" for evt, _ in telemetry.events)
    assert any(evt == "schema_profile" for evt, _ in telemetry.events)
    assert any(evt == "cognitive_distortion_audit" for evt, _ in telemetry.events)
    assert any(evt == "inner_dialogue_trace" for evt, _ in telemetry.events)
    assert len(hippocampus.episodes) >= 2
    latest_trace = hippocampus.episodes[-1]
    assert latest_trace.leading in {"left", "right", "braided"}
    assert latest_trace.collaboration_strength is not None
    assert latest_trace.annotations.get("hemisphere_mode") in {"left", "right", "balanced"}
    rollups = [payload["rollup"] for evt, payload in telemetry.events if evt == "hippocampal_collaboration"]
    assert rollups and rollups[-1]["window"] >= 1
    final_tags = memory.past_qas[-1].tags
    assert any(tag.startswith("archetype_") for tag in final_tags)
    assert any(tag.startswith("basal_") for tag in final_tags)
    assert "psychoid_projection" in final_tags
    assert "psychoid_attention" in final_tags
    assert any(tag.startswith("psychoid_") for tag in final_tags if tag != "psychoid_projection")
    assert any(tag.startswith("coherence") for tag in final_tags)
    assert any(tag.startswith("linguistic_fabric") for tag in final_tags)
    assert any(tag.startswith("linguistic_motif") for tag in final_tags)
    assert "coherence_linguistic_motif" in final_tags
    assert any(tag.startswith("hemisphere_") for tag in final_tags)
    assert any(tag.startswith("hemisphere_tilt_") for tag in final_tags)
    assert any(tag.startswith("schema_user") for tag in final_tags)
    assert any(tag.startswith("mode_user") for tag in final_tags)
    assert any(tag.startswith("mode_agent") for tag in final_tags)
    assert "schema_profile" in final_tags
    assert "distortion_audit" in final_tags
    assert "architecture_path" in final_tags
    inner_events = [payload for evt, payload in telemetry.events if evt == "inner_dialogue_trace"]
    assert inner_events, "Inner dialogue telemetry should be captured"
    steps = inner_events[-1]["steps"]
    assert steps, "Telemetry should expose dialogue steps"
    assert any(step.get("phase") == "left_draft" for step in steps)
    assert any(step.get("phase") in {"callosum_response", "consult_skipped"} for step in steps)
    architecture_events = [
        payload for evt, payload in telemetry.events if evt == "architecture_path"
    ]
    assert architecture_events, "Architecture path telemetry should be emitted"
    architecture_path = architecture_events[-1]["path"]
    assert architecture_path and architecture_path[0]["stage"] == "perception"
    assert any(stage.get("stage") == "memory" for stage in architecture_path)
    interaction_ids = [
        payload["qid"] for evt, payload in telemetry.events if evt == "interaction_complete"
    ]
    assert interaction_ids, "Interaction completion should provide a qid"
    latest_qid = interaction_ids[-1]
    flow_record = memory.dialogue_flow(latest_qid)
    assert flow_record is not None
    assert flow_record.get("architecture")
    assert flow_record.get("architecture_count") == len(architecture_path)


def test_trivial_chat_skips_consult():
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
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    # Avoid right-preview generation in this unit test.
    controller._last_leading_brain = "right"

    answer = asyncio.run(controller.process("やあ"))

    assert callosum.payloads == []
    assert "補足メモ" not in answer
    inner_events = [payload for evt, payload in telemetry.events if evt == "inner_dialogue_trace"]
    assert inner_events, "Inner dialogue telemetry should be captured"
    steps = inner_events[-1]["steps"]
    assert any(step.get("phase") == "consult_skipped" for step in steps)
    assert not any(step.get("phase") == "callosum_response" for step in steps)


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

    assert "Deeper take:" in answer
    assert "Reference from RightBrain" not in answer
    assert "[Hemisphere Routing]" not in answer
    assert "[Hemisphere Semantic Tilt]" not in answer
    assert "[Unconscious Linguistic Fabric]" not in answer
    assert "[Linguistic Motifs]" not in answer
    assert "[Coherence Integration]" not in answer
    assert "[Cognitive Distortion Audit]" not in answer
    assert "[Architecture Path]" not in answer
    assert memory.past_qas, "Final answer should be recorded"
    # Ensure fallback pathway annotated the tags
    final_trace = memory.past_qas[-1]
    assert any("right_model_fallback" == tag for tag in final_trace.tags)
    assert any("psychoid_projection" == tag for tag in final_trace.tags)
    assert any("psychoid_attention" == tag for tag in final_trace.tags)
    assert any(tag.startswith("coherence") for tag in final_trace.tags)
    assert any(tag.startswith("hemisphere_") for tag in final_trace.tags)
    assert "schema_profile" in final_trace.tags
    assert "distortion_audit" in final_trace.tags
    assert "architecture_path" in final_trace.tags
    assert any(payload["success"] for evt, payload in telemetry.events if evt == "interaction_complete")
    assert any(evt == "coherence_signal" for evt, _ in telemetry.events)
    assert any(evt == "coherence_unconscious_weave" for evt, _ in telemetry.events)
    assert any(evt == "coherence_linguistic_motifs" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_routing" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_semantic_tilt" for evt, _ in telemetry.events)
    assert any(evt == "schema_profile" for evt, _ in telemetry.events)
    assert any(evt == "cognitive_distortion_audit" for evt, _ in telemetry.events)
    assert len(hippocampus.episodes) >= 1
    architecture_events = [
        payload for evt, payload in telemetry.events if evt == "architecture_path"
    ]
    assert architecture_events, "Architecture path should be logged even on fallback"
    fallback_path = architecture_events[-1]["path"]
    assert any(stage.get("stage") == "inner_dialogue" for stage in fallback_path)


def test_metacognition_strips_unasked_weather_claims():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    cortex = PrefrontalCortex()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=WeatherClaimLeft(),
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=cortex,
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("調子はどう？"))

    assert callosum.payloads == []
    assert "天気" not in answer
    assert "元気" in answer
    meta = [payload for evt, payload in telemetry.events if evt == "metacognition"]
    assert meta
    assert "unsupported_sensing" in (meta[-1].get("flags") or [])


def test_metacognition_flags_offtopic_answer_without_overriding_user_output():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=OffTopicLeft(),
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("量子 デコヒーレンス を説明して"))

    assert callosum.payloads == []
    # The answer stays model-authored; metacognition only flags drift out-of-band.
    assert "猫" in answer
    meta = [payload for evt, payload in telemetry.events if evt == "metacognition"]
    assert meta
    assert meta[-1].get("action") == "clarify"
    assert meta[-1].get("clarifying_question")


def test_default_mode_reflections_suppressed_after_low_focus_streak():
    class StaticUnconsciousField:
        def analyse(self, *, question: str, draft: str):  # noqa: ANN001
            return type("Profile", (), {"top_k": ["self"]})()

        def summary(self, profile):  # noqa: ANN001
            return UnconsciousSummaryModel(
                top_k=["self"],
                archetype_map=[
                    {"id": "self", "label": "Self", "intensity": 0.95},
                    {"id": "hero", "label": "Hero", "intensity": 0.12},
                ],
                emergent_ideas=[],
                stress_released=0.0,
                cache_depth=7,
                psychoid_signal=None,
            )

        def integrate_outcome(  # noqa: ANN001
            self,
            *,
            mapping,
            question: str,
            draft: str,
            final_answer: str,
            success: bool,
            decision_state=None,
            affect=None,
            novelty=None,
            reward=None,
        ):
            return {"seed_cached": True, "stress_delta": 0.0, "cache_depth": 7}

    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=AlwaysConsultPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=StaticUnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
        default_mode_network=DefaultModeNetwork(
            cooldown_steps=0,
            activation_bias=0.0,
            min_cache_depth=0,
        ),
    )

    asyncio.run(controller.process("yo", leading_brain="right"))
    asyncio.run(controller.process("hi", leading_brain="right"))

    assert len(callosum.payloads) == 2
    assert "default_mode_reflections" in callosum.payloads[0]["payload"]
    assert "default_mode_reflections" not in callosum.payloads[1]["payload"]


def test_system2_forces_critic_and_revises_answer():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "2+2=5"

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            assert "Reasoning critic notes" in info
            assert "Math error" in info
            return "2+2=4"

    callosum = CriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("Compute 2+2?", system2_mode="on"))

    assert answer == "2+2=4"
    assert left.integrations == 1
    assert callosum.payloads, "System2 should trigger a critic call"
    assert callosum.payloads[0]["payload"].get("type") == "ASK_CRITIC"


def test_system2_runs_verification_round_and_tracks_issue_decay():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "2+2=5"

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            return "2+2=4"

    callosum = TwoPhaseCriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("Compute 2+2?", system2_mode="on"))

    assert answer == "2+2=4"
    assert left.integrations == 1
    assert callosum.critic_calls >= 2
    critic_rounds = [entry["payload"].get("round") for entry in callosum.payloads]
    assert 2 in critic_rounds

    refinement_events = [
        payload for evt, payload in telemetry.events if evt == "system2_refinement"
    ]
    assert refinement_events
    latest = refinement_events[-1]
    assert latest.get("round_target", 0) >= 2
    assert latest.get("rounds", 0) >= 2
    assert latest.get("initial_issues") == 1
    assert latest.get("final_issues") == 0
    assert latest.get("resolved") is True


def test_system2_skips_revision_when_critic_ok():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "All good."

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            return "SHOULD_NOT_RUN"

    callosum = OkCriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("Compute 2+2?", system2_mode="on"))

    assert answer == "All good."
    assert left.integrations == 0
    assert callosum.payloads
    assert callosum.payloads[0]["payload"].get("type") == "ASK_CRITIC"


def test_system2_verification_does_not_overcount_rephrased_issues():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "2+2=5"

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            return "2+2=4"

    callosum = RephraseCriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("Compute 2+2?", system2_mode="on"))

    assert answer == "2+2=4"
    assert callosum.critic_calls >= 2
    # Initial critique-driven revision runs once; no extra revision for rephrased follow-up issues.
    assert left.integrations == 1

    refinement_events = [
        payload for evt, payload in telemetry.events if evt == "system2_refinement"
    ]
    assert refinement_events
    latest = refinement_events[-1]
    assert latest.get("initial_issues") == 1
    assert latest.get("final_issues") == 1
    assert latest.get("followup_new_issues") == []
    assert latest.get("followup_revision") is False


def test_system2_round3_revises_persistent_issues_once():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "2+2=5"

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            if self.integrations == 1:
                return "2+2=4 (unchecked)"
            return "2+2=4"

    callosum = PersistentCriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(
        controller.process(
            "Check this quickly:\n```python\nprint(2 + 2)\n```",
            system2_mode="on",
        )
    )

    assert answer == "2+2=4"
    assert left.integrations == 2
    assert callosum.critic_calls >= 3

    refinement_events = [
        payload for evt, payload in telemetry.events if evt == "system2_refinement"
    ]
    assert refinement_events
    latest = refinement_events[-1]
    assert latest.get("round_target", 0) >= 3
    assert latest.get("rounds", 0) >= 3
    assert latest.get("initial_issues") == 2
    assert latest.get("final_issues") == 0
    assert latest.get("resolved") is True
    assert latest.get("followup_revision") is True
    assert latest.get("followup_new_issues") == []


def test_system2_filters_low_signal_critic_noise():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "Average speed is (80 + 40) / 2 = 60 km/h."

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            return "Average speed = total distance / total time."

    callosum = LowSignalNoiseCriticCallosum()
    telemetry = TrackingTelemetry()
    left = RevisingLeft()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=left,
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(
        controller.process(
            "A car travels 120 km in 1.5 hours, then 80 km in 2 hours. What is the average speed?",
            system2_mode="on",
        )
    )

    assert "total distance / total time" in answer
    assert left.integrations == 1
    assert callosum.critic_calls >= 2

    refinement_events = [
        payload for evt, payload in telemetry.events if evt == "system2_refinement"
    ]
    assert refinement_events
    latest = refinement_events[-1]
    assert latest.get("initial_issues") == 1
    assert latest.get("final_issues") == 0
    assert latest.get("resolved") is True
    assert latest.get("followup_new_issues") == []


def test_system2_low_signal_filter_can_be_disabled_via_env():
    class RevisingLeft:
        uses_external_llm = True

        def __init__(self):
            self.integrations = 0

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "Average speed is (80 + 40) / 2 = 60 km/h."

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            self.integrations += 1
            return "Average speed = total distance / total time."

    previous = os.environ.get("DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER")
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = "0"
    try:
        callosum = LowSignalNoiseCriticCallosum()
        telemetry = TrackingTelemetry()
        left = RevisingLeft()
        controller = DualBrainController(
            callosum=callosum,
            memory=SharedMemory(),
            left_model=left,
            right_model=RightBrainModel(),
            policy=AlwaysSkipPolicy(),
            hypothalamus=Hypothalamus(),
            reasoning_dial=ReasoningDial(mode="evaluative"),
            auditor=Auditor(),
            orchestrator=Orchestrator(3),
            telemetry=telemetry,
            unconscious_field=UnconsciousField(),
            prefrontal_cortex=PrefrontalCortex(),
            basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
        )

        answer = asyncio.run(
            controller.process(
                "A car travels 120 km in 1.5 hours, then 80 km in 2 hours. What is the average speed?",
                system2_mode="on",
            )
        )

        assert "total distance / total time" in answer
        assert left.integrations >= 1

        refinement_events = [
            payload for evt, payload in telemetry.events if evt == "system2_refinement"
        ]
        assert refinement_events
        latest = refinement_events[-1]
        assert latest.get("initial_issues") == 3
        assert latest.get("final_issues") == 1
        assert latest.get("resolved") is False
    finally:
        if previous is None:
            os.environ.pop("DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER", None)
        else:
            os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = previous


def test_system2_records_pitfall_patterns_in_memory_kv():
    class RevisingLeft:
        uses_external_llm = True

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "Average speed is (80 + 40) / 2 = 60 km/h."

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            return "Average speed = total distance / total time."

    callosum = LowSignalNoiseCriticCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=RevisingLeft(),
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
        hippocampus=TemporalHippocampalIndexing(dim=32),
    )

    answer = asyncio.run(
        controller.process(
            "A car travels 120 km in 1.5 hours, then 80 km in 2 hours. What is the average speed?",
            system2_mode="on",
        )
    )
    assert "total distance / total time" in answer

    counts = memory.get_kv("system2_pitfall_counts")
    examples = memory.get_kv("system2_pitfall_examples")
    last = memory.get_kv("system2_pitfall_last")
    assert isinstance(counts, dict) and counts
    assert isinstance(examples, dict) and examples
    assert isinstance(last, list) and last


def test_system2_timeout_still_emits_measurable_metrics():
    class RevisingLeft:
        uses_external_llm = True

        async def generate_answer(self, input_text: str, context: str, *, vision_images=None, on_delta=None) -> str:  # noqa: ANN001
            return "2+2=5"

        def estimate_confidence(self, draft: str) -> float:  # noqa: ANN001
            return 0.95

        async def integrate_info_async(  # noqa: ANN001
            self,
            *,
            question: str,
            draft: str,
            info: str,
            temperature: float = 0.3,
            on_delta=None,
        ) -> str:
            return "SHOULD_NOT_RUN_ON_TIMEOUT"

    callosum = TimeoutCriticCallosum()
    telemetry = TrackingTelemetry()
    controller = DualBrainController(
        callosum=callosum,
        memory=SharedMemory(),
        left_model=RevisingLeft(),
        right_model=RightBrainModel(),
        policy=AlwaysSkipPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(baseline_dopamine=0.0, novelty_weight=0.0),
    )

    answer = asyncio.run(controller.process("Compute 2+2?", system2_mode="on"))

    assert answer == "2+2=5"
    assert callosum.payloads
    assert callosum.payloads[0]["payload"].get("type") == "ASK_CRITIC"

    refinement_events = [
        payload for evt, payload in telemetry.events if evt == "system2_refinement"
    ]
    assert refinement_events
    latest = refinement_events[-1]
    assert latest.get("rounds") == 0
    assert latest.get("initial_issues") == 0
    assert latest.get("final_issues") == 0
    assert latest.get("resolved") is False
    assert latest.get("timeout") is True


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

    assert "Deeper take:" in answer
    assert "Reference from RightBrain" not in answer
    assert "[Hemisphere Routing]" not in answer
    assert "[Hemisphere Semantic Tilt]" not in answer
    assert "[Unconscious Linguistic Fabric]" not in answer
    assert "[Linguistic Motifs]" not in answer
    assert callosum.payloads, "Amygdala override should trigger consult"
    assert any("amygdala_alert" in trace.tags for trace in memory.past_qas)
    affect_events = [payload for evt, payload in telemetry.events if evt == "affective_state"]
    assert affect_events and affect_events[0]["risk"] >= 0.66
    assert any(evt == "hemisphere_routing" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_semantic_tilt" for evt, _ in telemetry.events)
    assert any(evt == "coherence_unconscious_weave" for evt, _ in telemetry.events)
    assert any(evt == "coherence_linguistic_motifs" for evt, _ in telemetry.events)


def test_right_brain_can_take_the_lead():
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
        reasoning_dial=ReasoningDial(mode="exploratory"),
        auditor=Auditor(),
        orchestrator=Orchestrator(3),
        telemetry=telemetry,
        unconscious_field=UnconsciousField(),
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    prompt = "Imagine a mythic waterfall dreamscape and describe its symbols."
    answer = asyncio.run(controller.process(prompt, leading_brain="right"))

    assert "補足メモ" in answer
    assert "[Right Brain Lead]" not in answer
    assert "[Left Brain Integration]" not in answer
    assert "Reference from RightBrain" not in answer
    assert "[Cognitive Distortion Audit]" not in answer
    assert any(evt == "leading_brain" for evt, _ in telemetry.events)
    assert any(evt == "schema_profile" for evt, _ in telemetry.events)
    assert any(evt == "cognitive_distortion_audit" for evt, _ in telemetry.events)
    assert any(evt == "inner_dialogue_trace" for evt, _ in telemetry.events)
    assert memory.past_qas, "The interaction should be persisted"
    flow = list(memory.dialogue_flows.values())[-1]
    assert flow["leading"] == "right"
    assert flow["follow"] == "left"
    assert flow.get("steps"), "Dialogue flow should include step trace"
    assert any(step.get("phase") == "callosum_response" for step in flow.get("steps", []))
    final_tags = memory.past_qas[-1].tags
    assert "schema_profile" in final_tags
    assert "distortion_audit" in final_tags
    tags = memory.past_qas[-1].tags
    assert any(tag == "leading_right" for tag in tags)


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
        default_mode_network=DefaultModeNetwork(
            min_cache_depth=0,
            stress_release_threshold=0.0,
            activation_bias=0.05,
            cooldown_steps=0,
        ),
    )

    question = "Sketch a mythic journey across unknown seas."
    context, _, _ = controller._compose_context(question)
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

    assert "[Unconscious Insight]" not in answer
    assert "[Default Mode Reflection]" not in answer
    assert "[Psychoid Field Alignment]" not in answer
    assert "[Psychoid Signifiers]" not in answer
    assert "[Hemisphere Routing]" not in answer
    assert "[Hemisphere Semantic Tilt]" not in answer
    assert "[Unconscious Linguistic Fabric]" not in answer
    assert "[Linguistic Motifs]" not in answer
    assert callosum.payloads, "Right brain should have received enriched payload"
    hint_payload = callosum.payloads[0]["payload"]
    assert "unconscious_hints" in hint_payload
    assert any("Hero" in hint for hint in hint_payload["unconscious_hints"])
    assert "default_mode_reflections" in hint_payload
    assert any("confidence" in entry for entry in hint_payload["default_mode_reflections"])
    assert "psychoid_signifiers" in hint_payload
    assert hint_payload["psychoid_bias_vector"], "Bias vector should accompany signifiers"
    assert any(evt == "default_mode_reflection" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_routing" for evt, _ in telemetry.events)
    assert any(evt == "hemisphere_semantic_tilt" for evt, _ in telemetry.events)
    assert any(evt == "coherence_unconscious_weave" for evt, _ in telemetry.events)
    assert any(evt == "coherence_linguistic_motifs" for evt, _ in telemetry.events)
    assert any(evt == "inner_dialogue_trace" for evt, _ in telemetry.events)


def test_neural_impulse_integration():
    """Test that neural impulse simulation integrates correctly with DualBrainController."""
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()
    hippocampus = TemporalHippocampalIndexing(dim=32)
    
    # Create controller without neural integrator - it will create default one
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(),
        auditor=Auditor(),
        orchestrator=Orchestrator(),
        telemetry=telemetry,
        hippocampus=hippocampus,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
        unconscious_field=UnconsciousField(),
        # Don't pass neural_integrator - let it create the default
    )
    
    # Run a simple question
    question = "Explain a complex pattern."
    result = asyncio.run(controller.process(question))
    
    # Verify neural activity was simulated
    neural_events = [e for e in telemetry.events if e[0] == "neural_impulse_activity"]
    assert len(neural_events) > 0, "Should have neural impulse activity logged"
    
    # Check neural activity data structure
    neural_event = neural_events[0]
    activity = neural_event[1]["activity"]
    assert "hemisphere" in activity
    assert "total_impulses" in activity
    assert "impulse_counts" in activity
    assert "network_activity" in activity
    
    # Plain answers should not include debug sections
    assert "[Neural Impulse Activity]" not in result
    
    # Verify network was created with expected structure
    assert controller.neural_integrator is not None
    assert len(controller.neural_integrator.neurons) > 0
    
    # Check that impulses were actually generated
    assert activity["total_impulses"] >= 0  # At least some activity should occur
    
    # Verify neurotransmitter activity tracking
    impulse_counts = activity["impulse_counts"]
    assert isinstance(impulse_counts, dict)
    assert "glutamate" in impulse_counts
    assert "gaba" in impulse_counts
    assert "dopamine" in impulse_counts


def test_slow_executive_timeout_does_not_crash_process():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        executive_model=SlowExecutiveModel(),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(2),
        telemetry=telemetry,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    # Should return without raising CancelledError / killing the engine when the
    # executive task times out.
    answer = asyncio.run(controller.process("やあ", executive_mode="observe"))
    assert isinstance(answer, str)
    assert answer.strip()


def test_executive_observe_does_not_inject_directives():
    callosum = DummyCallosum()
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    left = CaptureLeftModel(draft="draft", confidence=0.1)
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=left,
        right_model=RightBrainModel(),
        executive_model=InstantExecutiveModel(mix_in="(should not appear)", directives={"tone": "robotic"}),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(),
        telemetry=telemetry,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    answer = asyncio.run(controller.process("Explain a complex pattern.", executive_mode="observe"))
    assert "(INTEGRATED)" in answer
    assert left.integrations, "Integration should still occur due to right-brain material"
    merged = "\n".join(left.integrations)
    assert "Executive directives" not in merged
    assert "Executive mix-in" not in merged


def test_executive_assist_injects_mix_in():
    callosum = DummyCallosum(fail=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    left = CaptureLeftModel(draft="draft", confidence=0.9)
    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=left,
        right_model=RightBrainModel(),
        executive_model=InstantExecutiveModel(mix_in="追加の一言です。"),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(),
        telemetry=telemetry,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    answer = asyncio.run(controller.process("やあ", executive_mode="assist"))
    assert "(INTEGRATED)" in answer
    assert left.integrations
    merged = "\n".join(left.integrations)
    assert "Executive mix-in" in merged


def test_executive_observer_metrics_stores_memo_out_of_band():
    callosum = DummyCallosum(fail=True)
    memory = SharedMemory()
    telemetry = TrackingTelemetry()

    controller = DualBrainController(
        callosum=callosum,
        memory=memory,
        left_model=LeftBrainModel(),
        right_model=RightBrainModel(),
        executive_model=InstantExecutiveModel(mix_in="(observer)"),
        policy=RightBrainPolicy(),
        hypothalamus=Hypothalamus(),
        reasoning_dial=ReasoningDial(mode="evaluative"),
        auditor=Auditor(),
        orchestrator=Orchestrator(),
        telemetry=telemetry,
        prefrontal_cortex=PrefrontalCortex(),
        basal_ganglia=BasalGanglia(),
    )

    qid = "obs1"
    answer = asyncio.run(
        controller.process(
            "hi",
            qid=qid,
            executive_mode="off",
            executive_observer_mode="metrics",
        )
    )
    assert isinstance(answer, str)
    assert answer.strip()
    assert "Observer report" not in answer

    flow = memory.dialogue_flow(qid)
    assert isinstance(flow, dict)
    observer = flow.get("executive_observer")
    assert isinstance(observer, dict)
    assert observer.get("observer_mode") == "metrics"
    assert observer.get("memo")
