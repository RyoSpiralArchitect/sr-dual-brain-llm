"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DecisionOutcome:
    """Details about a single orchestration decision."""

    qid: str
    action: int
    temperature: float
    slot_ms: int
    state: Dict[str, Any]


class DualBrainController:
    """Coordinate left/right brain collaboration with adaptive heuristics."""

    def __init__(
        self,
        *,
        callosum,
        memory,
        left_model,
        right_model,
        policy,
        hypothalamus,
        reasoning_dial,
        auditor,
        orchestrator,
        default_timeout_ms: int = 6000,
        telemetry: Optional[Any] = None,
    ) -> None:
        self.callosum = callosum
        self.memory = memory
        self.left_model = left_model
        self.right_model = right_model
        self.policy = policy
        self.hypothalamus = hypothalamus
        self.reasoning_dial = reasoning_dial
        self.auditor = auditor
        self.orchestrator = orchestrator
        self.default_timeout_ms = default_timeout_ms
        self.telemetry = telemetry or _NullTelemetry()

    @staticmethod
    def _infer_question_type(question: str) -> str:
        lowered = question.lower()
        if any(key in lowered for key in ["why", "なぜ", "proof", "証明"]):
            return "hard"
        if any(key in lowered for key in ["how", "計算", "analyse", "分析"]):
            return "medium"
        return "easy"

    def _build_policy_state(self, question: str, draft: str, confidence: float) -> Dict[str, Any]:
        novelty = self.memory.novelty_score(question)
        return {
            "left_conf": confidence,
            "draft_len": len(draft),
            "novelty": novelty,
            "risk": max(0.0, min(1.0, 1.0 - confidence)),
            "q_type": self._infer_question_type(question),
        }

    def decide(self, question: str, draft: str, confidence: float) -> DecisionOutcome:
        state = self._build_policy_state(question, draft, confidence)
        action = self.policy.decide(state)
        action = self.reasoning_dial.adjust_decision(state, action)
        base_temp = self.hypothalamus.recommend_temperature(confidence)
        temperature = self.reasoning_dial.scale_temperature(base_temp)
        slot_ms = self.hypothalamus.recommend_slot_ms(state["risk"])
        qid = str(uuid.uuid4())
        return DecisionOutcome(qid=qid, action=action, temperature=temperature, slot_ms=slot_ms, state=state)

    async def process(self, question: str) -> str:
        context = self.memory.retrieve_related(question)
        draft = await self.left_model.generate_answer(question, context)
        confidence = self.left_model.estimate_confidence(draft)
        decision = self.decide(question, draft, confidence)
        self.telemetry.log(
            "policy_decision",
            qid=decision.qid,
            state=decision.state,
            action=decision.action,
            temperature=decision.temperature,
            slot_ms=decision.slot_ms,
        )

        if not self.orchestrator.register_request(decision.qid):
            return "Loop-killed"

        final_answer = draft
        success = False
        latency_ms = 0.0
        start = time.perf_counter()
        try:
            if decision.action == 0:
                success = True
            else:
                payload = {
                    "type": "ASK_DETAIL",
                    "qid": decision.qid,
                    "question": question,
                    "draft_sum": draft if len(draft) < 280 else draft[:280],
                    "temperature": decision.temperature,
                    "budget": self.reasoning_dial.pick_budget(),
                    "context": context,
                }
                timeout_ms = max(self.default_timeout_ms, decision.slot_ms * 12)
                original_slot = getattr(self.callosum, "slot_ms", decision.slot_ms)
                try:
                    self.callosum.slot_ms = decision.slot_ms
                    response = await self.callosum.ask_detail(payload, timeout_ms=timeout_ms)
                finally:
                    self.callosum.slot_ms = original_slot
                response_source = "callosum"
                detail_notes = response.get("notes_sum")
                if response.get("error") or not detail_notes:
                    response_source = "right_model_fallback"
                    try:
                        fallback = await self.right_model.deepen(
                            decision.qid,
                            question,
                            draft,
                            self.memory,
                            temperature=decision.temperature,
                            budget=payload["budget"],
                            context=context,
                        )
                    except Exception as exc:  # pragma: no cover - defensive guard
                        final_answer = draft + f"\n(Right brain error: {exc})"
                    else:
                        detail_notes = fallback.get("notes_sum")
                        decision.state["right_conf"] = fallback.get("confidence_r", 0.0)
                        success = True
                else:
                    decision.state["right_conf"] = response.get("confidence_r", 0.0)
                    success = True

                if detail_notes:
                    final_answer = self.left_model.integrate_info(draft, detail_notes)
                if decision.state.get("right_conf"):
                    decision.state["right_source"] = response_source
        except asyncio.TimeoutError:
            final_answer = draft + "\n(Right brain timeout: continuing with draft)"
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.orchestrator.clear(decision.qid)

        audit_result = self.auditor.check(final_answer)
        if not audit_result.get("ok", True):
            final_answer = draft + f"\n(Auditor veto: {audit_result.get('reason', 'unknown')})"
            success = False

        reward = 0.75 if success else 0.45
        if not audit_result.get("ok", True):
            reward = min(reward, 0.25)
        self.hypothalamus.update_feedback(reward=reward, latency_ms=latency_ms)

        tags = {decision.state.get("q_type", "unknown")}
        if decision.state.get("novelty", 0.0) < 0.4:
            tags.add("familiar")
        if decision.state.get("right_source"):
            tags.add(decision.state["right_source"])

        self.memory.store({"Q": question, "A": final_answer}, tags=tags)
        self.telemetry.log(
            "interaction_complete",
            qid=decision.qid,
            success=success,
            latency_ms=latency_ms,
            reward=reward,
            tags=list(tags),
        )
        return final_answer


class _NullTelemetry:
    """Default telemetry sink used when no observer is provided."""

    def log(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass
