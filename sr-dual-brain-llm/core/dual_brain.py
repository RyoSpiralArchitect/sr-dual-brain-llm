"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .amygdala import Amygdala
from .prefrontal_cortex import PrefrontalCortex
from .temporal_hippocampal_indexing import TemporalHippocampalIndexing


@dataclass
class DecisionOutcome:
    """Details about a single orchestration decision."""

    qid: str
    action: int
    temperature: float
    slot_ms: int
    state: Dict[str, Any]
    signals: Dict[str, Any]


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
        amygdala: Optional[Amygdala] = None,
        prefrontal: Optional[PrefrontalCortex] = None,
        hippocampus: Optional[TemporalHippocampalIndexing] = None,
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
        self.amygdala = amygdala or Amygdala()
        self.prefrontal = prefrontal or PrefrontalCortex()
        self.hippocampus = hippocampus or TemporalHippocampalIndexing()

    @staticmethod
    def _infer_question_type(question: str) -> str:
        lowered = question.lower()
        if any(key in lowered for key in ["why", "なぜ", "proof", "証明"]):
            return "hard"
        if any(key in lowered for key in ["how", "計算", "analyse", "分析"]):
            return "medium"
        return "easy"

    def _build_policy_state(
        self,
        question: str,
        draft: str,
        confidence: float,
        *,
        signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        novelty = signals.get("novelty", self.memory.novelty_score(question))
        amygdala_signal = signals.get("amygdala", {})
        return {
            "left_conf": confidence,
            "draft_len": len(draft),
            "novelty": novelty,
            "risk": max(0.0, min(1.0, 1.0 - confidence)),
            "q_type": self._infer_question_type(question),
            "amygdala_risk": float(amygdala_signal.get("risk", 0.0)),
            "amygdala_valence": float(amygdala_signal.get("valence", 0.0)),
            "amygdala_arousal": float(amygdala_signal.get("arousal", 0.0)),
            "conflict_signal": float(signals.get("conflict", 0.0)),
            "goal_focus": float(signals.get("goal_focus", 0.5)),
            "hippocampal_hits": int(signals.get("hippocampal_hits", 0)),
        }

    def decide(
        self,
        question: str,
        draft: str,
        confidence: float,
        *,
        signals: Dict[str, Any],
    ) -> DecisionOutcome:
        state = self._build_policy_state(question, draft, confidence, signals=signals)
        action = self.policy.decide(state)
        action = self.reasoning_dial.adjust_decision(state, action)
        base_temp = self.hypothalamus.recommend_temperature(confidence)
        temperature = self.reasoning_dial.scale_temperature(base_temp)
        if self.prefrontal:
            action = self.prefrontal.gate_consult(action, state)
            temperature = self.prefrontal.modulate_temperature(temperature, state)
        slot_ms = self.hypothalamus.recommend_slot_ms(
            max(state["risk"], state.get("amygdala_risk", 0.0))
        )
        qid = str(uuid.uuid4())
        return DecisionOutcome(
            qid=qid,
            action=action,
            temperature=temperature,
            slot_ms=slot_ms,
            state=state,
            signals=signals,
        )

    async def process(self, question: str) -> str:
        novelty = self.memory.novelty_score(question)
        hippocampal_hits = self.hippocampus.retrieve(question, topk=3)
        episodic_context = " | ".join(
            f"(sim={score:.2f}) {ep['text'].splitlines()[0][:120]}"
            for score, ep in hippocampal_hits
        )
        context = self.memory.retrieve_related(question)
        combined_context = "\n".join(filter(None, [episodic_context, context]))
        draft = await self.left_model.generate_answer(question, combined_context)
        confidence = self.left_model.estimate_confidence(draft)
        amygdala_signal = self.amygdala.analyze(f"{question}\n{draft}")
        prefrontal_metrics = self.prefrontal.update_goal_state(
            question,
            draft,
            combined_context,
            amygdala_signal=amygdala_signal,
            novelty=novelty,
        )
        signals = {
            "novelty": novelty,
            "amygdala": amygdala_signal,
            "conflict": prefrontal_metrics.get("conflict", 0.0),
            "goal_focus": prefrontal_metrics.get("goal_focus", 0.5),
            "hippocampal_hits": len(hippocampal_hits),
            "control_tone": prefrontal_metrics.get("control_tone", 0.6),
        }
        decision = self.decide(question, draft, confidence, signals=signals)
        self.telemetry.log(
            "policy_decision",
            qid=decision.qid,
            state=decision.state,
            action=decision.action,
            temperature=decision.temperature,
            slot_ms=decision.slot_ms,
            signals=decision.signals,
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
                    "context": combined_context,
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
                            context=combined_context,
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
        if decision.state.get("amygdala_risk", 0.0) > 0.6:
            tags.add("high_risk")

        self.memory.store({"Q": question, "A": final_answer}, tags=tags)
        self.hippocampus.index_episode(decision.qid, question, final_answer)
        self.telemetry.log(
            "interaction_complete",
            qid=decision.qid,
            success=success,
            latency_ms=latency_ms,
            reward=reward,
            tags=list(tags),
            signals=decision.signals,
        )
        self.prefrontal.integrate_feedback(reward, latency_ms, success=success)
        return final_answer


class _NullTelemetry:
    """Default telemetry sink used when no observer is provided."""

    def log(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass
