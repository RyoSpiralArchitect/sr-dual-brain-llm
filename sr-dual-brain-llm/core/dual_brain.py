"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .amygdala import Amygdala
from .temporal_hippocampal_indexing import TemporalHippocampalIndexing


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
        amygdala: Optional[Amygdala] = None,
        hippocampus: Optional[TemporalHippocampalIndexing] = None,
        unconscious_field: Optional[Any] = None,
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
        self.hippocampus = hippocampus
        self.unconscious_field = unconscious_field

    def _compose_context(self, question: str) -> str:
        """Blend working-memory recall with hippocampal episodic cues."""

        memory_context = self.memory.retrieve_related(question)
        hippocampal_context = ""
        if self.hippocampus is not None:
            hippocampal_context = self.hippocampus.retrieve_summary(question)

        segments = []
        if memory_context:
            segments.append(memory_context)
        if hippocampal_context:
            segments.append(f"[Hippocampal recall] {hippocampal_context}")
        return "\n".join(segments)

    def _sense_affect(self, question: str, draft: str) -> Dict[str, float]:
        """Approximate limbic evaluation inspired by the human amygdala."""

        metrics = self.amygdala.analyze(f"{question}\n{draft}") if self.amygdala else {}
        return {
            "valence": float(metrics.get("valence", 0.0)),
            "arousal": float(metrics.get("arousal", 0.0)),
            "risk": float(metrics.get("risk", 0.0)),
        }

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
        affect: Dict[str, float],
        novelty: float,
        consult_bias: float,
        hippocampal_density: int,
    ) -> Dict[str, Any]:
        adjusted_conf = max(
            0.0,
            min(1.0, confidence - max(0.0, consult_bias) - 0.25 * affect.get("risk", 0.0)),
        )
        return {
            "left_conf": adjusted_conf,
            "left_conf_raw": confidence,
            "draft_len": len(draft),
            "novelty": novelty,
            "risk": max(0.0, min(1.0, 1.0 - confidence)),
            "q_type": self._infer_question_type(question),
            "amygdala_risk": affect.get("risk", 0.0),
            "affect_valence": affect.get("valence", 0.0),
            "affect_arousal": affect.get("arousal", 0.0),
            "consult_bias": consult_bias,
            "hippocampal_density": hippocampal_density,
        }

    def decide(
        self,
        question: str,
        draft: str,
        confidence: float,
        affect: Dict[str, float],
        novelty: float,
        consult_bias: float,
        hippocampal_density: int,
    ) -> DecisionOutcome:
        state = self._build_policy_state(
            question,
            draft,
            confidence,
            affect,
            novelty,
            consult_bias,
            hippocampal_density,
        )
        action = self.policy.decide(state)
        action = self.reasoning_dial.adjust_decision(state, action)
        if state["amygdala_risk"] >= 0.66:
            action = max(action, 1)
            state["amygdala_override"] = True
        else:
            state["amygdala_override"] = False
        base_temp = self.hypothalamus.recommend_temperature(confidence)
        temperature = self.reasoning_dial.scale_temperature(base_temp)
        slot_ms = self.hypothalamus.recommend_slot_ms(state["risk"])
        qid = str(uuid.uuid4())
        return DecisionOutcome(qid=qid, action=action, temperature=temperature, slot_ms=slot_ms, state=state)

    async def process(self, question: str) -> str:
        context = self._compose_context(question)
        hippocampal_density = len(self.hippocampus.episodes) if self.hippocampus is not None else 0
        draft = await self.left_model.generate_answer(question, context)
        confidence = self.left_model.estimate_confidence(draft)
        affect = self._sense_affect(question, draft)
        novelty = self.memory.novelty_score(question)
        consult_bias = self.hypothalamus.bias_for_consult(novelty)
        decision = self.decide(
            question,
            draft,
            confidence,
            affect,
            novelty,
            consult_bias,
            hippocampal_density,
        )
        unconscious_profile = None
        if self.unconscious_field is not None:
            try:
                unconscious_profile = self.unconscious_field.analyse(
                    question=question, draft=draft
                )
            except Exception:  # pragma: no cover - defensive guard
                unconscious_profile = None
            else:
                summary = self.unconscious_field.summary(unconscious_profile)
                decision.state["unconscious_top"] = summary["top_k"][0] if summary["top_k"] else None
                self.telemetry.log(
                    "unconscious_field",
                    qid=decision.qid,
                    summary=summary,
                )
        self.telemetry.log(
            "affective_state",
            qid=decision.qid,
            valence=affect["valence"],
            arousal=affect["arousal"],
            risk=affect["risk"],
            novelty=novelty,
            hippocampal_density=hippocampal_density,
        )
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
        if decision.state.get("amygdala_override"):
            tags.add("amygdala_alert")
        if decision.state.get("affect_valence", 0.0) > 0.25:
            tags.add("positive_valence")
        elif decision.state.get("affect_valence", 0.0) < -0.25:
            tags.add("negative_valence")
        if unconscious_profile is not None and unconscious_profile.top_k:
            tags.add(f"archetype_{unconscious_profile.top_k[0]}")

        self.memory.store({"Q": question, "A": final_answer}, tags=tags)
        episodic_total = 0
        if self.hippocampus is not None:
            self.hippocampus.index_episode(decision.qid, question, final_answer)
            episodic_total = len(self.hippocampus.episodes)
        self.telemetry.log(
            "interaction_complete",
            qid=decision.qid,
            success=success,
            latency_ms=latency_ms,
            reward=reward,
            tags=list(tags),
            amygdala_override=decision.state.get("amygdala_override", False),
            hippocampal_total=episodic_total,
        )
        return final_answer


class _NullTelemetry:
    """Default telemetry sink used when no observer is provided."""

    def log(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass
