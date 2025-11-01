"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .prefrontal_cortex import PrefrontalCortex, FocusSummary
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
        prefrontal_cortex: Optional[PrefrontalCortex] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
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
        self.prefrontal_cortex = prefrontal_cortex
        self.basal_ganglia = basal_ganglia or BasalGanglia()

    def _compose_context(self, question: str) -> Tuple[str, FocusSummary | None]:
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
        combined = "\n".join(segments)
        focus: FocusSummary | None = None
        if self.prefrontal_cortex is not None:
            focus = self.prefrontal_cortex.synthesise_focus(
                question=question,
                memory_context=memory_context,
                hippocampal_context=hippocampal_context,
            )
            combined = self.prefrontal_cortex.gate_context(combined, focus)
        return combined, focus

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
        focus_metric: float,
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
            "prefrontal_focus": focus_metric,
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
        focus_metric: float,
    ) -> DecisionOutcome:
        state = self._build_policy_state(
            question,
            draft,
            confidence,
            affect,
            novelty,
            consult_bias,
            hippocampal_density,
            focus_metric,
        )
        action = self.policy.decide(state)
        action = self.reasoning_dial.adjust_decision(state, action)
        basal_signal = None
        if self.basal_ganglia is not None:
            basal_signal = self.basal_ganglia.evaluate(
                state=state,
                affect=affect,
                focus_metric=focus_metric,
            )
            state["basal_go"] = basal_signal.go_probability
            state["basal_inhibition"] = basal_signal.inhibition
            state["basal_dopamine"] = basal_signal.dopamine_level
            if basal_signal.note:
                state["basal_note"] = basal_signal.note
            if basal_signal.recommended_action == 1:
                action = max(action, 1)
            elif basal_signal.recommended_action == 0:
                action = min(action, 0)
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
        context, focus = self._compose_context(question)
        focus_metric = 0.0
        if self.prefrontal_cortex is not None and focus is not None:
            focus_metric = self.prefrontal_cortex.focus_metric(focus)
        hippocampal_density = len(self.hippocampus.episodes) if self.hippocampus is not None else 0
        draft = await self.left_model.generate_answer(question, context)
        confidence = self.left_model.estimate_confidence(draft)
        affect = self._sense_affect(question, draft)
        novelty = self.memory.novelty_score(question)
        consult_bias = self.hypothalamus.bias_for_consult(novelty)
        if self.prefrontal_cortex is not None and focus is not None:
            consult_bias = self.prefrontal_cortex.adjust_consult_bias(consult_bias, focus)
        decision = self.decide(
            question,
            draft,
            confidence,
            affect,
            novelty,
            consult_bias,
            hippocampal_density,
            focus_metric,
        )
        basal_signal = getattr(self.basal_ganglia, "last_signal", None)
        if focus is not None:
            decision.state["prefrontal_keywords"] = list(focus.keywords)
            decision.state["prefrontal_relevance"] = focus.relevance
            decision.state["prefrontal_hippocampal_overlap"] = focus.hippocampal_overlap
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
        if focus is not None:
            self.telemetry.log(
                "prefrontal_focus",
                qid=decision.qid,
                focus=focus.to_dict(),
                metric=focus_metric,
            )
        if basal_signal is not None:
            self.telemetry.log(
                "basal_ganglia",
                qid=decision.qid,
                signal=basal_signal.to_dict(),
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
                if focus is not None and focus.keywords:
                    payload["focus_keywords"] = list(focus.keywords[:5])
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
        if focus is not None and self.prefrontal_cortex is not None:
            tags.update(self.prefrontal_cortex.tags(focus))
        if basal_signal is not None:
            tags.update(self.basal_ganglia.tags(basal_signal))

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
        if self.basal_ganglia is not None and basal_signal is not None:
            self.basal_ganglia.integrate_feedback(reward=reward, latency_ms=latency_ms)
        return final_answer


class _NullTelemetry:
    """Default telemetry sink used when no observer is provided."""

    def log(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass
