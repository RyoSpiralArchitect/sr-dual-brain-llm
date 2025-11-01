"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .prefrontal_cortex import PrefrontalCortex, FocusSummary
from .default_mode_network import DefaultModeNetwork, DefaultModeReflection
from .temporal_hippocampal_indexing import TemporalHippocampalIndexing
from .psychoid_attention import PsychoidAttentionAdapter, PsychoidAttentionProjection
from .coherence_resonator import (
    CoherenceResonator,
    CoherenceSignal,
    HemisphericCoherence,
)
from .schema import PsychoidSignalModel, UnconsciousSummaryModel


_RIGHT_HEMISPHERE_KEYWORDS = {
    "story",
    "stories",
    "poem",
    "poetry",
    "metaphor",
    "imagine",
    "imagination",
    "dream",
    "vision",
    "art",
    "artistic",
    "myth",
    "mythic",
    "lyric",
    "narrative",
    "creative",
    "感情",
    "夢",
    "詩",
    "物語",
    "象徴",
    "直感",
    "比喩",
}

_LEFT_HEMISPHERE_KEYWORDS = {
    "analyze",
    "analysis",
    "explain",
    "detail",
    "formula",
    "proof",
    "derive",
    "compute",
    "calculation",
    "step",
    "structure",
    "data",
    "algorithm",
    "framework",
    "論理",
    "計算",
    "分析",
    "証明",
    "手順",
    "仕組み",
    "数式",
}

_RIGHT_HEMISPHERE_MARKERS = ["夢", "詩", "物語", "感情", "象徴", "幻想", "archetype"]
_LEFT_HEMISPHERE_MARKERS = ["計算", "分析", "証明", "構造", "論理", "手順", "データ"]

_RIGHT_HEMISPHERE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"compose (?:a )?(poem|story)",
        r"sketch .*journey",
        r"imagine(?:\s+or)?\s+",
        r"mythic\s+journey",
        r"write .*lyrics",
    ]
]

_LEFT_HEMISPHERE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"step[- ]by[- ]step",
        r"derive .*formula",
        r"calculate",
        r"provide an? analysis",
    ]
]


def _semantic_tokens(*chunks: str) -> List[str]:
    tokens: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        tokens.extend(re.findall(r"[\w']+", chunk.lower()))
    return tokens


def _format_section(title: str, lines: List[str]) -> str:
    body = "\n".join(lines) if lines else ""
    if body:
        return f"[{title}]\n{body}"
    return f"[{title}]"


@dataclass
class DecisionOutcome:
    """Details about a single orchestration decision."""

    qid: str
    action: int
    temperature: float
    slot_ms: int
    state: Dict[str, Any]


@dataclass
class HemisphericSignal:
    """Raw hemisphere cue scoring derived from the question and focus."""

    mode: str
    bias: float
    right_score: float
    left_score: float
    token_count: int

    def to_payload(self) -> Dict[str, float]:
        return {
            "mode": self.mode,
            "bias": float(self.bias),
            "right_score": float(self.right_score),
            "left_score": float(self.left_score),
            "token_count": float(self.token_count),
        }

    @property
    def total(self) -> float:
        return self.right_score + self.left_score


@dataclass
class CollaborationProfile:
    """Aggregated assessment of how strongly both hemispheres want to co-lead."""

    strength: float
    balance: float
    density: float
    focus_bonus: float
    token_count: int

    def to_payload(self) -> Dict[str, float]:
        return {
            "strength": float(self.strength),
            "balance": float(self.balance),
            "density": float(self.density),
            "focus_bonus": float(self.focus_bonus),
            "token_count": float(self.token_count),
        }


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
        default_mode_network: Optional[DefaultModeNetwork] = None,
        psychoid_attention_adapter: Optional[PsychoidAttentionAdapter] = None,
        coherence_resonator: Optional[CoherenceResonator] = None,
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
        self.default_mode_network = default_mode_network
        self.psychoid_adapter = psychoid_attention_adapter
        self.coherence_resonator = coherence_resonator or CoherenceResonator()
        self._last_leading_brain: Optional[str] = "right"

    def _prepare_psychoid_projection(
        self, psychoid_signal: Optional[PsychoidSignalModel], *, seq_len: int = 8
    ) -> Optional[PsychoidAttentionProjection]:
        if psychoid_signal is None:
            return None
        if self.psychoid_adapter is None:
            try:
                self.psychoid_adapter = PsychoidAttentionAdapter()
            except Exception:  # pragma: no cover - defensive guard
                return None
        try:
            return self.psychoid_adapter.build_projection(
                psychoid_signal, seq_len=seq_len
            )
        except Exception:  # pragma: no cover - defensive guard
            return None

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

    def _select_hemisphere_mode(
        self, question: str, focus: FocusSummary | None
    ) -> HemisphericSignal:
        tokens = {tok for tok in re.findall(r"[\w']+", question.lower())}
        if focus is not None and focus.keywords:
            tokens.update(tok.lower() for tok in focus.keywords)

        right_score = sum(1.0 for tok in tokens if tok in _RIGHT_HEMISPHERE_KEYWORDS)
        left_score = sum(1.0 for tok in tokens if tok in _LEFT_HEMISPHERE_KEYWORDS)

        for marker in _RIGHT_HEMISPHERE_MARKERS:
            if marker and marker in question:
                right_score += 1.25
        for marker in _LEFT_HEMISPHERE_MARKERS:
            if marker and marker in question:
                left_score += 1.25

        for pattern in _RIGHT_HEMISPHERE_PATTERNS:
            if pattern.search(question):
                right_score += 1.5
        for pattern in _LEFT_HEMISPHERE_PATTERNS:
            if pattern.search(question):
                left_score += 1.5

        total = right_score + left_score
        if total <= 0.0:
            return HemisphericSignal(
                mode="balanced",
                bias=0.0,
                right_score=right_score,
                left_score=left_score,
                token_count=len(tokens),
            )

        diff = right_score - left_score
        intensity = min(1.0, abs(diff) / max(total, 1.0))
        if diff > 0.8:
            mode = "right"
        elif diff < -0.8:
            mode = "left"
        else:
            mode = "balanced"
        return HemisphericSignal(
            mode=mode,
            bias=intensity,
            right_score=right_score,
            left_score=left_score,
            token_count=len(tokens),
        )

    def _compute_collaboration_profile(
        self, signal: HemisphericSignal, focus: FocusSummary | None
    ) -> CollaborationProfile:
        total = signal.total
        if total <= 0.0:
            return CollaborationProfile(
                strength=0.0,
                balance=0.0,
                density=0.0,
                focus_bonus=0.0,
                token_count=signal.token_count,
            )

        balance = 1.0 - abs(signal.right_score - signal.left_score) / max(total, 1.0)
        density = min(1.0, total / 6.0)
        focus_bonus = 0.0
        if focus is not None:
            focus_bonus = 0.12 * focus.relevance + 0.08 * focus.hippocampal_overlap
        strength = max(0.0, min(1.0, 0.55 * balance + 0.35 * density + focus_bonus))
        return CollaborationProfile(
            strength=strength,
            balance=balance,
            density=density,
            focus_bonus=focus_bonus,
            token_count=signal.token_count,
        )

    def _evaluate_semantic_tilt(
        self,
        *,
        question: str,
        final_answer: str,
        detail_notes: Optional[str],
    ) -> Dict[str, object]:
        tokens = _semantic_tokens(question, final_answer, detail_notes or "")
        token_set = set(tokens)
        right_hits = sorted(tok for tok in token_set if tok in _RIGHT_HEMISPHERE_KEYWORDS)
        left_hits = sorted(tok for tok in token_set if tok in _LEFT_HEMISPHERE_KEYWORDS)

        right_score = float(len(right_hits))
        left_score = float(len(left_hits))
        notes: List[str] = []

        combined_text = " ".join(chunk for chunk in [question, final_answer, detail_notes] if chunk)
        for marker in _RIGHT_HEMISPHERE_MARKERS:
            if marker and marker in combined_text:
                right_score += 0.75
                notes.append(f"right marker:{marker}")
        for marker in _LEFT_HEMISPHERE_MARKERS:
            if marker and marker in combined_text:
                left_score += 0.75
                notes.append(f"left marker:{marker}")

        for pattern in _RIGHT_HEMISPHERE_PATTERNS:
            if pattern.search(combined_text):
                right_score += 0.9
                notes.append(f"right pattern:{pattern.pattern}")
        for pattern in _LEFT_HEMISPHERE_PATTERNS:
            if pattern.search(combined_text):
                left_score += 0.9
                notes.append(f"left pattern:{pattern.pattern}")

        total = right_score + left_score
        if total <= 0.0:
            return {
                "mode": "balanced",
                "intensity": 0.0,
                "right_hits": right_hits,
                "left_hits": left_hits,
                "notes": notes,
                "scores": {"left": left_score, "right": right_score},
            }

        diff = right_score - left_score
        intensity = min(1.0, abs(diff) / max(total, 1.0))
        if diff > 0.6:
            mode = "right"
        elif diff < -0.6:
            mode = "left"
        else:
            mode = "balanced"
        return {
            "mode": mode,
            "intensity": intensity,
            "right_hits": right_hits,
            "left_hits": left_hits,
            "notes": notes,
            "scores": {"left": left_score, "right": right_score},
        }

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
        hemisphere_mode: str,
        hemisphere_bias: float,
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
            "hemisphere_mode": hemisphere_mode,
            "hemisphere_bias": hemisphere_bias,
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
        hemisphere_mode: str,
        hemisphere_bias: float,
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
            hemisphere_mode,
            hemisphere_bias,
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
        if hemisphere_mode == "right":
            temperature = min(0.95, temperature + 0.2 * (0.4 + hemisphere_bias))
        elif hemisphere_mode == "left":
            temperature = max(0.3, temperature - 0.2 * (0.4 + hemisphere_bias))
        state["hemisphere_temperature"] = temperature
        slot_ms = self.hypothalamus.recommend_slot_ms(state["risk"])
        qid = str(uuid.uuid4())
        return DecisionOutcome(qid=qid, action=action, temperature=temperature, slot_ms=slot_ms, state=state)

    async def process(self, question: str, *, leading_brain: Optional[str] = None) -> str:
        if self.coherence_resonator is not None:
            self.coherence_resonator.reset()
        requested_leading = (leading_brain or "").strip().lower()
        context, focus = self._compose_context(question)
        hemisphere_signal = self._select_hemisphere_mode(question, focus)
        hemisphere_mode = hemisphere_signal.mode
        hemisphere_bias = hemisphere_signal.bias
        collaboration_profile = self._compute_collaboration_profile(
            hemisphere_signal, focus
        )
        auto_selected_leading = False
        collaborative_lead = False
        selection_reason = "explicit_request"
        leading_style = "explicit_request"
        if requested_leading in {"left", "right"}:
            leading = requested_leading
            selection_reason = f"explicit_request_{leading}"
            leading_style = "explicit_request"
        else:
            collaborative_hint = (
                hemisphere_mode == "balanced"
                and hemisphere_bias < 0.45
                and collaboration_profile.strength >= 0.4
            )
            if hemisphere_mode == "right" and hemisphere_bias >= 0.35:
                leading = "right"
                selection_reason = "hemisphere_bias_right"
                leading_style = "hemisphere_bias"
            elif hemisphere_mode == "left" and hemisphere_bias >= 0.35:
                leading = "left"
                selection_reason = "hemisphere_bias_left"
                leading_style = "hemisphere_bias"
            else:
                if self._last_leading_brain == "right":
                    leading = "left"
                elif self._last_leading_brain == "left":
                    leading = "right"
                else:
                    leading = "right"
                selection_reason = "cooperative_rotation"
                leading_style = "rotation"
                if collaborative_hint:
                    collaborative_lead = True
                    selection_reason = "cooperative_rotation_balanced"
                    leading_style = (
                        "collaborative_braid_strong"
                        if collaboration_profile.strength >= 0.7
                        else "collaborative_braid"
                    )
            auto_selected_leading = True
        self._last_leading_brain = leading
        if self.coherence_resonator is not None:
            self.coherence_resonator.retune(hemisphere_mode, intensity=hemisphere_bias)
        focus_metric = 0.0
        if self.prefrontal_cortex is not None and focus is not None:
            focus_metric = self.prefrontal_cortex.focus_metric(focus)
        hippocampal_density = (
            len(self.hippocampus.episodes) if self.hippocampus is not None else 0
        )
        focus_keywords = list(focus.keywords) if focus and focus.keywords else None
        right_lead_notes: Optional[str] = None
        if leading == "right" or collaborative_lead:
            try:
                right_lead_notes = await self.right_model.generate_lead(
                    question,
                    context,
                )
            except Exception:  # pragma: no cover - defensive guard
                right_lead_notes = None
        left_lead_preview: Optional[str] = None
        draft = await self.left_model.generate_answer(question, context)
        if collaborative_lead:
            left_lead_preview = draft.split("\n\n", 1)[0][:320]
        left_coherence: Optional[HemisphericCoherence] = None
        right_coherence: Optional[HemisphericCoherence] = None
        coherence_signal: Optional[CoherenceSignal] = None
        if self.coherence_resonator is not None:
            left_coherence = self.coherence_resonator.capture_left(
                question=question,
                draft=draft,
                context=context,
                focus_keywords=focus_keywords or (),
                focus_metric=focus_metric,
            )
        confidence = self.left_model.estimate_confidence(draft)
        affect = self._sense_affect(question, draft)
        novelty = self.memory.novelty_score(question)
        consult_bias = self.hypothalamus.bias_for_consult(novelty)
        if self.prefrontal_cortex is not None and focus is not None:
            consult_bias = self.prefrontal_cortex.adjust_consult_bias(consult_bias, focus)
        if hemisphere_mode == "right":
            consult_bias = max(
                -0.35, consult_bias - (0.15 + 0.35 * hemisphere_bias)
            )
        elif hemisphere_mode == "left":
            consult_bias = min(
                0.35, consult_bias + (0.15 + 0.35 * hemisphere_bias)
            )
        force_right_lead = leading == "right"
        decision = self.decide(
            question,
            draft,
            confidence,
            affect,
            novelty,
            consult_bias,
            hippocampal_density,
            focus_metric,
            hemisphere_mode,
            hemisphere_bias,
        )
        if force_right_lead and decision.action == 0:
            decision.action = 1
            decision.state["right_forced_lead"] = True
        decision.state["hemisphere_mode"] = hemisphere_mode
        decision.state["hemisphere_bias_strength"] = hemisphere_bias
        decision.state["hemisphere_signal"] = hemisphere_signal.to_payload()
        decision.state["leading_brain"] = leading
        if auto_selected_leading:
            decision.state["leading_autoselected"] = True
        decision.state["leading_selection_reason"] = selection_reason
        decision.state["leading_style"] = leading_style
        if collaborative_lead:
            decision.state["leading_collaborative"] = True
        decision.state["collaboration_profile"] = collaboration_profile.to_payload()
        if left_lead_preview:
            decision.state["left_lead_preview"] = left_lead_preview
        if right_lead_notes:
            decision.state["right_lead_preview"] = right_lead_notes
        if left_coherence is not None:
            decision.state["coherence_left"] = left_coherence.to_payload()
        if self.coherence_resonator is not None:
            decision.state["hemisphere_weights"] = {
                "left": self.coherence_resonator.left_weight,
                "right": self.coherence_resonator.right_weight,
            }
        basal_signal = getattr(self.basal_ganglia, "last_signal", None)
        if focus is not None:
            decision.state["prefrontal_keywords"] = list(focus.keywords)
            decision.state["prefrontal_relevance"] = focus.relevance
            decision.state["prefrontal_hippocampal_overlap"] = focus.hippocampal_overlap
        unconscious_profile = None
        unconscious_summary: Optional[UnconsciousSummaryModel] = None
        default_mode_reflections: Optional[List[DefaultModeReflection]] = None
        psychoid_signal: Optional[PsychoidSignalModel] = None
        psychoid_projection: Optional[PsychoidAttentionProjection] = None
        if self.unconscious_field is not None:
            try:
                unconscious_profile = self.unconscious_field.analyse(
                    question=question, draft=draft
                )
            except Exception:  # pragma: no cover - defensive guard
                unconscious_profile = None
            else:
                summary = self.unconscious_field.summary(unconscious_profile)
                decision.state["unconscious_top"] = (
                    summary.top_k[0] if summary.top_k else None
                )
                decision.state["unconscious_cache_depth"] = summary.cache_depth
                if summary.emergent_ideas:
                    decision.state["unconscious_emergent"] = [
                        idea.dict() for idea in summary.emergent_ideas
                    ]
                if summary.stress_released:
                    decision.state["unconscious_stress_release"] = summary.stress_released
                psychoid_signal = summary.psychoid_signal
                if psychoid_signal:
                    decision.state["psychoid_bias"] = [
                        entry.dict() for entry in psychoid_signal.attention_bias
                    ]
                    decision.state["psychoid_tension"] = psychoid_signal.psychoid_tension
                    decision.state["psychoid_resonance"] = psychoid_signal.resonance
                    psychoid_projection = self._prepare_psychoid_projection(
                        psychoid_signal
                    )
                    if psychoid_projection:
                        decision.state["psychoid_attention_norm"] = psychoid_projection.norm
                self.telemetry.log(
                    "unconscious_field",
                    qid=decision.qid,
                    summary=summary.to_payload(),
                )
                if psychoid_signal:
                    self.telemetry.log(
                        "psychoid_signal",
                        qid=decision.qid,
                        signal=psychoid_signal.to_payload(),
                    )
                    if psychoid_projection:
                        self.telemetry.log(
                            "psychoid_attention_projection",
                            qid=decision.qid,
                            projection=psychoid_projection.to_payload(),
                        )
                unconscious_summary = summary
        if (
            self.default_mode_network is not None
            and unconscious_summary is not None
        ):
            try:
                reflections = self.default_mode_network.reflect(unconscious_summary)
            except Exception:  # pragma: no cover - defensive guard
                default_mode_reflections = None
            else:
                if reflections:
                    default_mode_reflections = reflections
                    decision.state["default_mode_reflections"] = [
                        ref.as_dict() for ref in reflections
                    ]
                    self.telemetry.log(
                        "default_mode_reflection",
                        qid=decision.qid,
                        reflections=[ref.as_dict() for ref in reflections],
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
            "hemisphere_routing",
            qid=decision.qid,
            mode=hemisphere_mode,
            intensity=hemisphere_bias,
        )
        self.telemetry.log(
            "hemisphere_signal",
            qid=decision.qid,
            signal=hemisphere_signal.to_payload(),
        )
        self.telemetry.log(
            "collaboration_signal",
            qid=decision.qid,
            profile=collaboration_profile.to_payload(),
            collaborative=collaborative_lead,
        )
        self.telemetry.log(
            "policy_decision",
            qid=decision.qid,
            state=decision.state,
            action=decision.action,
            temperature=decision.temperature,
            slot_ms=decision.slot_ms,
        )
        self.telemetry.log(
            "leading_brain",
            qid=decision.qid,
            leading=leading,
            preview=bool(right_lead_notes),
            forced=bool(decision.state.get("right_forced_lead", False)),
            auto=auto_selected_leading,
            reason=selection_reason,
            collaborative=collaborative_lead,
            style=leading_style,
            collaboration_strength=collaboration_profile.strength,
        )

        if not self.orchestrator.register_request(decision.qid, leader=leading):
            return "Loop-killed"

        final_answer = draft
        detail_notes: Optional[str] = None
        response_source = "lead" if right_lead_notes else ""
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
                    "hemisphere_mode": hemisphere_mode,
                    "hemisphere_bias": hemisphere_bias,
                }
                if focus is not None and focus.keywords:
                    payload["focus_keywords"] = list(focus.keywords[:5])
                if unconscious_summary is not None:
                    ideas = unconscious_summary.emergent_ideas
                    if ideas:
                        payload["unconscious_hints"] = [
                            f"{idea.label} ({idea.archetype})" for idea in ideas
                        ]
                        payload["unconscious_cache_depth"] = unconscious_summary.cache_depth
                        payload["unconscious_stress_released"] = unconscious_summary.stress_released
                if psychoid_signal:
                    chain = psychoid_signal.signifier_chain
                    if chain:
                        payload["psychoid_signifiers"] = list(chain[-6:])
                    bias_vector = psychoid_signal.bias_vector
                    if bias_vector:
                        payload["psychoid_bias_vector"] = [float(x) for x in bias_vector[:12]]
                    if psychoid_projection:
                        payload["psychoid_attention_bias"] = psychoid_projection.to_payload()
                if self.coherence_resonator is not None and left_coherence is not None:
                    vectorised = self.coherence_resonator.vectorise_left()
                    if vectorised:
                        payload["coherence_vector"] = vectorised
                if default_mode_reflections:
                    payload["default_mode_reflections"] = [
                        f"{ref.theme} (confidence {float(ref.confidence):.2f})"
                        for ref in default_mode_reflections
                    ]
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
                            psychoid_projection=(
                                psychoid_projection.to_payload()
                                if psychoid_projection
                                else None
                            ),
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
                elif right_lead_notes:
                    final_answer = self.left_model.integrate_info(draft, right_lead_notes)
                if decision.state.get("right_conf"):
                    decision.state["right_source"] = response_source
                if self.coherence_resonator is not None:
                    combined_detail = detail_notes or right_lead_notes
                    detail_origin = response_source if detail_notes else "lead"
                    if combined_detail:
                        right_coherence = self.coherence_resonator.capture_right(
                            question=question,
                            draft=draft,
                            detail_notes=combined_detail,
                            focus_keywords=focus_keywords or (),
                            psychoid_signal=psychoid_signal,
                            confidence=float(decision.state.get("right_conf", 0.0) or 0.0),
                            source=detail_origin,
                        )
                        decision.state["coherence_right"] = right_coherence.to_payload()
        except asyncio.TimeoutError:
            final_answer = draft + "\n(Right brain timeout: continuing with draft)"
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.orchestrator.clear(decision.qid)

        integrated_answer = final_answer
        if collaborative_lead:
            right_block = right_lead_notes or detail_notes or "(Right brain prelude unavailable)"
            left_block = left_lead_preview or draft
            final_answer = (
                "[Right Brain Prelude]\n"
                f"{right_block}\n\n"
                "[Left Brain Prelude]\n"
                f"{left_block}\n\n"
                "[Integrated Response]\n"
                f"{integrated_answer}"
            )
        elif leading == "right":
            lead_block = right_lead_notes or detail_notes or "(Right brain lead unavailable)"
            final_answer = (
                "[Right Brain Lead]\n"
                f"{lead_block}\n\n"
                "[Left Brain Integration]\n"
                f"{integrated_answer}"
            )
        else:
            final_answer = f"[Left Brain Lead]\n{integrated_answer}"

        semantic_tilt = self._evaluate_semantic_tilt(
            question=question,
            final_answer=final_answer,
            detail_notes=detail_notes,
        )
        decision.state["hemisphere_semantic_tilt"] = semantic_tilt
        self.telemetry.log(
            "hemisphere_semantic_tilt",
            qid=decision.qid,
            mode=semantic_tilt["mode"],
            intensity=semantic_tilt["intensity"],
            scores=semantic_tilt["scores"],
            right_hits=semantic_tilt["right_hits"],
            left_hits=semantic_tilt["left_hits"],
            initial_mode=hemisphere_mode,
        )
        tags = set()
        tags.add(f"leading_{leading}")
        if collaborative_lead:
            tags.add("leading_collaborative")
        strength = collaboration_profile.strength
        if strength >= 0.7:
            tags.add("collaboration_strong")
        elif strength >= 0.45:
            tags.add("collaboration_present")
        if collaborative_lead:
            tags.add("collaboration_braided")
        tags.add(
            f"collaboration_strength_{int(strength * 100):02d}"
        )
        tags.add(
            f"collaboration_balance_{int(collaboration_profile.balance * 100):02d}"
        )
        if leading == "right" and decision.action == 0:
            tags.add("right_lead_solo")
        if self.coherence_resonator is not None:
            self.coherence_resonator.retune(
                semantic_tilt["mode"],
                intensity=float(semantic_tilt["intensity"]),
            )
            projection_payload = (
                psychoid_projection.to_payload() if psychoid_projection else None
            )
            weave = self.coherence_resonator.capture_unconscious(
                question=question,
                draft=draft,
                final_answer=final_answer,
                summary=unconscious_summary,
                psychoid_signal=psychoid_signal,
            )
            if weave is not None:
                fabric_payload = weave.to_payload()
                decision.state["linguistic_fabric"] = fabric_payload
                self.telemetry.log(
                    "coherence_unconscious_weave",
                    qid=decision.qid,
                    fabric=fabric_payload,
                )
            motifs = self.coherence_resonator.capture_linguistic_motifs(
                question=question,
                draft=draft,
                final_answer=final_answer,
                unconscious_summary=unconscious_summary,
            )
            if motifs is not None:
                motifs_payload = motifs.to_payload()
                decision.state["linguistic_motifs"] = motifs_payload
                self.telemetry.log(
                    "coherence_linguistic_motifs",
                    qid=decision.qid,
                    motifs=motifs_payload,
                )
            coherence_signal = self.coherence_resonator.integrate(
                final_answer=final_answer,
                psychoid_projection=projection_payload,
                unconscious_summary=unconscious_summary,
            )
            if coherence_signal is not None:
                decision.state["coherence_combined"] = coherence_signal.combined_score
                decision.state["coherence_tension"] = coherence_signal.tension
                decision.state["coherence_notes"] = coherence_signal.notes
                decision.state["coherence_contributions"] = coherence_signal.contributions
                self.telemetry.log(
                    "coherence_signal",
                    qid=decision.qid,
                    signal=coherence_signal.to_payload(),
                )
                final_answer = self.coherence_resonator.annotate_answer(
                    final_answer, coherence_signal
                )
                coherence_tags = set(coherence_signal.tags())
                tags.update(coherence_tags)
                decision.state["coherence_tags"] = list(coherence_tags)

        audit_result = self.auditor.check(final_answer)
        if not audit_result.get("ok", True):
            final_answer = draft + f"\n(Auditor veto: {audit_result.get('reason', 'unknown')})"
            success = False

        reward = 0.75 if success else 0.45
        if not audit_result.get("ok", True):
            reward = min(reward, 0.25)
        self.hypothalamus.update_feedback(reward=reward, latency_ms=latency_ms)

        tags.add(decision.state.get("q_type", "unknown"))
        tags.add(f"hemisphere_{hemisphere_mode}")
        tags.add(f"hemisphere_bias_{int(hemisphere_bias * 100):02d}")
        tilt_mode = semantic_tilt["mode"]
        tags.add(f"hemisphere_tilt_{tilt_mode}")
        tags.add(
            f"hemisphere_tilt_bias_{int(float(semantic_tilt['intensity']) * 100):02d}"
        )
        if tilt_mode != hemisphere_mode:
            tags.add("hemisphere_shifted")
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
        if unconscious_summary is not None:
            if unconscious_summary.emergent_ideas:
                tags.add("unconscious_emergent")
                for idea in unconscious_summary.emergent_ideas:
                    archetype = idea.archetype
                    if archetype:
                        tags.add(f"emergent_{archetype}")
            if unconscious_summary.stress_released:
                tags.add("unconscious_stress_release")
            insights = []
            for idea in unconscious_summary.emergent_ideas:
                label = idea.label or idea.archetype or "Insight"
                archetype = idea.archetype or "unknown"
                intensity = float(idea.intensity)
                insights.append(
                    f"- {label} (archetype {archetype}, intensity {intensity:.2f})"
                )
            if insights:
                final_answer = f"{final_answer}\n\n[Unconscious Insight]\n" + "\n".join(insights)
            stress_value = float(unconscious_summary.stress_released or 0.0)
            if stress_value:
                final_answer = f"{final_answer}\n\n[Stress Released] {stress_value:.2f}"
        if psychoid_signal:
            tags.add("psychoid_projection")
            bias_entries = list(psychoid_signal.attention_bias)
            if bias_entries:
                top_bias = bias_entries[0]
                if top_bias.archetype:
                    tags.add(f"psychoid_{top_bias.archetype}")
            projection_lines = []
            for entry in bias_entries:
                projection_lines.append(
                    "- {label} ({archetype}) weight {weight:.2f} resonance {resonance:.2f}".format(
                        label=entry.label,
                        archetype=entry.archetype,
                        weight=float(entry.weight),
                        resonance=float(entry.resonance),
                    )
                )
            if projection_lines:
                final_answer = (
                    f"{final_answer}\n\n[Psychoid Field Alignment]\n" + "\n".join(projection_lines)
                )
            chain = psychoid_signal.signifier_chain
            if chain:
                final_answer = (
                    f"{final_answer}\n\n[Psychoid Signifiers]\n" + " -> ".join(chain[-6:])
                )
            if psychoid_projection:
                tags.add("psychoid_attention")
                bias_payload = psychoid_projection.to_payload()
                final_answer = (
                    f"{final_answer}\n\n[Psychoid Attention Bias]\n"
                    f"- norm {bias_payload.get('norm', 0.0):.3f}"
                    f" | temperature {bias_payload.get('temperature', 0.0):.2f}"
                    f" | clamp {bias_payload.get('clamp', 0.0):.2f}"
                )
        if default_mode_reflections:
            tags.add("default_mode_reflection")
            reflection_lines = []
            for ref in default_mode_reflections:
                primary = ref.primary_archetype or "unknown"
                tags.add(f"default_mode_{primary}")
                reflection_lines.append(
                    "- {theme} (confidence {confidence:.2f}, stress {stress:.2f}, cache {cache})".format(
                        theme=ref.theme,
                        confidence=float(ref.confidence),
                        stress=float(ref.stress_released),
                        cache=int(ref.cache_depth),
                    )
                )
            if reflection_lines:
                final_answer = (
                    f"{final_answer}\n\n[Default Mode Reflection]\n" + "\n".join(reflection_lines)
                )
        routing_lines = [
            f"- mode: {hemisphere_mode} (intensity {hemisphere_bias:.2f})",
            f"- policy action: {decision.action}",
            f"- temperature: {decision.temperature:.2f}",
        ]
        if self.coherence_resonator is not None:
            routing_lines.append(
                "- coherence weights left {left:.2f} | right {right:.2f}".format(
                    left=self.coherence_resonator.left_weight,
                    right=self.coherence_resonator.right_weight,
                )
            )
        tilt_lines = [
            "- semantic mode: {mode} (intensity {intensity:.2f})".format(
                mode=semantic_tilt["mode"],
                intensity=float(semantic_tilt["intensity"]),
            ),
            "- right hits: {hits}".format(
                hits=", ".join(semantic_tilt["right_hits"]) or "none",
            ),
            "- left hits: {hits}".format(
                hits=", ".join(semantic_tilt["left_hits"]) or "none",
            ),
        ]
        if semantic_tilt["notes"]:
            for note in semantic_tilt["notes"]:
                tilt_lines.append(f"- note: {note}")
        collab_lines = [
            "- strength: {strength:.2f}".format(
                strength=collaboration_profile.strength,
            ),
            "- balance: {balance:.2f}".format(
                balance=collaboration_profile.balance,
            ),
            "- density: {density:.2f}".format(
                density=collaboration_profile.density,
            ),
            "- focus bonus: {bonus:.2f}".format(
                bonus=collaboration_profile.focus_bonus,
            ),
            "- cue scores left {left:.2f} | right {right:.2f}".format(
                left=hemisphere_signal.left_score,
                right=hemisphere_signal.right_score,
            ),
            "- cue tokens: {count}".format(count=hemisphere_signal.token_count),
            f"- collaborative rotation engaged: {collaborative_lead}",
        ]
        meta_sections = [
            _format_section("Hemisphere Routing", routing_lines),
            _format_section("Hemisphere Semantic Tilt", tilt_lines),
            _format_section("Collaboration Profile", collab_lines),
        ]
        final_answer = f"{final_answer}\n\n" + "\n\n".join(meta_sections)
        if self.coherence_resonator is not None:
            projection_payload = (
                psychoid_projection.to_payload() if psychoid_projection else None
            )
            coherence_signal = self.coherence_resonator.integrate(
                final_answer=final_answer,
                psychoid_projection=projection_payload,
            )
            if coherence_signal is not None:
                decision.state["coherence_combined"] = coherence_signal.combined_score
                decision.state["coherence_tension"] = coherence_signal.tension
                decision.state["coherence_notes"] = coherence_signal.notes
                decision.state["coherence_contributions"] = coherence_signal.contributions
                self.telemetry.log(
                    "coherence_signal",
                    qid=decision.qid,
                    signal=coherence_signal.to_payload(),
                )
                final_answer = self.coherence_resonator.annotate_answer(
                    final_answer, coherence_signal
                )
                coherence_tags = set(coherence_signal.tags())
                tags.update(coherence_tags)
                decision.state["coherence_tags"] = list(coherence_tags)
        if focus is not None and self.prefrontal_cortex is not None:
            tags.update(self.prefrontal_cortex.tags(focus))
        if basal_signal is not None:
            tags.update(self.basal_ganglia.tags(basal_signal))

        follow_brain: Optional[str] = None
        if collaborative_lead:
            follow_brain = "braided"
        elif leading == "right":
            follow_brain = "left"
        elif detail_notes or decision.action != 0:
            follow_brain = "right"
        self.memory.record_dialogue_flow(
            decision.qid,
            leading_brain=leading,
            follow_brain=follow_brain,
            preview=right_lead_notes or detail_notes,
        )
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
        if self.unconscious_field is not None:
            outcome_meta = self.unconscious_field.integrate_outcome(
                mapping=unconscious_profile,
                question=question,
                draft=draft,
                final_answer=final_answer,
                success=success,
                decision_state=decision.state,
                affect=affect,
                novelty=novelty,
                reward=reward,
            )
            self.telemetry.log(
                "unconscious_outcome",
                qid=decision.qid,
                outcome=outcome_meta,
            )
        return final_answer


class _NullTelemetry:
    """Default telemetry sink used when no observer is provided."""

    def log(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass
