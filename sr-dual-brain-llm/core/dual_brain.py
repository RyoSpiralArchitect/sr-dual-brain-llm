"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia, BasalGangliaSignal
from .prefrontal_cortex import PrefrontalCortex, FocusSummary, SchemaProfile
from .default_mode_network import DefaultModeNetwork, DefaultModeReflection
from .director_reasoner import DirectorAdvice, DirectorReasonerModel
from .executive_reasoner import ExecutiveAdvice, ExecutiveReasonerModel
from .temporal_hippocampal_indexing import TemporalHippocampalIndexing
from .psychoid_attention import PsychoidAttentionAdapter, PsychoidAttentionProjection
from .coherence_resonator import (
    CoherenceResonator,
    CoherenceSignal,
    HemisphericCoherence,
)
from .schema import PsychoidSignalModel, UnconsciousSummaryModel
from .neural_impulse import (
    NeuralIntegrator,
    Neuron,
    NeurotransmitterType,
)
from .anterior_cingulate import AnteriorCingulateCortex, ConflictSignal
from .cerebellum import Cerebellum
from .insula import Insula, InteroceptiveState
from .predictive_coding import PredictiveCodingController, PredictionFrame
from .salience_network import SalienceNetwork, SalienceSignal
from .thalamus import ThalamicRelay, Thalamus
from .micro_critic import micro_criticise_reasoning
from .dual_brain_support import (
    _LEFT_HEMISPHERE_KEYWORDS,
    _LEFT_HEMISPHERE_MARKERS,
    _LEFT_HEMISPHERE_PATTERNS,
    _RIGHT_HEMISPHERE_KEYWORDS,
    _RIGHT_HEMISPHERE_MARKERS,
    _RIGHT_HEMISPHERE_PATTERNS,
    _detail_notes_redundant,
    _detect_system2_critic_unhealthy_reason,
    _env_flag,
    _env_float,
    _env_int,
    _filter_system2_issues,
    _format_section,
    _format_system2_revision_notes,
    _issue_overlap_with_previous,
    _issue_signal_score,
    _looks_like_coaching_notes,
    _normalise_issue_text,
    _normalise_issue_list,
    _novel_issue_items,
    _prioritise_issue_list,
    _resolve_system2_priority,
    _sanitize_user_answer,
    _semantic_tokens,
    _summarise_architecture_path,
    _trim_system2_draft,
    _truncate_text,
    CollaborationProfile,
    DecisionOutcome,
    HemisphericSignal,
    InnerDialogueStep,
)


@dataclass
class System2Settings:
    mode: str
    priority: str
    precision_priority: bool
    low_signal_filter: bool
    followup_new_issue_cap: int
    followup_new_issue_min_score: float
    followup_min_overlap: float
    followup_max_remaining: int
    followup_min_progress: int


@dataclass
class System2Status:
    capable: bool
    active: bool
    reason: str


@dataclass
class LeadingSelection:
    leading: str
    auto_selected: bool
    collaborative: bool
    selection_reason: str
    leading_style: str


@dataclass
class PerceptionStageResult:
    context: str
    focus: FocusSummary | None
    context_parts: Dict[str, str]
    is_trivial_chat: bool
    focus_metric: float
    context_signal_len: int
    question_affect: Dict[str, float]
    novelty: float
    q_type_hint: str
    insula_state: InteroceptiveState | None
    salience_signal: SalienceSignal | None
    thalamic_relay: ThalamicRelay | None
    hemisphere_signal: HemisphericSignal
    hemisphere_mode: str
    hemisphere_bias: float
    collaboration_profile: CollaborationProfile
    predictive_frame: PredictionFrame | None


@dataclass
class ContextFrame:
    context: str
    focus: FocusSummary | None
    focus_metric: float
    focus_keywords: List[str] | None


@dataclass
class ObserverSetup:
    executive_mode: str
    observer_mode: str
    use_director: bool
    use_metrics_observer: bool
    executive_task: asyncio.Task[ExecutiveAdvice] | None
    director_task: asyncio.Task[DirectorAdvice] | None


@dataclass
class DirectorStageResult:
    context_frame: ContextFrame
    director_task: asyncio.Task[DirectorAdvice] | None
    director_payload: Dict[str, Any] | None
    director_control: Dict[str, Any] | None


@dataclass
class DraftStageResult:
    draft: str
    right_lead_notes: str | None
    left_lead_preview: str | None
    left_coherence: HemisphericCoherence | None
    confidence: float
    affect: Dict[str, float]
    novelty: float
    consult_bias: float
    steps: List[InnerDialogueStep]


@dataclass
class RoutingStageResult:
    context_frame: ContextFrame
    decision: DecisionOutcome
    acc_signal: ConflictSignal | None
    basal_signal: BasalGangliaSignal | None
    system2_active: bool
    system2_reason: str
    system2_round_target: int
    director_task: asyncio.Task[DirectorAdvice] | None
    director_payload: Dict[str, Any] | None
    director_control: Dict[str, Any] | None
    director_max_chars: int | None
    director_append_question: str | None


@dataclass
class ConsultRequestPlan:
    payload: Dict[str, Any]
    request_meta: Dict[str, Any]
    timeout_ms: int
    timeout_ms_base: int
    timeout_multiplier: float
    timeout_max_ms: int
    system2_draft_limit: int
    q_type_hint: str


@dataclass
class RightConsultStageResult:
    detail_notes: Optional[str]
    response_source: str
    success: bool
    final_answer: str
    call_error: bool
    call_confidence: float
    fallback_used: bool
    fallback_confidence: float
    fallback_error: Optional[str]
    critic_verdict: Optional[str]
    critic_issues: List[str]
    critic_fixes: List[str]
    system2_initial_issue_count: int


@dataclass
class System2RefinementResult:
    final_answer: str
    system2_round_target: int
    system2_rounds_completed: int
    system2_final_issue_count: int
    system2_resolved: bool
    system2_followup_revision: bool
    system2_followup_new_issues: List[str]
    system2_followup_verdict: Optional[str]
    critic_needs_revision: bool
    critic_unhealthy: bool
    critic_unhealthy_reason: Optional[str]


@dataclass
class ExecutiveAdviceStageResult:
    executive_payload: Dict[str, Any] | None
    executive_directives: Dict[str, Any] | None


@dataclass
class ConsultSkippedStageResult:
    final_answer: str
    response_source: str
    success: bool
    executive_payload: Dict[str, Any] | None
    executive_directives: Dict[str, Any] | None
    steps: List[InnerDialogueStep]


@dataclass
class FinalAnswerStageResult:
    integrated_answer: str
    user_answer: str
    final_answer: str


@dataclass
class AuditStageResult:
    user_answer: str
    final_answer: str
    audit_result: Dict[str, Any]
    tags: set[str]
    success: bool


@dataclass
class MemoryPersistenceStageResult:
    memory_question: str
    memory_answer: str
    episodic_total: int
    hippocampal_rollup: Dict[str, float] | None
    hippocampal_lifecycle: Dict[str, float] | None
    hippocampal_forgetting: Dict[str, float] | None
    architecture_path: List[Dict[str, Any]]


@dataclass
class PostTurnObserverStageResult:
    executive_observer_payload: Dict[str, Any] | None


class DualBrainController:
    """Coordinate left/right brain collaboration with adaptive heuristics."""

    def __init__(
        self,
        *,
        callosum,
        memory,
        left_model,
        right_model,
        executive_model: ExecutiveReasonerModel | None = None,
        director_model: DirectorReasonerModel | None = None,
        policy,
        hypothalamus,
        reasoning_dial,
        auditor,
        orchestrator,
        default_timeout_ms: int = 45000,
        telemetry: Optional[Any] = None,
        amygdala: Optional[Amygdala] = None,
        hippocampus: Optional[TemporalHippocampalIndexing] = None,
        unconscious_field: Optional[Any] = None,
        prefrontal_cortex: Optional[PrefrontalCortex] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        default_mode_network: Optional[DefaultModeNetwork] = None,
        psychoid_attention_adapter: Optional[PsychoidAttentionAdapter] = None,
        coherence_resonator: Optional[CoherenceResonator] = None,
        neural_integrator: Optional[NeuralIntegrator] = None,
        anterior_cingulate_cortex: Optional[AnteriorCingulateCortex] = None,
        cerebellum: Optional[Cerebellum] = None,
        insula: Optional[Insula] = None,
        salience_network: Optional[SalienceNetwork] = None,
        thalamus: Optional[Thalamus] = None,
        predictive_coding: Optional[PredictiveCodingController] = None,
    ) -> None:
        self.callosum = callosum
        self.memory = memory
        self.left_model = left_model
        self.right_model = right_model
        self.executive_model = executive_model
        self.director_model = director_model
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
        self.anterior_cingulate_cortex = anterior_cingulate_cortex or AnteriorCingulateCortex()
        self.cerebellum = cerebellum or Cerebellum()
        self.insula = insula or Insula()
        self.salience_network = salience_network or SalienceNetwork()
        self.thalamus = thalamus or Thalamus()
        self.predictive_coding = predictive_coding or PredictiveCodingController()
        self._last_leading_brain: Optional[str] = "right"
        self._default_mode_low_focus_streak = 0
        
        # Neural impulse simulation
        self.neural_integrator = neural_integrator
        if self.neural_integrator is None:
            self.neural_integrator = self._create_default_neural_network()
    
    def _create_default_neural_network(self) -> NeuralIntegrator:
        """Create a default neural network for impulse simulation.
        
        This creates a simplified dual-hemisphere network with:
        - Left hemisphere pathway (analytical processing)
        - Right hemisphere pathway (holistic processing)
        - Inter-hemispheric connections (corpus callosum analog)
        - Modulatory pathways for affect and attention
        """
        integrator = NeuralIntegrator()
        
        # Create left hemisphere pathway (glutamatergic, excitatory)
        left_pathway = integrator.create_simple_pathway(
            "left_hemisphere",
            num_neurons=4,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
        )
        
        # Create right hemisphere pathway (also excitatory but separate)
        right_pathway = integrator.create_simple_pathway(
            "right_hemisphere",
            num_neurons=4,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
        )
        
        # Create modulatory neurons (dopaminergic for reward/motivation)
        basal_ganglia_neuron = Neuron(
            "basal_ganglia_modulator",
            base_neurotransmitter=NeurotransmitterType.DOPAMINE,
        )
        integrator.add_neuron(basal_ganglia_neuron)
        
        # Amygdala modulation (GABAergic for inhibition when risk detected)
        amygdala_neuron = Neuron(
            "amygdala_modulator",
            base_neurotransmitter=NeurotransmitterType.GABA,
        )
        integrator.add_neuron(amygdala_neuron)
        
        # Prefrontal attention (acetylcholine for attention/focus)
        prefrontal_neuron = Neuron(
            "prefrontal_modulator",
            base_neurotransmitter=NeurotransmitterType.ACETYLCHOLINE,
        )
        integrator.add_neuron(prefrontal_neuron)
        
        return integrator

    def _simulate_neural_activity(
        self,
        *,
        hemisphere: str,
        affect: Dict[str, float],
        novelty: float,
        focus_metric: float,
        num_steps: int = 10,
    ) -> Dict[str, Any]:
        """Simulate neural impulse activity for a processing phase.
        
        This models the biological neural activity that occurs during
        cognitive processing, with realistic action potentials and
        synaptic transmission.
        
        Args:
            hemisphere: "left" or "right" indicating which hemisphere is active
            affect: Affective state (valence, arousal, risk)
            novelty: Novelty score of the input
            focus_metric: Prefrontal focus strength
            num_steps: Number of simulation time steps
            
        Returns:
            Dictionary with neural activity metrics and impulse data
        """
        if self.neural_integrator is None:
            return {}
        
        # Determine which pathway to stimulate based on hemisphere
        pathway_id = f"{hemisphere}_hemisphere_neuron_0"
        
        # Stimulus strength based on affect and novelty
        arousal = float(affect.get("arousal", 0.0))
        risk = float(affect.get("risk", 0.0))
        
        # Higher arousal and novelty increase neural activity
        base_stimulus = 0.6 + 0.3 * arousal + 0.2 * novelty
        
        # Inject stimulus into the hemisphere pathway
        self.neural_integrator.inject_stimulus(
            pathway_id,
            strength=base_stimulus,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
        )
        
        # Modulate with basal ganglia (dopamine based on novelty/reward)
        if novelty > 0.5:
            self.neural_integrator.inject_stimulus(
                "basal_ganglia_modulator",
                strength=0.4 + 0.3 * novelty,
                neurotransmitter=NeurotransmitterType.DOPAMINE,
            )
        
        # Amygdala modulation (inhibition when high risk)
        if risk > 0.5:
            self.neural_integrator.inject_stimulus(
                "amygdala_modulator",
                strength=0.5 + 0.4 * risk,
                neurotransmitter=NeurotransmitterType.GABA,
            )
        
        # Prefrontal attention modulation
        if focus_metric > 0.3:
            self.neural_integrator.inject_stimulus(
                "prefrontal_modulator",
                strength=0.3 + 0.4 * focus_metric,
                neurotransmitter=NeurotransmitterType.ACETYLCHOLINE,
            )
        
        # Run simulation for specified steps
        all_impulses = []
        for _ in range(num_steps):
            new_impulses = self.neural_integrator.step(dt=0.001)
            all_impulses.extend(new_impulses)
        
        # Collect activity metrics
        network_activity = self.neural_integrator.get_network_activity()
        
        # Count impulses by neurotransmitter type
        impulse_counts = {
            "glutamate": 0,
            "gaba": 0,
            "dopamine": 0,
            "acetylcholine": 0,
            "serotonin": 0,
        }
        for impulse in all_impulses:
            nt_type = impulse.neurotransmitter.value
            if nt_type in impulse_counts:
                impulse_counts[nt_type] += 1
        
        return {
            "hemisphere": hemisphere,
            "total_impulses": len(all_impulses),
            "impulse_counts": impulse_counts,
            "network_activity": network_activity,
            "simulation_steps": num_steps,
            "stimulus_strength": base_stimulus,
        }

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

    def _compose_context(self, question: str) -> tuple[str, FocusSummary | None, dict[str, str]]:
        """Blend working-memory recall with hippocampal episodic cues."""

        cortex = self.prefrontal_cortex
        is_trivial = bool(cortex is not None and cortex.is_trivial_chat_turn(question))
        include_working_memory = bool(
            cortex is not None
            and cortex.should_include_working_memory(question)
        )
        include_long_term = bool(
            not is_trivial
            and (
                cortex is None
                or cortex.should_include_long_term_memory(question)
            )
        )

        working_memory_context = ""
        if include_working_memory and cortex is not None:
            working_memory_context = cortex.working_memory_context(turns=2)

        memory_context = self.memory.retrieve_related(question) if include_long_term else ""
        schema_context = ""
        if include_long_term:
            try:
                schema_context = self.memory.retrieve_schema_related(question)
            except Exception:  # pragma: no cover - optional feature
                schema_context = ""
        pitfall_context = ""
        if include_long_term:
            try:
                counts = self.memory.get_kv("system2_pitfall_counts")  # type: ignore[attr-defined]
                examples = self.memory.get_kv("system2_pitfall_examples")  # type: ignore[attr-defined]
                if isinstance(counts, dict) and isinstance(examples, dict) and counts:
                    scored = []
                    for key, value in counts.items():
                        try:
                            count = int(value)
                        except Exception:
                            continue
                        if count <= 0:
                            continue
                        scored.append((count, str(key)))
                    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
                    lines: List[str] = []
                    for count, key in scored[:4]:
                        example = str(examples.get(key) or "").strip()
                        if not example:
                            continue
                        example = _truncate_text(example, limit=140)
                        lines.append(f"- (x{count}) {example}")
                    if lines:
                        pitfall_context = _format_section("System2 pitfall memory", lines)
            except Exception:  # pragma: no cover - best-effort
                pitfall_context = ""
        hippocampal_context = ""
        replay_context = ""
        if include_long_term and self.hippocampus is not None:
            hippocampal_context = self.hippocampus.retrieve_summary(
                question, include_meta=False
            )
            replay_context = self.hippocampus.replay_summary(
                question,
                topk=1,
                max_chars=180,
                include_meta=False,
            )

        segments = []
        if working_memory_context:
            segments.append(_format_section("Working memory", working_memory_context.splitlines()))
        if memory_context:
            segments.append(memory_context)
        if schema_context:
            segments.append(f"[Schema memory] {schema_context}")
        if pitfall_context:
            segments.append(pitfall_context)
        if hippocampal_context:
            segments.append(f"[Hippocampal recall] {hippocampal_context}")
        if replay_context:
            segments.append(f"[Hippocampal replay] {replay_context}")
        combined = "\n".join(segments)
        parts = {
            "working_memory": working_memory_context,
            "memory": memory_context,
            "schema": schema_context,
            "pitfalls": pitfall_context,
            "hippocampal": hippocampal_context,
            "replay": replay_context,
        }
        focus: FocusSummary | None = None
        if self.prefrontal_cortex is not None:
            focus_memory_context = "\n".join(
                [part for part in (working_memory_context, memory_context) if part]
            )
            focus = self.prefrontal_cortex.synthesise_focus(
                question=question,
                memory_context=focus_memory_context,
                hippocampal_context=hippocampal_context,
            )
            combined = self.prefrontal_cortex.gate_context(combined, focus)
        return combined, focus, parts

    def _sense_affect(self, question: str, draft: str) -> Dict[str, float]:
        """Approximate limbic evaluation inspired by the human amygdala."""

        metrics = self.amygdala.analyze(f"{question}\n{draft}") if self.amygdala else {}
        return {
            "valence": float(metrics.get("valence", 0.0)),
            "arousal": float(metrics.get("arousal", 0.0)),
            "risk": float(metrics.get("risk", 0.0)),
        }

    def _rebuild_context_from_parts(
        self,
        *,
        question: str,
        context_parts: Dict[str, str],
    ) -> tuple[str, FocusSummary | None]:
        rebuilt: List[str] = []
        wm_ctx = str(context_parts.get("working_memory") or "")
        mem_ctx = str(context_parts.get("memory") or "")
        schema_ctx = str(context_parts.get("schema") or "")
        pitfall_ctx = str(context_parts.get("pitfalls") or "")
        hip_ctx = str(context_parts.get("hippocampal") or "")
        replay_ctx = str(context_parts.get("replay") or "")

        if wm_ctx:
            rebuilt.append(_format_section("Working memory", wm_ctx.splitlines()))
        if mem_ctx:
            rebuilt.append(mem_ctx)
        if schema_ctx:
            rebuilt.append(f"[Schema memory] {schema_ctx}")
        if pitfall_ctx:
            rebuilt.append(pitfall_ctx)
        if hip_ctx:
            rebuilt.append(f"[Hippocampal recall] {hip_ctx}")
        if replay_ctx:
            rebuilt.append(f"[Hippocampal replay] {replay_ctx}")

        context = "\n".join(rebuilt)
        focus: FocusSummary | None = None
        if self.prefrontal_cortex is not None:
            focus_memory_context = "\n".join([part for part in (wm_ctx, mem_ctx) if part])
            focus = self.prefrontal_cortex.synthesise_focus(
                question=question,
                memory_context=focus_memory_context,
                hippocampal_context=hip_ctx,
            )
            context = self.prefrontal_cortex.gate_context(context, focus)
        return context, focus

    @staticmethod
    def _infer_question_type(
        question: str,
        *,
        precision_priority: bool = False,
    ) -> str:
        q = str(question or "").strip()
        if not q:
            return "easy"

        has_qmark = ("?" in q) or ("？" in q)
        length = len(q)
        qmark_hard_len = 68 if precision_priority else 84
        qmark_medium_len = 20 if precision_priority else 26
        body_medium_len = 160 if precision_priority else 200
        delimiter_hits = len(re.findall(r"[,、，;；:：]", q))

        # Strong structural cues (language-agnostic).
        if "```" in q:
            return "hard"
        if re.search(r"\d", q) and re.search(r"[+\-*/^%]|=", q):
            return "hard"
        if any(sym in q for sym in ("=", "≠", "<", ">", "≥", "≤", "→", "⇒", "∴", "∵")):
            return "hard"
        if re.search(
            r"(推論|妥当性|妥当|論理|論証|検証|証明|演繹|帰納|矛盾|含意|したがって|"
            r"why|how|reason|justify|trade[- ]?off|counterexample|"
            r"therefore|validity|logical|inference|deduction|induction|proof)",
            q,
            flags=re.IGNORECASE,
        ):
            return "hard" if length >= 180 else "medium"
        if re.search(
            r"(de morgan|equivalent expression|equivalent form|logical equivalence|"
            r"等価|同値|書き換えて|書き換えよ|書き換える)",
            q,
            flags=re.IGNORECASE,
        ):
            return "medium"
        inline_code = re.search(r"`([^`]{8,})`", q)
        if inline_code and re.search(
            r"(review|correctness|edge[- ]?case|bug|fix|snippet)",
            q,
            flags=re.IGNORECASE,
        ):
            # Inline code + review language is a useful signal, but very short
            # snippets are often "easy" even when correctness is mentioned.
            code_len = len(inline_code.group(1))
            if length < 140 and code_len < 80:
                if precision_priority and re.search(
                    r"(correctness|edge[- ]?case|bug|failure|empty|zero|none|null)",
                    q,
                    flags=re.IGNORECASE,
                ):
                    return "medium"
                return "easy"
            return "hard" if (length >= 260 or code_len >= 220) else "medium"
        if (
            length >= 64
            and re.search(
                r"(compare|comparison|versus|\bvs\.?\b|complexity|causal|causation|"
                r"correlation|triage|root[- ]?cause|postmortem|preferable)",
                q,
                flags=re.IGNORECASE,
            )
        ):
            return "medium"
        if has_qmark and delimiter_hits >= 2 and length >= 42:
            return "medium"
        if delimiter_hits >= 4 and length >= 96:
            return "medium"

        # Complexity by size/structure (works for CJK + Latin).
        if q.count("\n") >= 2:
            return "hard" if length >= 360 else "medium"
        if precision_priority and q.count("\n") >= 1 and length >= 80:
            return "medium"
        if has_qmark and length >= qmark_hard_len:
            return "hard"
        if has_qmark and length >= qmark_medium_len:
            return "medium"
        if length >= body_medium_len:
            return "medium"
        return "easy"

    @staticmethod
    def _system2_round_budget(
        *,
        system2_mode: str,
        q_type: str,
        question: str,
        context_signal_len: int,
        focus_metric: float,
        priority: str = "balanced",
    ) -> int:
        rounds = 1
        q_type_norm = str(q_type or "").strip().lower()
        mode_norm = str(system2_mode or "").strip().lower()
        if q_type_norm == "hard":
            rounds = 3
        elif q_type_norm == "medium":
            rounds = 2

        q = str(question or "")
        if "```" in q:
            rounds = max(rounds, 3)
        elif re.search(r"\d", q) and re.search(r"[+\-*/^%=]|=", q):
            rounds = max(rounds, 2)
        elif len(q) >= 220:
            rounds = max(rounds, 2)

        if int(context_signal_len or 0) >= 420:
            rounds = max(rounds, 2)
        if float(focus_metric or 0.0) >= 0.7:
            rounds = max(rounds, 2)

        if mode_norm == "on":
            rounds = max(rounds, 2)

        priority_norm = str(priority or "balanced").strip().lower()
        if priority_norm == "precision":
            if q_type_norm == "hard":
                rounds = max(rounds, 3)
            elif q_type_norm == "medium":
                if mode_norm != "auto":
                    rounds = max(rounds, 3)
                else:
                    rounds = max(rounds, 2)
            elif len(q) >= 64:
                rounds = max(rounds, 2)
            if int(context_signal_len or 0) >= 220:
                rounds = max(rounds, 2)
            if float(focus_metric or 0.0) >= 0.45:
                rounds = max(rounds, 2)

        return max(1, min(3, rounds))

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

    def _compose_architecture_path(
        self,
        *,
        focus: FocusSummary | None,
        focus_metric: float,
        schema_profile: SchemaProfile | None,
        affect: Dict[str, float],
        novelty: float,
        interoception: Optional[InteroceptiveState],
        salience_signal: Optional[SalienceSignal],
        thalamic_relay: Optional[ThalamicRelay],
        predictive_frame: Optional[PredictionFrame],
        hemisphere_signal: HemisphericSignal,
        collaboration_profile: CollaborationProfile,
        decision: DecisionOutcome,
        leading: str,
        collaborative: bool,
        steps: Sequence[Dict[str, Any]],
        coherence_signal: Optional[CoherenceSignal],
        distortion_payload: Optional[Dict[str, Any]],
        tags: Sequence[str],
        hippocampal_rollup: Optional[Dict[str, float]],
        success: bool,
    ) -> List[Dict[str, Any]]:
        """Summarise the active architecture stages for telemetry and memory."""

        path: List[Dict[str, Any]] = []

        perception_modules = ["Amygdala"] if self.amygdala else []
        if self.insula is not None:
            perception_modules.append("Insula")
        if self.prefrontal_cortex is not None:
            perception_modules.append("PrefrontalCortex")
            perception_modules.append("SchemaProfiler")
        if self.thalamus is not None:
            perception_modules.append("Thalamus")
        perception_modules.append("SharedMemory")
        if self.hippocampus is not None:
            perception_modules.append("TemporalHippocampalIndexing")

        perception_entry: Dict[str, Any] = {
            "stage": "perception",
            "modules": perception_modules,
            "signals": {
                "affect": {
                    "valence": float(affect.get("valence", 0.0)),
                    "arousal": float(affect.get("arousal", 0.0)),
                    "risk": float(affect.get("risk", 0.0)),
                    "novelty": float(novelty),
                },
                "focus_metric": float(focus_metric),
                "hemisphere": hemisphere_signal.to_payload(),
                "collaboration": collaboration_profile.to_payload(),
            },
        }
        if interoception is not None:
            perception_entry["signals"]["interoception"] = interoception.to_payload()
        if thalamic_relay is not None:
            perception_entry["signals"]["thalamic_relay"] = thalamic_relay.to_payload()
        if focus is not None:
            perception_entry["focus"] = focus.to_dict()
        if schema_profile is not None:
            perception_entry["schema_profile"] = schema_profile.to_dict()
        path.append(perception_entry)

        if predictive_frame is not None:
            predictive_modules = ["PredictiveCodingController"]
            if self.neural_integrator is not None:
                predictive_modules.append("NeuralIntegrator")
            path.append(
                {
                    "stage": "predictive_routing",
                    "modules": predictive_modules,
                    "phase": predictive_frame.networks.phase_state,
                    "dominant_network": predictive_frame.networks.dominant_network,
                    "top_networks": list(predictive_frame.networks.top_networks),
                    "system2_pressure": float(predictive_frame.system2_pressure),
                    "system2_ready": bool(predictive_frame.system2_ready),
                    "prediction_error": predictive_frame.prediction_error.to_payload(),
                    "networks": predictive_frame.networks.to_payload(),
                }
            )

        dialogue_modules = ["LeftBrainModel", "ReasoningDial", "BasalGanglia"]
        if self.right_model is not None:
            dialogue_modules.append("RightBrainModel")
        dialogue_modules.append("CorpusCallosum")
        if self.unconscious_field is not None:
            dialogue_modules.append("UnconsciousField")

        phases = sorted(
            {
                phase
                for phase in (
                    step.get("phase") for step in steps if isinstance(step, dict)
                )
                if phase
            }
        )
        dialogue_entry: Dict[str, Any] = {
            "stage": "inner_dialogue",
            "modules": dialogue_modules,
            "leading": leading,
            "collaborative": collaborative,
            "policy_action": int(decision.action),
            "temperature": float(decision.temperature),
            "slot_ms": int(decision.slot_ms),
            "step_count": len(steps),
            "phases": phases,
        }
        path.append(dialogue_entry)

        integration_modules = ["CoherenceResonator", "Auditor"]
        if self.anterior_cingulate_cortex is not None:
            integration_modules.append("AnteriorCingulateCortex")
        if self.salience_network is not None:
            integration_modules.append("SalienceNetwork")
        if self.cerebellum is not None:
            integration_modules.append("Cerebellum")
        if self.default_mode_network is not None:
            integration_modules.append("DefaultModeNetwork")
        if self.psychoid_adapter is not None:
            integration_modules.append("PsychoidAttentionAdapter")

        integration_entry: Dict[str, Any] = {
            "stage": "integration",
            "modules": integration_modules,
            "success": bool(success),
        }
        if coherence_signal is not None:
            integration_entry["coherence"] = coherence_signal.to_payload()
        if salience_signal is not None:
            integration_entry["salience"] = salience_signal.to_payload()
        if distortion_payload is not None:
            integration_entry["distortion"] = distortion_payload
        path.append(integration_entry)

        memory_modules = ["SharedMemory"]
        if self.hippocampus is not None:
            memory_modules.append("TemporalHippocampalIndexing")
        if self.unconscious_field is not None:
            memory_modules.append("UnconsciousField")

        memory_entry: Dict[str, Any] = {
            "stage": "memory",
            "modules": memory_modules,
            "tags": list(tags),
        }
        if hippocampal_rollup is not None:
            memory_entry["hippocampal_rollup"] = hippocampal_rollup
        path.append(memory_entry)

        return path

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
        *,
        qid: Optional[str] = None,
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
        qid_value = str(qid) if qid else str(uuid.uuid4())
        return DecisionOutcome(
            qid=qid_value,
            action=action,
            temperature=temperature,
            slot_ms=slot_ms,
            state=state,
        )

    @staticmethod
    def _context_signal_len(context_parts: Dict[str, Any]) -> int:
        return sum(
            len(str(context_parts.get(key) or ""))
            for key in (
                "working_memory",
                "memory",
                "schema",
                "pitfalls",
                "hippocampal",
                "replay",
            )
        )

    def _build_system2_settings(self, system2_mode: str) -> System2Settings:
        system2_norm = str(system2_mode or "auto").strip().lower()
        if system2_norm in {"true", "1", "yes"}:
            system2_norm = "on"
        elif system2_norm in {"false", "0", "no"}:
            system2_norm = "off"
        if system2_norm not in {"auto", "on", "off"}:
            system2_norm = "auto"

        system2_priority = _resolve_system2_priority(system2_norm)
        precision_priority = system2_priority == "precision"
        return System2Settings(
            mode=system2_norm,
            priority=system2_priority,
            precision_priority=precision_priority,
            low_signal_filter=_env_flag("DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER", True),
            followup_new_issue_cap=_env_int(
                "DUALBRAIN_SYSTEM2_MAX_NEW_ISSUES",
                3 if precision_priority else 2,
                minimum=0,
                maximum=8,
            ),
            followup_new_issue_min_score=_env_float(
                "DUALBRAIN_SYSTEM2_FOLLOWUP_MIN_SCORE",
                1.2 if precision_priority else 1.6,
                minimum=-1.0,
                maximum=4.0,
            ),
            followup_min_overlap=_env_float(
                "DUALBRAIN_SYSTEM2_FOLLOWUP_MIN_OVERLAP",
                0.15 if precision_priority else 0.2,
                minimum=0.0,
                maximum=1.0,
            ),
            followup_max_remaining=_env_int(
                "DUALBRAIN_SYSTEM2_FOLLOWUP_MAX_REMAINING",
                3 if precision_priority else 2,
                minimum=1,
                maximum=12,
            ),
            followup_min_progress=_env_int(
                "DUALBRAIN_SYSTEM2_FOLLOWUP_MIN_PROGRESS",
                1,
                minimum=1,
                maximum=12,
            ),
        )

    def _run_perception_stage(
        self,
        question: str,
        *,
        qid: str,
        precision_priority: bool,
    ) -> PerceptionStageResult:
        context, focus, context_parts = self._compose_context(question)
        is_trivial_chat = bool(
            self.prefrontal_cortex is not None
            and self.prefrontal_cortex.is_trivial_chat_turn(question)
        )
        focus_metric = 0.0
        if self.prefrontal_cortex is not None and focus is not None:
            focus_metric = self.prefrontal_cortex.focus_metric(focus)

        context_signal_len = self._context_signal_len(context_parts)
        question_affect = self._sense_affect(question, "")
        novelty = self.memory.novelty_score(question)
        q_type_hint = self._infer_question_type(
            question,
            precision_priority=precision_priority,
        )

        insula_state: Optional[InteroceptiveState] = None
        salience_signal: Optional[SalienceSignal] = None
        thalamic_relay: Optional[ThalamicRelay] = None
        if self.insula is not None:
            insula_state = self.insula.assess(
                question=question,
                affect=question_affect,
                novelty=novelty,
                focus_metric=focus_metric,
                context_signal_len=context_signal_len,
                has_working_memory=bool(context_parts.get("working_memory")),
                has_long_term_memory=bool(
                    context_parts.get("memory")
                    or context_parts.get("schema")
                    or context_parts.get("pitfalls")
                    or context_parts.get("hippocampal")
                    or context_parts.get("replay")
                ),
                is_trivial_chat=is_trivial_chat,
            )
        if self.salience_network is not None and insula_state is not None:
            salience_signal = self.salience_network.evaluate(
                question=question,
                interoception=insula_state,
                focus_metric=focus_metric,
                q_type_hint=q_type_hint,
                is_trivial_chat=is_trivial_chat,
                has_working_memory=bool(context_parts.get("working_memory")),
                has_long_term_memory=bool(
                    context_parts.get("memory")
                    or context_parts.get("schema")
                    or context_parts.get("pitfalls")
                    or context_parts.get("replay")
                ),
                has_hippocampal_memory=bool(
                    context_parts.get("hippocampal")
                    or context_parts.get("replay")
                ),
            )
        if self.thalamus is not None and salience_signal is not None:
            thalamic_relay = self.thalamus.route(
                context_parts=context_parts,
                salience=salience_signal,
            )
            if not thalamic_relay.keep_working_memory:
                context_parts["working_memory"] = ""
            if not thalamic_relay.keep_long_term_memory:
                context_parts["memory"] = ""
            if not thalamic_relay.keep_schema_memory:
                context_parts["schema"] = ""
            if not thalamic_relay.keep_pitfall_memory:
                context_parts["pitfalls"] = ""
            if not thalamic_relay.keep_hippocampal_memory:
                context_parts["hippocampal"] = ""
                context_parts["replay"] = ""
            context, focus = self._rebuild_context_from_parts(
                question=question,
                context_parts=context_parts,
            )
            if self.prefrontal_cortex is not None and focus is not None:
                focus_metric = self.prefrontal_cortex.focus_metric(focus)
            else:
                focus_metric = 0.0
            context_signal_len = self._context_signal_len(context_parts)

        if insula_state is not None:
            try:
                self.telemetry.log(
                    "insula_state",
                    qid=qid,
                    state=insula_state.to_payload(),
                )
            except Exception:
                pass
        if salience_signal is not None:
            try:
                self.telemetry.log(
                    "salience_network",
                    qid=qid,
                    signal=salience_signal.to_payload(),
                )
            except Exception:
                pass
        if thalamic_relay is not None:
            try:
                self.telemetry.log(
                    "thalamic_relay",
                    qid=qid,
                    relay=thalamic_relay.to_payload(),
                )
            except Exception:
                pass
        if context_parts.get("replay"):
            try:
                self.telemetry.log(
                    "hippocampal_replay",
                    qid=qid,
                    replay=str(context_parts.get("replay") or ""),
                )
            except Exception:
                pass

        hemisphere_signal = self._select_hemisphere_mode(question, focus)
        hemisphere_mode = hemisphere_signal.mode
        hemisphere_bias = hemisphere_signal.bias
        collaboration_profile = self._compute_collaboration_profile(
            hemisphere_signal, focus
        )
        predictive_frame: Optional[PredictionFrame] = None
        if self.predictive_coding is not None:
            predictive_frame = self.predictive_coding.evaluate(
                question=question,
                q_type_hint=q_type_hint,
                precision_priority=precision_priority,
                focus_metric=focus_metric,
                affect=question_affect,
                novelty=novelty,
                hemisphere_mode=hemisphere_mode,
                hemisphere_bias=hemisphere_bias,
                collaboration_strength=collaboration_profile.strength,
                interoception=insula_state,
                salience_signal=salience_signal,
                thalamic_relay=thalamic_relay,
                context_signal_len=context_signal_len,
                has_working_memory=bool(context_parts.get("working_memory")),
                has_long_term_memory=bool(
                    context_parts.get("memory")
                    or context_parts.get("schema")
                    or context_parts.get("pitfalls")
                ),
                has_hippocampal_memory=bool(
                    context_parts.get("hippocampal")
                    or context_parts.get("replay")
                ),
                is_trivial_chat=is_trivial_chat,
            )
            try:
                self.telemetry.log(
                    "predictive_coding",
                    qid=qid,
                    frame=predictive_frame.to_payload(),
                )
            except Exception:
                pass

        return PerceptionStageResult(
            context=context,
            focus=focus,
            context_parts=context_parts,
            is_trivial_chat=is_trivial_chat,
            focus_metric=focus_metric,
            context_signal_len=context_signal_len,
            question_affect=question_affect,
            novelty=novelty,
            q_type_hint=q_type_hint,
            insula_state=insula_state,
            salience_signal=salience_signal,
            thalamic_relay=thalamic_relay,
            hemisphere_signal=hemisphere_signal,
            hemisphere_mode=hemisphere_mode,
            hemisphere_bias=hemisphere_bias,
            collaboration_profile=collaboration_profile,
            predictive_frame=predictive_frame,
        )

    def _resolve_system2_activation(
        self,
        question: str,
        *,
        perception: PerceptionStageResult,
        settings: System2Settings,
    ) -> System2Status:
        system2_capable = bool(
            getattr(self.left_model, "uses_external_llm", False)
            and getattr(self.right_model, "uses_external_llm", False)
        )
        system2_active = False
        system2_reason = "disabled"
        if settings.mode == "on":
            system2_active = True
            system2_reason = "forced_on" if system2_capable else "forced_on_unavailable"
        elif settings.mode == "off":
            system2_active = False
            system2_reason = "forced_off"
        else:
            if not system2_capable:
                system2_active = False
                system2_reason = "auto_unavailable"
            elif not perception.is_trivial_chat:
                q = str(question or "")
                has_qmark = ("?" in q) or ("？" in q)
                line_count = q.count("\n") + 1
                bullet_hits = len(
                    re.findall(
                        r"(?m)^\s*(?:[-*•]|[0-9]{1,2}[.)]|[①-⑳])\s+",
                        q,
                    )
                )
                wm_len = len(str(perception.context_parts.get("working_memory") or ""))
                mem_len = len(str(perception.context_parts.get("memory") or ""))
                hip_len = len(str(perception.context_parts.get("hippocampal") or ""))
                replay_len = len(str(perception.context_parts.get("replay") or ""))
                context_signal_len = wm_len + mem_len + hip_len + replay_len
                focus_relevance = (
                    float(perception.focus.relevance) if perception.focus is not None else 0.0
                )
                focus_overlap = (
                    float(perception.focus.hippocampal_overlap)
                    if perception.focus is not None
                    else 0.0
                )
                delimiter_hits = len(re.findall(r"[,、，;；:：]", q))

                if perception.q_type_hint in {"medium", "hard"}:
                    system2_active = True
                    system2_reason = f"q_type_{perception.q_type_hint}"
                elif (
                    perception.salience_signal is not None
                    and perception.salience_signal.system2_gate
                    and perception.salience_signal.dominant_network
                    in {"executive_control", "memory_recall"}
                ):
                    system2_active = True
                    system2_reason = f"salience_{perception.salience_signal.dominant_network}"
                elif (
                    perception.predictive_frame is not None
                    and perception.predictive_frame.system2_ready
                ):
                    system2_active = True
                    if (
                        float(perception.predictive_frame.prediction_error.overall)
                        >= 0.40
                    ):
                        system2_reason = (
                            f"predictive_{perception.predictive_frame.prediction_error.dominant_channel}"
                        )
                    else:
                        system2_reason = (
                            f"predictive_{perception.predictive_frame.networks.dominant_network}"
                        )
                elif "```" in q or any(
                    sym in q for sym in ("=", "≠", "<", ">", "≥", "≤", "→", "⇒", "∴", "∵")
                ):
                    system2_active = True
                    system2_reason = "symbolic"
                elif has_qmark and re.search(r"\d", q):
                    system2_active = True
                    system2_reason = "numeric_question"
                elif has_qmark and len(q) >= 80:
                    system2_active = True
                    system2_reason = "long_question"
                elif line_count >= 3 and bullet_hits >= 2 and len(q) >= 60:
                    system2_active = True
                    system2_reason = "structured_list"
                elif line_count >= 4 and len(q) >= 110:
                    system2_active = True
                    system2_reason = "multiline_long"
                elif (
                    context_signal_len >= 260
                    and 48 <= len(q) <= 180
                    and (delimiter_hits >= 2 or line_count >= 2)
                    and (focus_relevance >= 0.08 or focus_overlap >= 0.08)
                ):
                    system2_active = True
                    system2_reason = "context_heavy"
                elif settings.precision_priority and has_qmark and len(q) >= 16:
                    system2_active = True
                    system2_reason = "precision_short_question"
                elif settings.precision_priority and line_count >= 2 and len(q) >= 45:
                    system2_active = True
                    system2_reason = "precision_multiline"

        return System2Status(
            capable=system2_capable,
            active=system2_active,
            reason=system2_reason,
        )

    def _log_system2_mode(
        self,
        *,
        qid: str,
        settings: System2Settings,
        status: System2Status,
    ) -> None:
        try:
            self.telemetry.log(
                "system2_mode",
                qid=qid,
                mode=settings.mode,
                enabled=bool(status.active),
                reason=status.reason,
                priority=settings.priority,
                low_signal_filter=bool(settings.low_signal_filter),
                followup_new_issue_cap=int(settings.followup_new_issue_cap),
                followup_new_issue_min_score=float(
                    settings.followup_new_issue_min_score
                ),
                followup_min_overlap=float(settings.followup_min_overlap),
                followup_max_remaining=int(settings.followup_max_remaining),
                followup_min_progress=int(settings.followup_min_progress),
            )
        except Exception:  # pragma: no cover - telemetry is best-effort
            pass

    def _select_leading_stage(
        self,
        *,
        requested_leading: str,
        system2_active: bool,
        system2_mode: str,
        hemisphere_mode: str,
        hemisphere_bias: float,
        collaboration_profile: CollaborationProfile,
    ) -> LeadingSelection:
        auto_selected_leading = False
        collaborative_lead = False
        selection_reason = "explicit_request"
        leading_style = "explicit_request"
        collaborative_hint = False
        if requested_leading in {"left", "right"}:
            leading = requested_leading
            selection_reason = f"explicit_request_{leading}"
        else:
            collaborative_hint = (
                hemisphere_mode == "balanced"
                and hemisphere_bias < 0.45
                and collaboration_profile.strength >= 0.4
            )

            if system2_active:
                leading = "left"
                selection_reason = (
                    "system2_forced" if system2_mode == "on" else "system2_auto"
                )
                leading_style = "system2"
                auto_selected_leading = True
            else:
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

        return LeadingSelection(
            leading=leading,
            auto_selected=auto_selected_leading,
            collaborative=collaborative_lead,
            selection_reason=selection_reason,
            leading_style=leading_style,
        )

    def _build_context_frame(
        self,
        *,
        question: str,
        context_parts: Dict[str, str],
    ) -> ContextFrame:
        context, focus = self._rebuild_context_from_parts(
            question=question,
            context_parts=context_parts,
        )
        if self.prefrontal_cortex is not None and focus is not None:
            focus_metric = self.prefrontal_cortex.focus_metric(focus)
            focus_keywords = list(focus.keywords) if focus.keywords else None
        else:
            focus = None
            focus_metric = 0.0
            focus_keywords = None
        return ContextFrame(
            context=context,
            focus=focus,
            focus_metric=focus_metric,
            focus_keywords=focus_keywords,
        )

    def _apply_director_memory_control(
        self,
        *,
        question: str,
        context_parts: Dict[str, str],
        control: Dict[str, Any],
    ) -> ContextFrame:
        mem = control.get("memory")
        mem_obj: Dict[str, Any] = mem if isinstance(mem, dict) else {}
        wm = str(mem_obj.get("working_memory") or "auto").strip().lower()
        lt = str(mem_obj.get("long_term") or "auto").strip().lower()

        cortex = self.prefrontal_cortex
        if wm == "drop":
            context_parts["working_memory"] = ""
        elif wm == "keep" and not context_parts.get("working_memory") and cortex is not None:
            context_parts["working_memory"] = cortex.working_memory_context(turns=2)

        if lt == "drop":
            context_parts["memory"] = ""
            context_parts["schema"] = ""
            context_parts["pitfalls"] = ""
            context_parts["hippocampal"] = ""
            context_parts["replay"] = ""
        elif lt == "keep":
            if not context_parts.get("memory"):
                context_parts["memory"] = self.memory.retrieve_related(question)
            if not context_parts.get("schema"):
                try:
                    context_parts["schema"] = self.memory.retrieve_schema_related(question)
                except Exception:  # pragma: no cover - optional feature
                    context_parts["schema"] = ""
            if not context_parts.get("hippocampal") and self.hippocampus is not None:
                context_parts["hippocampal"] = self.hippocampus.retrieve_summary(
                    question, include_meta=False
                )
            if not context_parts.get("replay") and self.hippocampus is not None:
                context_parts["replay"] = self.hippocampus.replay_summary(
                    question,
                    topk=1,
                    max_chars=180,
                    include_meta=False,
                )

        return self._build_context_frame(
            question=question,
            context_parts=context_parts,
        )

    def _prepare_observer_stage(
        self,
        *,
        question: str,
        context: str,
        focus: FocusSummary | None,
        focus_keywords: List[str] | None,
        context_parts: Dict[str, str],
        is_trivial_chat: bool,
        hemisphere_mode: str,
        hemisphere_bias: float,
        leading: str,
        collaborative_lead: bool,
        collaboration_profile: CollaborationProfile,
        executive_mode: str,
        executive_observer_mode: str,
    ) -> ObserverSetup:
        executive_mode_norm = str(executive_mode or "off").strip().lower()
        if executive_mode_norm not in {"off", "observe", "assist", "polish"}:
            executive_mode_norm = "observe"
        executive_observer_mode_norm = str(executive_observer_mode or "off").strip().lower()
        if executive_observer_mode_norm not in {"off", "metrics", "director", "both"}:
            executive_observer_mode_norm = "off"
        use_director = executive_observer_mode_norm in {"director", "both"}
        use_metrics_observer = executive_observer_mode_norm in {"metrics", "both"}

        executive_task: asyncio.Task[ExecutiveAdvice] | None = None
        if self.executive_model is not None and executive_mode_norm != "off":
            try:
                executive_task = asyncio.create_task(
                    self.executive_model.advise(
                        question=question,
                        context=context,
                        focus_keywords=focus_keywords,
                    )
                )
            except Exception:  # pragma: no cover - defensive guard
                executive_task = None

        director_task: asyncio.Task[DirectorAdvice] | None = None
        if self.director_model is not None and use_director:
            try:
                has_qmark = ("?" in (question or "")) or ("？" in (question or ""))
                director_task = asyncio.create_task(
                    self.director_model.advise(
                        question=question,
                        context=context,
                        signals={
                            "is_trivial_chat": is_trivial_chat,
                            "question_len": len(question or ""),
                            "question_has_qmark": has_qmark,
                            "hemisphere_mode": hemisphere_mode,
                            "hemisphere_bias": hemisphere_bias,
                            "leading": leading,
                            "collaborative": collaborative_lead,
                            "collaboration_strength": collaboration_profile.strength,
                            "focus_relevance": (focus.relevance if focus else None),
                            "hippocampal_overlap": (
                                focus.hippocampal_overlap if focus else None
                            ),
                            "has_working_memory": bool(context_parts.get("working_memory")),
                            "has_long_term": bool(
                                context_parts.get("memory")
                                or context_parts.get("schema")
                                or context_parts.get("hippocampal")
                                or context_parts.get("replay")
                            ),
                        },
                    )
                )
            except Exception:  # pragma: no cover - defensive guard
                director_task = None

        return ObserverSetup(
            executive_mode=executive_mode_norm,
            observer_mode=executive_observer_mode_norm,
            use_director=use_director,
            use_metrics_observer=use_metrics_observer,
            executive_task=executive_task,
            director_task=director_task,
        )

    async def _run_pre_draft_director_stage(
        self,
        *,
        question: str,
        qid: str,
        is_trivial_chat: bool,
        context_frame: ContextFrame,
        context_parts: Dict[str, str],
        director_task: asyncio.Task[DirectorAdvice] | None,
        phase_add: Callable[[str, float], None],
    ) -> DirectorStageResult:
        director_payload: Dict[str, Any] | None = None
        director_control: Dict[str, Any] | None = None
        if director_task is None:
            return DirectorStageResult(
                context_frame=context_frame,
                director_task=director_task,
                director_payload=director_payload,
                director_control=director_control,
            )

        advice: DirectorAdvice | None = None
        if is_trivial_chat:
            advice = DirectorAdvice(
                memo=(
                    "(Director memo / fast-path)\n"
                    "- trivial chat: keep WM (if any), drop long-term, skip consult"
                ),
                control={
                    "consult": "skip",
                    "temperature": 0.45,
                    "max_chars": 280,
                    "memory": {"working_memory": "keep", "long_term": "drop"},
                    "append_clarifying_question": None,
                },
                confidence=0.35,
                latency_ms=0.0,
                source="fast_path",
            )
            director_task.cancel()
            director_task = None
        else:
            try:
                director_wait_started = time.perf_counter()
                advice = await asyncio.wait_for(
                    asyncio.shield(director_task), timeout=1.6
                )
                phase_add("executive", director_wait_started)
            except asyncio.TimeoutError:
                phase_add("executive", director_wait_started)
                advice = None
            except asyncio.CancelledError:
                advice = None
            except Exception:  # pragma: no cover - defensive guard
                advice = None
            if advice is None:
                director_task.cancel()
                director_task = None
                advice = DirectorAdvice(
                    memo=(
                        "(Director memo / timeout fallback)\n"
                        "- Proceeding with heuristic steering (no external director output)."
                    ),
                    control={
                        "consult": "auto",
                        "temperature": None,
                        "max_chars": None,
                        "memory": {"working_memory": "auto", "long_term": "auto"},
                        "append_clarifying_question": None,
                    },
                    confidence=0.15,
                    latency_ms=1600.0,
                    source="timeout_fallback",
                )

        director_payload = advice.to_payload()
        director_payload["observer_mode"] = "director"
        director_control = advice.control if isinstance(advice.control, dict) else {}
        self.telemetry.log(
            "director_reasoner",
            qid=qid,
            confidence=float(advice.confidence),
            latency_ms=float(advice.latency_ms),
            source=str(advice.source),
            phase="pre",
        )
        context_frame = self._apply_director_memory_control(
            question=question,
            context_parts=context_parts,
            control=director_control,
        )
        return DirectorStageResult(
            context_frame=context_frame,
            director_task=director_task,
            director_payload=director_payload,
            director_control=director_control,
        )

    async def _run_left_draft_stage(
        self,
        *,
        question: str,
        context: str,
        focus: FocusSummary | None,
        leading: str,
        collaborative_lead: bool,
        system2_active: bool,
        vision_images: Optional[Sequence[Dict[str, Any]]],
        delta_cb: Optional[Callable[[str], Any]],
        stream_final_only: bool,
        emitted_any: bool,
        focus_keywords: List[str] | None,
        focus_metric: float,
        hemisphere_mode: str,
        hemisphere_bias: float,
        phase_add: Callable[[str, float], None],
    ) -> DraftStageResult:
        steps: List[InnerDialogueStep] = []
        right_lead_notes: Optional[str] = None
        right_preview_task: asyncio.Task[str] | None = None
        if (leading == "right" or collaborative_lead) and not system2_active:
            try:
                right_preview_task = asyncio.create_task(
                    self.right_model.generate_lead(question, context, on_delta=None)
                )
            except Exception:  # pragma: no cover - defensive guard
                right_preview_task = None

        left_context = context
        if system2_active:
            system2_hint = (
                "[System2 reasoning mode]\n"
                "- Prioritize logical correctness over tone.\n"
                "- Make assumptions explicit.\n"
                "- If needed, include short step-by-step reasoning, but avoid any meta about tools/prompts.\n"
                "- Keep the final answer self-contained.\n"
            )
            left_context = (f"{context}\n\n{system2_hint}".strip() if context else system2_hint)

        left_draft_started = time.perf_counter()
        draft = await self.left_model.generate_answer(
            question,
            left_context,
            vision_images=vision_images,
            on_delta=(
                delta_cb
                if delta_cb and not stream_final_only and not emitted_any
                else None
            ),
        )
        phase_add("left_draft", left_draft_started)

        if right_preview_task is not None:
            if right_preview_task.done():
                try:
                    right_lead_notes = right_preview_task.result()
                except asyncio.CancelledError:
                    right_lead_notes = None
                except Exception:  # pragma: no cover - defensive guard
                    right_lead_notes = None
            else:
                right_preview_task.cancel()

            preview_meta = {
                "requested": True,
                "collaborative": collaborative_lead,
                "leading_mode": leading,
                "available": bool(right_lead_notes),
            }
            if right_lead_notes:
                preview_meta["length"] = len(right_lead_notes)
            steps.append(
                InnerDialogueStep(
                    phase="right_preview",
                    role="right",
                    content=_truncate_text(right_lead_notes),
                    metadata=preview_meta,
                )
            )

        left_lead_preview = draft.split("\n\n", 1)[0][:320] if collaborative_lead else None
        left_coherence: Optional[HemisphericCoherence] = None
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
            consult_bias = self.prefrontal_cortex.adjust_consult_bias(
                consult_bias, focus
            )
        if hemisphere_mode == "right":
            consult_bias = max(
                -0.35, consult_bias - (0.15 + 0.35 * hemisphere_bias)
            )
        elif hemisphere_mode == "left":
            consult_bias = min(
                0.35, consult_bias + (0.15 + 0.35 * hemisphere_bias)
            )

        steps.append(
            InnerDialogueStep(
                phase="left_draft",
                role="left",
                content=_truncate_text(draft),
                metadata={
                    "confidence": round(confidence, 4),
                    "novelty": round(novelty, 4),
                    "consult_bias": round(consult_bias, 4),
                    "valence": affect["valence"],
                    "arousal": affect["arousal"],
                    "hemisphere_mode": hemisphere_mode,
                    "focus_metric": round(focus_metric, 4),
                    **(
                        {"left_coherence": round(left_coherence.score(), 4)}
                        if left_coherence is not None
                        else {}
                    ),
                },
            )
        )
        return DraftStageResult(
            draft=draft,
            right_lead_notes=right_lead_notes,
            left_lead_preview=left_lead_preview,
            left_coherence=left_coherence,
            confidence=confidence,
            affect=affect,
            novelty=novelty,
            consult_bias=consult_bias,
            steps=steps,
        )

    async def _run_routing_stage(
        self,
        *,
        question: str,
        decision: DecisionOutcome,
        draft: str,
        confidence: float,
        affect: Dict[str, float],
        salience_signal: SalienceSignal | None,
        context_parts: Dict[str, str],
        context_frame: ContextFrame,
        system2_mode: str,
        system2_priority: str,
        system2_active: bool,
        system2_capable: bool,
        system2_reason: str,
        precision_priority: bool,
        system2_low_signal_filter: bool,
        director_task: asyncio.Task[DirectorAdvice] | None,
        director_payload: Dict[str, Any] | None,
        director_control: Dict[str, Any] | None,
        explicit_leading: bool,
        force_right_lead: bool,
        is_trivial_chat: bool,
        phase_add: Callable[[str, float], None],
        context_signal_len: int,
    ) -> RoutingStageResult:
        focus_metric = context_frame.focus_metric
        acc_signal_pre: ConflictSignal | None = None
        acc_override_consult = _env_flag("DUALBRAIN_ACC_OVERRIDE_CONSULT", False)
        acc_system2_bump = _env_flag("DUALBRAIN_ACC_SYSTEM2_BUMP", False)
        acc_temp_drop = _env_float(
            "DUALBRAIN_ACC_TEMPERATURE_DROP",
            0.0,
            minimum=0.0,
            maximum=0.6,
        )
        acc_control_enabled = bool(
            (acc_override_consult or acc_system2_bump or acc_temp_drop > 0.0)
            and not is_trivial_chat
            and draft
            and str(draft).strip()
            and self.anterior_cingulate_cortex is not None
        )
        if acc_control_enabled:
            micro_pre = None
            try:
                micro_pre = micro_criticise_reasoning(question, draft)
            except Exception:
                micro_pre = None
            if micro_pre is not None:
                try:
                    acc_signal_pre = self.anterior_cingulate_cortex.monitor(
                        question=question,
                        draft=draft,
                        left_confidence=float(confidence or 0.0),
                        micro=micro_pre,
                    )
                except Exception:  # pragma: no cover - best-effort
                    acc_signal_pre = None
            if acc_signal_pre is not None and micro_pre is not None:
                acc_payload_pre = acc_signal_pre.to_payload()
                decision.state["acc_conflict_pre"] = acc_payload_pre
                try:
                    self.telemetry.log(
                        "acc_control_signal",
                        qid=decision.qid,
                        signal=acc_payload_pre,
                    )
                except Exception:  # pragma: no cover - telemetry best-effort
                    pass

                threshold = _env_float(
                    "DUALBRAIN_ACC_CONFLICT_THRESHOLD",
                    0.75,
                    minimum=0.0,
                    maximum=1.0,
                )
                min_micro_conf = _env_float(
                    "DUALBRAIN_ACC_MICRO_MIN_CONFIDENCE",
                    0.86,
                    minimum=0.0,
                    maximum=1.0,
                )
                max_micro_issues = _env_int(
                    "DUALBRAIN_ACC_MICRO_MAX_ISSUES",
                    6,
                    minimum=1,
                    maximum=12,
                )
                eligible = bool(
                    micro_pre.verdict == "issues"
                    and float(micro_pre.confidence_r) >= float(min_micro_conf)
                    and len(micro_pre.issues or []) <= int(max_micro_issues)
                    and float(acc_signal_pre.conflict_level) >= float(threshold)
                )
                if eligible:
                    if acc_temp_drop > 0.0:
                        base_temp = float(decision.temperature)
                        new_temp = max(
                            0.1,
                            min(
                                0.95,
                                base_temp - float(acc_temp_drop) * float(acc_signal_pre.conflict_level),
                            ),
                        )
                        if new_temp < base_temp:
                            decision.temperature = new_temp
                            decision.state["acc_temperature"] = {
                                "base": float(base_temp),
                                "adjusted": float(new_temp),
                                "drop": float(base_temp - new_temp),
                                "conflict": float(acc_signal_pre.conflict_level),
                                "domain": str(micro_pre.domain),
                            }
                    if acc_override_consult and decision.action == 0 and self.right_model is not None:
                        decision.action = 1
                        decision.state["acc_override_consult"] = True
                    if (
                        acc_system2_bump
                        and system2_mode == "auto"
                        and not system2_active
                        and system2_capable
                    ):
                        system2_active = True
                        system2_reason = "acc_bump"
                        decision.state["acc_system2_bump"] = True
                        try:
                            self.telemetry.log(
                                "acc_system2_bump",
                                qid=decision.qid,
                                conflict=float(acc_signal_pre.conflict_level),
                                micro_domain=str(micro_pre.domain),
                                micro_confidence=float(micro_pre.confidence_r),
                                micro_issues=int(len(micro_pre.issues or [])),
                            )
                        except Exception:  # pragma: no cover - telemetry best-effort
                            pass

        basal_loop_signal: BasalGangliaSignal | None = None
        if self.basal_ganglia is not None and (
            acc_signal_pre is not None or salience_signal is not None
        ):
            try:
                basal_loop_signal = self.basal_ganglia.evaluate(
                    state=decision.state,
                    affect=affect,
                    focus_metric=focus_metric,
                    conflict_level=(
                        float(acc_signal_pre.conflict_level)
                        if acc_signal_pre is not None
                        else 0.0
                    ),
                    salience_level=(
                        float(salience_signal.level)
                        if salience_signal is not None
                        else 0.0
                    ),
                )
            except Exception:  # pragma: no cover - best-effort
                basal_loop_signal = None
            if basal_loop_signal is not None:
                decision.state["basal_loop_signal"] = basal_loop_signal.to_dict()
                if (
                    basal_loop_signal.recommended_action == 1
                    and decision.action == 0
                    and self.right_model is not None
                    and (
                        (
                            acc_signal_pre is not None
                            and str(acc_signal_pre.recommended_control or "").lower()
                            in {"consult", "system2"}
                        )
                        or (
                            salience_signal is not None
                            and salience_signal.dominant_network == "memory_recall"
                            and salience_signal.system2_gate
                        )
                    )
                ):
                    decision.action = 1
                    decision.state["basal_loop_override"] = "consult"
                try:
                    self.telemetry.log(
                        "basal_ganglia_loop",
                        qid=decision.qid,
                        signal=basal_loop_signal.to_dict(),
                    )
                except Exception:  # pragma: no cover - telemetry best-effort
                    pass

        if precision_priority:
            decision.state["q_type"] = self._infer_question_type(
                question,
                precision_priority=True,
            )
        decision.state["system2_enabled"] = bool(system2_active)
        decision.state["system2_mode"] = system2_mode
        decision.state["system2_reason"] = system2_reason
        decision.state["system2_priority"] = system2_priority
        decision.state["system2_low_signal_filter"] = bool(system2_low_signal_filter)
        if force_right_lead and decision.action == 0:
            decision.action = 1
            decision.state["right_forced_lead"] = True
        if system2_active and decision.action == 0:
            decision.action = 1
            decision.state["system2_forced_consult"] = True

        system2_round_target = 1
        if system2_active:
            decision.temperature = max(0.1, min(0.6, float(decision.temperature)))
            decision.state["system2_temperature"] = decision.temperature
            system2_round_target = self._system2_round_budget(
                system2_mode=system2_mode,
                q_type=str(decision.state.get("q_type") or ""),
                question=question,
                context_signal_len=context_signal_len,
                focus_metric=focus_metric,
                priority=system2_priority,
            )
            decision.state["system2_round_target"] = system2_round_target
            system2_round_target_min = _env_int(
                "DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN",
                0,
                minimum=0,
                maximum=3,
            )
            if system2_round_target_min and system2_round_target < int(system2_round_target_min):
                system2_round_target = int(system2_round_target_min)
                decision.state["system2_round_target_min"] = int(system2_round_target_min)
                decision.state["system2_round_target"] = system2_round_target

        if (
            self.prefrontal_cortex is not None
            and not explicit_leading
            and self.prefrontal_cortex.is_trivial_chat_turn(question)
            and not system2_active
        ):
            decision.action = 0
            decision.state["prefrontal_override"] = "trivial_chat"

        if director_task is not None and director_payload is None:
            advice: DirectorAdvice | None = None
            if director_task.done():
                try:
                    advice = director_task.result()
                except asyncio.CancelledError:
                    advice = None
                except Exception:  # pragma: no cover - defensive guard
                    advice = None
            else:
                try:
                    director_wait_started = time.perf_counter()
                    advice = await asyncio.wait_for(
                        asyncio.shield(director_task),
                        timeout=0.9,
                    )
                    phase_add("executive", director_wait_started)
                except asyncio.TimeoutError:
                    phase_add("executive", director_wait_started)
                    advice = None
                except asyncio.CancelledError:
                    advice = None
                except Exception:  # pragma: no cover - defensive guard
                    advice = None

            if advice is not None:
                director_payload = advice.to_payload()
                director_payload["observer_mode"] = "director"
                director_control = advice.control if isinstance(advice.control, dict) else {}
                self.telemetry.log(
                    "director_reasoner",
                    qid=decision.qid,
                    confidence=float(advice.confidence),
                    latency_ms=float(advice.latency_ms),
                    source=str(advice.source),
                    phase="post",
                )
                context_frame = self._apply_director_memory_control(
                    question=question,
                    context_parts=context_parts,
                    control=director_control,
                )
            else:
                director_task.cancel()
                director_task = None

        director_max_chars: int | None = None
        director_append_question: str | None = None
        if director_control is not None:
            consult = str(director_control.get("consult") or "auto").strip().lower()
            if consult == "skip":
                decision.action = 0
                decision.state["director_consult"] = "skip"
            elif consult == "force" and not is_trivial_chat:
                decision.action = max(1, decision.action)
                decision.state["director_consult"] = "force"

            temp_raw = director_control.get("temperature")
            if temp_raw is not None and temp_raw != "":
                try:
                    temp_value = float(temp_raw)
                except Exception:
                    temp_value = None
                if temp_value is not None:
                    decision.temperature = max(0.1, min(0.95, temp_value))
                    decision.state["director_temperature"] = decision.temperature

            max_chars_raw = director_control.get("max_chars")
            if max_chars_raw is not None and max_chars_raw != "":
                try:
                    director_max_chars = int(max_chars_raw)
                except Exception:
                    director_max_chars = None
                if director_max_chars is not None:
                    director_max_chars = max(80, min(2400, director_max_chars))
                    decision.state["director_max_chars"] = director_max_chars

            append_q = str(director_control.get("append_clarifying_question") or "").strip()
            if append_q:
                director_append_question = append_q[:240]
                decision.state["director_append_question"] = director_append_question

            context_frame = self._apply_director_memory_control(
                question=question,
                context_parts=context_parts,
                control=director_control,
            )
            mem = director_control.get("memory")
            mem_obj: Dict[str, Any] = mem if isinstance(mem, dict) else {}
            wm = str(mem_obj.get("working_memory") or "auto").strip().lower()
            lt = str(mem_obj.get("long_term") or "auto").strip().lower()
            if wm == "drop":
                decision.state["director_memory_working"] = "drop"
            if lt == "drop":
                decision.state["director_memory_long_term"] = "drop"

        return RoutingStageResult(
            context_frame=context_frame,
            decision=decision,
            acc_signal=acc_signal_pre,
            basal_signal=basal_loop_signal,
            system2_active=system2_active,
            system2_reason=system2_reason,
            system2_round_target=system2_round_target,
            director_task=director_task,
            director_payload=director_payload,
            director_control=director_control,
            director_max_chars=director_max_chars,
            director_append_question=director_append_question,
        )

    def _build_consult_request(
        self,
        *,
        question: str,
        decision: DecisionOutcome,
        draft: str,
        context: str,
        focus: FocusSummary | None,
        system2_active: bool,
        system2_mode: str,
        hemisphere_mode: str,
        hemisphere_bias: float,
        unconscious_summary: UnconsciousSummaryModel | None,
        psychoid_signal: PsychoidSignalModel | None,
        psychoid_projection: PsychoidAttentionProjection | None,
        left_coherence: HemisphericCoherence | None,
        default_mode_reflections: List[DefaultModeReflection] | None,
        suppress_default_mode_reflections: bool,
    ) -> ConsultRequestPlan:
        payload_type = "ASK_CRITIC" if system2_active else "ASK_DETAIL"
        payload: Dict[str, Any] = {
            "type": payload_type,
            "qid": decision.qid,
            "question": question,
            "draft_sum": draft if len(draft) < 280 else draft[:280],
            "temperature": decision.temperature,
            "budget": self.reasoning_dial.pick_budget(),
            "context": context,
            "hemisphere_mode": hemisphere_mode,
            "hemisphere_bias": hemisphere_bias,
        }
        q_type_hint = str(decision.state.get("q_type") or "").strip().lower()
        system2_draft_limit = 2400
        if system2_active:
            if system2_mode == "auto":
                if q_type_hint == "hard":
                    system2_draft_limit = 1700
                elif q_type_hint == "medium":
                    system2_draft_limit = 1400
                else:
                    system2_draft_limit = 1200
            elif q_type_hint == "medium":
                system2_draft_limit = 2200
            payload["draft"] = _trim_system2_draft(
                draft,
                limit=system2_draft_limit,
            )
            payload["system2"] = True
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
        if default_mode_reflections and not suppress_default_mode_reflections:
            payload["default_mode_reflections"] = [
                f"{ref.theme} (confidence {float(ref.confidence):.2f})"
                for ref in default_mode_reflections
            ]

        budget_norm = str(payload.get("budget") or "small").strip().lower()
        timeout_scale = 60 if system2_active else 80
        timeout_floor = 6000.0
        if system2_active:
            if q_type_hint == "hard":
                timeout_scale = 56
                timeout_floor = 4200.0
            elif q_type_hint == "medium":
                timeout_scale = 48
                timeout_floor = 3600.0
            else:
                timeout_scale = 42
                timeout_floor = 3200.0
            if system2_mode == "auto":
                timeout_scale = max(
                    32,
                    int(round(float(timeout_scale) * 0.78)),
                )
                timeout_floor = max(2200.0, timeout_floor - 900.0)
        if budget_norm == "large" and not system2_active:
            timeout_scale = 95
        timeout_ms_base = int(
            min(
                float(self.default_timeout_ms),
                max(timeout_floor, float(decision.slot_ms) * float(timeout_scale)),
            )
        )
        timeout_multiplier = _env_float(
            "DUALBRAIN_TIMEOUT_MULTIPLIER",
            1.0,
            minimum=0.25,
            maximum=4.0,
        )
        if system2_active:
            timeout_multiplier *= _env_float(
                "DUALBRAIN_SYSTEM2_TIMEOUT_MULTIPLIER",
                1.0,
                minimum=0.25,
                maximum=4.0,
            )
        timeout_multiplier = max(0.25, min(8.0, float(timeout_multiplier)))
        timeout_max_ms = _env_int(
            "DUALBRAIN_TIMEOUT_MAX_MS",
            int(round(float(self.default_timeout_ms) * 4.0)),
            minimum=int(self.default_timeout_ms),
            maximum=240000,
        )
        timeout_ms = int(
            min(
                float(timeout_max_ms),
                float(timeout_ms_base) * float(timeout_multiplier),
            )
        )

        request_meta = {
            "temperature": round(decision.temperature, 3),
            "budget": payload["budget"],
            "slot_ms": decision.slot_ms,
            "timeout_ms": timeout_ms,
            "hemisphere_mode": hemisphere_mode,
            "hemisphere_bias": round(hemisphere_bias, 4),
            "focus_keywords": len(payload.get("focus_keywords", [])),
            "unconscious_hints": len(payload.get("unconscious_hints", [])),
            "psychoid_bias": bool(payload.get("psychoid_attention_bias")),
            "default_mode_refs": len(payload.get("default_mode_reflections", [])),
        }
        return ConsultRequestPlan(
            payload=payload,
            request_meta=request_meta,
            timeout_ms=timeout_ms,
            timeout_ms_base=timeout_ms_base,
            timeout_multiplier=float(timeout_multiplier),
            timeout_max_ms=timeout_max_ms,
            system2_draft_limit=system2_draft_limit,
            q_type_hint=q_type_hint,
        )

    async def _run_right_consult_stage(
        self,
        *,
        question: str,
        decision: DecisionOutcome,
        draft: str,
        final_answer: str,
        context: str,
        consult_plan: ConsultRequestPlan,
        system2_active: bool,
        system2_low_signal_filter: bool,
        psychoid_projection: PsychoidAttentionProjection | None,
        ask_callosum_with_timeout: Callable[..., Any],
        phase_add: Callable[[str, float], None],
    ) -> RightConsultStageResult:
        payload = consult_plan.payload
        consult_started = time.perf_counter()
        response = await ask_callosum_with_timeout(
            payload,
            timeout_ms=consult_plan.timeout_ms,
        )
        response_source = "callosum"
        detail_notes: Optional[str]
        critic_verdict: Optional[str] = None
        critic_issues: List[str] = []
        critic_fixes: List[str] = []
        system2_initial_issue_count = 0
        if system2_active:
            critic_verdict = str(response.get("verdict") or "").strip().lower() or None
            critic_issues = _filter_system2_issues(
                _normalise_issue_list(response.get("issues"), limit=12),
                question=question,
                filter_enabled=system2_low_signal_filter,
                keep_at_least=1,
            )
            critic_fixes = _normalise_issue_list(response.get("fixes"), limit=12)
            decision.state["right_role"] = "critic"
            critic_kind = response.get("critic_kind")
            if critic_kind:
                decision.state["critic_kind"] = str(critic_kind)
            if critic_verdict:
                decision.state["critic_verdict"] = critic_verdict
            if critic_issues:
                decision.state["critic_issues"] = critic_issues
            if critic_fixes:
                decision.state["critic_fixes"] = critic_fixes
            detail_notes = response.get("critic_sum")
            system2_initial_issue_count = len(critic_issues)
        else:
            detail_notes = response.get("notes_sum")

        call_error = bool(response.get("error"))
        call_confidence = float(response.get("confidence_r", 0.0) or 0.0)
        fallback_used = False
        fallback_confidence = 0.0
        fallback_error: Optional[str] = None
        success = False
        if call_error or not detail_notes:
            response_source = "right_model_fallback"
            fallback_used = True
            try:
                if system2_active:
                    fallback = await self.right_model.criticise_reasoning(
                        decision.qid,
                        question,
                        payload.get("draft") or draft,
                        temperature=max(0.1, min(0.35, decision.temperature)),
                        context=context,
                        psychoid_projection=(
                            psychoid_projection.to_payload()
                            if psychoid_projection
                            else None
                        ),
                    )
                else:
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
                fallback_error = str(exc)
                final_answer = draft
            else:
                if system2_active:
                    decision.state["right_role"] = "critic"
                    critic_verdict = str(fallback.get("verdict") or "").strip().lower() or None
                    critic_kind = fallback.get("critic_kind")
                    if critic_kind:
                        decision.state["critic_kind"] = str(critic_kind)
                    if critic_verdict:
                        decision.state["critic_verdict"] = critic_verdict
                    critic_issues = _filter_system2_issues(
                        _normalise_issue_list(
                            fallback.get("issues"),
                            limit=12,
                        ),
                        question=question,
                        filter_enabled=system2_low_signal_filter,
                        keep_at_least=1,
                    )
                    critic_fixes = _normalise_issue_list(
                        fallback.get("fixes"),
                        limit=12,
                    )
                    if critic_issues:
                        decision.state["critic_issues"] = critic_issues
                    if critic_fixes:
                        decision.state["critic_fixes"] = critic_fixes
                    detail_notes = fallback.get("critic_sum")
                    system2_initial_issue_count = len(critic_issues)
                else:
                    detail_notes = fallback.get("notes_sum")
                fallback_confidence = float(
                    fallback.get("confidence_r", 0.0) or 0.0
                )
                decision.state["right_conf"] = fallback_confidence
                success = True
        else:
            decision.state["right_conf"] = call_confidence
            success = True
        phase_add("right_consult", consult_started)

        return RightConsultStageResult(
            detail_notes=detail_notes,
            response_source=response_source,
            success=success,
            final_answer=final_answer,
            call_error=call_error,
            call_confidence=call_confidence,
            fallback_used=fallback_used,
            fallback_confidence=fallback_confidence,
            fallback_error=fallback_error,
            critic_verdict=critic_verdict,
            critic_issues=critic_issues,
            critic_fixes=critic_fixes,
            system2_initial_issue_count=system2_initial_issue_count,
        )

    async def _await_executive_advice(
        self,
        *,
        decision: DecisionOutcome,
        executive_task: asyncio.Task[ExecutiveAdvice] | None,
        executive_mode: str,
        phase_add: Callable[[str, float], None],
    ) -> ExecutiveAdviceStageResult:
        if executive_task is None:
            return ExecutiveAdviceStageResult(
                executive_payload=None,
                executive_directives=None,
            )

        timeout_ms = max(1500, min(9000, decision.slot_ms * 12))
        if executive_mode == "observe":
            timeout_ms = max(1200, min(4500, decision.slot_ms * 8))
        try:
            executive_wait_started = time.perf_counter()
            advice = await asyncio.wait_for(executive_task, timeout=timeout_ms / 1000.0)
            phase_add("executive", executive_wait_started)
        except asyncio.TimeoutError:
            phase_add("executive", executive_wait_started)
            advice = None
        except asyncio.CancelledError:
            advice = None
        except Exception:  # pragma: no cover - defensive guard
            advice = None

        if advice is None:
            return ExecutiveAdviceStageResult(
                executive_payload=None,
                executive_directives=None,
            )

        executive_payload = advice.to_payload()
        directives = advice.directives
        executive_directives = dict(directives) if isinstance(directives, dict) else None
        self.telemetry.log(
            "executive_reasoner",
            qid=decision.qid,
            confidence=float(advice.confidence),
            latency_ms=float(advice.latency_ms),
            source=str(advice.source),
        )
        return ExecutiveAdviceStageResult(
            executive_payload=executive_payload,
            executive_directives=executive_directives,
        )

    async def _run_consult_skipped_stage(
        self,
        *,
        question: str,
        decision: DecisionOutcome,
        draft: str,
        leading: str,
        explicit_leading: bool,
        right_lead_notes: Optional[str],
        vision_images: Optional[Sequence[Dict[str, Any]]],
        executive_task: asyncio.Task[ExecutiveAdvice] | None,
        executive_mode: str,
        emit_reset_if_needed: Callable[[], Any],
        delta_cb: Optional[Callable[[str], Any]],
        stream_final_only: bool,
        phase_add: Callable[[str, float], None],
    ) -> ConsultSkippedStageResult:
        retained = "left_draft"
        final_answer = draft
        response_source = "left_only"
        if (
            explicit_leading
            and right_lead_notes
            and leading == "right"
            and not vision_images
        ):
            retained = "right_preview"
            final_answer = right_lead_notes
            response_source = "lead"
            if _looks_like_coaching_notes(question, final_answer):
                retained = "left_draft"
                final_answer = draft
                response_source = "left_only"

        steps = [
            InnerDialogueStep(
                phase="consult_skipped",
                role="left",
                content=(
                    "Policy retained right-brain preview"
                    if retained == "right_preview"
                    else "Policy retained left-brain draft"
                ),
                metadata={"reason": f"policy_action_0_{retained}"},
            )
        ]

        executive_result = await self._await_executive_advice(
            decision=decision,
            executive_task=executive_task,
            executive_mode=executive_mode,
            phase_add=phase_add,
        )
        executive_payload = executive_result.executive_payload
        executive_directives = executive_result.executive_directives

        can_polish = bool(
            executive_mode == "polish"
            and bool(executive_directives)
            and getattr(self.left_model, "uses_external_llm", False)
        )
        if (
            can_polish
            and executive_payload
            and float(executive_payload.get("confidence") or 0.0) < 0.35
        ):
            can_polish = False

        integration_parts: List[str] = []
        if executive_payload and executive_mode == "assist":
            mix_in = str(executive_payload.get("mix_in") or "").strip()
            if mix_in and not _looks_like_coaching_notes(question, mix_in):
                if not _detail_notes_redundant(final_answer, mix_in):
                    integration_parts.append("Executive mix-in (user-facing):\n" + mix_in)
        if can_polish and executive_directives:
            try:
                directives_json = json.dumps(executive_directives, ensure_ascii=False)
            except Exception:
                directives_json = str(executive_directives)
            integration_parts.append(
                "Executive directives (do not output directly):\n" + directives_json
            )
        if integration_parts:
            await emit_reset_if_needed()
            integration_started = time.perf_counter()
            final_answer = await self.left_model.integrate_info_async(
                question=question,
                draft=final_answer,
                info="\n\n".join(integration_parts),
                temperature=max(0.2, min(0.6, decision.temperature)),
                on_delta=None if stream_final_only else delta_cb,
            )
            phase_add("integration", integration_started)

        return ConsultSkippedStageResult(
            final_answer=final_answer,
            response_source=response_source,
            success=True,
            executive_payload=executive_payload,
            executive_directives=executive_directives,
            steps=steps,
        )

    async def _run_system2_refinement_stage(
        self,
        *,
        question: str,
        decision: DecisionOutcome,
        draft: str,
        final_answer: str,
        payload: Dict[str, Any],
        context: str,
        psychoid_projection: PsychoidAttentionProjection | None,
        system2_mode: str,
        system2_round_target: int,
        system2_draft_limit: int,
        timeout_ms_base: int,
        timeout_multiplier: float,
        timeout_max_ms: int,
        system2_priority: str,
        system2_low_signal_filter: bool,
        system2_followup_new_issue_cap: int,
        system2_followup_new_issue_min_score: float,
        system2_followup_min_overlap: float,
        system2_followup_max_remaining: int,
        system2_followup_min_progress: int,
        critic_issues: List[str],
        system2_initial_issue_count: int,
        critic_needs_revision: bool,
        system2_critic_unhealthy: bool,
        system2_critic_unhealthy_reason: Optional[str],
        ask_callosum_with_timeout: Callable[..., Any],
        emit_reset_if_needed: Callable[[], Any],
        delta_cb: Optional[Callable[[str], Any]],
        stream_final_only: bool,
        phase_add: Callable[[str, float], None],
    ) -> System2RefinementResult:
        system2_rounds_completed = 1
        system2_final_issue_count = len(critic_issues)
        system2_resolved = False if system2_critic_unhealthy else not critic_needs_revision
        system2_followup_revision = False
        system2_followup_new_issues: List[str] = []
        system2_followup_verdict: Optional[str] = None

        should_verify = bool(
            critic_needs_revision
            and system2_round_target >= 2
            and final_answer
            and final_answer.strip()
        )
        if should_verify:
            verify_draft_limit = max(
                1200,
                min(2400, int(system2_draft_limit) + 300),
            )
            verify_payload = dict(payload)
            verify_payload["type"] = "ASK_CRITIC"
            verify_payload["qid"] = decision.qid
            verify_payload["system2"] = True
            verify_payload["round"] = 2
            verify_payload["draft"] = _trim_system2_draft(
                final_answer,
                limit=verify_draft_limit,
            )
            verify_payload["draft_sum"] = verify_payload["draft"][:280]

            verify_timeout_floor = 2600.0 if system2_mode == "auto" else 3200.0
            verify_timeout_ms_base = int(
                min(
                    float(self.default_timeout_ms),
                    max(
                        verify_timeout_floor,
                        float(timeout_ms_base) * 0.7,
                    ),
                )
            )
            verify_timeout_ms = int(
                min(
                    float(timeout_max_ms),
                    float(verify_timeout_ms_base) * float(timeout_multiplier),
                )
            )
            verify_call_error = False
            verify_fallback_used = False
            verify_response: Dict[str, Any] = {}
            try:
                verify_response = await ask_callosum_with_timeout(
                    verify_payload,
                    timeout_ms=verify_timeout_ms,
                )
            except Exception:
                verify_call_error = True
                verify_response = {}

            verify_verdict = str(
                verify_response.get("verdict") or ""
            ).strip().lower()
            verify_issues = _filter_system2_issues(
                _normalise_issue_list(
                    verify_response.get("issues"),
                    limit=12,
                ),
                question=question,
                filter_enabled=system2_low_signal_filter,
            )
            verify_detail = str(
                verify_response.get("critic_sum") or ""
            ).strip()
            if verify_call_error or bool(verify_response.get("error")):
                verify_fallback_used = True
            if verify_fallback_used or (
                not verify_detail
                and verify_verdict not in {"ok", "issues"}
            ):
                try:
                    verify_fallback = await self.right_model.criticise_reasoning(
                        decision.qid,
                        question,
                        verify_payload.get("draft") or final_answer,
                        temperature=max(0.1, min(0.3, decision.temperature)),
                        context=context,
                        psychoid_projection=(
                            psychoid_projection.to_payload()
                            if psychoid_projection
                            else None
                        ),
                    )
                except Exception:
                    verify_fallback = {}
                else:
                    verify_verdict = str(
                        verify_fallback.get("verdict") or ""
                    ).strip().lower()
                    verify_issues = _filter_system2_issues(
                        _normalise_issue_list(
                            verify_fallback.get("issues"),
                            limit=12,
                        ),
                        question=question,
                        filter_enabled=system2_low_signal_filter,
                    )
                    verify_detail = str(
                        verify_fallback.get("critic_sum") or ""
                    ).strip()

            system2_rounds_completed = max(system2_rounds_completed, 2)
            system2_followup_verdict = verify_verdict or None
            followup_new_issues_raw = _novel_issue_items(
                verify_issues,
                critic_issues,
            )
            if system2_low_signal_filter:
                system2_followup_new_issues = _prioritise_issue_list(
                    followup_new_issues_raw,
                    question=question,
                    limit=max(1, int(system2_followup_new_issue_cap))
                    if int(system2_followup_new_issue_cap) > 0
                    else 1,
                    min_score=float(system2_followup_new_issue_min_score),
                    keep_at_least=0,
                )
                if int(system2_followup_new_issue_cap) <= 0:
                    system2_followup_new_issues = []
            else:
                if int(system2_followup_new_issue_cap) <= 0:
                    system2_followup_new_issues = []
                else:
                    system2_followup_new_issues = followup_new_issues_raw[
                        : int(system2_followup_new_issue_cap)
                    ]
            if system2_followup_new_issues and critic_issues:
                system2_followup_new_issues = [
                    item
                    for item in system2_followup_new_issues
                    if _issue_overlap_with_previous(item, critic_issues)
                    >= float(system2_followup_min_overlap)
                ]
            verify_issue_count_raw = len(verify_issues)
            verify_issue_count_calibrated = verify_issue_count_raw
            if critic_issues and verify_issue_count_raw > len(critic_issues):
                max_growth = len(system2_followup_new_issues)
                verify_issue_count_calibrated = min(
                    verify_issue_count_raw,
                    len(critic_issues) + max_growth,
                )

            has_verify_signal = bool(
                verify_verdict in {"ok", "issues"}
                or verify_issues
                or verify_detail
            )
            if has_verify_signal:
                system2_final_issue_count = verify_issue_count_calibrated
                system2_resolved = (
                    verify_verdict == "ok"
                    or system2_final_issue_count == 0
                )
            else:
                system2_final_issue_count = len(critic_issues)
                system2_resolved = False
            decision.state["system2_issue_count_verify_raw"] = int(
                verify_issue_count_raw
            )
            decision.state["system2_issue_count_verify_calibrated"] = int(
                verify_issue_count_calibrated
            )
            followup_progress = max(
                0,
                int(system2_initial_issue_count)
                - int(verify_issue_count_calibrated),
            )
            followup_eligible = bool(
                int(verify_issue_count_calibrated) > 0
                and followup_progress >= int(system2_followup_min_progress)
                and int(verify_issue_count_calibrated)
                <= int(system2_followup_max_remaining)
            )
            decision.state["system2_followup_progress"] = int(
                followup_progress
            )
            decision.state["system2_followup_eligible"] = bool(
                followup_eligible
            )

            followup_focus_issues: List[str] = []
            followup_instruction = ""
            if (
                verify_detail
                and not system2_resolved
                and system2_round_target >= 3
                and followup_eligible
            ):
                if system2_followup_new_issues:
                    followup_focus_issues = list(system2_followup_new_issues[:8])
                    followup_instruction = (
                        "Apply only the newly discovered issues and preserve already-correct parts."
                    )
                elif verify_issues and system2_initial_issue_count >= 2:
                    followup_focus_issues = list(verify_issues[:8])
                    followup_instruction = (
                        "Apply minimal edits to resolve remaining unresolved issues without broad rewrites."
                    )

            if followup_focus_issues and verify_detail:
                focus_block = "\n".join(
                    f"- {item}" for item in followup_focus_issues
                )
                followup_info = (
                    "Reasoning critic follow-up (internal; do not output directly).\n"
                    f"{followup_instruction}\n"
                    "Issue focus list:\n"
                    f"{focus_block}\n\n"
                    "Critic details:\n"
                    f"{verify_detail}"
                )
                await emit_reset_if_needed()
                integration_started = time.perf_counter()
                followup_temperature = max(
                    0.15, min(0.32, float(decision.temperature))
                )
                final_answer = await self.left_model.integrate_info_async(
                    question=question,
                    draft=final_answer,
                    info=followup_info,
                    temperature=followup_temperature,
                    on_delta=None if stream_final_only else delta_cb,
                )
                phase_add("integration", integration_started)
                system2_followup_revision = True
                system2_resolved = False

            should_round3_verify = bool(
                system2_round_target >= 3
                and final_answer
                and final_answer.strip()
                and not system2_resolved
                and bool(system2_followup_revision)
            )
            if should_round3_verify:
                round3_payload = dict(payload)
                round3_payload["type"] = "ASK_CRITIC"
                round3_payload["qid"] = decision.qid
                round3_payload["system2"] = True
                round3_payload["round"] = 3
                round3_payload["draft"] = _trim_system2_draft(
                    final_answer,
                    limit=verify_draft_limit,
                )
                round3_payload["draft_sum"] = round3_payload["draft"][:280]

                round3_timeout_floor = (
                    2200.0 if system2_mode == "auto" else 2600.0
                )
                round3_timeout_ms_base = int(
                    min(
                        float(self.default_timeout_ms),
                        max(
                            round3_timeout_floor,
                            float(verify_timeout_ms_base) * 0.65,
                        ),
                    )
                )
                round3_timeout_ms = int(
                    min(
                        float(timeout_max_ms),
                        float(round3_timeout_ms_base) * float(timeout_multiplier),
                    )
                )
                round3_call_error = False
                round3_fallback_used = False
                round3_response: Dict[str, Any] = {}
                try:
                    round3_response = await ask_callosum_with_timeout(
                        round3_payload,
                        timeout_ms=round3_timeout_ms,
                    )
                except Exception:
                    round3_call_error = True
                    round3_response = {}

                round3_verdict = str(
                    round3_response.get("verdict") or ""
                ).strip().lower()
                round3_issues = _filter_system2_issues(
                    _normalise_issue_list(
                        round3_response.get("issues"),
                        limit=12,
                    ),
                    question=question,
                    filter_enabled=system2_low_signal_filter,
                )
                round3_detail = str(
                    round3_response.get("critic_sum") or ""
                ).strip()
                if round3_call_error or bool(round3_response.get("error")):
                    round3_fallback_used = True
                if round3_fallback_used or (
                    not round3_detail
                    and round3_verdict not in {"ok", "issues"}
                ):
                    try:
                        round3_fallback = await self.right_model.criticise_reasoning(
                            decision.qid,
                            question,
                            round3_payload.get("draft") or final_answer,
                            temperature=max(0.1, min(0.25, decision.temperature)),
                            context=context,
                            psychoid_projection=(
                                psychoid_projection.to_payload()
                                if psychoid_projection
                                else None
                            ),
                        )
                    except Exception:
                        round3_fallback = {}
                    else:
                        round3_verdict = str(
                            round3_fallback.get("verdict") or ""
                        ).strip().lower()
                        round3_issues = _filter_system2_issues(
                            _normalise_issue_list(
                                round3_fallback.get("issues"),
                                limit=12,
                            ),
                            question=question,
                            filter_enabled=system2_low_signal_filter,
                        )
                        round3_detail = str(
                            round3_fallback.get("critic_sum") or ""
                        ).strip()

                round3_new_issues = _novel_issue_items(
                    round3_issues,
                    verify_issues,
                )
                if system2_low_signal_filter:
                    round3_new_issues = _prioritise_issue_list(
                        round3_new_issues,
                        question=question,
                        limit=max(1, int(system2_followup_new_issue_cap))
                        if int(system2_followup_new_issue_cap) > 0
                        else 1,
                        min_score=float(system2_followup_new_issue_min_score),
                        keep_at_least=0,
                    )
                    if int(system2_followup_new_issue_cap) <= 0:
                        round3_new_issues = []
                else:
                    if int(system2_followup_new_issue_cap) <= 0:
                        round3_new_issues = []
                    else:
                        round3_new_issues = round3_new_issues[
                            : int(system2_followup_new_issue_cap)
                        ]
                if round3_new_issues and verify_issues:
                    round3_new_issues = [
                        item
                        for item in round3_new_issues
                        if _issue_overlap_with_previous(item, verify_issues)
                        >= float(system2_followup_min_overlap)
                    ]
                round3_issue_count_raw = len(round3_issues)
                round3_issue_count_calibrated = round3_issue_count_raw
                if verify_issues and round3_issue_count_raw > len(verify_issues):
                    max_growth = len(round3_new_issues)
                    round3_issue_count_calibrated = min(
                        round3_issue_count_raw,
                        len(verify_issues) + max_growth,
                    )

                has_round3_signal = bool(
                    round3_verdict in {"ok", "issues"}
                    or round3_issues
                    or round3_detail
                )
                if has_round3_signal:
                    system2_final_issue_count = round3_issue_count_calibrated
                    system2_resolved = (
                        round3_verdict == "ok"
                        or system2_final_issue_count == 0
                    )
                    if round3_verdict:
                        system2_followup_verdict = round3_verdict
                system2_rounds_completed = max(system2_rounds_completed, 3)
                decision.state["system2_issue_count_round3_raw"] = int(
                    round3_issue_count_raw
                )
                decision.state["system2_issue_count_round3_calibrated"] = int(
                    round3_issue_count_calibrated
                )
                if round3_new_issues:
                    decision.state["system2_round3_new_issues"] = round3_new_issues
                if round3_verdict:
                    decision.state["system2_round3_verdict"] = round3_verdict

        decision.state["system2_rounds"] = int(system2_rounds_completed)
        decision.state["system2_issue_count_initial"] = int(
            system2_initial_issue_count
        )
        decision.state["system2_issue_count_final"] = int(
            system2_final_issue_count
        )
        decision.state["system2_resolved"] = bool(system2_resolved)
        if system2_followup_verdict:
            decision.state["system2_followup_verdict"] = (
                system2_followup_verdict
            )
        if system2_followup_new_issues:
            decision.state["system2_followup_new_issues"] = (
                system2_followup_new_issues
            )
        if system2_followup_revision:
            decision.state["system2_followup_revision"] = True
        try:
            self.telemetry.log(
                "system2_refinement",
                qid=decision.qid,
                rounds=int(system2_rounds_completed),
                round_target=int(system2_round_target),
                initial_issues=int(system2_initial_issue_count),
                final_issues=int(system2_final_issue_count),
                critic_kind=decision.state.get("critic_kind"),
                verify_issues_raw=decision.state.get(
                    "system2_issue_count_verify_raw"
                ),
                verify_issues_calibrated=decision.state.get(
                    "system2_issue_count_verify_calibrated"
                ),
                round3_issues_raw=decision.state.get(
                    "system2_issue_count_round3_raw"
                ),
                round3_issues_calibrated=decision.state.get(
                    "system2_issue_count_round3_calibrated"
                ),
                priority=system2_priority,
                low_signal_filter=bool(system2_low_signal_filter),
                followup_new_issue_cap=int(system2_followup_new_issue_cap),
                followup_new_issue_min_score=float(
                    system2_followup_new_issue_min_score
                ),
                followup_min_overlap=float(system2_followup_min_overlap),
                followup_max_remaining=int(system2_followup_max_remaining),
                followup_min_progress=int(system2_followup_min_progress),
                critic_unhealthy=bool(system2_critic_unhealthy),
                critic_unhealthy_reason=system2_critic_unhealthy_reason,
                round_target_requested=decision.state.get(
                    "system2_round_target_requested"
                ),
                resolved=bool(system2_resolved),
                followup_revision=bool(system2_followup_revision),
                followup_new_issues=list(system2_followup_new_issues),
                followup_verdict=system2_followup_verdict,
            )
        except Exception:  # pragma: no cover - telemetry best-effort
            pass

        try:
            pitfall_raw: List[str] = []
            for key in (
                "critic_issues",
                "system2_followup_new_issues",
                "system2_round3_new_issues",
            ):
                value = decision.state.get(key)
                if isinstance(value, list):
                    pitfall_raw.extend(str(item) for item in value if item)

            pitfall_scored = []
            seen_norm: set[str] = set()
            for item in pitfall_raw[:18]:
                text = str(item or "").strip()
                if not text:
                    continue
                if text.lstrip().lower().startswith("(fallback)"):
                    continue
                norm = _normalise_issue_text(text)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                score = _issue_signal_score(text, question=question)
                pitfall_scored.append((score, norm, text))

            pitfall_scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
            pitfall_patterns = [
                _truncate_text(row[2], limit=160)
                for row in pitfall_scored[:6]
                if row[0] > 0.0
            ]
            if pitfall_patterns:
                decision.state["system2_pitfall_patterns"] = pitfall_patterns
                counts = self.memory.get_kv("system2_pitfall_counts", {})
                examples = self.memory.get_kv("system2_pitfall_examples", {})
                if not isinstance(counts, dict):
                    counts = {}
                if not isinstance(examples, dict):
                    examples = {}
                for _, norm, original in pitfall_scored[:6]:
                    counts[norm] = int(counts.get(norm, 0) or 0) + 1
                    if norm not in examples:
                        examples[norm] = _truncate_text(original, limit=180)
                self.memory.put_kv("system2_pitfall_counts", counts)
                self.memory.put_kv("system2_pitfall_examples", examples)
                self.memory.put_kv("system2_pitfall_last", pitfall_patterns)
        except Exception:  # pragma: no cover - best-effort
            pass

        return System2RefinementResult(
            final_answer=final_answer,
            system2_round_target=system2_round_target,
            system2_rounds_completed=system2_rounds_completed,
            system2_final_issue_count=system2_final_issue_count,
            system2_resolved=system2_resolved,
            system2_followup_revision=system2_followup_revision,
            system2_followup_new_issues=system2_followup_new_issues,
            system2_followup_verdict=system2_followup_verdict,
            critic_needs_revision=critic_needs_revision,
            critic_unhealthy=system2_critic_unhealthy,
            critic_unhealthy_reason=system2_critic_unhealthy_reason,
        )

    def _coerce_director_max_chars(
        self,
        director_max_chars: int | None,
    ) -> int | None:
        if director_max_chars is None:
            return None
        try:
            max_chars = int(director_max_chars)
        except Exception:
            return None
        return max(80, min(2400, max_chars))

    def _apply_director_output_constraints(
        self,
        user_answer: str,
        *,
        director_max_chars: int | None,
        director_append_question: str | None,
    ) -> str:
        max_chars = self._coerce_director_max_chars(director_max_chars)
        if director_append_question and director_append_question not in user_answer:
            base = user_answer.strip()
            suffix = director_append_question.strip()
            sep = "\n\n" if base and suffix else ""
            if max_chars is not None and max_chars > 0:
                reserved = len(sep) + len(suffix)
                if reserved >= max_chars:
                    if len(suffix) > max_chars:
                        cutoff = max(0, max_chars - 3)
                        suffix = suffix[:cutoff].rstrip() + "..."
                    user_answer = suffix
                else:
                    allowed_base = max_chars - reserved
                    if len(base) > allowed_base:
                        cutoff = max(0, allowed_base - 3)
                        base = base[:cutoff].rstrip() + "..."
                    user_answer = f"{base}{sep}{suffix}".strip()
            else:
                user_answer = f"{base}{sep}{suffix}".strip()

        if max_chars is not None and len(user_answer) > max_chars:
            cutoff = max(0, max_chars - 3)
            user_answer = user_answer[:cutoff].rstrip() + "..."
        return user_answer

    def _assemble_final_answer_stage(
        self,
        *,
        final_answer: str,
        emit_debug_sections: bool,
        collaborative_lead: bool,
        leading: str,
        right_lead_notes: str | None,
        detail_notes: str | None,
        left_lead_preview: str | None,
        draft: str,
        director_max_chars: int | None,
        director_append_question: str | None,
    ) -> FinalAnswerStageResult:
        integrated_answer = final_answer
        user_answer = integrated_answer
        if not emit_debug_sections:
            user_answer = _sanitize_user_answer(user_answer)
            user_answer = self._apply_director_output_constraints(
                user_answer,
                director_max_chars=director_max_chars,
                director_append_question=director_append_question,
            )

        if emit_debug_sections:
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
        else:
            final_answer = user_answer

        return FinalAnswerStageResult(
            integrated_answer=integrated_answer,
            user_answer=user_answer,
            final_answer=final_answer,
        )

    def _run_metacognition_audit_stage(
        self,
        *,
        question: str,
        draft: str,
        user_answer: str,
        final_answer: str,
        focus_keywords: Sequence[str] | None,
        context_parts: Dict[str, str],
        is_trivial_chat: bool,
        emit_debug_sections: bool,
        director_max_chars: int | None,
        director_append_question: str | None,
        tags: set[str],
        success: bool,
        qid: str,
        phase_add: Callable[[str, float], None],
    ) -> AuditStageResult:
        metacognition_started = time.perf_counter()
        audit_result = self.auditor.check(
            user_answer,
            question=question,
            focus_keywords=tuple(focus_keywords or ()),
            working_memory_context=str(context_parts.get("working_memory") or ""),
            is_trivial_chat=bool(is_trivial_chat),
            allow_debug=bool(emit_debug_sections),
        )
        phase_add("metacognition", metacognition_started)
        metacognition = audit_result.get("metacognition")
        if isinstance(metacognition, dict):
            try:
                self.telemetry.log(
                    "metacognition",
                    qid=qid,
                    **metacognition,
                )
            except Exception:  # pragma: no cover - telemetry is best-effort
                pass
            flags = metacognition.get("flags")
            if isinstance(flags, list):
                for flag in flags:
                    if flag:
                        tags.add(f"metacognition_{flag}")
            action = metacognition.get("action")
            if action:
                tags.add(f"metacognition_action_{action}")

        revised_answer = audit_result.get("revised_answer")
        if isinstance(revised_answer, str) and revised_answer.strip():
            user_answer = revised_answer.strip()
            final_answer = user_answer

        # Preserve director-provided clarifying questions even when the
        # metacognition layer rewrites the answer (e.g., repetition cleanup).
        if (
            not emit_debug_sections
            and director_append_question
            and director_append_question not in (user_answer or "")
        ):
            constrained = self._apply_director_output_constraints(
                user_answer or "",
                director_max_chars=director_max_chars,
                director_append_question=director_append_question,
            )
            if constrained != user_answer:
                user_answer = constrained
                final_answer = user_answer

        if not audit_result.get("ok", True):
            user_answer = draft
            final_answer = user_answer
            success = False

        return AuditStageResult(
            user_answer=user_answer,
            final_answer=final_answer,
            audit_result=audit_result,
            tags=tags,
            success=success,
        )

    def _persist_memory_stage(
        self,
        *,
        question: str,
        user_answer: str,
        decision: DecisionOutcome,
        leading: str,
        collaboration_profile: CollaborationProfile,
        selection_reason: str,
        tags: set[str],
        steps_payload: List[Dict[str, Any]],
        hemisphere_mode: str,
        hemisphere_bias: float,
        collaborative_lead: bool,
        leading_style: str,
        reward: float,
        success: bool,
        salience_signal: SalienceSignal | None,
        focus: FocusSummary | None,
        focus_metric: float,
        schema_profile: SchemaProfile | None,
        affect: Dict[str, float],
        novelty: float,
        insula_state: InteroceptiveState | None,
        thalamic_relay: ThalamicRelay | None,
        predictive_frame: PredictionFrame | None,
        hemisphere_signal: HemisphericSignal,
        coherence_signal: CoherenceSignal | None,
        distortion_payload: Dict[str, object] | None,
    ) -> MemoryPersistenceStageResult:
        # Always store "clean" memory, even when the user requested debug/meta output.
        memory_question = _sanitize_user_answer(question) or str(question or "").strip()
        memory_answer = _sanitize_user_answer(user_answer) or str(user_answer or "").strip()
        episodic_total = 0
        hippocampal_rollup: Dict[str, float] | None = None
        hippocampal_lifecycle: Dict[str, float] | None = None
        hippocampal_forgetting: Dict[str, float] | None = None

        conflict_resolved = bool(
            decision.state.get("system2_resolved")
            or (
                isinstance(decision.state.get("cerebellum_micro"), dict)
                and decision.state.get("cerebellum_micro", {}).get("resolved")
            )
        )
        if self.hippocampus is not None:
            self.hippocampus.index_episode(
                decision.qid,
                memory_question,
                memory_answer,
                leading=leading,
                collaboration_strength=collaboration_profile.strength,
                selection_reason=selection_reason,
                tags=tags,
                metadata={
                    "hemisphere_mode": hemisphere_mode,
                    "hemisphere_bias": hemisphere_bias,
                    "collaborative": collaborative_lead,
                    "leading_style": leading_style,
                    "inner_dialogue_trace": steps_payload,
                    "inner_dialogue_steps": len(steps_payload),
                    **(
                        {"system2_pitfall_patterns": decision.state.get("system2_pitfall_patterns")}
                        if decision.state.get("system2_pitfall_patterns")
                        else {}
                    ),
                    **(
                        {"schema_profile": schema_profile.to_dict()}
                        if schema_profile is not None
                        else {}
                    ),
                    **(
                        {"distortion_report": distortion_payload}
                        if distortion_payload is not None
                        else {}
                    ),
                    "reward": reward,
                    "success": success,
                    "conflict_resolved": conflict_resolved,
                    "system2_used": bool(decision.state.get("system2_enabled")),
                    "salience_level": (
                        float(salience_signal.level)
                        if salience_signal is not None
                        else 0.0
                    ),
                },
            )
            hippocampal_lifecycle = self.hippocampus.consolidate_feedback(
                qid=decision.qid,
                reward=reward,
                success=success,
                conflict_resolved=conflict_resolved,
                system2_used=bool(decision.state.get("system2_enabled")),
                salience_level=(
                    float(salience_signal.level)
                    if salience_signal is not None
                    else 0.0
                ),
            )
            hippocampal_forgetting = self.hippocampus.forget_stale()
            episodic_total = len(self.hippocampus.episodes)
            hippocampal_rollup = self.hippocampus.collaboration_rollup()
            if hippocampal_lifecycle is not None:
                decision.state["hippocampal_lifecycle"] = hippocampal_lifecycle
            if hippocampal_forgetting is not None:
                decision.state["hippocampal_forgetting"] = hippocampal_forgetting

        if memory_question and memory_answer:
            self.memory.store(
                {"Q": memory_question, "A": memory_answer}, tags=tags, qid=decision.qid
            )
        if self.prefrontal_cortex is not None:
            self.prefrontal_cortex.remember_working_memory(
                question=memory_question,
                answer=memory_answer,
                qid=decision.qid,
            )

        architecture_path = self._compose_architecture_path(
            focus=focus,
            focus_metric=focus_metric,
            schema_profile=schema_profile,
            affect=affect,
            novelty=novelty,
            interoception=insula_state,
            salience_signal=salience_signal,
            thalamic_relay=thalamic_relay,
            predictive_frame=predictive_frame,
            hemisphere_signal=hemisphere_signal,
            collaboration_profile=collaboration_profile,
            decision=decision,
            leading=leading,
            collaborative=collaborative_lead,
            steps=steps_payload,
            coherence_signal=coherence_signal,
            distortion_payload=distortion_payload,
            tags=sorted(tags),
            hippocampal_rollup=hippocampal_rollup,
            success=success,
        )
        if architecture_path:
            decision.state["architecture_path"] = architecture_path
            self.telemetry.log(
                "architecture_path",
                qid=decision.qid,
                path=architecture_path,
            )
        if self.hippocampus is not None and self.hippocampus.episodes and architecture_path:
            self.hippocampus.episodes[-1].annotations["architecture_path"] = architecture_path

        return MemoryPersistenceStageResult(
            memory_question=memory_question,
            memory_answer=memory_answer,
            episodic_total=episodic_total,
            hippocampal_rollup=hippocampal_rollup,
            hippocampal_lifecycle=hippocampal_lifecycle,
            hippocampal_forgetting=hippocampal_forgetting,
            architecture_path=architecture_path,
        )

    async def _run_post_turn_observer_stage(
        self,
        *,
        question: str,
        context: str,
        focus: FocusSummary | None,
        focus_keywords: Sequence[str] | None,
        user_answer: str,
        decision: DecisionOutcome,
        director_payload: Dict[str, Any] | None,
        use_metrics_observer: bool,
        architecture_path: List[Dict[str, Any]],
        coherence_signal: CoherenceSignal | None,
        leading: str,
        collaborative_lead: bool,
        selection_reason: str,
        phase_add: Callable[[str, float], None],
    ) -> PostTurnObserverStageResult:
        executive_observer_payload: Dict[str, Any] | None = director_payload
        if self.executive_model is None or not use_metrics_observer:
            return PostTurnObserverStageResult(
                executive_observer_payload=executive_observer_payload
            )

        active_modules: List[str] = []
        if architecture_path:
            mods = set()
            for stage in architecture_path:
                if not isinstance(stage, dict):
                    continue
                stage_mods = stage.get("modules")
                if not isinstance(stage_mods, list):
                    continue
                for mod in stage_mods:
                    if mod:
                        mods.add(str(mod))
            active_modules = sorted(mods)

        sections: List[str] = []
        for marker in ("[Working memory]", "[Schema memory]", "[Hippocampal recall]"):
            if marker in (context or ""):
                sections.append(marker.strip("[]"))
        if "[Hippocampal replay]" in (context or ""):
            sections.append("Hippocampal replay")

        coh_combined = None
        coh_tension = None
        coh_mode = None
        if coherence_signal is not None:
            coh_combined = float(coherence_signal.combined_score)
            coh_tension = float(coherence_signal.tension)
            coh_mode = str(coherence_signal.mode)

        focus_payload = ""
        if focus is not None:
            kws = list(focus.keywords[:4]) if focus.keywords else []
            focus_payload = (
                f"focus_keywords={kws} focus_relevance={focus.relevance:.2f} "
                f"hippocampal_overlap={focus.hippocampal_overlap:.2f}"
            ).strip()

        answer_snip = user_answer.replace("\n", " ").strip()
        if len(answer_snip) > 420:
            answer_snip = answer_snip[:420] + "..."

        ctx_snip = (context or "").replace("\n", " ").strip()
        if len(ctx_snip) > 420:
            ctx_snip = ctx_snip[:420] + "..."

        observer_context = (
            "Observer report (internal; never show to the user):\n"
            f"- final_answer_snippet: {answer_snip}\n"
            f"- policy: action={decision.action} temp={decision.temperature:.2f} slot_ms={decision.slot_ms}\n"
            f"- coherence: combined={coh_combined} tension={coh_tension} mode={coh_mode}\n"
            f"- leading={leading} collaborative={bool(collaborative_lead)} reason={selection_reason}\n"
            f"- memory_sections_in_context: {sections}\n"
            f"- active_modules: {active_modules[:18]}\n"
            f"- {focus_payload}\n"
            f"- context_snippet: {ctx_snip}\n"
        ).strip()

        try:
            observer_task = asyncio.create_task(
                self.executive_model.advise(
                    question=question,
                    context=observer_context,
                    focus_keywords=focus_keywords,
                )
            )
        except Exception:  # pragma: no cover - defensive guard
            observer_task = None

        if observer_task is not None:
            try:
                observer_wait_started = time.perf_counter()
                observer_advice = await asyncio.wait_for(observer_task, timeout=2.5)
                phase_add("executive", observer_wait_started)
            except asyncio.TimeoutError:
                phase_add("executive", observer_wait_started)
                observer_advice = None
            except asyncio.CancelledError:
                observer_advice = None
            except Exception:  # pragma: no cover - defensive guard
                observer_advice = None
            if observer_advice is not None:
                metrics_payload = observer_advice.to_payload()
                metrics_payload["observer_mode"] = "metrics"
                if executive_observer_payload and executive_observer_payload.get("memo"):
                    combined_directives: Dict[str, Any] = {
                        "director": executive_observer_payload.get("directives"),
                        "observer": metrics_payload.get("directives"),
                    }
                    combined_memo = (
                        f"{executive_observer_payload.get('memo')}\n\n"
                        "---\n(post-turn)\n"
                        f"{metrics_payload.get('memo')}"
                    ).strip()
                    executive_observer_payload = {
                        "memo": combined_memo[:2400],
                        "mix_in": "",
                        "directives": combined_directives,
                        "confidence": float(metrics_payload.get("confidence") or 0.0),
                        "latency_ms": float(metrics_payload.get("latency_ms") or 0.0),
                        "source": "combined",
                        "observer_mode": "both",
                    }
                else:
                    executive_observer_payload = metrics_payload
                self.telemetry.log(
                    "executive_observer",
                    qid=decision.qid,
                    confidence=float(observer_advice.confidence),
                    latency_ms=float(observer_advice.latency_ms),
                    source=str(observer_advice.source),
                    observer_mode="metrics",
                )

        return PostTurnObserverStageResult(
            executive_observer_payload=executive_observer_payload
        )

    async def process(
        self,
        question: str,
        *,
        leading_brain: Optional[str] = None,
        qid: Optional[str] = None,
        answer_mode: str = "plain",
        system2_mode: str = "auto",
        vision_images: Optional[Sequence[Dict[str, Any]]] = None,
        on_delta: Optional[Callable[[str], Any]] = None,
        on_reset: Optional[Callable[[], Any]] = None,
        executive_mode: str = "off",
        executive_observer_mode: str = "off",
    ) -> str:
        mode = str(answer_mode or "plain").strip().lower()
        emit_debug_sections = mode in {"debug", "annotated", "meta"}
        if self.coherence_resonator is not None:
            self.coherence_resonator.reset()
        requested_leading = (leading_brain or "").strip().lower()
        qid_value = str(qid) if qid else str(uuid.uuid4())
        system2_settings = self._build_system2_settings(system2_mode)
        system2_norm = system2_settings.mode
        system2_priority = system2_settings.priority
        precision_priority = system2_settings.precision_priority
        system2_low_signal_filter = system2_settings.low_signal_filter
        system2_followup_new_issue_cap = system2_settings.followup_new_issue_cap
        system2_followup_new_issue_min_score = (
            system2_settings.followup_new_issue_min_score
        )
        system2_followup_min_overlap = system2_settings.followup_min_overlap
        system2_followup_max_remaining = system2_settings.followup_max_remaining
        system2_followup_min_progress = system2_settings.followup_min_progress

        perception = self._run_perception_stage(
            question,
            qid=qid_value,
            precision_priority=precision_priority,
        )
        context = perception.context
        focus = perception.focus
        context_parts = perception.context_parts
        is_trivial_chat = perception.is_trivial_chat
        focus_metric = perception.focus_metric
        context_signal_len = perception.context_signal_len
        question_affect = perception.question_affect
        novelty = perception.novelty
        insula_state = perception.insula_state
        salience_signal = perception.salience_signal
        thalamic_relay = perception.thalamic_relay
        hemisphere_signal = perception.hemisphere_signal
        hemisphere_mode = perception.hemisphere_mode
        hemisphere_bias = perception.hemisphere_bias
        collaboration_profile = perception.collaboration_profile
        predictive_frame = perception.predictive_frame

        system2_status = self._resolve_system2_activation(
            question,
            perception=perception,
            settings=system2_settings,
        )
        system2_capable = system2_status.capable
        system2_active = system2_status.active
        system2_reason = system2_status.reason
        self._log_system2_mode(
            qid=qid_value,
            settings=system2_settings,
            status=system2_status,
        )

        schema_profile: Optional[SchemaProfile] = None
        distortion_payload: Optional[Dict[str, object]] = None
        leading_selection = self._select_leading_stage(
            requested_leading=requested_leading,
            system2_active=system2_active,
            system2_mode=system2_norm,
            hemisphere_mode=hemisphere_mode,
            hemisphere_bias=hemisphere_bias,
            collaboration_profile=collaboration_profile,
        )
        leading = leading_selection.leading
        auto_selected_leading = leading_selection.auto_selected
        collaborative_lead = leading_selection.collaborative
        selection_reason = leading_selection.selection_reason
        leading_style = leading_selection.leading_style
        self._last_leading_brain = leading
        if self.coherence_resonator is not None:
            self.coherence_resonator.retune(hemisphere_mode, intensity=hemisphere_bias)
        context_frame = self._build_context_frame(
            question=question,
            context_parts=context_parts,
        )
        context = context_frame.context
        focus = context_frame.focus
        focus_metric = context_frame.focus_metric
        focus_keywords = context_frame.focus_keywords
        hippocampal_density = (
            len(self.hippocampus.episodes) if self.hippocampus is not None else 0
        )

        observer_setup = self._prepare_observer_stage(
            question=question,
            context=context,
            focus=focus,
            focus_keywords=focus_keywords,
            context_parts=context_parts,
            is_trivial_chat=is_trivial_chat,
            hemisphere_mode=hemisphere_mode,
            hemisphere_bias=hemisphere_bias,
            leading=leading,
            collaborative_lead=collaborative_lead,
            collaboration_profile=collaboration_profile,
            executive_mode=executive_mode,
            executive_observer_mode=executive_observer_mode,
        )
        executive_mode_norm = observer_setup.executive_mode
        use_metrics_observer = observer_setup.use_metrics_observer
        executive_task = observer_setup.executive_task
        director_task = observer_setup.director_task

        executive_payload: Dict[str, Any] | None = None
        executive_directives: Dict[str, Any] | None = None
        director_payload: Dict[str, Any] | None = None
        director_control: Dict[str, Any] | None = None
        director_max_chars: Optional[int] = None
        director_append_question: Optional[str] = None
        inner_steps: List[InnerDialogueStep] = []
        stream_final_only = bool(on_delta and on_reset)
        emitted_any = False
        phase_latencies_ms: Dict[str, float] = {}

        def _phase_add(name: str, started_at: float) -> None:
            elapsed = (time.perf_counter() - started_at) * 1000.0
            phase_latencies_ms[name] = phase_latencies_ms.get(name, 0.0) + elapsed

        director_stage = await self._run_pre_draft_director_stage(
            question=question,
            qid=qid_value,
            is_trivial_chat=is_trivial_chat,
            context_frame=context_frame,
            context_parts=context_parts,
            director_task=director_task,
            phase_add=_phase_add,
        )
        context_frame = director_stage.context_frame
        context = context_frame.context
        focus = context_frame.focus
        focus_metric = context_frame.focus_metric
        focus_keywords = context_frame.focus_keywords
        director_task = director_stage.director_task
        director_payload = director_stage.director_payload
        director_control = director_stage.director_control

        async def _emit_delta(text: str) -> None:
            nonlocal emitted_any
            if not on_delta or not text:
                return
            emitted_any = True
            maybe = on_delta(text)
            if asyncio.iscoroutine(maybe):
                await maybe

        async def _emit_reset_if_needed() -> None:
            nonlocal emitted_any
            if not on_reset or not emitted_any:
                return
            maybe = on_reset()
            if asyncio.iscoroutine(maybe):
                await maybe
            emitted_any = False

        async def _ask_callosum_with_timeout(
            payload_obj: Dict[str, Any],
            *,
            timeout_ms: int,
        ) -> Dict[str, Any]:
            original_slot = getattr(self.callosum, "slot_ms", decision.slot_ms)
            hard_timeout_s = max(0.35, float(timeout_ms) / 1000.0 + 0.2)
            try:
                self.callosum.slot_ms = decision.slot_ms
                return await asyncio.wait_for(
                    self.callosum.ask_detail(payload_obj, timeout_ms=timeout_ms),
                    timeout=hard_timeout_s,
                )
            finally:
                self.callosum.slot_ms = original_slot

        delta_cb = _emit_delta if on_delta else None
        draft_stage = await self._run_left_draft_stage(
            question=question,
            context=context,
            focus=focus,
            leading=leading,
            collaborative_lead=collaborative_lead,
            system2_active=system2_active,
            vision_images=vision_images,
            delta_cb=delta_cb,
            stream_final_only=stream_final_only,
            emitted_any=emitted_any,
            focus_keywords=focus_keywords,
            focus_metric=focus_metric,
            hemisphere_mode=hemisphere_mode,
            hemisphere_bias=hemisphere_bias,
            phase_add=_phase_add,
        )
        draft = draft_stage.draft
        right_lead_notes = draft_stage.right_lead_notes
        left_lead_preview = draft_stage.left_lead_preview
        left_coherence = draft_stage.left_coherence
        confidence = draft_stage.confidence
        affect = draft_stage.affect
        novelty = draft_stage.novelty
        consult_bias = draft_stage.consult_bias
        inner_steps.extend(draft_stage.steps)

        right_coherence: Optional[HemisphericCoherence] = None
        coherence_signal: Optional[CoherenceSignal] = None
        explicit_leading = requested_leading in {"left", "right"}
        force_right_lead = requested_leading == "right"
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
            qid=qid_value,
        )
        if insula_state is not None:
            decision.state["interoception"] = insula_state.to_payload()
        if salience_signal is not None:
            decision.state["salience_signal"] = salience_signal.to_payload()
        if thalamic_relay is not None:
            decision.state["thalamic_relay"] = thalamic_relay.to_payload()
        if predictive_frame is not None:
            decision.state["predictive_coding"] = predictive_frame.to_payload()
            decision.state["network_state_profile"] = predictive_frame.networks.to_payload()
        routing_stage = await self._run_routing_stage(
            question=question,
            decision=decision,
            draft=draft,
            confidence=confidence,
            affect=affect,
            salience_signal=salience_signal,
            context_parts=context_parts,
            context_frame=context_frame,
            system2_mode=system2_norm,
            system2_priority=system2_priority,
            system2_active=system2_active,
            system2_capable=system2_capable,
            system2_reason=system2_reason,
            precision_priority=precision_priority,
            system2_low_signal_filter=system2_low_signal_filter,
            director_task=director_task,
            director_payload=director_payload,
            director_control=director_control,
            explicit_leading=explicit_leading,
            force_right_lead=force_right_lead,
            is_trivial_chat=is_trivial_chat,
            phase_add=_phase_add,
            context_signal_len=context_signal_len,
        )
        context_frame = routing_stage.context_frame
        context = context_frame.context
        focus = context_frame.focus
        focus_metric = context_frame.focus_metric
        focus_keywords = context_frame.focus_keywords
        decision = routing_stage.decision
        acc_signal_pre = routing_stage.acc_signal
        basal_signal = routing_stage.basal_signal
        system2_active = routing_stage.system2_active
        system2_reason = routing_stage.system2_reason
        system2_round_target = routing_stage.system2_round_target
        director_task = routing_stage.director_task
        director_payload = routing_stage.director_payload
        director_control = routing_stage.director_control
        director_max_chars = routing_stage.director_max_chars
        director_append_question = routing_stage.director_append_question
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
        if focus is not None:
            decision.state["prefrontal_keywords"] = list(focus.keywords)
            decision.state["prefrontal_relevance"] = focus.relevance
            decision.state["prefrontal_hippocampal_overlap"] = focus.hippocampal_overlap
        
        # Simulate neural impulse activity for the leading hemisphere
        neural_activity: Optional[Dict[str, Any]] = None
        if self.neural_integrator is not None:
            try:
                neural_activity = self._simulate_neural_activity(
                    hemisphere=leading,
                    affect=affect,
                    novelty=novelty,
                    focus_metric=focus_metric,
                    num_steps=10,
                )
                decision.state["neural_activity"] = neural_activity
                self.telemetry.log(
                    "neural_impulse_activity",
                    qid=decision.qid,
                    activity=neural_activity,
                )
            except Exception:  # pragma: no cover - defensive guard
                neural_activity = None
        
        unconscious_profile = None
        unconscious_summary: Optional[UnconsciousSummaryModel] = None
        default_mode_reflections: Optional[List[DefaultModeReflection]] = None
        suppress_default_mode_reflections = False
        task_positive_load = 0.0
        task_positive_mode = "idle"
        if salience_signal is not None and salience_signal.dominant_network in {
            "executive_control",
            "memory_recall",
        }:
            task_positive_load = max(task_positive_load, float(salience_signal.level))
            task_positive_mode = salience_signal.dominant_network
        if acc_signal_pre is not None:
            if float(acc_signal_pre.adaptation_signal) >= task_positive_load:
                task_positive_mode = str(acc_signal_pre.recommended_control or "conflict")
            task_positive_load = max(
                task_positive_load,
                float(acc_signal_pre.adaptation_signal),
            )
        if basal_signal is not None and (
            (
                salience_signal is not None
                and salience_signal.dominant_network in {"executive_control", "memory_recall"}
            )
            or (
                acc_signal_pre is not None
                and str(acc_signal_pre.recommended_control or "").lower()
                in {"consult", "system2"}
            )
        ):
            if float(basal_signal.gating_balance) >= task_positive_load:
                task_positive_mode = str(basal_signal.dominant_pathway or "basal")
            task_positive_load = max(
                task_positive_load,
                float(basal_signal.gating_balance),
            )
        if predictive_frame is not None:
            predictive_task_positive_load = float(
                predictive_frame.networks.task_positive_load
            )
            if predictive_task_positive_load >= task_positive_load:
                task_positive_mode = str(
                    predictive_frame.networks.task_positive_mode or "attention"
                )
            task_positive_load = max(task_positive_load, predictive_task_positive_load)
        decision.state["task_positive_network"] = {
            "load": float(task_positive_load),
            "mode": task_positive_mode,
        }
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
                        idea.to_payload() for idea in summary.emergent_ideas
                    ]
                if summary.stress_released:
                    decision.state["unconscious_stress_release"] = summary.stress_released
                psychoid_signal = summary.psychoid_signal
                if psychoid_signal:
                    decision.state["psychoid_bias"] = [
                        entry.to_payload() for entry in psychoid_signal.attention_bias
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
        # When executive focus stays low for multiple "easy" turns, treat DMN-style
        # reflections as internal-only (telemetry ok) to avoid "poetic inertia"
        # dominating the right-brain consult prompt.
        low_focus = bool(focus is not None and float(focus.relevance) < 0.02)
        low_arousal = float(affect.get("arousal", 0.0) or 0.0) <= 0.05
        is_easy = str(decision.state.get("q_type") or "").lower() == "easy"
        if low_focus and low_arousal and is_easy:
            self._default_mode_low_focus_streak += 1
        else:
            self._default_mode_low_focus_streak = 0
        suppress_default_mode_reflections = self._default_mode_low_focus_streak >= 2 or bool(
            thalamic_relay is not None and thalamic_relay.suppress_default_mode
        ) or bool(task_positive_load >= 0.62) or bool(
            predictive_frame is not None and predictive_frame.networks.suppress_default_mode
        )
        try:
            self.telemetry.log(
                "task_positive_network",
                qid=decision.qid,
                load=float(task_positive_load),
                mode=task_positive_mode,
                suppressed=bool(
                    task_positive_load >= 0.62
                    or (
                        predictive_frame is not None
                        and predictive_frame.networks.suppress_default_mode
                    )
                ),
            )
        except Exception:  # pragma: no cover - telemetry best-effort
            pass
        if suppress_default_mode_reflections:
            decision.state["default_mode_suppressed"] = True
        if task_positive_load >= 0.62:
            decision.state["task_positive_suppressed_default_mode"] = True
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
        response_source = ""
        success = False
        latency_ms = 0.0
        call_error = False
        call_confidence = 0.0
        fallback_confidence = 0.0
        fallback_error: Optional[str] = None
        start = time.perf_counter()
        try:
            if decision.action == 0:
                skipped_stage = await self._run_consult_skipped_stage(
                    question=question,
                    decision=decision,
                    draft=draft,
                    leading=leading,
                    explicit_leading=explicit_leading,
                    right_lead_notes=right_lead_notes,
                    vision_images=vision_images,
                    executive_task=executive_task,
                    executive_mode=executive_mode_norm,
                    emit_reset_if_needed=_emit_reset_if_needed,
                    delta_cb=delta_cb,
                    stream_final_only=stream_final_only,
                    phase_add=_phase_add,
                )
                final_answer = skipped_stage.final_answer
                response_source = skipped_stage.response_source
                success = skipped_stage.success
                executive_payload = skipped_stage.executive_payload
                executive_directives = skipped_stage.executive_directives
                inner_steps.extend(skipped_stage.steps)
            else:
                consult_plan = self._build_consult_request(
                    question=question,
                    decision=decision,
                    draft=draft,
                    context=context,
                    focus=focus,
                    system2_active=system2_active,
                    system2_mode=system2_norm,
                    hemisphere_mode=hemisphere_mode,
                    hemisphere_bias=hemisphere_bias,
                    unconscious_summary=unconscious_summary,
                    psychoid_signal=psychoid_signal,
                    psychoid_projection=psychoid_projection,
                    left_coherence=left_coherence,
                    default_mode_reflections=default_mode_reflections,
                    suppress_default_mode_reflections=suppress_default_mode_reflections,
                )
                payload = consult_plan.payload
                q_type_hint = consult_plan.q_type_hint
                system2_draft_limit = consult_plan.system2_draft_limit
                timeout_ms = consult_plan.timeout_ms
                timeout_ms_base = consult_plan.timeout_ms_base
                timeout_multiplier = consult_plan.timeout_multiplier
                timeout_max_ms = consult_plan.timeout_max_ms
                if timeout_multiplier != 1.0:
                    decision.state["timeout_multiplier"] = float(timeout_multiplier)
                    decision.state["timeout_max_ms"] = int(timeout_max_ms)
                    decision.state["timeout_ms_base"] = int(timeout_ms_base)
                inner_steps.append(
                    InnerDialogueStep(
                        phase="callosum_request",
                        role="coordinator",
                        content=_truncate_text(payload.get("draft_sum", ""), 120),
                        metadata=consult_plan.request_meta,
                    )
                )
                consult_stage = await self._run_right_consult_stage(
                    question=question,
                    decision=decision,
                    draft=draft,
                    final_answer=final_answer,
                    context=context,
                    consult_plan=consult_plan,
                    system2_active=system2_active,
                    system2_low_signal_filter=system2_low_signal_filter,
                    psychoid_projection=psychoid_projection,
                    ask_callosum_with_timeout=_ask_callosum_with_timeout,
                    phase_add=_phase_add,
                )
                detail_notes = consult_stage.detail_notes
                response_source = consult_stage.response_source
                success = consult_stage.success
                final_answer = consult_stage.final_answer
                call_error = consult_stage.call_error
                call_confidence = consult_stage.call_confidence
                fallback_used = consult_stage.fallback_used
                fallback_confidence = consult_stage.fallback_confidence
                fallback_error = consult_stage.fallback_error
                critic_verdict = consult_stage.critic_verdict
                critic_issues = consult_stage.critic_issues
                critic_fixes = consult_stage.critic_fixes
                system2_rounds_completed = 1 if system2_active else 0
                system2_initial_issue_count = consult_stage.system2_initial_issue_count
                system2_final_issue_count = 0
                system2_resolved = False
                system2_followup_revision = False
                system2_followup_new_issues: list[str] = []
                system2_followup_verdict: Optional[str] = None

                system2_critic_unhealthy = False
                system2_critic_unhealthy_reason: Optional[str] = None
                suppress_critic_revision = False
                if system2_active:
                    system2_critic_unhealthy_reason = _detect_system2_critic_unhealthy_reason(
                        critic_kind=str(decision.state.get("critic_kind") or ""),
                        issues=critic_issues,
                        critic_sum=str(detail_notes or ""),
                    )
                    system2_critic_unhealthy = bool(system2_critic_unhealthy_reason)
                    decision.state["system2_critic_unhealthy"] = bool(system2_critic_unhealthy)
                    if system2_critic_unhealthy:
                        suppress_critic_revision = True
                        previous_round_target = int(system2_round_target)
                        if previous_round_target > 1:
                            decision.state["system2_round_target_requested"] = previous_round_target
                        system2_round_target = 1
                        decision.state["system2_round_target"] = 1
                        decision.state["system2_critic_unhealthy_reason"] = (
                            system2_critic_unhealthy_reason
                        )
                        try:
                            self.telemetry.log(
                                "system2_critic_health",
                                qid=decision.qid,
                                healthy=False,
                                reason=system2_critic_unhealthy_reason,
                                critic_kind=decision.state.get("critic_kind"),
                                round_target=int(system2_round_target),
                            )
                        except Exception:  # pragma: no cover - telemetry best-effort
                            pass

                integrated_detail = detail_notes
                critic_needs_revision = False
                if system2_active:
                    critic_needs_revision = bool(critic_issues) and not suppress_critic_revision
                    if not critic_needs_revision:
                        integrated_detail = None
                elif detail_notes and _detail_notes_redundant(draft, detail_notes):
                    integrated_detail = None
                if integrated_detail and _looks_like_coaching_notes(question, integrated_detail):
                    integrated_detail = None

                executive_result = await self._await_executive_advice(
                    decision=decision,
                    executive_task=executive_task,
                    executive_mode=executive_mode_norm,
                    phase_add=_phase_add,
                )
                if executive_result.executive_payload is not None:
                    executive_payload = executive_result.executive_payload
                    executive_directives = executive_result.executive_directives

                integration_parts: List[str] = []
                has_right_material = False
                has_mix_in_material = False
                if integrated_detail:
                    if system2_active:
                        integration_parts.append(
                            _format_system2_revision_notes(
                                question=question,
                                issues=critic_issues,
                                fixes=critic_fixes,
                                critic_sum=str(integrated_detail),
                            )
                        )
                    else:
                        integration_parts.append(str(integrated_detail))
                    has_right_material = True
                elif right_lead_notes and not detail_notes:
                    if not _looks_like_coaching_notes(question, right_lead_notes):
                        integration_parts.append(str(right_lead_notes))
                        has_right_material = True

                can_polish = (
                    executive_mode_norm == "polish"
                    and bool(executive_directives)
                    and getattr(self.left_model, "uses_external_llm", False)
                )
                if can_polish:
                    if executive_payload and float(executive_payload.get("confidence") or 0.0) < 0.35:
                        can_polish = False

                if executive_payload and executive_mode_norm == "assist":
                    mix_in = str(executive_payload.get("mix_in") or "").strip()
                    if mix_in and not _looks_like_coaching_notes(question, mix_in):
                        if not _detail_notes_redundant(draft, mix_in):
                            integration_parts.append("Executive mix-in (user-facing):\n" + mix_in)
                            has_mix_in_material = True

                if can_polish and executive_directives:
                    try:
                        directives_json = json.dumps(executive_directives, ensure_ascii=False)
                    except Exception:
                        directives_json = str(executive_directives)
                    integration_parts.append(
                        "Executive directives (do not output directly):\n" + directives_json
                    )

                if has_right_material or has_mix_in_material or can_polish:
                    if integration_parts:
                        await _emit_reset_if_needed()
                        integration_started = time.perf_counter()
                        integration_temperature = max(0.2, min(0.6, decision.temperature))
                        if system2_active and integrated_detail:
                            integration_temperature = max(
                                0.15, min(0.35, float(decision.temperature))
                            )
                        final_answer = await self.left_model.integrate_info_async(
                            question=question,
                            draft=draft,
                            info="\n\n".join(integration_parts),
                            temperature=integration_temperature,
                            on_delta=None if stream_final_only else delta_cb,
                        )
                        _phase_add("integration", integration_started)
                if system2_active:
                    refinement = await self._run_system2_refinement_stage(
                        question=question,
                        decision=decision,
                        draft=draft,
                        final_answer=final_answer,
                        payload=payload,
                        context=context,
                        psychoid_projection=psychoid_projection,
                        system2_mode=system2_norm,
                        system2_round_target=system2_round_target,
                        system2_draft_limit=system2_draft_limit,
                        timeout_ms_base=timeout_ms_base,
                        timeout_multiplier=timeout_multiplier,
                        timeout_max_ms=timeout_max_ms,
                        system2_priority=system2_priority,
                        system2_low_signal_filter=system2_low_signal_filter,
                        system2_followup_new_issue_cap=system2_followup_new_issue_cap,
                        system2_followup_new_issue_min_score=system2_followup_new_issue_min_score,
                        system2_followup_min_overlap=system2_followup_min_overlap,
                        system2_followup_max_remaining=system2_followup_max_remaining,
                        system2_followup_min_progress=system2_followup_min_progress,
                        critic_issues=critic_issues,
                        system2_initial_issue_count=system2_initial_issue_count,
                        critic_needs_revision=critic_needs_revision,
                        system2_critic_unhealthy=system2_critic_unhealthy,
                        system2_critic_unhealthy_reason=system2_critic_unhealthy_reason,
                        ask_callosum_with_timeout=_ask_callosum_with_timeout,
                        emit_reset_if_needed=_emit_reset_if_needed,
                        delta_cb=delta_cb,
                        stream_final_only=stream_final_only,
                        phase_add=_phase_add,
                    )
                    final_answer = refinement.final_answer
                    system2_round_target = refinement.system2_round_target
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
                response_meta: Dict[str, Any] = {
                    "call_error": call_error,
                    "call_confidence": round(call_confidence, 4),
                    "fallback_used": fallback_used,
                    "fallback_confidence": round(fallback_confidence, 4),
                    "has_detail": bool(detail_notes),
                    "final_source": response_source,
                }
                if system2_active:
                    response_meta["system2"] = True
                    response_meta["critic_verdict"] = critic_verdict
                    response_meta["critic_issues"] = len(critic_issues)
                    response_meta["system2_rounds"] = int(
                        decision.state.get("system2_rounds") or 1
                    )
                    response_meta["system2_issue_initial"] = int(
                        decision.state["system2_issue_count_initial"]
                        if "system2_issue_count_initial" in decision.state
                        else len(critic_issues)
                    )
                    response_meta["system2_issue_final"] = int(
                        decision.state["system2_issue_count_final"]
                        if "system2_issue_count_final" in decision.state
                        else len(critic_issues)
                    )
                    response_meta["system2_resolved"] = bool(
                        decision.state.get("system2_resolved")
                    )
                if fallback_error:
                    response_meta["fallback_error"] = fallback_error
                if detail_notes:
                    response_meta["detail_length"] = len(detail_notes)
                if right_lead_notes and not detail_notes:
                    response_meta["used_preview"] = True
                if right_coherence is not None:
                    response_meta["right_coherence"] = round(
                        right_coherence.score(), 4
                    )
                inner_steps.append(
                    InnerDialogueStep(
                        phase="callosum_response",
                        role="right",
                        content=_truncate_text(detail_notes or right_lead_notes),
                        metadata=response_meta,
                    )
                )
        except asyncio.TimeoutError:
            final_answer = draft
            if system2_active:
                # Count the timed-out critic attempt as a completed round to keep
                # System2 observability stable (avoid ambiguous 0-round records).
                timeout_rounds = 1
                timeout_initial_issues = 0
                timeout_final_issues = 0
                timeout_resolved = False
                timeout_recovered = False
                timeout_recovery_source: Optional[str] = None
                timeout_followup_revision = False
                micro_result = None
                try:
                    micro_result = micro_criticise_reasoning(question, draft)
                except Exception:
                    micro_result = None

                if micro_result is not None:
                    timeout_recovered = True
                    timeout_recovery_source = f"micro:{micro_result.domain}"
                    timeout_rounds = 1
                    timeout_initial_issues = len(micro_result.issues)
                    timeout_final_issues = timeout_initial_issues
                    timeout_resolved = (
                        micro_result.verdict == "ok" or timeout_initial_issues == 0
                    )
                    decision.state["right_role"] = "critic"
                    decision.state["critic_kind"] = "micro_timeout_fallback"
                    decision.state["critic_verdict"] = micro_result.verdict
                    decision.state["critic_issues"] = list(micro_result.issues)
                    if micro_result.fixes:
                        decision.state["critic_fixes"] = list(micro_result.fixes)
                    decision.state["right_conf"] = float(
                        micro_result.confidence_r
                    )
                    decision.state["right_source"] = "micro_timeout_fallback"
                    if micro_result.issues:
                        try:
                            await _emit_reset_if_needed()
                            integration_started = time.perf_counter()
                            final_answer = await self.left_model.integrate_info_async(
                                question=question,
                                draft=draft,
                                info=(
                                    "Reasoning critic notes (internal; do not output directly). "
                                    "Revise the draft to address issues and improve correctness. "
                                    "Apply minimal precise edits and avoid introducing new assumptions:\n"
                                    + str(micro_result.critic_sum or "")
                                ),
                                temperature=max(0.2, min(0.6, decision.temperature)),
                                on_delta=None if stream_final_only else delta_cb,
                            )
                            _phase_add("integration", integration_started)
                            timeout_followup_revision = True
                            post_micro = micro_criticise_reasoning(
                                question,
                                final_answer,
                            )
                            if post_micro is not None:
                                timeout_final_issues = len(post_micro.issues)
                                timeout_resolved = (
                                    post_micro.verdict == "ok"
                                    or timeout_final_issues == 0
                                )
                        except Exception:
                            timeout_followup_revision = False
                    success = True
                    inner_steps.append(
                        InnerDialogueStep(
                            phase="system2_timeout_recovery",
                            role="critic",
                            content=_truncate_text(micro_result.critic_sum, 180),
                            metadata={
                                "source": timeout_recovery_source,
                                "initial_issues": timeout_initial_issues,
                                "final_issues": timeout_final_issues,
                                "resolved": timeout_resolved,
                                "revision": timeout_followup_revision,
                            },
                        )
                    )

                decision.state["system2_rounds"] = int(timeout_rounds)
                decision.state["system2_issue_count_initial"] = int(
                    timeout_initial_issues
                )
                decision.state["system2_issue_count_final"] = int(
                    timeout_final_issues
                )
                decision.state["system2_resolved"] = bool(timeout_resolved)
                decision.state["system2_timeout"] = True
                decision.state["system2_timeout_recovered"] = bool(timeout_recovered)
                if timeout_recovery_source:
                    decision.state["system2_timeout_recovery_source"] = (
                        timeout_recovery_source
                    )
                if timeout_followup_revision:
                    decision.state["system2_followup_revision"] = True
                try:
                    self.telemetry.log(
                        "system2_refinement",
                        qid=decision.qid,
                        rounds=int(timeout_rounds),
                        round_target=int(system2_round_target),
                        initial_issues=int(timeout_initial_issues),
                        final_issues=int(timeout_final_issues),
                        critic_kind=(
                            decision.state.get("critic_kind")
                            if timeout_recovered
                            else "timeout"
                        ),
                        low_signal_filter=bool(system2_low_signal_filter),
                        followup_new_issue_cap=int(system2_followup_new_issue_cap),
                        followup_new_issue_min_score=float(
                            system2_followup_new_issue_min_score
                        ),
                        followup_min_overlap=float(system2_followup_min_overlap),
                        followup_max_remaining=int(system2_followup_max_remaining),
                        followup_min_progress=int(system2_followup_min_progress),
                        resolved=bool(timeout_resolved),
                        followup_revision=bool(timeout_followup_revision),
                        followup_new_issues=[],
                        followup_verdict="timeout",
                        timeout=True,
                        timeout_recovered=bool(timeout_recovered),
                        timeout_recovery_source=timeout_recovery_source,
                    )
                except Exception:  # pragma: no cover - telemetry best-effort
                    pass
            inner_steps.append(
                InnerDialogueStep(
                    phase="callosum_timeout",
                    role="coordinator",
                    content="Right brain consult timed out",
                    metadata={
                        "timeout_ms": timeout_ms,
                        "slot_ms": decision.slot_ms,
                        "recovered": bool(
                            decision.state.get("system2_timeout_recovered", False)
                        ),
                    },
                )
            )
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.orchestrator.clear(decision.qid)

        if executive_payload is None and executive_task is not None:
            advice: ExecutiveAdvice | None = None
            if executive_task.done():
                try:
                    advice = executive_task.result()
                except asyncio.CancelledError:
                    advice = None
                except Exception:  # pragma: no cover - defensive guard
                    advice = None
            else:
                tail_timeout = 6.0 if executive_mode_norm in {"polish", "assist"} else 2.0
                try:
                    executive_wait_started = time.perf_counter()
                    advice = await asyncio.wait_for(executive_task, timeout=tail_timeout)
                    _phase_add("executive", executive_wait_started)
                except asyncio.CancelledError:
                    advice = None
                except Exception:
                    advice = None

            if advice is not None:
                executive_payload = advice.to_payload()
                directives = advice.directives
                if isinstance(directives, dict):
                    executive_directives = dict(directives)
                self.telemetry.log(
                    "executive_reasoner",
                    qid=decision.qid,
                    confidence=float(advice.confidence),
                    latency_ms=float(advice.latency_ms),
                    source=str(advice.source),
                )

        # ACC conflict monitoring + cerebellar micro-correction:
        # - only when System2 is inactive,
        # - and only in System2 auto mode (avoid surprising extra LLM passes when forced off),
        # - and only when the left model can do safe integration (external LLM available).
        if (
            not system2_active
            and system2_norm == "auto"
            and not is_trivial_chat
            and final_answer
            and str(final_answer).strip()
        ):
            acc_enabled = _env_flag("DUALBRAIN_ACC_MONITOR", True)
            cerebellum_enabled = _env_flag("DUALBRAIN_CEREBELLUM_MICRO_CORRECTION", True)
            micro_result = None
            if acc_enabled or cerebellum_enabled:
                try:
                    micro_result = micro_criticise_reasoning(question, final_answer)
                except Exception:
                    micro_result = None

            acc_signal = None
            if acc_enabled and self.anterior_cingulate_cortex is not None:
                try:
                    acc_signal = self.anterior_cingulate_cortex.monitor(
                        question=question,
                        draft=final_answer,
                        left_confidence=float(confidence or 0.0),
                        micro=micro_result,
                    )
                except Exception:  # pragma: no cover - best-effort
                    acc_signal = None
                if acc_signal is not None:
                    acc_payload = acc_signal.to_payload()
                    decision.state["acc_conflict"] = acc_payload
                    try:
                        self.telemetry.log(
                            "acc_conflict_monitor",
                            qid=decision.qid,
                            signal=acc_payload,
                        )
                    except Exception:  # pragma: no cover - telemetry best-effort
                        pass

            can_micro_correct = bool(
                cerebellum_enabled
                and micro_result is not None
                and getattr(self.left_model, "uses_external_llm", False)
                and self.cerebellum is not None
            )
            if can_micro_correct and micro_result is not None:
                cerebellar_forecast = self.cerebellum.forecast(
                    micro_result,
                    conflict=acc_signal,
                    system2_active=bool(system2_active),
                    consult_planned=bool(detail_notes or decision.action != 0),
                )
                decision.state["cerebellum_forecast"] = (
                    cerebellar_forecast.to_payload()
                )
                try:
                    self.telemetry.log(
                        "cerebellum_forward_model",
                        qid=decision.qid,
                        forecast=cerebellar_forecast.to_payload(),
                    )
                except Exception:  # pragma: no cover - telemetry best-effort
                    pass
                min_conf = _env_float(
                    "DUALBRAIN_CEREBELLUM_MICRO_MIN_CONFIDENCE",
                    float(getattr(self.cerebellum, "min_confidence", 0.88)),
                    minimum=0.0,
                    maximum=1.0,
                )
                max_issues = _env_int(
                    "DUALBRAIN_CEREBELLUM_MICRO_MAX_ISSUES",
                    int(getattr(self.cerebellum, "max_issues", 4)),
                    minimum=1,
                    maximum=12,
                )
                if (
                    micro_result.verdict == "issues"
                    and float(micro_result.confidence_r) >= float(min_conf)
                    and len(micro_result.issues) <= int(max_issues)
                    and cerebellar_forecast.recommended_path == "micro_correct"
                ):
                    correction_started = time.perf_counter()
                    cerebellum_applied = False
                    resolved = False
                    initial_issues = len(micro_result.issues)
                    final_issues = initial_issues
                    try:
                        await _emit_reset_if_needed()
                        revised = await self.left_model.integrate_info_async(
                            question=question,
                            draft=final_answer,
                            info=self.cerebellum.build_internal_notes(
                                micro_result,
                                forecast=cerebellar_forecast,
                            ),
                            temperature=max(
                                0.15, min(0.35, float(decision.temperature))
                            ),
                            on_delta=None if stream_final_only else delta_cb,
                        )
                    except Exception:  # pragma: no cover - defensive guard
                        revised = final_answer
                    else:
                        if revised and revised.strip() and revised != final_answer:
                            cerebellum_applied = True
                            final_answer = revised
                        try:
                            post_micro = micro_criticise_reasoning(question, final_answer)
                        except Exception:
                            post_micro = None
                        if post_micro is not None:
                            final_issues = len(post_micro.issues)
                            resolved = bool(
                                post_micro.verdict == "ok" or final_issues == 0
                            )
                    finally:
                        _phase_add("cerebellum", correction_started)

                    decision.state["cerebellum_micro"] = {
                        "applied": bool(cerebellum_applied),
                        "domain": str(micro_result.domain),
                        "initial_issues": int(initial_issues),
                        "final_issues": int(final_issues),
                        "resolved": bool(resolved),
                        "confidence": float(micro_result.confidence_r),
                        "forward_path": str(cerebellar_forecast.recommended_path),
                        "predicted_gain": float(cerebellar_forecast.predicted_gain),
                    }
                    try:
                        self.telemetry.log(
                            "cerebellum_micro_correction",
                            qid=decision.qid,
                            applied=bool(cerebellum_applied),
                            domain=str(micro_result.domain),
                            initial_issues=int(initial_issues),
                            final_issues=int(final_issues),
                            resolved=bool(resolved),
                            confidence=float(micro_result.confidence_r),
                            forward_path=str(cerebellar_forecast.recommended_path),
                            predicted_gain=float(cerebellar_forecast.predicted_gain),
                        )
                    except Exception:  # pragma: no cover - telemetry best-effort
                        pass
                    inner_steps.append(
                        InnerDialogueStep(
                            phase="cerebellum_micro",
                            role="cerebellum",
                            content=_truncate_text(micro_result.critic_sum, 180),
                            metadata={
                                "applied": bool(cerebellum_applied),
                                "domain": str(micro_result.domain),
                                "initial_issues": int(initial_issues),
                                "final_issues": int(final_issues),
                                "resolved": bool(resolved),
                                "confidence": float(micro_result.confidence_r),
                                "forward_path": str(cerebellar_forecast.recommended_path),
                                "predicted_gain": float(cerebellar_forecast.predicted_gain),
                            },
                        )
                    )

        answer_stage = self._assemble_final_answer_stage(
            final_answer=final_answer,
            emit_debug_sections=emit_debug_sections,
            collaborative_lead=collaborative_lead,
            leading=leading,
            right_lead_notes=right_lead_notes,
            detail_notes=detail_notes,
            left_lead_preview=left_lead_preview,
            draft=draft,
            director_max_chars=director_max_chars,
            director_append_question=director_append_question,
        )
        integrated_answer = answer_stage.integrated_answer
        user_answer = answer_stage.user_answer
        final_answer = answer_stage.final_answer

        semantic_tilt = self._evaluate_semantic_tilt(
            question=question,
            final_answer=user_answer,
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
                final_answer=user_answer,
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
                final_answer=user_answer,
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
                final_answer=user_answer,
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
                if emit_debug_sections:
                    final_answer = self.coherence_resonator.annotate_answer(
                        final_answer, coherence_signal
                    )
                coherence_tags = set(coherence_signal.tags())
                tags.update(coherence_tags)
                decision.state["coherence_tags"] = list(coherence_tags)
                if coherence_signal.distortions is not None:
                    distortion_payload = coherence_signal.distortions.to_payload()
                else:
                    distortion_payload = None

        inner_steps.append(
            InnerDialogueStep(
                phase="integration",
                role="integrator",
                content=_truncate_text(final_answer),
                metadata={
                    "leading": leading,
                    "collaborative": collaborative_lead,
                    "response_source": response_source or "left_only",
                    "success": success,
                    "latency_ms": round(latency_ms, 3),
                    "action": decision.action,
                    **(
                        {"coherence": round(coherence_signal.combined_score, 4)}
                        if coherence_signal is not None
                        else {}
                    ),
                    **(
                        {"right_coherence": round(right_coherence.score(), 4)}
                        if right_coherence is not None
                        else {}
                    ),
                },
            )
        )

        audit_stage = self._run_metacognition_audit_stage(
            question=question,
            draft=draft,
            user_answer=user_answer,
            final_answer=final_answer,
            focus_keywords=focus_keywords,
            context_parts=context_parts,
            is_trivial_chat=is_trivial_chat,
            emit_debug_sections=emit_debug_sections,
            director_max_chars=director_max_chars,
            director_append_question=director_append_question,
            tags=tags,
            success=success,
            qid=decision.qid,
            phase_add=_phase_add,
        )
        user_answer = audit_stage.user_answer
        final_answer = audit_stage.final_answer
        audit_result = audit_stage.audit_result
        tags = audit_stage.tags
        success = audit_stage.success

        if stream_final_only and delta_cb and not emitted_any and final_answer:
            chunk_size = 512
            for i in range(0, len(final_answer), chunk_size):
                await _emit_delta(final_answer[i : i + chunk_size])

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
            stress_value = float(unconscious_summary.stress_released or 0.0)
            if emit_debug_sections:
                if insights:
                    final_answer = f"{final_answer}\n\n[Unconscious Insight]\n" + "\n".join(insights)
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
            chain = psychoid_signal.signifier_chain
            if psychoid_projection:
                tags.add("psychoid_attention")
                bias_payload = psychoid_projection.to_payload()
                if emit_debug_sections:
                    final_answer = (
                        f"{final_answer}\n\n[Psychoid Attention Bias]\n"
                        f"- norm {bias_payload.get('norm', 0.0):.3f}"
                        f" | temperature {bias_payload.get('temperature', 0.0):.2f}"
                        f" | clamp {bias_payload.get('clamp', 0.0):.2f}"
                    )
            if emit_debug_sections:
                if projection_lines:
                    final_answer = (
                        f"{final_answer}\n\n[Psychoid Field Alignment]\n" + "\n".join(projection_lines)
                    )
                if chain:
                    final_answer = (
                        f"{final_answer}\n\n[Psychoid Signifiers]\n" + " -> ".join(chain[-6:])
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
            if emit_debug_sections and reflection_lines:
                final_answer = (
                    f"{final_answer}\n\n[Default Mode Reflection]\n" + "\n".join(reflection_lines)
                )
        if emit_debug_sections:
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
        if distortion_payload is not None:
            self.telemetry.log(
                "cognitive_distortion_audit",
                qid=decision.qid,
                report=distortion_payload,
            )
        if self.prefrontal_cortex is not None and schema_profile is None:
            try:
                schema_profile = self.prefrontal_cortex.profile_turn(
                    question=question,
                    answer=user_answer,
                    focus=focus,
                    affect=affect,
                )
            except Exception:  # pragma: no cover - defensive guard
                schema_profile = None
        if schema_profile is not None:
            schema_payload = schema_profile.to_dict()
            decision.state["schema_profile"] = schema_payload
            self.telemetry.log(
                "schema_profile",
                qid=decision.qid,
                profile=schema_payload,
            )
            tags.update(schema_profile.tags())
        if focus is not None and self.prefrontal_cortex is not None:
            tags.update(self.prefrontal_cortex.tags(focus))
        if basal_signal is not None:
            tags.update(self.basal_ganglia.tags(basal_signal))

        if inner_steps:
            inner_steps[-1].content = _truncate_text(user_answer)
            inner_steps[-1].metadata.update(
                {
                    "finalised": True,
                    "reward": round(reward, 3),
                    "tag_count": len(tags),
                }
            )
        steps_payload = [step.to_payload() for step in inner_steps]
        decision.state["inner_dialogue_trace"] = steps_payload
        self.telemetry.log(
            "inner_dialogue_trace",
            qid=decision.qid,
            steps=steps_payload,
            leading=leading,
            collaborative=collaborative_lead,
            action=decision.action,
        )
        tags.add("inner_dialogue_trace")
        tags.add(f"inner_steps_{len(steps_payload)}")
        
        # Add neural impulse activity summary if available
        if neural_activity:
            tags.add("neural_impulse_simulation")
            if emit_debug_sections:
                impulse_lines = [
                    f"- Hemisphere: {neural_activity['hemisphere']}",
                    f"- Total impulses: {neural_activity['total_impulses']}",
                    f"- Stimulus strength: {neural_activity['stimulus_strength']:.2f}",
                ]
                impulse_counts = neural_activity.get("impulse_counts", {})
                if impulse_counts:
                    count_parts = []
                    for nt, count in impulse_counts.items():
                        if count > 0:
                            count_parts.append(f"{nt}={count}")
                    if count_parts:
                        impulse_lines.append(
                            f"- Neurotransmitter activity: {', '.join(count_parts)}"
                        )

                network_activity = neural_activity.get("network_activity", {})
                if network_activity:
                    impulse_lines.append(
                        f"- Active neurons: {network_activity.get('active_neurons', 0):.0f}/{network_activity.get('network_size', 0):.0f}"
                    )
                    impulse_lines.append(
                        f"- Average membrane potential: {network_activity.get('avg_membrane_potential', 0):.1f} mV"
                    )

                final_answer = (
                    f"{final_answer}\n\n[Neural Impulse Activity]\n"
                    + "\n".join(impulse_lines)
                )

        tags_with_architecture = set(tags)
        tags_with_architecture.add("architecture_path")
        architecture_preview = self._compose_architecture_path(
            focus=focus,
            focus_metric=focus_metric,
            schema_profile=schema_profile,
            affect=affect,
            novelty=novelty,
            interoception=insula_state,
            salience_signal=salience_signal,
            thalamic_relay=thalamic_relay,
            predictive_frame=predictive_frame,
            hemisphere_signal=hemisphere_signal,
            collaboration_profile=collaboration_profile,
            decision=decision,
            leading=leading,
            collaborative=collaborative_lead,
            steps=steps_payload,
            coherence_signal=coherence_signal,
            distortion_payload=distortion_payload,
            tags=sorted(tags_with_architecture),
            hippocampal_rollup=None,
            success=success,
        )
        architecture_summary = _summarise_architecture_path(architecture_preview)
        if architecture_summary:
            tags = tags_with_architecture
            decision.state["architecture_path_preview"] = architecture_preview
            decision.state["architecture_path_summary"] = architecture_summary
            if emit_debug_sections:
                final_answer = (
                    f"{final_answer}\n\n[Architecture Path]\n"
                    + "\n".join(architecture_summary)
                )

        follow_brain: Optional[str] = None
        if collaborative_lead:
            follow_brain = "braided"
        elif leading == "right":
            follow_brain = "left"
        elif detail_notes or decision.action != 0:
            follow_brain = "right"
        memory_stage = self._persist_memory_stage(
            question=question,
            user_answer=user_answer,
            decision=decision,
            leading=leading,
            collaboration_profile=collaboration_profile,
            selection_reason=selection_reason,
            tags=tags,
            steps_payload=steps_payload,
            hemisphere_mode=hemisphere_mode,
            hemisphere_bias=hemisphere_bias,
            collaborative_lead=collaborative_lead,
            leading_style=leading_style,
            reward=reward,
            success=success,
            salience_signal=salience_signal,
            focus=focus,
            focus_metric=focus_metric,
            schema_profile=schema_profile,
            affect=affect,
            novelty=novelty,
            insula_state=insula_state,
            thalamic_relay=thalamic_relay,
            predictive_frame=predictive_frame,
            hemisphere_signal=hemisphere_signal,
            coherence_signal=coherence_signal,
            distortion_payload=distortion_payload,
        )
        episodic_total = memory_stage.episodic_total
        hippocampal_rollup = memory_stage.hippocampal_rollup
        hippocampal_lifecycle = memory_stage.hippocampal_lifecycle
        hippocampal_forgetting = memory_stage.hippocampal_forgetting
        architecture_path = memory_stage.architecture_path

        observer_stage = await self._run_post_turn_observer_stage(
            question=question,
            context=context,
            focus=focus,
            focus_keywords=focus_keywords,
            user_answer=user_answer,
            decision=decision,
            director_payload=director_payload,
            use_metrics_observer=use_metrics_observer,
            architecture_path=architecture_path,
            coherence_signal=coherence_signal,
            leading=leading,
            collaborative_lead=collaborative_lead,
            selection_reason=selection_reason,
            phase_add=_phase_add,
        )
        executive_observer_payload = observer_stage.executive_observer_payload
        self.memory.record_dialogue_flow(
            decision.qid,
            leading_brain=leading,
            follow_brain=follow_brain,
            preview=_truncate_text(right_lead_notes or detail_notes),
            executive=executive_payload,
            executive_observer=executive_observer_payload,
            steps=steps_payload,
            architecture=architecture_path,
        )
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
        if phase_latencies_ms:
            phases = {
                key: round(value, 3)
                for key, value in sorted(phase_latencies_ms.items())
                if value > 0.0
            }
            if phases:
                accounted_ms = round(sum(phases.values()), 3)
                other_ms = round(max(0.0, float(latency_ms) - accounted_ms), 3)
                decision.state["latency_phases_ms"] = phases
                self.telemetry.log(
                    "latency_breakdown",
                    qid=decision.qid,
                    phases=phases,
                    accounted_ms=accounted_ms,
                    other_ms=other_ms,
                    total_ms=round(float(latency_ms), 3),
                )
        if hippocampal_rollup is not None:
            self.telemetry.log(
                "hippocampal_collaboration",
                qid=decision.qid,
                rollup=hippocampal_rollup,
            )
        if hippocampal_lifecycle is not None:
            self.telemetry.log(
                "hippocampal_lifecycle",
                qid=decision.qid,
                lifecycle=hippocampal_lifecycle,
            )
        if hippocampal_forgetting is not None:
            self.telemetry.log(
                "hippocampal_forgetting",
                qid=decision.qid,
                forgetting=hippocampal_forgetting,
            )
        if self.basal_ganglia is not None and basal_signal is not None:
            self.basal_ganglia.integrate_feedback(
                reward=reward,
                latency_ms=latency_ms,
                conflict_resolved=bool(
                    decision.state.get("system2_resolved")
                    or (
                        isinstance(decision.state.get("cerebellum_micro"), dict)
                        and decision.state.get("cerebellum_micro", {}).get("resolved")
                    )
                ),
                system2_used=bool(decision.state.get("system2_enabled")),
            )
        if self.unconscious_field is not None:
            outcome_meta = self.unconscious_field.integrate_outcome(
                mapping=unconscious_profile,
                question=question,
                draft=draft,
                final_answer=user_answer,
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
