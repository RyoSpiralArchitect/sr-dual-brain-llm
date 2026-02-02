"""High-level orchestration helpers for the dual-brain control loop."""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
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


def _truncate_text(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    condensed = " ".join(text.split())
    if len(condensed) <= limit:
        return condensed
    return condensed[: max(0, limit - 3)] + "..."


def _normalise_similarity_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _similarity_ratio(a: str, b: str) -> float:
    na = _normalise_similarity_text(a)
    nb = _normalise_similarity_text(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()

def _looks_like_coaching_notes(question: str, text: str) -> bool:
    """Detect "writing coach" style content that should not leak into user replies.

    We only suppress these patterns when the user is *not* asking for writing feedback.
    """

    q = str(question or "").strip().lower()
    # If the user explicitly requests wording advice / proofreading, allow coaching.
    allow_markers = (
        "rewrite",
        "rephrase",
        "proofread",
        "improve this",
        "make it sound",
        "添削",
        "言い換え",
        "言い方",
        "表現",
        "文章",
        "文面",
        "返事",
    )
    if any(marker in q for marker in allow_markers):
        return False

    t = str(text or "").strip()
    if not t:
        return False

    lower = t.lower()
    coaching_markers = (
        "add a ",
        "add an ",
        "consider ",
        "if the user",
        "you should",
        "try to",
        "make sure to",
        "time of day",
        "もう少し",
        "〜したほうが",
        "したほうが",
        "した方が",
        "時間帯",
        "推測",
        "遊び心",
    )
    if any(marker in lower for marker in coaching_markers):
        return True

    for line in t.splitlines():
        line_norm = line.strip().lower()
        if not line_norm:
            continue
        if line_norm.startswith(("- add ", "- consider ", "add ", "consider ")):
            return True
        if line_norm.startswith(("- 「", "-『", "- \"", "- '")) and ("例" in line_norm or "もう少し" in line_norm):
            return True
    return False

def _looks_like_internal_debug_line(line: str) -> bool:
    if not line:
        return False
    raw = line.strip()
    if not raw:
        return False
    lower = raw.lower()
    if lower.startswith("qid "):
        return True
    if lower.startswith("architecture path") or lower.startswith("[architecture path"):
        return True
    if lower.startswith("telemetry (raw)") or lower.startswith("[telemetry"):
        return True
    if lower.startswith("[") and any(
        token in lower
        for token in (
            "left brain",
            "right brain",
            "coherence",
            "unconscious",
            "linguistic",
            "cognitive",
            "psychoid",
            "hemisphere",
            "collaboration",
            "architecture",
        )
    ):
        return True
    if "brain timeout" in lower and "draft" in lower:
        return True
    return False


def _looks_like_writing_coach_line(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False
    lower = raw.lower()

    # English coach patterns (avoid blocking normal "consider X" advice).
    if "if the user" in lower:
        return True
    if lower.startswith(("- add a ", "- add an ", "add a ", "add an ")):
        return True
    if lower.startswith(("consider noting the time", "- consider noting the time")):
        return True

    # Japanese coach patterns observed in the wild.
    if raw.startswith("-") and ("もう少し" in raw or "時間帯" in raw or "推測" in raw or "例：" in raw):
        return True
    return False


def _sanitize_user_answer(answer: str) -> str:
    if not answer:
        return ""
    lines: List[str] = []
    for line in str(answer).splitlines():
        if _looks_like_internal_debug_line(line) or _looks_like_writing_coach_line(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or str(answer).strip()


def _detail_notes_redundant(draft: str, detail_notes: str) -> bool:
    if not draft or not detail_notes:
        return False

    # If right-brain notes come as a bullet list, treat as additive by default.
    if re.search(r"(^|\n)\s*[-•*]\s+\S", detail_notes):
        return False

    ratio = _similarity_ratio(draft, detail_notes)
    if ratio >= 0.62:
        return len(detail_notes) <= len(draft) * 2.2

    # Conversational paraphrases can be less similar yet still redundant.
    if ratio >= 0.55 and len(detail_notes) <= len(draft) * 1.6:
        return True

    return False


def _format_modules(modules: Sequence[str] | None) -> str:
    if not modules:
        return "∅"
    return ", ".join(modules)


def _summarise_architecture_stage(idx: int, stage: Dict[str, Any]) -> str:
    name = stage.get("stage", f"stage_{idx}")
    modules = _format_modules(stage.get("modules"))
    descriptors: List[str] = []

    if name == "perception":
        signals = stage.get("signals", {})
        affect = signals.get("affect", {}) if isinstance(signals, dict) else {}
        valence = float(affect.get("valence", 0.0))
        arousal = float(affect.get("arousal", 0.0))
        risk = float(affect.get("risk", 0.0))
        novelty = float(affect.get("novelty", 0.0))
        descriptors.append(
            "affect v{:+.2f}/a{:+.2f}/r{:+.2f}/n{:.2f}".format(
                valence, arousal, risk, novelty
            )
        )
        focus = stage.get("focus", {}) or {}
        if isinstance(focus, dict) and focus.get("keywords"):
            keywords = focus.get("keywords", [])
            descriptors.append(
                "focus {}".format(", ".join(str(kw) for kw in keywords[:3]))
            )
            if len(keywords) > 3:
                descriptors.append(f"(+{len(keywords) - 3} more)")
        if isinstance(focus, dict):
            if "relevance" in focus:
                descriptors.append(f"rel {float(focus['relevance']):.2f}")
            if "hippocampal_overlap" in focus:
                descriptors.append(
                    f"hip {float(focus['hippocampal_overlap']):.2f}"
                )
        hemisphere = signals.get("hemisphere") if isinstance(signals, dict) else {}
        if isinstance(hemisphere, dict):
            mode = hemisphere.get("mode", "?")
            bias = float(hemisphere.get("bias", 0.0))
            descriptors.append(f"hemisphere {mode}:{bias:.2f}")
        collaboration = (
            signals.get("collaboration") if isinstance(signals, dict) else {}
        )
        if isinstance(collaboration, dict) and collaboration.get("strength") is not None:
            strength = float(collaboration.get("strength", 0.0))
            balance = float(collaboration.get("balance", 0.0))
            descriptors.append(f"collab {strength:.2f}/{balance:.2f}")
        schema_profile = stage.get("schema_profile")
        if isinstance(schema_profile, dict):
            user_schemas = schema_profile.get("user_schemas", [])
            if user_schemas:
                descriptors.append(
                    "schemas {}".format(", ".join(user_schemas[:2]))
                )
            user_modes = schema_profile.get("user_modes", [])
            if user_modes:
                descriptors.append("user modes {}".format(", ".join(user_modes[:2])))
            agent_modes = schema_profile.get("agent_modes", [])
            if agent_modes:
                descriptors.append(
                    "agent modes {}".format(", ".join(agent_modes[:2]))
                )
            if "confidence" in schema_profile:
                descriptors.append(
                    f"schema conf {float(schema_profile['confidence']):.2f}"
                )
    elif name == "inner_dialogue":
        leading = stage.get("leading", "?")
        descriptors.append(f"leading {leading}")
        if stage.get("collaborative"):
            descriptors.append("braided")
        step_count = int(stage.get("step_count", 0))
        descriptors.append(f"steps {step_count}")
        phases = stage.get("phases") or []
        if phases:
            phase_list = ", ".join(str(phase) for phase in list(phases)[:4])
            descriptors.append(f"phases {phase_list}")
        temperature = stage.get("temperature")
        if temperature is not None:
            descriptors.append(f"temp {float(temperature):.2f}")
        slot_ms = stage.get("slot_ms")
        if slot_ms is not None:
            descriptors.append(f"slot {int(slot_ms)}ms")
    elif name == "integration":
        descriptors.append("success" if stage.get("success") else "retry")
        coherence = stage.get("coherence")
        if isinstance(coherence, dict):
            if "combined" in coherence:
                descriptors.append(f"coh {float(coherence['combined']):.2f}")
            if "tension" in coherence:
                descriptors.append(f"ten {float(coherence['tension']):.2f}")
        distortion = stage.get("distortion")
        if isinstance(distortion, dict):
            flags = distortion.get("flags", [])
            if flags:
                descriptors.append(
                    "distortions {}".format(", ".join(flags[:3]))
                )
            if "score" in distortion:
                descriptors.append(f"distortion {float(distortion['score']):.2f}")
    elif name == "memory":
        tags = stage.get("tags", []) or []
        descriptors.append(f"tags {len(tags)}")
        rollup = stage.get("hippocampal_rollup") or {}
        if isinstance(rollup, dict) and rollup:
            if "avg_strength" in rollup:
                descriptors.append(f"avg {float(rollup['avg_strength']):.2f}")
            if "strength_coverage" in rollup:
                descriptors.append(
                    f"coverage {float(rollup['strength_coverage']):.2f}"
                )
            lead_parts: List[str] = []
            for key, label in (
                ("lead_left", "L"),
                ("lead_right", "R"),
                ("lead_braided", "B"),
            ):
                val = rollup.get(key)
                if val:
                    lead_parts.append(f"{label}{float(val):.2f}")
            if lead_parts:
                descriptors.append("lead mix " + " ".join(lead_parts))

    descriptor_text = "; ".join(descriptors)
    if descriptor_text:
        return f"{idx}. {name}: {modules} | {descriptor_text}"
    return f"{idx}. {name}: {modules}"


def _summarise_architecture_path(path: Sequence[Dict[str, Any]]) -> List[str]:
    return [_summarise_architecture_stage(idx, stage) for idx, stage in enumerate(path, 1)]


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
        

@dataclass
class InnerDialogueStep:
    """Single step captured during an inner dialogue exchange."""

    phase: str
    role: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "phase": self.phase,
            "role": self.role,
            "ts": float(self.timestamp),
        }
        if self.content:
            payload["content"] = self.content
        if self.metadata:
            payload["meta"] = dict(self.metadata)
        return payload


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
        hippocampal_context = ""
        if include_long_term and self.hippocampus is not None:
            hippocampal_context = self.hippocampus.retrieve_summary(
                question, include_meta=False
            )

        segments = []
        if working_memory_context:
            segments.append(_format_section("Working memory", working_memory_context.splitlines()))
        if memory_context:
            segments.append(memory_context)
        if schema_context:
            segments.append(f"[Schema memory] {schema_context}")
        if hippocampal_context:
            segments.append(f"[Hippocampal recall] {hippocampal_context}")
        combined = "\n".join(segments)
        parts = {
            "working_memory": working_memory_context,
            "memory": memory_context,
            "schema": schema_context,
            "hippocampal": hippocampal_context,
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

    @staticmethod
    def _infer_question_type(question: str) -> str:
        q = str(question or "").strip()
        if not q:
            return "easy"

        has_qmark = ("?" in q) or ("？" in q)
        length = len(q)

        # Strong structural cues (language-agnostic).
        if "```" in q:
            return "hard"
        if re.search(r"\d", q) and re.search(r"[+\-*/^%]|=", q):
            return "hard"
        if any(sym in q for sym in ("=", "≠", "<", ">", "≥", "≤", "→", "⇒", "∴", "∵")):
            return "hard"

        # Complexity by size/structure (works for CJK + Latin).
        if q.count("\n") >= 2:
            return "hard" if length >= 360 else "medium"
        if has_qmark and length >= 90:
            return "hard"
        if has_qmark and length >= 32:
            return "medium"
        if length >= 240:
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

    def _compose_architecture_path(
        self,
        *,
        focus: FocusSummary | None,
        focus_metric: float,
        schema_profile: SchemaProfile | None,
        affect: Dict[str, float],
        novelty: float,
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
        if self.prefrontal_cortex is not None:
            perception_modules.append("PrefrontalCortex")
            perception_modules.append("SchemaProfiler")
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
        if focus is not None:
            perception_entry["focus"] = focus.to_dict()
        if schema_profile is not None:
            perception_entry["schema_profile"] = schema_profile.to_dict()
        path.append(perception_entry)

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
        context, focus, context_parts = self._compose_context(question)
        is_trivial_chat = bool(
            self.prefrontal_cortex is not None
            and self.prefrontal_cortex.is_trivial_chat_turn(question)
        )
        system2_norm = str(system2_mode or "auto").strip().lower()
        if system2_norm in {"true", "1", "yes"}:
            system2_norm = "on"
        elif system2_norm in {"false", "0", "no"}:
            system2_norm = "off"
        if system2_norm not in {"auto", "on", "off"}:
            system2_norm = "auto"

        system2_capable = bool(
            getattr(self.left_model, "uses_external_llm", False)
            and getattr(self.right_model, "uses_external_llm", False)
        )
        system2_active = False
        system2_reason = "disabled"
        if system2_norm == "on":
            system2_active = True
            system2_reason = "forced_on" if system2_capable else "forced_on_unavailable"
        elif system2_norm == "off":
            system2_active = False
            system2_reason = "forced_off"
        else:
            if not system2_capable:
                system2_active = False
                system2_reason = "auto_unavailable"
            elif not is_trivial_chat:
                q = str(question or "")
                has_qmark = ("?" in q) or ("？" in q)
                wm_len = len(str(context_parts.get("working_memory") or ""))
                mem_len = len(str(context_parts.get("memory") or ""))
                hip_len = len(str(context_parts.get("hippocampal") or ""))
                context_signal_len = wm_len + mem_len + hip_len

                q_type_hint = self._infer_question_type(q)
                if q_type_hint in {"medium", "hard"}:
                    system2_active = True
                    system2_reason = f"q_type_{q_type_hint}"
                # Structural cues: math/code-like inputs and long/loaded questions.
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
                elif has_qmark and context_signal_len >= 240:
                    # A short question riding on heavy prior context → treat as System2.
                    system2_active = True
                    system2_reason = "context_heavy"

        try:
            self.telemetry.log(
                "system2_mode",
                qid=qid_value,
                mode=system2_norm,
                enabled=bool(system2_active),
                reason=system2_reason,
            )
        except Exception:  # pragma: no cover - telemetry is best-effort
            pass

        hemisphere_signal = self._select_hemisphere_mode(question, focus)
        hemisphere_mode = hemisphere_signal.mode
        hemisphere_bias = hemisphere_signal.bias
        collaboration_profile = self._compute_collaboration_profile(
            hemisphere_signal, focus
        )
        schema_profile: Optional[SchemaProfile] = None
        distortion_payload: Optional[Dict[str, object]] = None
        auto_selected_leading = False
        collaborative_lead = False
        selection_reason = "explicit_request"
        leading_style = "explicit_request"
        collaborative_hint = False
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

            if system2_active:
                # System2 is builder/critic: keep the draft on the left to avoid a poetic right prelude.
                leading = "left"
                selection_reason = (
                    "system2_forced" if system2_norm == "on" else "system2_auto"
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

        executive_mode_norm = str(executive_mode or "off").strip().lower()
        if executive_mode_norm not in {"off", "observe", "assist", "polish"}:
            executive_mode_norm = "observe"
        executive_observer_mode_norm = str(executive_observer_mode or "off").strip().lower()
        if executive_observer_mode_norm not in {"off", "metrics", "director", "both"}:
            executive_observer_mode_norm = "off"
        use_director = executive_observer_mode_norm in {"director", "both"}
        use_metrics_observer = executive_observer_mode_norm in {"metrics", "both"}
        executive_task: asyncio.Task[ExecutiveAdvice] | None = None
        executive_payload: Dict[str, Any] | None = None
        executive_directives: Dict[str, Any] | None = None
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
        director_payload: Dict[str, Any] | None = None
        director_control: Dict[str, Any] | None = None
        director_max_chars: Optional[int] = None
        director_append_question: Optional[str] = None
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
                            "hippocampal_overlap": (focus.hippocampal_overlap if focus else None),
                            "has_working_memory": bool(context_parts.get("working_memory")),
                            "has_long_term": bool(
                                context_parts.get("memory")
                                or context_parts.get("schema")
                                or context_parts.get("hippocampal")
                            ),
                        },
                    )
                )
            except Exception:  # pragma: no cover - defensive guard
                director_task = None
        inner_steps: List[InnerDialogueStep] = []
        right_lead_notes: Optional[str] = None
        stream_final_only = bool(on_delta and on_reset)
        emitted_any = False
        right_preview_task: asyncio.Task[str] | None = None

        def _rebuild_context_from_parts() -> None:
            nonlocal context, focus, focus_metric, focus_keywords

            rebuilt: List[str] = []
            wm_ctx = str(context_parts.get("working_memory") or "")
            mem_ctx = str(context_parts.get("memory") or "")
            schema_ctx = str(context_parts.get("schema") or "")
            hip_ctx = str(context_parts.get("hippocampal") or "")

            if wm_ctx:
                rebuilt.append(_format_section("Working memory", wm_ctx.splitlines()))
            if mem_ctx:
                rebuilt.append(mem_ctx)
            if schema_ctx:
                rebuilt.append(f"[Schema memory] {schema_ctx}")
            if hip_ctx:
                rebuilt.append(f"[Hippocampal recall] {hip_ctx}")

            context = "\n".join(rebuilt)
            if self.prefrontal_cortex is not None:
                focus_memory_context = "\n".join([part for part in (wm_ctx, mem_ctx) if part])
                focus = self.prefrontal_cortex.synthesise_focus(
                    question=question,
                    memory_context=focus_memory_context,
                    hippocampal_context=hip_ctx,
                )
                context = self.prefrontal_cortex.gate_context(context, focus)
                focus_metric = self.prefrontal_cortex.focus_metric(focus)
                focus_keywords = list(focus.keywords) if focus.keywords else None
            else:
                focus = None
                focus_metric = 0.0
                focus_keywords = None

        def _apply_director_memory(control: Dict[str, Any]) -> None:
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
                context_parts["hippocampal"] = ""
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

            _rebuild_context_from_parts()

        # Director is a pre-turn steering module; to avoid "regen" behavior we apply it
        # before drafting whenever it's enabled.
        if director_task is not None and director_payload is None:
            advice: DirectorAdvice | None = None
            if is_trivial_chat:
                # Avoid paying external LLM latency on lightweight turns; structural cues
                # already give a good "executive control" default.
                advice = DirectorAdvice(
                    memo="(Director memo / fast-path)\n- trivial chat: keep WM (if any), drop long-term, skip consult",
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
                    # Give the director a real chance (network calls often exceed 350ms).
                    advice = await asyncio.wait_for(
                        asyncio.shield(director_task), timeout=1.6
                    )
                except asyncio.TimeoutError:
                    advice = None
                except asyncio.CancelledError:
                    advice = None
                except Exception:  # pragma: no cover - defensive guard
                    advice = None
                if advice is None:
                    director_task.cancel()
                    director_task = None
                    advice = DirectorAdvice(
                        memo="(Director memo / timeout fallback)\n- Proceeding with heuristic steering (no external director output).",
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

            if advice is not None:
                director_payload = advice.to_payload()
                director_payload["observer_mode"] = "director"
                director_control = advice.control if isinstance(advice.control, dict) else {}
                self.telemetry.log(
                    "director_reasoner",
                    qid=qid_value,
                    confidence=float(advice.confidence),
                    latency_ms=float(advice.latency_ms),
                    source=str(advice.source),
                    phase="pre",
                )
                _apply_director_memory(director_control)

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

        delta_cb = _emit_delta if on_delta else None
        if (leading == "right" or collaborative_lead) and not system2_active:
            try:
                # Run the right-brain prelude in parallel with the left draft. We keep it
                # internal (no streaming) to avoid "split voice" in the chat UI.
                right_preview_task = asyncio.create_task(
                    self.right_model.generate_lead(question, context, on_delta=None)
                )
            except Exception:  # pragma: no cover - defensive guard
                right_preview_task = None
        left_lead_preview: Optional[str] = None

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
            inner_steps.append(
                InnerDialogueStep(
                    phase="right_preview",
                    role="right",
                    content=_truncate_text(right_lead_notes),
                    metadata=preview_meta,
                )
            )
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
        inner_steps.append(
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
        decision.state["system2_enabled"] = bool(system2_active)
        decision.state["system2_mode"] = system2_norm
        decision.state["system2_reason"] = system2_reason
        if force_right_lead and decision.action == 0:
            decision.action = 1
            decision.state["right_forced_lead"] = True
        if system2_active and decision.action == 0:
            decision.action = 1
            decision.state["system2_forced_consult"] = True
        if system2_active:
            decision.temperature = max(0.1, min(0.6, float(decision.temperature)))
            decision.state["system2_temperature"] = decision.temperature
        if (
            self.prefrontal_cortex is not None
            and not explicit_leading
            and self.prefrontal_cortex.is_trivial_chat_turn(question)
            and not system2_active
        ):
            decision.action = 0
            decision.state["prefrontal_override"] = "trivial_chat"

        # If the director wasn't applied pre-draft (e.g., disabled, missing, or timed out),
        # try a short post-draft fetch and still apply its steering for the remainder of the turn.
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
                    advice = await asyncio.wait_for(asyncio.shield(director_task), timeout=0.9)
                except asyncio.TimeoutError:
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
                _apply_director_memory(director_control)
            else:
                director_task.cancel()
                director_task = None

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

            # Apply memory steering (again) to reflect director decisions in observer reports
            # and subsequent consult calls. Idempotent when already applied pre-draft.
            _apply_director_memory(director_control)
            mem = director_control.get("memory")
            mem_obj: Dict[str, Any] = mem if isinstance(mem, dict) else {}
            wm = str(mem_obj.get("working_memory") or "auto").strip().lower()
            lt = str(mem_obj.get("long_term") or "auto").strip().lower()
            if wm == "drop":
                decision.state["director_memory_working"] = "drop"
            if lt == "drop":
                decision.state["director_memory_long_term"] = "drop"
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
        suppress_default_mode_reflections = self._default_mode_low_focus_streak >= 2
        if suppress_default_mode_reflections:
            decision.state["default_mode_suppressed"] = True
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
                inner_steps.append(
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
                )
                if executive_task is not None:
                    timeout_ms = max(1500, min(9000, decision.slot_ms * 12))
                    if executive_mode_norm == "observe":
                        timeout_ms = max(1200, min(4500, decision.slot_ms * 8))
                    try:
                        advice = await asyncio.wait_for(executive_task, timeout=timeout_ms / 1000.0)
                    except asyncio.TimeoutError:
                        advice = None
                    except asyncio.CancelledError:
                        advice = None
                    except Exception:  # pragma: no cover - defensive guard
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

                can_polish = bool(
                    executive_mode_norm == "polish"
                    and bool(executive_directives)
                    and getattr(self.left_model, "uses_external_llm", False)
                )
                if can_polish and executive_payload and float(executive_payload.get("confidence") or 0.0) < 0.35:
                    can_polish = False

                integration_parts: List[str] = []
                if executive_payload and executive_mode_norm == "assist":
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
                    await _emit_reset_if_needed()
                    final_answer = await self.left_model.integrate_info_async(
                        question=question,
                        draft=final_answer,
                        info="\n\n".join(integration_parts),
                        temperature=max(0.2, min(0.6, decision.temperature)),
                        on_delta=None if stream_final_only else delta_cb,
                    )
                success = True
            else:
                payload_type = "ASK_CRITIC" if system2_active else "ASK_DETAIL"
                payload = {
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
                if system2_active:
                    draft_full = draft
                    if len(draft_full) > 2400:
                        draft_full = draft_full[:2397].rstrip() + "..."
                    payload["draft"] = draft_full
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
                if budget_norm == "large" and not system2_active:
                    timeout_scale = 95
                timeout_ms = int(
                    min(
                        float(self.default_timeout_ms),
                        max(6000.0, float(decision.slot_ms) * float(timeout_scale)),
                    )
                )
                original_slot = getattr(self.callosum, "slot_ms", decision.slot_ms)
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
                inner_steps.append(
                    InnerDialogueStep(
                        phase="callosum_request",
                        role="coordinator",
                        content=_truncate_text(payload.get("draft_sum", ""), 120),
                        metadata=request_meta,
                    )
                )
                try:
                    self.callosum.slot_ms = decision.slot_ms
                    response = await self.callosum.ask_detail(payload, timeout_ms=timeout_ms)
                finally:
                    self.callosum.slot_ms = original_slot
                response_source = "callosum"
                critic_verdict: Optional[str] = None
                critic_issues: list[str] = []
                critic_fixes: list[str] = []
                if system2_active:
                    critic_verdict = str(response.get("verdict") or "").strip().lower() or None
                    issues_raw = response.get("issues")
                    fixes_raw = response.get("fixes")
                    if isinstance(issues_raw, list):
                        critic_issues = [str(item).strip() for item in issues_raw if str(item).strip()]
                    if isinstance(fixes_raw, list):
                        critic_fixes = [str(item).strip() for item in fixes_raw if str(item).strip()]
                    decision.state["right_role"] = "critic"
                    if critic_verdict:
                        decision.state["critic_verdict"] = critic_verdict
                    if critic_issues:
                        decision.state["critic_issues"] = critic_issues[:12]
                    if critic_fixes:
                        decision.state["critic_fixes"] = critic_fixes[:12]
                    detail_notes = response.get("critic_sum")
                else:
                    detail_notes = response.get("notes_sum")
                call_error = bool(response.get("error"))
                call_confidence = float(response.get("confidence_r", 0.0) or 0.0)
                fallback_used = False
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
                            if critic_verdict:
                                decision.state["critic_verdict"] = critic_verdict
                            issues_raw = fallback.get("issues")
                            fixes_raw = fallback.get("fixes")
                            if isinstance(issues_raw, list):
                                critic_issues = [str(item).strip() for item in issues_raw if str(item).strip()]
                                if critic_issues:
                                    decision.state["critic_issues"] = critic_issues[:12]
                            if isinstance(fixes_raw, list):
                                critic_fixes = [str(item).strip() for item in fixes_raw if str(item).strip()]
                                if critic_fixes:
                                    decision.state["critic_fixes"] = critic_fixes[:12]
                            detail_notes = fallback.get("critic_sum")
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

                integrated_detail = detail_notes
                critic_needs_revision = False
                if system2_active:
                    verdict_norm = (critic_verdict or "").strip().lower()
                    critic_needs_revision = verdict_norm == "issues" or bool(critic_issues)
                    if not critic_needs_revision:
                        integrated_detail = None
                elif detail_notes and _detail_notes_redundant(draft, detail_notes):
                    integrated_detail = None
                if integrated_detail and _looks_like_coaching_notes(question, integrated_detail):
                    integrated_detail = None

                if executive_task is not None:
                    timeout_ms = max(1500, min(9000, decision.slot_ms * 12))
                    if executive_mode_norm == "observe":
                        timeout_ms = max(1200, min(4500, decision.slot_ms * 8))
                    try:
                        advice = await asyncio.wait_for(executive_task, timeout=timeout_ms / 1000.0)
                    except asyncio.TimeoutError:
                        advice = None
                    except asyncio.CancelledError:
                        advice = None
                    except Exception:  # pragma: no cover - defensive guard
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

                integration_parts: List[str] = []
                has_right_material = False
                has_mix_in_material = False
                if integrated_detail:
                    if system2_active:
                        integration_parts.append(
                            "Reasoning critic notes (internal; do not output directly). "
                            "Revise the draft to address issues and improve correctness:\n"
                            + str(integrated_detail)
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
                        final_answer = await self.left_model.integrate_info_async(
                            question=question,
                            draft=draft,
                            info="\n\n".join(integration_parts),
                            temperature=max(0.2, min(0.6, decision.temperature)),
                            on_delta=None if stream_final_only else delta_cb,
                        )
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
            inner_steps.append(
                InnerDialogueStep(
                    phase="callosum_timeout",
                    role="coordinator",
                    content="Right brain consult timed out",
                    metadata={"timeout_ms": timeout_ms, "slot_ms": decision.slot_ms},
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
                    advice = await asyncio.wait_for(executive_task, timeout=tail_timeout)
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

        integrated_answer = final_answer
        user_answer = integrated_answer
        if not emit_debug_sections:
            user_answer = _sanitize_user_answer(user_answer)
            max_chars: Optional[int] = None
            if director_max_chars is not None:
                try:
                    max_chars = int(director_max_chars)
                except Exception:
                    max_chars = None
                if max_chars is not None:
                    max_chars = max(80, min(2400, max_chars))

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

        audit_result = self.auditor.check(
            user_answer,
            question=question,
            focus_keywords=tuple(focus_keywords or ()),
            working_memory_context=str(context_parts.get("working_memory") or ""),
            is_trivial_chat=bool(is_trivial_chat),
            allow_debug=bool(emit_debug_sections),
        )
        metacognition = audit_result.get("metacognition")
        if isinstance(metacognition, dict):
            try:
                self.telemetry.log(
                    "metacognition",
                    qid=decision.qid,
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
        # Ensure director-provided clarifying question remains present even when the
        # metacognition layer rewrites the answer (e.g., repetition cleanup).
        if (
            not emit_debug_sections
            and director_append_question
            and director_append_question not in (user_answer or "")
        ):
            base = (user_answer or "").strip()
            suffix = director_append_question.strip()
            sep = "\n\n" if base and suffix else ""
            max_chars: Optional[int] = None
            if director_max_chars is not None:
                try:
                    max_chars = int(director_max_chars)
                except Exception:
                    max_chars = None
                if max_chars is not None:
                    max_chars = max(80, min(2400, max_chars))
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
            final_answer = user_answer

        if not audit_result.get("ok", True):
            user_answer = draft
            final_answer = user_answer
            success = False

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
        # Always store "clean" memory, even when the user requested debug/meta output.
        memory_question = _sanitize_user_answer(question) or str(question or "").strip()
        memory_answer = _sanitize_user_answer(user_answer) or str(user_answer or "").strip()
        episodic_total = 0
        hippocampal_rollup: Optional[Dict[str, float]] = None
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
                        {"schema_profile": schema_profile.to_dict()}
                        if schema_profile is not None
                        else {}
                    ),
                    **(
                        {"distortion_report": distortion_payload}
                        if distortion_payload is not None
                        else {}
                    ),
                },
            )
            episodic_total = len(self.hippocampus.episodes)
            hippocampal_rollup = self.hippocampus.collaboration_rollup()

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

        executive_observer_payload: Dict[str, Any] | None = director_payload
        if self.executive_model is not None and use_metrics_observer:
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
                    observer_advice = await asyncio.wait_for(observer_task, timeout=2.5)
                except asyncio.TimeoutError:
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
        if hippocampal_rollup is not None:
            self.telemetry.log(
                "hippocampal_collaboration",
                qid=decision.qid,
                rollup=hippocampal_rollup,
            )
        if self.basal_ganglia is not None and basal_signal is not None:
            self.basal_ganglia.integrate_feedback(reward=reward, latency_ms=latency_ms)
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
