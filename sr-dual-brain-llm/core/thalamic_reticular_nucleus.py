"""Model-free activation gates for explicitly registered optional modules.

The thalamic reticular nucleus analogy describes selective inhibition. This
component decides whether a module may run; modules retain their own algorithms
and local guards. Evidence approval, persistence and transport are outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
import re
from types import MappingProxyType
from typing import Callable, Mapping


CONVERSATION_SELECTIVE_CONSULTATION = "conversation_selective_consultation"
ACC_MONITOR = "acc_monitor"
CEREBELLAR_CORRECTION = "cerebellar_correction"
DEFAULT_MODE_REFLECTION_INJECTION = "default_mode_reflection_injection"
SYSTEM2_STALLED_FOLLOWUP = "system2_stalled_followup"
EXECUTIVE_ADVICE = "executive_advice"
DIRECTOR_ADVICE = "director_advice"
EXECUTIVE_METRICS_OBSERVATION = "executive_metrics_observation"
ACC_CONSULT_PROMOTION = "acc_consult_promotion"
ACC_SYSTEM2_PROMOTION = "acc_system2_promotion"
ACC_TEMPERATURE_ADJUSTMENT = "acc_temperature_adjustment"
BASAL_CONSULT_PROMOTION = "basal_consult_promotion"
SPECULATIVE_PREVIEW_SUPPRESSION = "speculative_preview_suppression"
TASK_GOAL_ORCHESTRATION = "task_goal_orchestration"
COMPLETION_REPLANNING = "completion_replanning"


class GateMode(str, Enum):
    ON = "on"
    OFF = "off"
    AUTO = "auto"


@dataclass(frozen=True)
class GateConfig:
    mode: GateMode
    source: str
    valid: bool


@dataclass(frozen=True)
class GateFacts:
    eligible: bool
    blocked_reason: str = "outside_scope"
    has_draft: bool = False
    is_trivial: bool = False
    confidence: float | None = None
    risk: float | None = None
    approved_records: int = 0
    reference_resolved: bool = False
    literal_replay_requested: bool = False
    full_source_replayed: bool = False
    bounded_clarification_complete: bool = False
    micro_known: bool = False
    micro_issues: int = 0
    micro_confidence: float | None = None
    has_input: bool = False
    reflection_count: int = 0
    reflection_confidence: float | None = None
    task_positive_load: float | None = None
    conflict_level: float | None = None
    critic_structured: bool = False
    critic_confidence: float | None = None
    remaining_issues: int = 0
    memory_pressure: bool = False
    memory_competition: bool = False
    remaining_budget_ms: float | None = None
    estimated_latency_ms: float | None = None
    recovery_probe_allowed: bool = False
    recovery_kind: str = 'probe'
    recovery_output_blocked: bool = False
    recovery_confirmation_required: bool = False
    local_controls_ready: bool = False
    record_choice_pending: bool = False
    task_goal_count: int = 0
    generation_incomplete: bool = False
    generation_length_limited: bool = False
    remaining_output_tokens: int | None = None


AutoPolicy = Callable[[GateFacts], tuple[bool, str]]


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    phase: str
    policy_version: str
    mode_env: str
    legacy_env: str | None
    default_mode: GateMode
    auto_policy: AutoPolicy


@dataclass(frozen=True)
class GateDecision:
    module: str
    phase: str
    mode: GateMode
    source: str
    eligible: bool
    enabled: bool
    reason: str
    policy_version: str

    def to_payload(self) -> dict[str, str | bool]:
        """Return only fixed metadata; no source text, IDs or numeric signals."""
        return {
            "module": self.module,
            "phase": self.phase,
            "mode": self.mode.value,
            "source": self.source,
            "eligible": self.eligible,
            "enabled": self.enabled,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


# Callers supply these codes for hard eligibility failures. Unknown text never
# enters decision telemetry. Changes to auto reasons accompany policy versions.
BLOCKED_REASONS = frozenset({
    "outside_scope", "conversation_context_disabled", "answerability_gate_unavailable",
    "primary_write_pending", "explicit_system2_mode", "precision_priority",
    "system2_unavailable", "right_lead", "director_override", "amygdala_override",
    "acc_override", "system2_active", "model_unavailable", "module_unavailable",
    "micro_unavailable",
    "caller_mode_off", "no_reflections", "default_mode_suppressed",
    "followup_not_eligible", "signal_unavailable", "adjustment_zero",
    "turn_deadline_elapsed",
    "completion_replanning_unavailable",
})
AUTO_REASONS = frozenset({
    "no_draft", "trivial_turn", "unknown_signal", "risk_or_uncertainty",
    "literal_replay_not_requested", "source_replay_incomplete",
    "unsupported_record_count", "reference_unresolved", "supported_literal_replay",
    "observation_needed", "observation_not_needed", "micro_unknown",
    "no_micro_issues", "micro_issue_limit", "micro_confidence_low",
    "micro_correction_candidate",
    "no_input", "optional_observation_needed", "no_reflections", "reflection_signal_low",
    "reflection_injection_needed", "conflict_control_needed",
    "critic_signal_uncertain", "stalled_checklist_needed",
    "memory_consultation_needed",
    "time_budget_exhausted", "insufficient_time_budget", "invalid_timing_signal", "bounded_recovery_probe",
    "bounded_recovery_confirmation", "unusable_recovery_output", "invalid_recovery_signal", "recovery_unconfirmed",
    "local_controls_ready",
    "bounded_record_clarification",
    "record_choice_pending", "preview_context_needed",
    "explicit_task_goals", "task_goals_unavailable",
    "length_limited_replanning", "replanning_not_needed",
})
_CONFIG_SOURCES = frozenset({"default", "canonical", "legacy", "explicit"})
_MODULE_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_METADATA_CODE = re.compile(r"[a-z][a-z0-9_]*")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_GATE_SELF_IDS = frozenset({"thalamic_reticular_nucleus", "module_gate"})


def _bounded_code(value: object, pattern: re.Pattern[str], limit: int) -> bool:
    return type(value) is str and 0 < len(value) <= limit and pattern.fullmatch(value) is not None


def _validated_registry(registry: Mapping[str, ModuleSpec]) -> Mapping[str, ModuleSpec]:
    """Copy explicitly supplied trusted policies and enforce registration rules."""
    copied = dict(registry)
    environment_names: set[str] = set()
    for module_id, spec in copied.items():
        if not _bounded_code(module_id, _MODULE_ID, 64):
            raise ValueError("Module IDs must be bounded lowercase snake-case identifiers")
        if module_id in _GATE_SELF_IDS:
            raise ValueError("The module gate cannot register itself")
        if not isinstance(spec, ModuleSpec) or spec.module_id != module_id:
            raise ValueError("Registry keys must match their ModuleSpec module IDs")
        if not _bounded_code(spec.phase, _METADATA_CODE, 32):
            raise ValueError("Module phases must be bounded metadata codes")
        if not _bounded_code(spec.policy_version, _METADATA_CODE, 64):
            raise ValueError("Policy versions must be bounded metadata codes")
        if spec.mode_env != f"DUALBRAIN_MODULE_{module_id.upper()}_MODE":
            raise ValueError("Canonical environment names must derive from the module ID")
        if not isinstance(spec.default_mode, GateMode):
            raise ValueError("Module default modes must be GateMode values")
        if not callable(spec.auto_policy):
            raise ValueError("Module auto policies must be explicit callable objects")
        for environment_name in (spec.mode_env, spec.legacy_env):
            if environment_name is None:
                continue
            if not _bounded_code(environment_name, _ENVIRONMENT_NAME, 128):
                raise ValueError("Environment names must be bounded uppercase identifiers")
            if environment_name in environment_names:
                raise ValueError("Module environment names must be unique")
            environment_names.add(environment_name)
    return MappingProxyType(copied)


def _finite(value: float | None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _auto_selective_consultation(facts: GateFacts) -> tuple[bool, str]:
    if not facts.has_draft:
        return False, "no_draft"
    if facts.is_trivial:
        return False, "trivial_turn"
    if not (_finite(facts.confidence) and _finite(facts.risk)):
        return False, "unknown_signal"
    if facts.confidence < 0.85 or facts.risk != 0:
        return False, "risk_or_uncertainty"
    if facts.bounded_clarification_complete is True:
        if type(facts.approved_records) is not int or facts.approved_records != 0:
            return False, "unsupported_record_count"
        if not facts.reference_resolved:
            return False, "reference_unresolved"
        return True, "bounded_record_clarification"
    if not facts.literal_replay_requested:
        return False, "literal_replay_not_requested"
    if not facts.full_source_replayed:
        return False, "source_replay_incomplete"
    if type(facts.approved_records) is not int or facts.approved_records != 1:
        return False, "unsupported_record_count"
    if not facts.reference_resolved:
        return False, "reference_unresolved"
    return True, "supported_literal_replay"


def _auto_acc_monitor(facts: GateFacts) -> tuple[bool, str]:
    if not facts.has_draft:
        return False, "no_draft"
    if facts.is_trivial:
        return False, "trivial_turn"
    if (facts.micro_known
            or (_finite(facts.risk) and facts.risk > 0)
            or (_finite(facts.confidence) and facts.confidence < 0.85)):
        return True, "observation_needed"
    if not (_finite(facts.risk) and _finite(facts.confidence)):
        return False, "unknown_signal"
    return False, "observation_not_needed"


def _auto_cerebellar_correction(facts: GateFacts) -> tuple[bool, str]:
    if not facts.micro_known:
        return False, "micro_unknown"
    if type(facts.micro_issues) is not int:
        return False, "unknown_signal"
    if facts.micro_issues < 1:
        return False, "no_micro_issues"
    if facts.micro_issues > 4:
        return False, "micro_issue_limit"
    if not _finite(facts.micro_confidence):
        return False, "unknown_signal"
    if facts.micro_confidence < 0.88:
        return False, "micro_confidence_low"
    return True, "micro_correction_candidate"


def _auto_optional_observation(facts: GateFacts) -> tuple[bool, str]:
    if not facts.has_input:
        return False, "no_input"
    if facts.is_trivial:
        return False, "trivial_turn"
    budget, latency = facts.remaining_budget_ms, facts.estimated_latency_ms
    if any(value is not None and (not _finite(value) or value < 0) for value in (budget, latency)):
        return False, "invalid_timing_signal"
    if budget is not None and budget <= 50.:
        return False, "time_budget_exhausted"
    if (facts.recovery_output_blocked is True or facts.recovery_confirmation_required is True
            or budget is not None and latency is not None and latency + 50. >= budget):
        if facts.recovery_probe_allowed is True:
            if facts.recovery_kind not in {'probe', 'confirmation'}:
                return False, "invalid_recovery_signal"
            return True, "bounded_recovery_confirmation" if facts.recovery_kind == 'confirmation' else "bounded_recovery_probe"
        return False, ("unusable_recovery_output" if facts.recovery_output_blocked is True
                       else "recovery_unconfirmed" if facts.recovery_confirmation_required is True else "insufficient_time_budget")
    # Unknown estimate/window preserves v1's nontrivial-input cold probe.
    return True, "optional_observation_needed"


def _auto_director_advice(facts: GateFacts) -> tuple[bool, str]:
    if facts.local_controls_ready is True and facts.has_input and not facts.is_trivial:
        return False, "local_controls_ready"
    return _auto_optional_observation(facts)


def _auto_metrics_observation(facts: GateFacts) -> tuple[bool, str]:
    if not facts.has_draft:
        return False, "no_draft"
    return _auto_optional_observation(facts)


def _auto_reflection_injection(facts: GateFacts) -> tuple[bool, str]:
    if type(facts.reflection_count) is not int or facts.reflection_count < 1:
        return False, "no_reflections"
    if not all(_finite(value) for value in (
        facts.reflection_confidence, facts.task_positive_load, facts.risk,
    )):
        return False, "unknown_signal"
    if (facts.reflection_confidence < 0.6 or facts.task_positive_load >= 0.62
            or facts.risk != 0):
        return False, "reflection_signal_low"
    return True, "reflection_injection_needed"


def _auto_conflict_control(facts: GateFacts) -> tuple[bool, str]:
    if facts.memory_competition is True and _finite(facts.conflict_level) and facts.conflict_level >= 0.75:
        return True, "conflict_control_needed"
    if not facts.micro_known:
        return False, "micro_unknown"
    if not all(_finite(value) for value in (facts.micro_confidence, facts.conflict_level)):
        return False, "unknown_signal"
    if (type(facts.micro_issues) is not int or not 1 <= facts.micro_issues <= 6
            or facts.micro_confidence < 0.86 or facts.conflict_level < 0.75):
        return False, "risk_or_uncertainty"
    return True, "conflict_control_needed"


def _auto_stalled_followup(facts: GateFacts) -> tuple[bool, str]:
    if (not facts.critic_structured or not _finite(facts.critic_confidence)
            or facts.critic_confidence < 0.85
            or type(facts.remaining_issues) is not int
            or not 1 <= facts.remaining_issues <= 6):
        return False, "critic_signal_uncertain"
    return True, "stalled_checklist_needed"


def _auto_basal_consult_promotion(facts: GateFacts) -> tuple[bool, str]:
    if facts.is_trivial:
        return False, "trivial_turn"
    if facts.memory_pressure:
        return True, "memory_consultation_needed"
    return _auto_conflict_control(facts)


def _auto_preview_suppression(facts: GateFacts) -> tuple[bool, str]:
    # This is a pre-draft fact, never the eventual policy action or answer.
    if facts.record_choice_pending is True:
        return True, "record_choice_pending"
    return False, "preview_context_needed"


def _auto_task_goals(facts: GateFacts) -> tuple[bool, str]:
    if type(facts.task_goal_count) is int and 2 <= facts.task_goal_count <= 4:
        return True, "explicit_task_goals"
    return False, "task_goals_unavailable"


def _auto_completion_replanning(facts: GateFacts) -> tuple[bool, str]:
    if (facts.generation_incomplete is True and facts.generation_length_limited is True
            and type(facts.task_goal_count) is int and 2 <= facts.task_goal_count <= 4
            and type(facts.remaining_output_tokens) is int and facts.remaining_output_tokens > 0):
        return True, 'length_limited_replanning'
    return False, 'replanning_not_needed'


DEFAULT_REGISTRY: Mapping[str, ModuleSpec] = MappingProxyType({
    COMPLETION_REPLANNING: ModuleSpec(
        module_id=COMPLETION_REPLANNING, phase='integration',
        policy_version='completion_replanning_v1',
        mode_env='DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE', legacy_env=None,
        default_mode=GateMode.OFF, auto_policy=_auto_completion_replanning,
    ),
    TASK_GOAL_ORCHESTRATION: ModuleSpec(
        module_id=TASK_GOAL_ORCHESTRATION,
        phase="pre_draft", policy_version="explicit_task_goals_v1",
        mode_env="DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE", legacy_env=None,
        default_mode=GateMode.OFF, auto_policy=_auto_task_goals,
    ),
    SPECULATIVE_PREVIEW_SUPPRESSION: ModuleSpec(
        module_id=SPECULATIVE_PREVIEW_SUPPRESSION,
        phase="pre_draft", policy_version="pending_record_choice_v1",
        mode_env="DUALBRAIN_MODULE_SPECULATIVE_PREVIEW_SUPPRESSION_MODE", legacy_env=None,
        default_mode=GateMode.OFF, auto_policy=_auto_preview_suppression,
    ),
    CONVERSATION_SELECTIVE_CONSULTATION: ModuleSpec(
        module_id=CONVERSATION_SELECTIVE_CONSULTATION,
        phase="routing", policy_version="literal_or_choice_clarification_v3",
        mode_env="DUALBRAIN_MODULE_CONVERSATION_SELECTIVE_CONSULTATION_MODE",
        legacy_env="DUALBRAIN_CONVERSATION_SELECTIVE_CONSULT",
        default_mode=GateMode.OFF, auto_policy=_auto_selective_consultation,
    ),
    ACC_MONITOR: ModuleSpec(
        module_id=ACC_MONITOR,
        phase="conflict_regulation", policy_version="acc_observation_v1",
        mode_env="DUALBRAIN_MODULE_ACC_MONITOR_MODE",
        legacy_env="DUALBRAIN_ACC_MONITOR",
        default_mode=GateMode.ON, auto_policy=_auto_acc_monitor,
    ),
    CEREBELLAR_CORRECTION: ModuleSpec(
        module_id=CEREBELLAR_CORRECTION,
        phase="conflict_regulation", policy_version="cerebellar_micro_v1",
        mode_env="DUALBRAIN_MODULE_CEREBELLAR_CORRECTION_MODE",
        legacy_env="DUALBRAIN_CEREBELLUM_MICRO_CORRECTION",
        default_mode=GateMode.ON, auto_policy=_auto_cerebellar_correction,
    ),
    DEFAULT_MODE_REFLECTION_INJECTION: ModuleSpec(
        module_id=DEFAULT_MODE_REFLECTION_INJECTION,
        phase="unconscious", policy_version="reflection_injection_v1",
        mode_env="DUALBRAIN_MODULE_DEFAULT_MODE_REFLECTION_INJECTION_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_reflection_injection,
    ),
    SYSTEM2_STALLED_FOLLOWUP: ModuleSpec(
        module_id=SYSTEM2_STALLED_FOLLOWUP,
        phase="system2_refinement", policy_version="stalled_checklist_v1",
        mode_env="DUALBRAIN_MODULE_SYSTEM2_STALLED_FOLLOWUP_MODE",
        legacy_env="DUALBRAIN_SYSTEM2_STALLED_FOLLOWUP",
        default_mode=GateMode.ON, auto_policy=_auto_stalled_followup,
    ),
    EXECUTIVE_ADVICE: ModuleSpec(
        module_id=EXECUTIVE_ADVICE,
        phase="observer_preparation", policy_version="confirmed_recovery_v4",
        mode_env="DUALBRAIN_MODULE_EXECUTIVE_ADVICE_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_optional_observation,
    ),
    DIRECTOR_ADVICE: ModuleSpec(
        module_id=DIRECTOR_ADVICE,
        phase="observer_preparation", policy_version="local_controls_v6",
        mode_env="DUALBRAIN_MODULE_DIRECTOR_ADVICE_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_director_advice,
    ),
    EXECUTIVE_METRICS_OBSERVATION: ModuleSpec(
        module_id=EXECUTIVE_METRICS_OBSERVATION,
        phase="post_turn_observation", policy_version="compact_delivery_v5",
        mode_env="DUALBRAIN_MODULE_EXECUTIVE_METRICS_OBSERVATION_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_metrics_observation,
    ),
    ACC_CONSULT_PROMOTION: ModuleSpec(
        module_id=ACC_CONSULT_PROMOTION,
        phase="routing", policy_version="acc_memory_review_v2",
        mode_env="DUALBRAIN_MODULE_ACC_CONSULT_PROMOTION_MODE",
        legacy_env="DUALBRAIN_ACC_OVERRIDE_CONSULT",
        default_mode=GateMode.OFF, auto_policy=_auto_conflict_control,
    ),
    ACC_SYSTEM2_PROMOTION: ModuleSpec(
        module_id=ACC_SYSTEM2_PROMOTION,
        phase="routing", policy_version="acc_memory_review_v2",
        mode_env="DUALBRAIN_MODULE_ACC_SYSTEM2_PROMOTION_MODE",
        legacy_env="DUALBRAIN_ACC_SYSTEM2_BUMP",
        default_mode=GateMode.OFF, auto_policy=_auto_conflict_control,
    ),
    ACC_TEMPERATURE_ADJUSTMENT: ModuleSpec(
        module_id=ACC_TEMPERATURE_ADJUSTMENT,
        phase="routing", policy_version="acc_conflict_control_v1",
        mode_env="DUALBRAIN_MODULE_ACC_TEMPERATURE_ADJUSTMENT_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_conflict_control,
    ),
    BASAL_CONSULT_PROMOTION: ModuleSpec(
        module_id=BASAL_CONSULT_PROMOTION,
        phase="routing", policy_version="basal_consult_promotion_v1",
        mode_env="DUALBRAIN_MODULE_BASAL_CONSULT_PROMOTION_MODE", legacy_env=None,
        default_mode=GateMode.ON, auto_policy=_auto_basal_consult_promotion,
    ),
})


def _parse_mode(value: object, *, legacy: bool) -> GateMode | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"on", "1", "true", "yes"}:
        return GateMode.ON
    if normalized in {"off", "0", "false", "no"}:
        return GateMode.OFF
    if normalized == "auto" and not legacy:
        return GateMode.AUTO
    return None


class ThalamicReticularNucleus:
    """Resolve configuration and pure policies without executing any module.

    The registry is an immutable copy of an explicit mapping. Contributions add
    reviewed specifications here, rather than having modules register themselves.
    """

    def __init__(self, registry: Mapping[str, ModuleSpec] | None = None) -> None:
        self._registry = _validated_registry(DEFAULT_REGISTRY if registry is None else registry)

    @property
    def registry(self) -> Mapping[str, ModuleSpec]:
        return self._registry

    def configure(
        self, module_id: str, environment: Mapping[str, str] | None = None,
    ) -> GateConfig:
        spec = self.registry.get(module_id)
        if spec is None:
            return GateConfig(GateMode.OFF, "default", False)
        environment = os.environ if environment is None else environment
        if spec.mode_env in environment:
            mode = _parse_mode(environment[spec.mode_env], legacy=False)
            return GateConfig(mode or GateMode.OFF, "canonical", mode is not None)
        if spec.legacy_env is not None and spec.legacy_env in environment:
            mode = _parse_mode(environment[spec.legacy_env], legacy=True)
            return GateConfig(mode or GateMode.OFF, "legacy", mode is not None)
        return GateConfig(spec.default_mode, "default", True)

    def decide(
        self, module_id: str, *, facts: GateFacts, config: GateConfig | None = None,
    ) -> GateDecision:
        spec = self.registry.get(module_id)
        if spec is None:
            return GateDecision(
                module="unknown", phase="unknown", mode=GateMode.OFF,
                source="default", eligible=False, enabled=False,
                reason="unknown_module", policy_version="unknown",
            )
        config = self.configure(module_id) if config is None else config
        valid = config.valid is True and isinstance(config.mode, GateMode)
        mode = config.mode if isinstance(config.mode, GateMode) else GateMode.OFF
        source = config.source if type(config.source) is str and config.source in _CONFIG_SOURCES else "explicit"
        eligible = bool(facts.eligible)
        enabled = False
        if not valid:
            reason = "invalid_configuration"
        elif mode is GateMode.OFF:
            reason = "forced_off"
        elif not eligible:
            reason = facts.blocked_reason if facts.blocked_reason in BLOCKED_REASONS else "outside_scope"
        elif mode is GateMode.ON:
            enabled, reason = True, "forced_on"
        else:
            try:
                enabled, reason = spec.auto_policy(facts)
                if type(enabled) is not bool or reason not in AUTO_REASONS:
                    enabled, reason = False, "policy_error"
            except Exception:
                enabled, reason = False, "policy_error"
        return GateDecision(
            module=spec.module_id, phase=spec.phase, mode=mode, source=source,
            eligible=eligible, enabled=enabled, reason=reason,
            policy_version=spec.policy_version,
        )


__all__ = [
    "ACC_MONITOR", "CEREBELLAR_CORRECTION", "CONVERSATION_SELECTIVE_CONSULTATION",
    "GateMode", "GateConfig", "GateFacts", "ModuleSpec", "GateDecision",
    "ThalamicReticularNucleus", "DEFAULT_REGISTRY",
    "DEFAULT_MODE_REFLECTION_INJECTION", "SYSTEM2_STALLED_FOLLOWUP",
    "EXECUTIVE_ADVICE", "DIRECTOR_ADVICE", "EXECUTIVE_METRICS_OBSERVATION",
    "ACC_CONSULT_PROMOTION", "ACC_SYSTEM2_PROMOTION", "ACC_TEMPERATURE_ADJUSTMENT",
    "BASAL_CONSULT_PROMOTION",
    "SPECULATIVE_PREVIEW_SUPPRESSION",
    "TASK_GOAL_ORCHESTRATION",
    "COMPLETION_REPLANNING",
]
