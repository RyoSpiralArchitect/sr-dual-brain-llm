"""Predictive-coding inspired network-state estimation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .insula import InteroceptiveState
from .salience_network import SalienceSignal
from .thalamus import ThalamicRelay


def _clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


_REFLECTIVE_PATTERN = re.compile(
    r"(reflect|reflection|metaphor|story|poem|imagine|dream|feel|symbol|"
    r"内省|比喩|物語|詩|夢|感じ|象徴|想像)",
    flags=re.IGNORECASE,
)

_MEMORY_PATTERN = re.compile(
    r"(remember|recall|previous|earlier|history|past|before|log|trace|"
    r"前回|履歴|過去|以前|ログ|思い出)",
    flags=re.IGNORECASE,
)

_REVIEW_PATTERN = re.compile(
    r"(review|correctness|edge[- ]?case|bug|failure|fix|invalid|unsafe|"
    r"code review|snippet|empty|zero|none|null|"
    r"妥当|正しい|不具合|バグ|失敗|壊れる|レビュー)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PredictionErrorSignal:
    language: float
    executive_control: float
    salience: float
    default_mode: float
    attention: float
    valuation: float
    memory: float
    overall: float
    dominant_channel: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "language": float(_clamp01(self.language)),
            "executive_control": float(_clamp01(self.executive_control)),
            "salience": float(_clamp01(self.salience)),
            "default_mode": float(_clamp01(self.default_mode)),
            "attention": float(_clamp01(self.attention)),
            "valuation": float(_clamp01(self.valuation)),
            "memory": float(_clamp01(self.memory)),
            "overall": float(_clamp01(self.overall)),
            "dominant_channel": self.dominant_channel,
        }


@dataclass(frozen=True)
class NetworkStateProfile:
    language: float
    executive_control: float
    salience: float
    default_mode: float
    attention: float
    valuation: float
    task_positive_load: float
    task_positive_mode: str
    default_mode_load: float
    dominant_network: str
    top_networks: Tuple[str, ...]
    phase_state: str
    suppress_default_mode: bool
    consult_bias: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "language": float(_clamp01(self.language)),
            "executive_control": float(_clamp01(self.executive_control)),
            "salience": float(_clamp01(self.salience)),
            "default_mode": float(_clamp01(self.default_mode)),
            "attention": float(_clamp01(self.attention)),
            "valuation": float(_clamp01(self.valuation)),
            "task_positive_load": float(_clamp01(self.task_positive_load)),
            "task_positive_mode": self.task_positive_mode,
            "default_mode_load": float(_clamp01(self.default_mode_load)),
            "dominant_network": self.dominant_network,
            "top_networks": list(self.top_networks),
            "phase_state": self.phase_state,
            "suppress_default_mode": bool(self.suppress_default_mode),
            "consult_bias": float(max(-1.0, min(1.0, self.consult_bias))),
        }


@dataclass(frozen=True)
class PredictionFrame:
    q_type_hint: str
    top_down: Mapping[str, float]
    bottom_up: Mapping[str, float]
    prediction_error: PredictionErrorSignal
    networks: NetworkStateProfile
    system2_pressure: float
    system2_ready: bool
    notes: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "q_type_hint": self.q_type_hint,
            "top_down": {key: float(_clamp01(val)) for key, val in self.top_down.items()},
            "bottom_up": {
                key: float(_clamp01(val)) for key, val in self.bottom_up.items()
            },
            "prediction_error": self.prediction_error.to_payload(),
            "networks": self.networks.to_payload(),
            "system2_pressure": float(_clamp01(self.system2_pressure)),
            "system2_ready": bool(self.system2_ready),
            "notes": list(self.notes),
        }


class PredictiveCodingController:
    """Estimate network engagement and prediction error for routing."""

    def evaluate(
        self,
        *,
        question: str,
        q_type_hint: str,
        precision_priority: bool,
        focus_metric: float,
        affect: Mapping[str, float],
        novelty: float,
        hemisphere_mode: str,
        hemisphere_bias: float,
        collaboration_strength: float,
        interoception: InteroceptiveState | None,
        salience_signal: SalienceSignal | None,
        thalamic_relay: ThalamicRelay | None,
        context_signal_len: int,
        has_working_memory: bool,
        has_long_term_memory: bool,
        has_hippocampal_memory: bool,
        is_trivial_chat: bool,
    ) -> PredictionFrame:
        q = str(question or "").strip()
        q_lower = q.lower()
        q_type_norm = str(q_type_hint or "easy").strip().lower()
        q_type_weight = (
            0.85 if q_type_norm == "hard" else 0.60 if q_type_norm == "medium" else 0.20
        )
        focus_metric = _clamp01(focus_metric)
        novelty = _clamp01(novelty)
        hemisphere_bias = _clamp01(hemisphere_bias)
        collaboration_strength = _clamp01(collaboration_strength)

        risk = _clamp01(affect.get("risk", 0.0))
        arousal = _clamp01(affect.get("arousal", 0.0))

        line_count = q.count("\n") + (1 if q else 0)
        delimiter_hits = len(re.findall(r"[,、，;；:：]", q))
        has_digits = bool(re.search(r"\d", q))
        inline_code = bool(re.search(r"`[^`]{4,}`", q))
        has_code = inline_code or ("```" in q)
        has_symbolic = has_code or bool(
            re.search(r"[=<>≤≥+\-*/%^]|→|⇒|∴|∵", q)
        )
        reflective_prompt = bool(_REFLECTIVE_PATTERN.search(q))
        memory_probe = bool(_MEMORY_PATTERN.search(q))
        review_prompt = bool(_REVIEW_PATTERN.search(q))

        left_bias = hemisphere_bias if hemisphere_mode == "left" else 0.0
        right_bias = hemisphere_bias if hemisphere_mode == "right" else 0.0
        balanced_bias = (
            max(0.0, 1.0 - hemisphere_bias) if hemisphere_mode == "balanced" else 0.0
        )

        salience_level = (
            float(salience_signal.level)
            if salience_signal is not None
            else float(interoception.salience) if interoception is not None else 0.0
        )
        relay_gain = (
            float(thalamic_relay.relay_gain) if thalamic_relay is not None else 0.0
        )
        relay_target = (
            str(thalamic_relay.target_network or "idle")
            if thalamic_relay is not None
            else "idle"
        )

        top_down = {
            "language": _clamp01(
                0.26 + 0.18 * left_bias + 0.08 * balanced_bias + 0.12 * (1.0 - q_type_weight)
            ),
            "executive_control": _clamp01(
                0.18
                + 0.34 * q_type_weight
                + 0.14 * focus_metric
                + (0.10 if precision_priority else 0.0)
                + 0.08 * left_bias
            ),
            "salience": _clamp01(
                0.20 + 0.26 * novelty + 0.24 * risk + 0.10 * arousal + 0.10 * q_type_weight
            ),
            "default_mode": _clamp01(
                0.14
                + 0.22 * right_bias
                + 0.14 * collaboration_strength
                + (0.16 if reflective_prompt else 0.0)
                + 0.06 * balanced_bias
            ),
            "attention": _clamp01(
                0.18
                + 0.24 * focus_metric
                + 0.18 * q_type_weight
                + (0.14 if precision_priority else 0.0)
                + (0.10 if has_symbolic else 0.0)
            ),
            "valuation": _clamp01(
                0.10 + 0.34 * risk + 0.22 * novelty + 0.14 * arousal + 0.10 * right_bias
            ),
            "memory": _clamp01(
                0.10
                + (0.18 if has_long_term_memory else 0.0)
                + (0.16 if has_hippocampal_memory else 0.0)
                + 0.18 * novelty
                + (0.12 if memory_probe else 0.0)
            ),
        }

        bottom_up = {
            "language": _clamp01(
                (
                    0.34 * float(salience_signal.language_score)
                    if salience_signal is not None
                    else 0.18
                )
                + 0.20 * (1.0 - min(1.0, has_symbolic * 1.0))
                + 0.10 * left_bias
                + 0.08 * balanced_bias
                + 0.10 * min(1.0, len(q_lower) / 180.0)
            ),
            "executive_control": _clamp01(
                (0.40 * float(salience_signal.executive_score) if salience_signal is not None else 0.0)
                + (0.18 * float(interoception.uncertainty) if interoception is not None else 0.0)
                + 0.18 * q_type_weight
                + (0.10 if review_prompt else 0.0)
                + (0.08 if relay_target == "executive_control" else 0.0)
            ),
            "salience": _clamp01(
                0.60 * salience_level
                + (0.18 * float(interoception.urgency) if interoception is not None else 0.0)
                + 0.12 * risk
                + 0.10 * novelty
            ),
            "default_mode": _clamp01(
                (0.42 * float(salience_signal.default_mode_score) if salience_signal is not None else 0.0)
                + (0.22 if reflective_prompt else 0.0)
                + 0.12 * collaboration_strength
                + (0.10 * float(interoception.stability) if interoception is not None else 0.0)
                - (0.22 if thalamic_relay is not None and thalamic_relay.suppress_default_mode else 0.0)
            ),
            "attention": _clamp01(
                0.24 * focus_metric
                + (0.20 if has_symbolic else 0.0)
                + (0.10 if has_digits else 0.0)
                + (0.12 if line_count >= 2 else 0.0)
                + (0.08 if delimiter_hits >= 2 else 0.0)
                + 0.12 * relay_gain
                + 0.14 * q_type_weight
            ),
            "valuation": _clamp01(
                0.42 * risk
                + 0.22 * arousal
                + 0.18 * novelty
                + 0.12 * salience_level
            ),
            "memory": _clamp01(
                (0.28 * float(salience_signal.memory_score) if salience_signal is not None else 0.0)
                + (0.20 if memory_probe else 0.0)
                + (0.14 if has_working_memory else 0.0)
                + (0.18 if has_long_term_memory else 0.0)
                + (0.18 if has_hippocampal_memory else 0.0)
                + (0.12 if relay_target == "memory_recall" else 0.0)
                + 0.10 * min(1.0, float(context_signal_len) / 320.0)
            ),
        }

        if is_trivial_chat:
            for key in ("executive_control", "salience", "attention", "valuation", "memory"):
                bottom_up[key] = _clamp01(bottom_up[key] * 0.55)
            bottom_up["language"] = _clamp01(max(bottom_up["language"], 0.55))
            bottom_up["default_mode"] = _clamp01(max(bottom_up["default_mode"], 0.42))

        errors = {
            key: _clamp01(abs(float(top_down[key]) - float(bottom_up[key])))
            for key in top_down.keys()
        }
        dominant_error = max(errors, key=errors.get)
        overall_error = _clamp01(
            0.20 * errors["executive_control"]
            + 0.18 * errors["attention"]
            + 0.16 * errors["salience"]
            + 0.14 * errors["memory"]
            + 0.12 * errors["default_mode"]
            + 0.10 * errors["language"]
            + 0.10 * errors["valuation"]
        )
        prediction_error = PredictionErrorSignal(
            language=errors["language"],
            executive_control=errors["executive_control"],
            salience=errors["salience"],
            default_mode=errors["default_mode"],
            attention=errors["attention"],
            valuation=errors["valuation"],
            memory=errors["memory"],
            overall=overall_error,
            dominant_channel=dominant_error,
        )

        networks = {
            "language": _clamp01(
                0.66 * bottom_up["language"] + 0.18 * top_down["language"] - 0.10 * errors["executive_control"]
            ),
            "executive_control": _clamp01(
                0.68 * bottom_up["executive_control"]
                + 0.16 * top_down["executive_control"]
                + 0.16 * errors["executive_control"]
                + (0.06 if review_prompt else 0.0)
                + 0.06 * errors["memory"]
            ),
            "salience": _clamp01(
                0.52 * bottom_up["salience"]
                + 0.18 * top_down["salience"]
                + 0.14 * errors["salience"]
                + 0.08 * risk
            ),
            "default_mode": _clamp01(
                0.64 * bottom_up["default_mode"]
                + 0.20 * top_down["default_mode"]
                - 0.18 * max(
                    bottom_up["executive_control"],
                    bottom_up["attention"],
                )
                + (0.08 if reflective_prompt else 0.0)
                + 0.06 * right_bias
            ),
            "attention": _clamp01(
                0.76 * bottom_up["attention"]
                + 0.14 * top_down["attention"]
                + 0.14 * errors["attention"]
                + (0.10 if has_symbolic else 0.0)
                + (0.08 if review_prompt else 0.0)
                + 0.08 * errors["memory"]
                + (0.06 if relay_target == "memory_recall" else 0.0)
            ),
            "valuation": _clamp01(
                0.54 * bottom_up["valuation"]
                + 0.18 * top_down["valuation"]
                + 0.12 * errors["valuation"]
                - 0.10 * max(bottom_up["attention"], bottom_up["executive_control"])
            ),
        }

        top_networks = tuple(
            name for name, _score in sorted(
                networks.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        )
        dominant_network = top_networks[0] if top_networks else "language"
        if networks["attention"] >= networks["executive_control"]:
            task_positive_mode = "attention"
            task_positive_load = networks["attention"]
        else:
            task_positive_mode = "executive_control"
            task_positive_load = networks["executive_control"]
        default_mode_load = networks["default_mode"]
        suppress_default_mode = bool(
            (thalamic_relay is not None and thalamic_relay.suppress_default_mode)
            or task_positive_load >= 0.56
            or networks["salience"] >= 0.62
        )

        if overall_error >= 0.54:
            phase_state = "prediction_error"
        elif task_positive_load >= 0.56:
            phase_state = "task_positive"
        elif default_mode_load >= 0.48 and not suppress_default_mode:
            phase_state = "reflective"
        else:
            phase_state = "stabilizing"

        consult_bias = 0.0
        if dominant_network in {"executive_control", "attention", "salience"}:
            consult_bias += 0.10 * networks[dominant_network]
        if dominant_error in {"executive_control", "attention", "memory"}:
            consult_bias += 0.08 * errors[dominant_error]
        if dominant_network == "default_mode" and not suppress_default_mode:
            consult_bias -= 0.06 * default_mode_load
        consult_bias = max(-0.25, min(0.25, consult_bias))

        system2_pressure = _clamp01(
            max(
                0.48 * task_positive_load
                + 0.22 * networks["salience"]
                + 0.20 * overall_error
                + (0.10 if precision_priority else 0.0),
                0.34 * bottom_up["memory"]
                + 0.26 * errors["memory"]
                + 0.20 * q_type_weight
                + (0.10 if relay_target == "memory_recall" else 0.0),
                0.42 * task_positive_load
                + 0.18 * overall_error
                + 0.10 * errors["memory"]
                + (0.18 if review_prompt else 0.0)
                + (0.12 if has_symbolic else 0.0)
            )
        )
        system2_ready = bool(
            not is_trivial_chat
            and (
                system2_pressure >= (0.50 if precision_priority else 0.57)
                or (
                    dominant_network in {"executive_control", "attention", "salience"}
                    and task_positive_load >= 0.55
                    and overall_error >= 0.34
                )
                or (
                    review_prompt
                    and has_symbolic
                    and dominant_network in {"executive_control", "attention"}
                    and task_positive_load >= 0.45
                    and (
                        errors["memory"] >= 0.18
                        or errors["executive_control"] >= 0.18
                    )
                )
            )
        )

        network_profile = NetworkStateProfile(
            language=networks["language"],
            executive_control=networks["executive_control"],
            salience=networks["salience"],
            default_mode=networks["default_mode"],
            attention=networks["attention"],
            valuation=networks["valuation"],
            task_positive_load=task_positive_load,
            task_positive_mode=task_positive_mode,
            default_mode_load=default_mode_load,
            dominant_network=dominant_network,
            top_networks=top_networks,
            phase_state=phase_state,
            suppress_default_mode=suppress_default_mode,
            consult_bias=consult_bias,
        )

        notes = [
            f"q_type:{q_type_norm}",
            f"phase:{phase_state}",
            f"dominant:{dominant_network}",
            f"error:{dominant_error}",
        ]
        if relay_target != "idle":
            notes.append(f"relay:{relay_target}")
        if system2_ready:
            notes.append("system2_ready")

        return PredictionFrame(
            q_type_hint=q_type_norm,
            top_down=top_down,
            bottom_up=bottom_up,
            prediction_error=prediction_error,
            networks=network_profile,
            system2_pressure=system2_pressure,
            system2_ready=system2_ready,
            notes=tuple(notes),
        )


__all__ = [
    "NetworkStateProfile",
    "PredictionErrorSignal",
    "PredictionFrame",
    "PredictiveCodingController",
]
