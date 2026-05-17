#!/usr/bin/env python3
"""Benchmark latent motif handling and plain-answer leakage for the unconscious field."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_system2 import _filter_questions, _load_questions, _parse_question_paths  # noqa: E402
from engine_stdio import EngineSession, _extract_metrics  # noqa: E402

INTERNAL_SECTION_MARKERS = (
    "[Unconscious Insight]",
    "[Stress Released]",
    "[Psychoid Attention Bias]",
    "[Psychoid Field Alignment]",
    "[Psychoid Signifiers]",
    "[Default Mode Reflection]",
    "[Coherence Integration]",
    "[Unconscious Linguistic Fabric]",
    "[Linguistic Motifs]",
    "[Hemisphere Routing]",
    "[Hemisphere Semantic Tilt]",
    "[Collaboration Profile]",
    "[Neural Impulse Activity]",
    "[Architecture Path]",
)

ARCHETYPE_IDS = (
    "self",
    "persona",
    "shadow",
    "syzygy",
    "sage",
    "trickster",
    "hero",
    "great_mother",
    "child",
)

ARCHETYPE_CUES = {
    "self": {
        "center",
        "integration",
        "integrative",
        "balance",
        "equilibrium",
        "whole",
    },
    "persona": {
        "persona",
        "mask",
        "interface",
        "boundary",
        "settings",
        "screen",
        "ux",
        "role",
    },
    "shadow": {
        "shadow",
        "legacy",
        "fear",
        "risk",
        "brittle",
        "avoid",
        "avoiding",
        "anxiety",
        "hidden",
    },
    "syzygy": {
        "syzygy",
        "mirror",
        "reflect",
        "reflection",
        "twin",
        "pair",
        "surface",
    },
    "sage": {
        "sage",
        "mentor",
        "patient",
        "teach",
        "junior",
        "debug",
        "guide",
    },
    "trickster": {
        "trickster",
        "disruption",
        "reversal",
        "chaos",
        "misdirection",
        "unexpected",
    },
    "hero": {
        "hero",
        "bridge",
        "journey",
        "path",
        "trial",
        "overcome",
        "roadmap",
        "fixes",
    },
    "great_mother": {
        "great_mother",
        "forest",
        "nurturing",
        "safety",
        "grounded",
        "soil",
        "care",
    },
    "child": {
        "child",
        "sprout",
        "experiment",
        "prototype",
        "new",
        "dawn",
        "rebirth",
    },
}


def _last_event(events: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for ev in reversed(events or []):
        if ev.get("event") == name:
            return ev
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return float(count) / float(total)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalise_trace_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _tokenise_cues(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, list):
            for item in value:
                tokens.update(_tokenise_cues(item))
            continue
        text = str(value or "").lower()
        tokens.update(_normalise_trace_token(part) for part in re.findall(r"[\w-]+", text))
    return {tok for tok in tokens if tok}


def _archetype_ids_from_motifs(motifs: List[str]) -> List[str]:
    found: set[str] = set()
    for motif in motifs:
        norm = _normalise_trace_token(motif)
        for archetype_id in ARCHETYPE_IDS:
            if (
                norm == archetype_id
                or norm.endswith(f"_{archetype_id}")
                or f"_{archetype_id}_" in norm
                or norm.startswith(f"{archetype_id}_")
            ):
                found.add(archetype_id)
    return sorted(found)


def _cue_archetypes(*, question: str, tags: List[Any]) -> Dict[str, Any]:
    tokens = _tokenise_cues(question, tags)
    hits: Dict[str, List[str]] = {}
    for archetype_id, cues in ARCHETYPE_CUES.items():
        matched = sorted(tokens.intersection(cues))
        if matched:
            hits[archetype_id] = matched
    return {
        "ids": sorted(hits),
        "hits": hits,
    }


def _activation_entropy(values: List[float]) -> Optional[float]:
    clean = [max(0.0, float(value)) for value in values if math.isfinite(float(value))]
    total = sum(clean)
    if total <= 0.0 or len(clean) <= 1:
        return None
    probs = [value / total for value in clean if value > 0.0]
    if not probs:
        return None
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return float(entropy / math.log(len(clean)))


def _archetype_activation_trace(
    *,
    unconscious_summary: Dict[str, Any],
    top_k: List[str],
    motifs: List[str],
    attention_bias: List[Any],
    question: str,
    tags: List[Any],
) -> Dict[str, Any]:
    raw_map = _as_list(unconscious_summary.get("archetype_map"))
    activations: List[Dict[str, Any]] = []
    for item in raw_map:
        payload = _as_dict(item)
        archetype_id = _normalise_trace_token(payload.get("id"))
        if not archetype_id:
            continue
        activations.append(
            {
                "id": archetype_id,
                "label": str(payload.get("label") or archetype_id),
                "intensity": _safe_float(payload.get("intensity")),
            }
        )
    activations.sort(
        key=lambda item: (
            item.get("intensity") if item.get("intensity") is not None else -1.0
        ),
        reverse=True,
    )

    intensities = [
        float(item["intensity"])
        for item in activations
        if item.get("intensity") is not None
    ]
    top1_margin = None
    if len(intensities) >= 2:
        top1_margin = float(intensities[0] - intensities[1])

    top_ids = {_normalise_trace_token(item) for item in top_k}
    motif_ids = set(_archetype_ids_from_motifs(motifs))
    attention_ids = {
        _normalise_trace_token(_as_dict(item).get("archetype"))
        for item in attention_bias
    }
    attention_ids.discard("")
    cue = _cue_archetypes(question=question, tags=tags)
    cue_ids = set(cue["ids"])
    cue_top_hits = sorted(cue_ids.intersection(top_ids))
    cue_motif_hits = sorted(cue_ids.intersection(motif_ids))
    cue_attention_hits = sorted(cue_ids.intersection(attention_ids))

    if not cue_ids:
        cue_alignment = "unlabeled"
        cue_distance = None
    elif cue_top_hits:
        cue_alignment = "top_k_aligned"
        cue_distance = 0
    elif cue_motif_hits:
        cue_alignment = "motif_only"
        cue_distance = 1
    elif cue_attention_hits:
        cue_alignment = "psychoid_only"
        cue_distance = 1
    else:
        cue_alignment = "divergent"
        cue_distance = 2

    entropy = _activation_entropy(intensities)
    ambiguous = bool(
        entropy is not None
        and (
            entropy >= 0.92
            or (top1_margin is not None and top1_margin <= 0.035)
        )
    )
    motif_without_top = sorted(motif_ids.difference(top_ids))
    return {
        "cue": cue,
        "cue_alignment": cue_alignment,
        "cue_distance": cue_distance,
        "cue_top_k_hits": cue_top_hits,
        "cue_motif_hits": cue_motif_hits,
        "cue_attention_hits": cue_attention_hits,
        "cue_missing_from_top_k": sorted(cue_ids.difference(top_ids)),
        "motif_archetypes": sorted(motif_ids),
        "attention_archetypes": sorted(attention_ids),
        "motif_without_top_k": motif_without_top,
        "motif_top_k_divergent": bool(motif_without_top),
        "top1_margin": top1_margin,
        "activation_entropy": entropy,
        "ambiguous_activation": ambiguous,
        "top_activations": activations[:5],
    }


def _internal_leak_markers(answer: str) -> List[str]:
    lowered = str(answer or "").lower()
    return [marker for marker in INTERNAL_SECTION_MARKERS if marker.lower() in lowered]


def _latent_grounding_score(
    *,
    top_k_count: int,
    emergent_count: int,
    psychoid_resonance: Optional[float],
    weave_score: Optional[float],
    motif_score: Optional[float],
) -> float:
    top_component = min(1.0, max(0.0, float(top_k_count) / 3.0))
    emergent_component = min(1.0, max(0.0, float(emergent_count) / 2.0))
    resonance_component = max(0.0, min(1.0, psychoid_resonance or 0.0))
    weave_component = max(0.0, min(1.0, weave_score or 0.0))
    motif_component = max(0.0, min(1.0, motif_score or 0.0))
    return float(
        round(
            0.2 * top_component
            + 0.2 * emergent_component
            + 0.2 * resonance_component
            + 0.25 * weave_component
            + 0.15 * motif_component,
            6,
        )
    )


def _incubation_pressure(
    *,
    cache_depth: Optional[int],
    emergent_count: int,
    stress_released: Optional[float],
    psychoid_resonance: Optional[float],
    signifier_chain_len: int,
    repeated_loops: Optional[int],
) -> float:
    cache_component = min(1.0, max(0.0, float(cache_depth or 0) / 4.0))
    chain_component = min(1.0, max(0.0, float(signifier_chain_len) / 6.0))
    loop_component = min(1.0, max(0.0, float(repeated_loops or 0) / 4.0))
    stress_component = max(0.0, min(1.0, stress_released or 0.0))
    resonance_component = max(0.0, min(1.0, psychoid_resonance or 0.0))
    emergent_release = min(1.0, max(0.0, float(emergent_count) / 2.0))
    unreleased = 1.0 - 0.35 * emergent_release
    pressure = (
        0.32 * cache_component
        + 0.22 * chain_component
        + 0.18 * loop_component
        + 0.14 * stress_component
        + 0.14 * resonance_component
    )
    return float(round(max(0.0, min(1.0, pressure * unreleased)), 6))


def _harvest_attempt_details(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for item in _as_list(summary.get("harvest_attempts")):
        payload = _as_dict(item)
        if not payload:
            continue
        details.append(
            {
                "archetype": str(payload.get("archetype") or ""),
                "label": str(payload.get("label") or ""),
                "intensity": _safe_float(payload.get("intensity")),
                "novelty": _safe_float(payload.get("novelty")),
                "incubation_rounds": _safe_int(payload.get("incubation_rounds")),
                "trigger_similarity": _safe_float(payload.get("trigger_similarity")),
                "threshold": _safe_float(payload.get("threshold")),
                "threshold_gap": _safe_float(payload.get("threshold_gap")),
                "threshold_margin": _safe_float(payload.get("threshold_margin")),
                "emerged": bool(payload.get("emerged")),
                "similarity_pass": bool(payload.get("similarity_pass")),
                "incubation_pass": bool(payload.get("incubation_pass")),
                "intensity_pass": bool(payload.get("intensity_pass")),
                "failure_reasons": [
                    str(reason) for reason in _as_list(payload.get("failure_reasons"))
                ],
                "status": str(payload.get("status") or ""),
                "origin": str(payload.get("origin") or ""),
            }
        )
    return details


def _closest_near_miss(attempts: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [
        attempt
        for attempt in attempts
        if not attempt.get("emerged")
        and _safe_float(attempt.get("threshold_gap")) is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda attempt: (
            float(_safe_float(attempt.get("threshold_gap")) or 0.0),
            -float(_safe_float(attempt.get("trigger_similarity")) or 0.0),
        ),
    )


def _extract_case_signals(
    events: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    answer: str,
    *,
    question: str = "",
    tags: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    unconscious_summary = _as_dict(_last_event(events, "unconscious_field").get("summary"))
    psychoid_event_signal = _as_dict(_last_event(events, "psychoid_signal").get("signal"))
    psychoid_signal = psychoid_event_signal or _as_dict(unconscious_summary.get("psychoid_signal"))
    projection = _as_dict(_last_event(events, "psychoid_attention_projection").get("projection"))
    dmn = _last_event(events, "default_mode_reflection")
    task_positive = _last_event(events, "task_positive_network")
    coherence_event_signal = _as_dict(_last_event(events, "coherence_signal").get("signal"))
    coherence_metrics = _as_dict(metrics.get("coherence"))
    coherence = coherence_event_signal or coherence_metrics
    weave = _as_dict(_last_event(events, "coherence_unconscious_weave").get("fabric"))
    if not weave:
        weave = _as_dict(coherence.get("unconscious"))
    motifs = _as_dict(_last_event(events, "coherence_linguistic_motifs").get("motifs"))
    if not motifs:
        motifs = _as_dict(coherence.get("motifs"))

    top_k = [str(item) for item in _as_list(unconscious_summary.get("top_k"))]
    emergent_ideas = _as_list(unconscious_summary.get("emergent_ideas"))
    harvest_attempts = _harvest_attempt_details(unconscious_summary)
    closest_near_miss = _closest_near_miss(harvest_attempts)
    unconscious_motifs = [str(item) for item in _as_list(unconscious_summary.get("motifs"))]
    attention_bias = _as_list(psychoid_signal.get("attention_bias"))
    signifier_chain = [str(item) for item in _as_list(psychoid_signal.get("signifier_chain"))]
    reflections = _as_list(dmn.get("reflections"))

    psychoid_resonance = _safe_float(psychoid_signal.get("resonance"))
    psychoid_tension = _safe_float(psychoid_signal.get("psychoid_tension"))
    weave_score = _safe_float(weave.get("score"))
    motif_score = _safe_float(motifs.get("score"))
    cache_depth = _safe_int(unconscious_summary.get("cache_depth"))
    stress_released = _safe_float(unconscious_summary.get("stress_released"))
    repeated_loops = _safe_int(motifs.get("repeated_loops"))
    leak_markers = _internal_leak_markers(answer)
    archetype_trace = _archetype_activation_trace(
        unconscious_summary=unconscious_summary,
        top_k=top_k,
        motifs=unconscious_motifs,
        attention_bias=attention_bias,
        question=question,
        tags=tags or [],
    )
    incubation_pressure = _incubation_pressure(
        cache_depth=cache_depth,
        emergent_count=len(emergent_ideas),
        stress_released=stress_released,
        psychoid_resonance=psychoid_resonance,
        signifier_chain_len=len(signifier_chain),
        repeated_loops=repeated_loops,
    )
    grounding_score = _latent_grounding_score(
        top_k_count=len(top_k),
        emergent_count=len(emergent_ideas),
        psychoid_resonance=psychoid_resonance,
        weave_score=weave_score,
        motif_score=motif_score,
    )

    return {
        "unconscious": {
            "top_k": top_k,
            "top_count": len(top_k),
            "emergent_ideas": len(emergent_ideas),
            "harvest_attempts": harvest_attempts,
            "harvest_attempt_count": len(harvest_attempts),
            "harvest_near_miss_count": sum(
                1 for attempt in harvest_attempts if not attempt.get("emerged")
            ),
            "closest_near_miss": closest_near_miss,
            "stress_released": stress_released,
            "cache_depth": cache_depth,
            "motifs": unconscious_motifs,
            "score": grounding_score,
            "incubation_pressure": incubation_pressure,
            "unreleased_cache": bool((cache_depth or 0) > 0 and not emergent_ideas),
        },
        "archetype_trace": archetype_trace,
        "psychoid": {
            "present": bool(psychoid_signal),
            "attention_bias_count": len(attention_bias),
            "signifier_chain_len": len(signifier_chain),
            "resonance": psychoid_resonance,
            "tension": psychoid_tension,
            "projection_norm": _safe_float(projection.get("norm")),
            "projection_chain_length": _safe_int(projection.get("chain_length")),
        },
        "default_mode": {
            "reflection_count": len(reflections),
            "suppressed": bool(task_positive.get("suppressed")),
            "task_positive_load": _safe_float(task_positive.get("load")),
            "task_positive_mode": task_positive.get("mode"),
        },
        "coherence": {
            "combined": _safe_float(coherence.get("combined")),
            "tension": _safe_float(coherence.get("tension")),
            "mode": coherence.get("mode"),
            "linguistic_depth": _safe_float(coherence.get("linguistic_depth")),
            "unconscious_weave_score": weave_score,
            "unconscious_weave_emergent_count": _safe_int(weave.get("emergent_count")),
            "motif_score": motif_score,
            "motif_repeated_loops": repeated_loops,
        },
        "leakage": {
            "has_internal_leak": bool(leak_markers),
            "markers": leak_markers,
        },
    }


def _summarise_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_cases = [case for case in cases if not case.get("error")]
    total = len(cases)
    ok_total = len(ok_cases)

    def floats(path: tuple[str, ...]) -> List[float]:
        out: List[float] = []
        for case in ok_cases:
            value: Any = case
            for part in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
            number = _safe_float(value)
            if number is not None:
                out.append(number)
        return out

    leak_cases = sum(1 for case in ok_cases if case.get("leakage", {}).get("has_internal_leak"))
    psychoid_cases = sum(1 for case in ok_cases if case.get("psychoid", {}).get("present"))
    default_mode_cases = sum(
        1 for case in ok_cases if (case.get("default_mode", {}).get("reflection_count") or 0) > 0
    )
    suppressed_cases = sum(1 for case in ok_cases if case.get("default_mode", {}).get("suppressed"))
    emergent_cases = sum(
        1 for case in ok_cases if (case.get("unconscious", {}).get("emergent_ideas") or 0) > 0
    )
    unreleased_cache_cases = sum(
        1 for case in ok_cases if case.get("unconscious", {}).get("unreleased_cache")
    )
    weave_cases = sum(
        1
        for case in ok_cases
        if case.get("coherence", {}).get("unconscious_weave_score") is not None
    )
    motif_cases = sum(
        1 for case in ok_cases if case.get("coherence", {}).get("motif_score") is not None
    )
    cue_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("cue_alignment") != "unlabeled"
    )
    cue_top_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("cue_alignment") == "top_k_aligned"
    )
    cue_motif_only_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("cue_alignment") == "motif_only"
    )
    cue_divergent_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("cue_alignment") == "divergent"
    )
    motif_divergent_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("motif_top_k_divergent")
    )
    ambiguous_activation_cases = sum(
        1
        for case in ok_cases
        if case.get("archetype_trace", {}).get("ambiguous_activation")
    )
    cache_depths = [
        value
        for value in (_safe_int(case.get("unconscious", {}).get("cache_depth")) for case in ok_cases)
        if value is not None
    ]

    return {
        "total_cases": total,
        "ok_cases": ok_total,
        "error_cases": total - ok_total,
        "leak_cases": leak_cases,
        "leak_rate": _rate(leak_cases, ok_total),
        "psychoid_signal_rate": _rate(psychoid_cases, ok_total),
        "default_mode_reflection_rate": _rate(default_mode_cases, ok_total),
        "default_mode_suppressed_rate": _rate(suppressed_cases, ok_total),
        "emergent_idea_rate": _rate(emergent_cases, ok_total),
        "unreleased_cache_rate": _rate(unreleased_cache_cases, ok_total),
        "unconscious_weave_rate": _rate(weave_cases, ok_total),
        "linguistic_motif_rate": _rate(motif_cases, ok_total),
        "archetype_cue_cases": cue_cases,
        "archetype_cue_top_k_alignment_rate": _rate(cue_top_cases, cue_cases),
        "archetype_cue_motif_only_rate": _rate(cue_motif_only_cases, cue_cases),
        "archetype_cue_divergent_rate": _rate(cue_divergent_cases, cue_cases),
        "archetype_motif_top_k_divergence_rate": _rate(motif_divergent_cases, ok_total),
        "archetype_ambiguous_activation_rate": _rate(ambiguous_activation_cases, ok_total),
        "avg_archetype_top1_margin": _mean(floats(("archetype_trace", "top1_margin"))),
        "avg_archetype_activation_entropy": _mean(
            floats(("archetype_trace", "activation_entropy"))
        ),
        "avg_unconscious_score": _mean(floats(("unconscious", "score"))),
        "avg_incubation_pressure": _mean(
            floats(("unconscious", "incubation_pressure"))
        ),
        "avg_cache_depth": _mean([float(value) for value in cache_depths]),
        "max_cache_depth": max(cache_depths) if cache_depths else None,
        "avg_coherence_combined": _mean(floats(("coherence", "combined"))),
        "avg_coherence_tension": _mean(floats(("coherence", "tension"))),
        "avg_weave_score": _mean(floats(("coherence", "unconscious_weave_score"))),
        "avg_motif_score": _mean(floats(("coherence", "motif_score"))),
        "avg_motif_repeated_loops": _mean(floats(("coherence", "motif_repeated_loops"))),
        "avg_psychoid_resonance": _mean(floats(("psychoid", "resonance"))),
        "avg_psychoid_tension": _mean(floats(("psychoid", "tension"))),
        "avg_signifier_chain_len": _mean(floats(("psychoid", "signifier_chain_len"))),
        "avg_attention_bias_count": _mean(floats(("psychoid", "attention_bias_count"))),
        "avg_task_positive_load": _mean(floats(("default_mode", "task_positive_load"))),
        "avg_latency_ms": _mean(floats(("latency_ms",))),
    }


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


async def _run_case(
    *,
    session: EngineSession,
    question_entry: Dict[str, Any],
    index: int,
    run_id: str,
    answer_mode: str,
    leading_brain: str,
    system2_mode: str,
    executive_mode: str,
    executive_observer_mode: str,
) -> Dict[str, Any]:
    qid = f"{run_id}-c{index:03d}"
    question = str(question_entry.get("question") or "")
    session.telemetry.clear()
    started = time.perf_counter()
    error_text: Optional[str] = None
    answer = ""
    try:
        answer = await session.controller.process(
            question,
            qid=qid,
            answer_mode=answer_mode,
            leading_brain=(None if leading_brain == "auto" else leading_brain),
            system2_mode=system2_mode,
            executive_mode=executive_mode,
            executive_observer_mode=executive_observer_mode,
        )
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    events = list(session.telemetry.events)
    metrics = _extract_metrics(events) if events else {}
    tags = question_entry.get("tags") if isinstance(question_entry.get("tags"), list) else []
    signals = _extract_case_signals(
        events,
        metrics,
        answer,
        question=question,
        tags=tags,
    )
    return {
        "index": index,
        "id": question_entry.get("id") or f"q{index:03d}",
        "qid": qid,
        "question": question,
        "tags": tags,
        "answer_mode": answer_mode,
        "system2_mode": system2_mode,
        "latency_ms": _safe_float(metrics.get("latency_ms")) or elapsed_ms,
        "telemetry_events": len(events),
        "error": error_text,
        "answer_preview": answer[:280] if answer else "",
        **signals,
    }


async def _run(args: argparse.Namespace) -> int:
    question_paths = _parse_question_paths(args.questions)
    if not question_paths:
        raise ValueError("No --questions paths provided.")
    missing_paths = [path for path in question_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Questions file(s) not found: {paths}".format(
                paths=", ".join(str(path) for path in missing_paths)
            )
        )

    questions: List[Dict[str, Any]] = []
    for path in question_paths:
        questions.extend(_load_questions(path))
    questions = _filter_questions(
        questions,
        only_ids=getattr(args, "only_ids", None),
        only_tags=getattr(args, "only_tags", None),
    )
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(questions)
    if args.limit is not None and args.limit > 0:
        questions = questions[: int(args.limit)]
    if not questions:
        raise RuntimeError("No benchmark questions to run.")

    low_signal_filter = str(args.low_signal_filter or "on").strip().lower()
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = "1" if low_signal_filter == "on" else "0"

    run_id = time.strftime("unconscious_creativity_%Y%m%d_%H%M%S")
    session_id = str(args.session_id or run_id).strip() or run_id
    question_set_label = ",".join(str(path) for path in question_paths)
    print(f"[unconscious-bench] run_id={run_id}")
    print(f"[unconscious-bench] session_id={session_id}")
    print(f"[unconscious-bench] questions={len(questions)}")

    session = await EngineSession.create(session_id=session_id)
    try:
        llm_capable = bool(
            getattr(session.left, "uses_external_llm", False)
            and getattr(session.right, "uses_external_llm", False)
        )
        if not llm_capable:
            print("[unconscious-bench] warning: external LLM not configured for both hemispheres.")

        cases: List[Dict[str, Any]] = []
        for idx, entry in enumerate(questions, 1):
            case = await _run_case(
                session=session,
                question_entry=entry,
                index=idx,
                run_id=run_id,
                answer_mode=args.answer_mode,
                leading_brain=args.leading_brain,
                system2_mode=args.system2_mode,
                executive_mode=args.executive_mode,
                executive_observer_mode=args.executive_observer_mode,
            )
            cases.append(case)
            print(
                "[unconscious-bench] {idx:03d}/{total} id={id} latent={latent} "
                "coh={coh} weave={weave} motif={motif} trace={trace} "
                "incub={incub} leak={leak} error={error}".format(
                    idx=idx,
                    total=len(questions),
                    id=case.get("id"),
                    latent=case.get("unconscious", {}).get("score"),
                    coh=case.get("coherence", {}).get("combined"),
                    weave=case.get("coherence", {}).get("unconscious_weave_score"),
                    motif=case.get("coherence", {}).get("motif_score"),
                    trace=case.get("archetype_trace", {}).get("cue_alignment"),
                    incub=case.get("unconscious", {}).get("incubation_pressure"),
                    leak=case.get("leakage", {}).get("has_internal_leak"),
                    error=("yes" if case.get("error") else "no"),
                )
            )

        summary = _summarise_cases(cases)
        report = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question_set": question_set_label,
            "question_sets": [str(path) for path in question_paths],
            "config": {
                "session_id": session_id,
                "answer_mode": args.answer_mode,
                "leading_brain": args.leading_brain,
                "system2_mode": args.system2_mode,
                "executive_mode": args.executive_mode,
                "executive_observer_mode": args.executive_observer_mode,
                "low_signal_filter": low_signal_filter,
                "question_count": len(questions),
                "only_ids": getattr(args, "only_ids", None),
                "only_tags": getattr(args, "only_tags", None),
                "llm_capable": llm_capable,
            },
            "summary": summary,
            "cases": cases,
        }

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[unconscious-bench] wrote report: {output_path}")

        history_path = Path(args.history).expanduser().resolve() if args.history else None
        if history_path is not None:
            _append_history(
                history_path,
                {
                    "run_id": run_id,
                    "timestamp": report["timestamp"],
                    "question_set": question_set_label,
                    "summary": summary,
                    "config": report["config"],
                },
            )
            print(f"[unconscious-bench] appended history: {history_path}")

        print(
            "[unconscious-bench] summary ok={ok}/{total} leaks={leaks} "
            "latent={latent} incub={incub} coherence={coherence} "
            "weave={weave} motif={motif} cue_top={cue_top}".format(
                ok=summary.get("ok_cases"),
                total=summary.get("total_cases"),
                leaks=summary.get("leak_cases"),
                latent=summary.get("avg_unconscious_score"),
                incub=summary.get("avg_incubation_pressure"),
                coherence=summary.get("avg_coherence_combined"),
                weave=summary.get("avg_weave_score"),
                motif=summary.get("avg_motif_score"),
                cue_top=summary.get("archetype_cue_top_k_alignment_rate"),
            )
        )
        if args.expect_no_leaks and summary.get("leak_cases"):
            print("[unconscious-bench] internal debug markers leaked into answers.")
            return 3
    finally:
        await session.close()
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "examples" / "unconscious_creativity_benchmark_questions.json"),
        help="Comma separated paths to unconscious creativity benchmark question sets.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_creativity_last.json"),
    )
    parser.add_argument(
        "--history",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_creativity_history.jsonl"),
        help="Append compact run summary to JSONL history (set empty to disable).",
    )
    parser.add_argument("--session-id", default="")
    parser.add_argument("--answer-mode", choices=["plain", "debug", "annotated", "meta"], default="plain")
    parser.add_argument("--system2-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--leading-brain", choices=["auto", "left", "right"], default="auto")
    parser.add_argument(
        "--executive-mode",
        choices=["off", "observe", "assist", "polish"],
        default="off",
    )
    parser.add_argument(
        "--executive-observer-mode",
        choices=["off", "metrics", "director", "both"],
        default="off",
    )
    parser.add_argument("--low-signal-filter", choices=["on", "off"], default="on")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--only-tags", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expect-no-leaks", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.history is not None and str(args.history).strip() == "":
        args.history = None
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
