#!/usr/bin/env python3
"""Benchmark latent seed incubation across multi-turn unconscious-field sequences."""

from __future__ import annotations

import argparse
import asyncio
import json
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

from benchmark_unconscious_creativity import (  # noqa: E402
    _as_dict,
    _as_list,
    _extract_case_signals,
    _last_event,
    _mean,
    _rate,
    _safe_float,
    _safe_int,
)
from engine_stdio import EngineSession, _extract_metrics  # noqa: E402


def _load_sequences(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("sequences")
    if not isinstance(raw, list):
        raise ValueError("Sequence file must be a JSON array or an object with 'sequences'.")

    out: List[Dict[str, Any]] = []
    for seq_idx, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Sequence entry #{seq_idx} must be an object.")
        turns = item.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Sequence entry #{seq_idx} missing non-empty 'turns'.")
        seq_id = str(item.get("id") or f"seq{seq_idx:03d}").strip()
        if not seq_id:
            raise ValueError(f"Sequence entry #{seq_idx} has empty id.")
        parsed_turns: List[Dict[str, Any]] = []
        for turn_idx, turn in enumerate(turns, 1):
            if not isinstance(turn, dict):
                raise ValueError(f"Sequence {seq_id} turn #{turn_idx} must be an object.")
            question = str(turn.get("question") or "").strip()
            if not question:
                raise ValueError(f"Sequence {seq_id} turn #{turn_idx} missing question.")
            parsed_turns.append(
                {
                    "id": str(turn.get("id") or f"t{turn_idx:02d}"),
                    "role": str(turn.get("role") or "turn").strip().lower(),
                    "question": question,
                    "tags": turn.get("tags") if isinstance(turn.get("tags"), list) else [],
                    "target_archetypes": (
                        turn.get("target_archetypes")
                        if isinstance(turn.get("target_archetypes"), list)
                        else []
                    ),
                    "system2_mode": (
                        str(turn.get("system2_mode")).strip().lower()
                        if turn.get("system2_mode") is not None
                        else None
                    ),
                }
            )
        out.append(
            {
                "id": seq_id,
                "title": str(item.get("title") or seq_id),
                "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
                "target_archetypes": (
                    item.get("target_archetypes")
                    if isinstance(item.get("target_archetypes"), list)
                    else []
                ),
                "turns": parsed_turns,
            }
        )
    return out


def _filter_sequences(
    sequences: List[Dict[str, Any]],
    *,
    only_ids: str | None,
    only_tags: str | None,
) -> List[Dict[str, Any]]:
    ids = {tok.strip() for tok in str(only_ids or "").split(",") if tok.strip()}
    tags = {
        tok.strip().lower()
        for tok in str(only_tags or "").split(",")
        if tok.strip()
    }
    out = list(sequences)
    if ids:
        out = [seq for seq in out if str(seq.get("id") or "").strip() in ids]
    if tags:
        filtered: List[Dict[str, Any]] = []
        for seq in out:
            raw_tags = seq.get("tags") if isinstance(seq.get("tags"), list) else []
            norm = {str(tag).strip().lower() for tag in raw_tags if str(tag).strip()}
            if norm.intersection(tags):
                filtered.append(seq)
        out = filtered
    return out


def _emergent_details(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = _as_dict(_last_event(events, "unconscious_field").get("summary"))
    details: List[Dict[str, Any]] = []
    for item in _as_list(summary.get("emergent_ideas")):
        payload = _as_dict(item)
        if not payload:
            continue
        details.append(
            {
                "archetype": str(payload.get("archetype") or ""),
                "label": str(payload.get("label") or ""),
                "intensity": _safe_float(payload.get("intensity")),
                "incubation_rounds": _safe_int(payload.get("incubation_rounds")),
                "trigger_similarity": _safe_float(payload.get("trigger_similarity")),
                "origin": str(payload.get("origin") or ""),
            }
        )
    return details


def _outcome_meta(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _as_dict(_last_event(events, "unconscious_outcome").get("outcome"))


def _normalise_origin_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _origin_matches(left: Any, right: Any) -> bool:
    lhs = _normalise_origin_text(left)
    rhs = _normalise_origin_text(right)
    if not lhs or not rhs:
        return False
    if lhs in rhs or rhs in lhs:
        return True
    min_len = min(len(lhs), len(rhs), 96)
    return bool(min_len >= 40 and lhs[:min_len] == rhs[:min_len])


def _cached_seed_snapshot(
    *,
    turn_index: int,
    turn_id: str,
    role: str,
    qid: str,
    question: str,
    outcome: Dict[str, Any],
    signals: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not outcome.get("seed_cached"):
        return None
    trace = _as_dict(signals.get("archetype_trace"))
    top_activations = _as_list(trace.get("top_activations"))
    top = _as_dict(top_activations[0]) if top_activations else {}
    archetype = str(top.get("id") or "")
    if not archetype:
        top_k = _as_list(_as_dict(signals.get("unconscious")).get("top_k"))
        archetype = str(top_k[0] if top_k else "")
    if not archetype:
        archetype = "unknown"
    return {
        "turn_index": int(turn_index),
        "turn_id": str(turn_id),
        "role": str(role),
        "qid": str(qid),
        "archetype": archetype,
        "label": str(top.get("label") or archetype),
        "intensity": _safe_float(top.get("intensity")),
        "cache_depth_after": _safe_int(outcome.get("cache_depth")),
        "origin": str(question or "")[:160],
        "cue_alignment": trace.get("cue_alignment"),
        "cue_ids": _as_list(_as_dict(trace.get("cue")).get("ids")),
        "top_k": _as_list(_as_dict(signals.get("unconscious")).get("top_k")),
    }


def _sequence_targets(sequence_entry: Dict[str, Any]) -> List[str]:
    targets = [str(item) for item in _as_list(sequence_entry.get("target_archetypes"))]
    for turn in _as_list(sequence_entry.get("turns")):
        targets.extend(str(item) for item in _as_list(_as_dict(turn).get("target_archetypes")))
    return sorted({item for item in targets if item})


async def _run_turn(
    *,
    session: EngineSession,
    sequence_entry: Dict[str, Any],
    turn_entry: Dict[str, Any],
    turn_index: int,
    run_id: str,
    answer_mode: str,
    leading_brain: str,
    default_system2_mode: str,
    executive_mode: str,
    executive_observer_mode: str,
) -> Dict[str, Any]:
    sequence_id = str(sequence_entry.get("id") or "seq")
    turn_id = str(turn_entry.get("id") or f"t{turn_index:02d}")
    qid = f"{run_id}-{sequence_id}-{turn_id}"
    question = str(turn_entry.get("question") or "")
    role = str(turn_entry.get("role") or "turn")
    tags = list(_as_list(sequence_entry.get("tags"))) + list(_as_list(turn_entry.get("tags")))
    system2_mode = str(turn_entry.get("system2_mode") or default_system2_mode).strip().lower()
    if system2_mode not in {"auto", "on", "off"}:
        system2_mode = default_system2_mode

    session.telemetry.clear()
    started = time.perf_counter()
    answer = ""
    error_text: Optional[str] = None
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
    signals = _extract_case_signals(
        events,
        metrics,
        answer,
        question=question,
        tags=tags,
    )
    emergent = _emergent_details(events)
    outcome = _outcome_meta(events)
    signals["unconscious"]["emergent_details"] = emergent
    signals["unconscious"]["emergent_archetypes"] = sorted(
        {
            str(item.get("archetype") or "")
            for item in emergent
            if str(item.get("archetype") or "")
        }
    )
    cached_seed = _cached_seed_snapshot(
        turn_index=turn_index,
        turn_id=turn_id,
        role=role,
        qid=qid,
        question=question,
        outcome=outcome,
        signals=signals,
    )
    return {
        "turn_index": turn_index,
        "id": turn_id,
        "qid": qid,
        "role": role,
        "question": question,
        "tags": tags,
        "target_archetypes": _as_list(turn_entry.get("target_archetypes")),
        "answer_mode": answer_mode,
        "system2_mode": system2_mode,
        "latency_ms": _safe_float(metrics.get("latency_ms")) or elapsed_ms,
        "telemetry_events": len(events),
        "unconscious_outcome": outcome,
        "cached_seed": cached_seed,
        "error": error_text,
        "answer_preview": answer[:280] if answer else "",
        **signals,
    }


def _emergent_events_with_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for turn in turns:
        payload = _as_dict(turn)
        for detail in _as_list(payload.get("unconscious", {}).get("emergent_details")):
            detail_payload = _as_dict(detail)
            if not detail_payload:
                continue
            events.append(
                {
                    "turn_index": _safe_int(payload.get("turn_index")),
                    "turn_id": payload.get("id"),
                    "role": payload.get("role"),
                    "archetype": str(detail_payload.get("archetype") or ""),
                    "label": str(detail_payload.get("label") or ""),
                    "intensity": _safe_float(detail_payload.get("intensity")),
                    "incubation_rounds": _safe_int(detail_payload.get("incubation_rounds")),
                    "trigger_similarity": _safe_float(detail_payload.get("trigger_similarity")),
                    "origin": str(detail_payload.get("origin") or ""),
                }
            )
    return events


def _near_miss_state(attempt: Dict[str, Any], turn: Dict[str, Any]) -> str:
    gap = _safe_float(attempt.get("threshold_gap"))
    archetype = str(attempt.get("archetype") or "")
    unconscious = _as_dict(turn.get("unconscious"))
    top_k = [str(item) for item in _as_list(unconscious.get("top_k"))]
    emergent_archetypes = {
        str(item)
        for item in _as_list(unconscious.get("emergent_archetypes"))
        if str(item)
    }
    if emergent_archetypes and archetype not in emergent_archetypes:
        return "absorbed_by_emergent_archetype"
    if not bool(attempt.get("incubation_pass")):
        return "waiting_incubation"
    if not bool(attempt.get("intensity_pass")):
        return "fading_intensity"
    if top_k and archetype and top_k[0] != archetype and (gap or 0.0) > 0.08:
        return "off_axis_top1_dominant"
    if gap is None:
        return "unknown"
    if gap <= 0.03:
        return "near_surface"
    if gap <= 0.12:
        return "mid_depth"
    return "deep_below_threshold"


def _near_miss_events_with_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for turn in turns:
        payload = _as_dict(turn)
        unconscious = _as_dict(payload.get("unconscious"))
        top_k = [str(item) for item in _as_list(unconscious.get("top_k"))]
        for attempt in _as_list(unconscious.get("harvest_attempts")):
            attempt_payload = _as_dict(attempt)
            if not attempt_payload or attempt_payload.get("emerged"):
                continue
            archetype = str(attempt_payload.get("archetype") or "")
            events.append(
                {
                    "turn_index": _safe_int(payload.get("turn_index")),
                    "turn_id": payload.get("id"),
                    "role": payload.get("role"),
                    "archetype": archetype,
                    "label": str(attempt_payload.get("label") or ""),
                    "intensity": _safe_float(attempt_payload.get("intensity")),
                    "novelty": _safe_float(attempt_payload.get("novelty")),
                    "incubation_rounds": _safe_int(attempt_payload.get("incubation_rounds")),
                    "trigger_similarity": _safe_float(
                        attempt_payload.get("trigger_similarity")
                    ),
                    "threshold": _safe_float(attempt_payload.get("threshold")),
                    "threshold_gap": _safe_float(attempt_payload.get("threshold_gap")),
                    "threshold_margin": _safe_float(attempt_payload.get("threshold_margin")),
                    "similarity_pass": bool(attempt_payload.get("similarity_pass")),
                    "incubation_pass": bool(attempt_payload.get("incubation_pass")),
                    "intensity_pass": bool(attempt_payload.get("intensity_pass")),
                    "failure_reasons": [
                        str(reason)
                        for reason in _as_list(attempt_payload.get("failure_reasons"))
                    ],
                    "status": str(attempt_payload.get("status") or ""),
                    "origin": str(attempt_payload.get("origin") or ""),
                    "current_top1": top_k[0] if top_k else None,
                    "current_top1_matches_seed": bool(top_k and top_k[0] == archetype),
                    "state": _near_miss_state(attempt_payload, payload),
                }
            )
    return events


def _closest_near_miss_event(events: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [
        event
        for event in events
        if _safe_float(event.get("threshold_gap")) is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda event: (
            float(_safe_float(event.get("threshold_gap")) or 0.0),
            -float(_safe_float(event.get("trigger_similarity")) or 0.0),
        ),
    )


def _match_emergent_to_seed(
    emergent: Dict[str, Any],
    seeds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    emergent_turn = _safe_int(emergent.get("turn_index"))
    prior = [
        seed
        for seed in seeds
        if emergent_turn is None
        or (_safe_int(seed.get("turn_index")) or 0) < emergent_turn
    ]
    if not prior:
        return {
            "seed": None,
            "match_type": "unmatched",
            "match_confidence": 0.0,
            "same_archetype": False,
            "origin_match": False,
        }

    emergent_archetype = str(emergent.get("archetype") or "")
    scored: List[tuple[float, Dict[str, Any], bool, bool]] = []
    for seed in prior:
        same_archetype = bool(emergent_archetype and seed.get("archetype") == emergent_archetype)
        origin_match = _origin_matches(emergent.get("origin"), seed.get("origin"))
        seed_turn = _safe_int(seed.get("turn_index")) or 0
        recency = float(seed_turn) / max(1.0, float(emergent_turn or seed_turn or 1))
        score = recency * 0.1
        if same_archetype:
            score += 0.62
        if origin_match:
            score += 0.33
        scored.append((score, seed, same_archetype, origin_match))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, seed, same_archetype, origin_match = scored[0]
    if same_archetype and origin_match:
        match_type = "same_archetype_and_origin"
        confidence = 1.0
    elif same_archetype:
        match_type = "same_archetype"
        confidence = 0.76
    elif origin_match:
        match_type = "origin_only"
        confidence = 0.58
    else:
        match_type = "nearest_prior"
        confidence = 0.25
    return {
        "seed": seed,
        "match_type": match_type,
        "match_confidence": confidence,
        "same_archetype": same_archetype,
        "origin_match": origin_match,
        "score": round(float(score), 6),
    }


def _sequence_lineage(sequence: Dict[str, Any]) -> Dict[str, Any]:
    turns = [_as_dict(turn) for turn in _as_list(sequence.get("turns"))]
    cached_seeds = [
        _as_dict(turn.get("cached_seed"))
        for turn in turns
        if _as_dict(turn.get("cached_seed"))
    ]
    emergent_events = _emergent_events_with_turns(turns)
    links: List[Dict[str, Any]] = []
    for emergent in emergent_events:
        match = _match_emergent_to_seed(emergent, cached_seeds)
        seed = _as_dict(match.get("seed"))
        links.append(
            {
                "emergent": emergent,
                "seed": seed or None,
                "seed_archetype": seed.get("archetype") if seed else None,
                "emergent_archetype": emergent.get("archetype"),
                "archetype_transition": (
                    f"{seed.get('archetype')}->{emergent.get('archetype')}"
                    if seed
                    else f"None->{emergent.get('archetype')}"
                ),
                "same_archetype": bool(match.get("same_archetype")),
                "origin_match": bool(match.get("origin_match")),
                "match_type": match.get("match_type"),
                "match_confidence": match.get("match_confidence"),
                "match_score": match.get("score"),
            }
        )

    transitions = [str(link.get("archetype_transition")) for link in links]
    same = sum(1 for link in links if link.get("same_archetype"))
    origin = sum(1 for link in links if link.get("origin_match"))
    return {
        "cached_seed_count": len(cached_seeds),
        "cached_seed_archetypes": sorted(
            {
                str(seed.get("archetype") or "")
                for seed in cached_seeds
                if str(seed.get("archetype") or "")
            }
        ),
        "cached_seeds": cached_seeds,
        "emergent_count": len(emergent_events),
        "emergent_archetypes": sorted(
            {
                str(event.get("archetype") or "")
                for event in emergent_events
                if str(event.get("archetype") or "")
            }
        ),
        "links": links,
        "archetype_transitions": sorted(set(transitions)),
        "same_archetype_links": same,
        "origin_matched_links": origin,
        "same_archetype_link_rate": _rate(same, len(links)),
        "origin_matched_link_rate": _rate(origin, len(links)),
    }


def _sequence_observation(sequence: Dict[str, Any]) -> Dict[str, Any]:
    turns = _as_list(sequence.get("turns"))
    ok_turns = [turn for turn in turns if not _as_dict(turn).get("error")]
    emergent_turns = [
        turn
        for turn in ok_turns
        if (_as_dict(turn).get("unconscious", {}).get("emergent_ideas") or 0) > 0
    ]
    leak_turns = [
        turn
        for turn in ok_turns
        if _as_dict(turn).get("leakage", {}).get("has_internal_leak")
    ]
    seed_cached_turns = [
        turn
        for turn in ok_turns
        if _as_dict(turn).get("unconscious_outcome", {}).get("seed_cached")
    ]
    targets = set(str(item) for item in _as_list(sequence.get("target_archetypes")) if str(item))
    emergent_archetypes = sorted(
        {
            str(item)
            for turn in emergent_turns
            for item in _as_list(_as_dict(turn).get("unconscious", {}).get("emergent_archetypes"))
            if str(item)
        }
    )
    target_hits = sorted(targets.intersection(emergent_archetypes))
    pressures = [
        value
        for value in (
            _safe_float(_as_dict(turn).get("unconscious", {}).get("incubation_pressure"))
            for turn in ok_turns
        )
        if value is not None
    ]
    cache_depths = [
        value
        for value in (
            _safe_int(_as_dict(turn).get("unconscious", {}).get("cache_depth"))
            for turn in ok_turns
        )
        if value is not None
    ]
    first_emergent = emergent_turns[0] if emergent_turns else None
    lineage = _as_dict(sequence.get("lineage")) or _sequence_lineage(sequence)
    near_miss_events = _near_miss_events_with_turns([_as_dict(turn) for turn in turns])
    echo_near_miss_events = [
        event
        for event in near_miss_events
        if str(event.get("role") or "").startswith("echo")
    ]
    closest_near_miss = _closest_near_miss_event(near_miss_events)
    closest_echo_near_miss = _closest_near_miss_event(echo_near_miss_events)
    return {
        "turns": len(turns),
        "ok_turns": len(ok_turns),
        "error_turns": len(turns) - len(ok_turns),
        "leak_turns": len(leak_turns),
        "seed_cached_turns": len(seed_cached_turns),
        "emergent_turns": len(emergent_turns),
        "emerged": bool(emergent_turns),
        "first_emergent_turn_index": (
            _safe_int(_as_dict(first_emergent).get("turn_index")) if first_emergent else None
        ),
        "first_emergent_role": _as_dict(first_emergent).get("role") if first_emergent else None,
        "emergent_archetypes": emergent_archetypes,
        "target_archetypes": sorted(targets),
        "target_emergent_hits": target_hits,
        "target_emerged": bool(target_hits) if targets else None,
        "cached_seed_archetypes": lineage.get("cached_seed_archetypes"),
        "seed_to_emergent_transitions": lineage.get("archetype_transitions"),
        "seed_emergent_same_archetype_rate": lineage.get("same_archetype_link_rate"),
        "seed_emergent_origin_match_rate": lineage.get("origin_matched_link_rate"),
        "near_miss_attempts": len(near_miss_events),
        "echo_near_miss_attempts": len(echo_near_miss_events),
        "closest_near_miss": closest_near_miss,
        "closest_echo_near_miss": closest_echo_near_miss,
        "closest_near_miss_gap": (
            _safe_float(closest_near_miss.get("threshold_gap"))
            if closest_near_miss
            else None
        ),
        "closest_echo_near_miss_gap": (
            _safe_float(closest_echo_near_miss.get("threshold_gap"))
            if closest_echo_near_miss
            else None
        ),
        "peak_incubation_pressure": max(pressures) if pressures else None,
        "final_incubation_pressure": pressures[-1] if pressures else None,
        "initial_incubation_pressure": pressures[0] if pressures else None,
        "pressure_delta": (
            float(pressures[-1] - pressures[0]) if len(pressures) >= 2 else None
        ),
        "peak_cache_depth": max(cache_depths) if cache_depths else None,
        "final_cache_depth": cache_depths[-1] if cache_depths else None,
        "lineage": lineage,
    }


def _summarise_turns_by_role(turns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    roles = sorted({str(turn.get("role") or "turn") for turn in turns})
    out: Dict[str, Dict[str, Any]] = {}
    for role in roles:
        role_turns = [turn for turn in turns if str(turn.get("role") or "turn") == role]
        ok_turns = [turn for turn in role_turns if not turn.get("error")]
        cue_turns = [
            turn
            for turn in ok_turns
            if turn.get("archetype_trace", {}).get("cue_alignment") != "unlabeled"
        ]
        out[role] = {
            "turns": len(role_turns),
            "ok_turns": len(ok_turns),
            "emergent_rate": _rate(
                sum(
                    1
                    for turn in ok_turns
                    if (turn.get("unconscious", {}).get("emergent_ideas") or 0) > 0
                ),
                len(ok_turns),
            ),
            "seed_cached_rate": _rate(
                sum(
                    1
                    for turn in ok_turns
                    if turn.get("unconscious_outcome", {}).get("seed_cached")
                ),
                len(ok_turns),
            ),
            "harvest_attempt_turn_rate": _rate(
                sum(
                    1
                    for turn in ok_turns
                    if (turn.get("unconscious", {}).get("harvest_attempt_count") or 0) > 0
                ),
                len(ok_turns),
            ),
            "near_miss_turn_rate": _rate(
                sum(
                    1
                    for turn in ok_turns
                    if (turn.get("unconscious", {}).get("harvest_near_miss_count") or 0) > 0
                ),
                len(ok_turns),
            ),
            "leak_rate": _rate(
                sum(
                    1
                    for turn in ok_turns
                    if turn.get("leakage", {}).get("has_internal_leak")
                ),
                len(ok_turns),
            ),
            "cue_top_k_alignment_rate": _rate(
                sum(
                    1
                    for turn in cue_turns
                    if turn.get("archetype_trace", {}).get("cue_alignment")
                    == "top_k_aligned"
                ),
                len(cue_turns),
            ),
            "cue_motif_only_rate": _rate(
                sum(
                    1
                    for turn in cue_turns
                    if turn.get("archetype_trace", {}).get("cue_alignment")
                    == "motif_only"
                ),
                len(cue_turns),
            ),
            "cue_psychoid_only_rate": _rate(
                sum(
                    1
                    for turn in cue_turns
                    if turn.get("archetype_trace", {}).get("cue_alignment")
                    == "psychoid_only"
                ),
                len(cue_turns),
            ),
            "avg_incubation_pressure": _mean(
                [
                    value
                    for value in (
                        _safe_float(turn.get("unconscious", {}).get("incubation_pressure"))
                        for turn in ok_turns
                    )
                    if value is not None
                ]
            ),
            "avg_closest_near_miss_gap": _mean(
                [
                    float(value)
                    for value in (
                        _safe_float(
                            _as_dict(
                                turn.get("unconscious", {}).get("closest_near_miss")
                            ).get("threshold_gap")
                        )
                        for turn in ok_turns
                    )
                    if value is not None
                ]
            ),
            "avg_cache_depth": _mean(
                [
                    float(value)
                    for value in (
                        _safe_int(turn.get("unconscious", {}).get("cache_depth"))
                        for turn in ok_turns
                    )
                    if value is not None
                ]
            ),
        }
    return out


def _count_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _summarise_sequences(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    observations = [_as_dict(seq.get("observation")) for seq in sequences]
    turns = [
        turn
        for seq in sequences
        for turn in _as_list(seq.get("turns"))
        if isinstance(turn, dict)
    ]
    ok_sequences = [
        seq
        for seq, obs in zip(sequences, observations)
        if not obs.get("error_turns")
    ]
    sequence_total = len(sequences)
    target_observations = [obs for obs in observations if obs.get("target_emerged") is not None]
    lineages = [_as_dict(obs.get("lineage")) for obs in observations]
    links = [
        _as_dict(link)
        for lineage in lineages
        for link in _as_list(lineage.get("links"))
        if isinstance(link, dict)
    ]
    transition_counts: Dict[str, int] = {}
    for link in links:
        transition = str(link.get("archetype_transition") or "")
        if not transition:
            continue
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    same_links = sum(1 for link in links if link.get("same_archetype"))
    origin_links = sum(1 for link in links if link.get("origin_match"))
    near_miss_events = [
        event
        for seq in sequences
        for event in _near_miss_events_with_turns(
            [_as_dict(turn) for turn in _as_list(seq.get("turns"))]
        )
    ]
    echo_near_miss_events = [
        event
        for event in near_miss_events
        if str(event.get("role") or "").startswith("echo")
    ]
    closest_echo_gaps = [
        float(obs.get("closest_echo_near_miss_gap"))
        for obs in observations
        if obs.get("closest_echo_near_miss_gap") is not None
    ]
    return {
        "total_sequences": sequence_total,
        "ok_sequences": len(ok_sequences),
        "error_sequences": sequence_total - len(ok_sequences),
        "total_turns": len(turns),
        "ok_turns": sum(1 for turn in turns if not turn.get("error")),
        "leak_sequences": sum(1 for obs in observations if (obs.get("leak_turns") or 0) > 0),
        "emergent_sequences": sum(1 for obs in observations if obs.get("emerged")),
        "emergent_sequence_rate": _rate(
            sum(1 for obs in observations if obs.get("emerged")),
            sequence_total,
        ),
        "target_emergent_sequence_rate": _rate(
            sum(1 for obs in target_observations if obs.get("target_emerged")),
            len(target_observations),
        ),
        "seed_cached_sequence_rate": _rate(
            sum(1 for obs in observations if (obs.get("seed_cached_turns") or 0) > 0),
            sequence_total,
        ),
        "seed_to_emergent_links": len(links),
        "seed_to_emergent_same_archetype_rate": _rate(same_links, len(links)),
        "seed_to_emergent_origin_match_rate": _rate(origin_links, len(links)),
        "seed_to_emergent_transition_counts": dict(sorted(transition_counts.items())),
        "sequences_with_seed_emergent_same_archetype": sum(
            1
            for obs in observations
            if (obs.get("seed_emergent_same_archetype_rate") or 0.0) > 0.0
        ),
        "sequences_with_seed_emergent_origin_match": sum(
            1
            for obs in observations
            if (obs.get("seed_emergent_origin_match_rate") or 0.0) > 0.0
        ),
        "near_miss_attempts": len(near_miss_events),
        "echo_near_miss_attempts": len(echo_near_miss_events),
        "sequences_with_near_miss": sum(
            1 for obs in observations if (obs.get("near_miss_attempts") or 0) > 0
        ),
        "sequences_with_echo_near_miss": sum(
            1 for obs in observations if (obs.get("echo_near_miss_attempts") or 0) > 0
        ),
        "avg_closest_near_miss_gap": _mean(
            [
                float(obs.get("closest_near_miss_gap"))
                for obs in observations
                if obs.get("closest_near_miss_gap") is not None
            ]
        ),
        "avg_closest_echo_near_miss_gap": _mean(closest_echo_gaps),
        "min_closest_echo_near_miss_gap": (
            min(closest_echo_gaps) if closest_echo_gaps else None
        ),
        "max_closest_echo_near_miss_gap": (
            max(closest_echo_gaps) if closest_echo_gaps else None
        ),
        "near_miss_state_counts": _count_by_key(near_miss_events, "state"),
        "echo_near_miss_state_counts": _count_by_key(echo_near_miss_events, "state"),
        "near_miss_archetype_counts": _count_by_key(near_miss_events, "archetype"),
        "echo_near_miss_archetype_counts": _count_by_key(
            echo_near_miss_events,
            "archetype",
        ),
        "avg_seed_cached_turns": _mean(
            [
                float(obs.get("seed_cached_turns"))
                for obs in observations
                if obs.get("seed_cached_turns") is not None
            ]
        ),
        "avg_first_emergent_turn_index": _mean(
            [
                float(obs.get("first_emergent_turn_index"))
                for obs in observations
                if obs.get("first_emergent_turn_index") is not None
            ]
        ),
        "avg_peak_incubation_pressure": _mean(
            [
                float(obs.get("peak_incubation_pressure"))
                for obs in observations
                if obs.get("peak_incubation_pressure") is not None
            ]
        ),
        "avg_pressure_delta": _mean(
            [
                float(obs.get("pressure_delta"))
                for obs in observations
                if obs.get("pressure_delta") is not None
            ]
        ),
        "avg_final_cache_depth": _mean(
            [
                float(obs.get("final_cache_depth"))
                for obs in observations
                if obs.get("final_cache_depth") is not None
            ]
        ),
        "max_peak_cache_depth": max(
            [
                int(obs.get("peak_cache_depth"))
                for obs in observations
                if obs.get("peak_cache_depth") is not None
            ],
            default=None,
        ),
        "turns_by_role": _summarise_turns_by_role(turns),
    }


async def _run_sequence(
    *,
    sequence_entry: Dict[str, Any],
    run_id: str,
    session_prefix: str,
    answer_mode: str,
    leading_brain: str,
    system2_mode: str,
    executive_mode: str,
    executive_observer_mode: str,
) -> Dict[str, Any]:
    sequence_id = str(sequence_entry.get("id") or "sequence")
    session_id = f"{session_prefix}-{run_id}-{sequence_id}"
    session = await EngineSession.create(session_id=session_id)
    turns: List[Dict[str, Any]] = []
    try:
        for idx, turn in enumerate(_as_list(sequence_entry.get("turns")), 1):
            turns.append(
                await _run_turn(
                    session=session,
                    sequence_entry=sequence_entry,
                    turn_entry=_as_dict(turn),
                    turn_index=idx,
                    run_id=run_id,
                    answer_mode=answer_mode,
                    leading_brain=leading_brain,
                    default_system2_mode=system2_mode,
                    executive_mode=executive_mode,
                    executive_observer_mode=executive_observer_mode,
                )
            )
    finally:
        await session.close()
    sequence_case = {
        "id": sequence_id,
        "title": sequence_entry.get("title") or sequence_id,
        "tags": _as_list(sequence_entry.get("tags")),
        "target_archetypes": _sequence_targets(sequence_entry),
        "session_id": session_id,
        "turns": turns,
    }
    sequence_case["lineage"] = _sequence_lineage(sequence_case)
    sequence_case["observation"] = _sequence_observation(sequence_case)
    return sequence_case


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


async def _run(args: argparse.Namespace) -> int:
    sequence_paths = [Path(tok).expanduser().resolve() for tok in str(args.sequences).split(",") if tok.strip()]
    if not sequence_paths:
        raise ValueError("No --sequences paths provided.")
    missing_paths = [path for path in sequence_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Sequence file(s) not found: {paths}".format(
                paths=", ".join(str(path) for path in missing_paths)
            )
        )

    sequences: List[Dict[str, Any]] = []
    for path in sequence_paths:
        sequences.extend(_load_sequences(path))
    sequences = _filter_sequences(
        sequences,
        only_ids=args.only_ids,
        only_tags=args.only_tags,
    )
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(sequences)
    if args.limit is not None and args.limit > 0:
        sequences = sequences[: int(args.limit)]
    if not sequences:
        raise RuntimeError("No incubation sequences to run.")

    low_signal_filter = str(args.low_signal_filter or "on").strip().lower()
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = "1" if low_signal_filter == "on" else "0"

    run_id = time.strftime("unconscious_incubation_%Y%m%d_%H%M%S")
    print(f"[incubation-bench] run_id={run_id}")
    print(f"[incubation-bench] sequences={len(sequences)}")
    report_sequences: List[Dict[str, Any]] = []
    for idx, sequence in enumerate(sequences, 1):
        case = await _run_sequence(
            sequence_entry=sequence,
            run_id=run_id,
            session_prefix=args.session_prefix,
            answer_mode=args.answer_mode,
            leading_brain=args.leading_brain,
            system2_mode=args.system2_mode,
            executive_mode=args.executive_mode,
            executive_observer_mode=args.executive_observer_mode,
        )
        report_sequences.append(case)
        obs = case["observation"]
        print(
            "[incubation-bench] {idx:03d}/{total} id={id} emerged={emerged} "
            "first={first_role}@{first_idx} target_hits={target_hits} "
            "transitions={transitions} echo_gap={echo_gap} cache={cache} "
            "pressure={pressure} leaks={leaks}".format(
                idx=idx,
                total=len(sequences),
                id=case.get("id"),
                emerged=obs.get("emerged"),
                first_role=obs.get("first_emergent_role"),
                first_idx=obs.get("first_emergent_turn_index"),
                target_hits=obs.get("target_emergent_hits"),
                transitions=obs.get("seed_to_emergent_transitions"),
                echo_gap=obs.get("closest_echo_near_miss_gap"),
                cache=obs.get("final_cache_depth"),
                pressure=obs.get("peak_incubation_pressure"),
                leaks=obs.get("leak_turns"),
            )
        )

    summary = _summarise_sequences(report_sequences)
    report = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sequence_sets": [str(path) for path in sequence_paths],
        "config": {
            "session_prefix": args.session_prefix,
            "answer_mode": args.answer_mode,
            "leading_brain": args.leading_brain,
            "system2_mode": args.system2_mode,
            "executive_mode": args.executive_mode,
            "executive_observer_mode": args.executive_observer_mode,
            "low_signal_filter": low_signal_filter,
            "sequence_count": len(report_sequences),
            "only_ids": args.only_ids,
            "only_tags": args.only_tags,
        },
        "summary": summary,
        "sequences": report_sequences,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[incubation-bench] wrote report: {output_path}")

    history_path = Path(args.history).expanduser().resolve() if args.history else None
    if history_path is not None:
        _append_history(
            history_path,
            {
                "run_id": run_id,
                "timestamp": report["timestamp"],
                "sequence_sets": report["sequence_sets"],
                "summary": summary,
                "config": report["config"],
            },
        )
        print(f"[incubation-bench] appended history: {history_path}")

    print(
        "[incubation-bench] summary ok={ok}/{total} emergence={emergence} "
        "target_emergence={target} avg_first={first} avg_pressure={pressure} "
        "lineage_same={lineage_same} lineage_origin={lineage_origin} "
        "avg_echo_gap={echo_gap} max_cache={cache}".format(
            ok=summary.get("ok_sequences"),
            total=summary.get("total_sequences"),
            emergence=summary.get("emergent_sequence_rate"),
            target=summary.get("target_emergent_sequence_rate"),
            first=summary.get("avg_first_emergent_turn_index"),
            pressure=summary.get("avg_peak_incubation_pressure"),
            lineage_same=summary.get("seed_to_emergent_same_archetype_rate"),
            lineage_origin=summary.get("seed_to_emergent_origin_match_rate"),
            echo_gap=summary.get("avg_closest_echo_near_miss_gap"),
            cache=summary.get("max_peak_cache_depth"),
        )
    )
    if args.expect_no_leaks and summary.get("leak_sequences"):
        print("[incubation-bench] internal debug markers leaked into answers.")
        return 3
    if args.expect_emergence and not summary.get("emergent_sequences"):
        print("[incubation-bench] no emergent ideas observed.")
        return 4
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences",
        default=str(PROJECT_ROOT / "examples" / "unconscious_incubation_benchmark_sequences.json"),
        help="Comma separated paths to incubation sequence JSON files.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_incubation_last.json"),
    )
    parser.add_argument(
        "--history",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_incubation_history.jsonl"),
        help="Append compact run summary to JSONL history (set empty to disable).",
    )
    parser.add_argument("--session-prefix", default="unconscious-incubation")
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
    parser.add_argument("--expect-emergence", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.history is not None and str(args.history).strip() == "":
        args.history = None
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
