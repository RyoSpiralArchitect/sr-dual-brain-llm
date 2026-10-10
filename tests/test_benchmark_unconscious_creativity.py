import math
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_unconscious_creativity import (  # noqa: E402
    _closest_near_miss,
    _extract_case_signals,
    _internal_leak_markers,
    _summarise_cases,
)
import benchmark_unconscious_creativity as creativity  # noqa: E402
import benchmark_unconscious_incubation as incubation  # noqa: E402


@pytest.mark.parametrize("runner", ["creativity", "incubation"])
def test_unconscious_runners_measure_wall_time_and_preserve_scope(monkeypatch, runner):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(creativity.time, "perf_counter", lambda: next(clock))

    async def answer(*_args, **_kwargs):
        return "Fixture answer"

    session = SimpleNamespace(
        controller=SimpleNamespace(process=answer),
        telemetry=SimpleNamespace(clear=lambda: None, events=[
            {"event": "interaction_complete", "latency_ms": 1.0},
        ]),
    )
    common = dict(session=session, run_id="wall-fixture", answer_mode="plain",
                  leading_brain="auto", executive_mode="off", executive_observer_mode="off")
    if runner == "creativity":
        row = asyncio.run(creativity._run_case(
            **common, question_entry={"id": "one", "question": "Why?"}, index=1, system2_mode="off"))
        summary = creativity._summarise_cases([row])
    else:
        row = asyncio.run(incubation._run_turn(
            **common, sequence_entry={"id": "seq"}, turn_entry={"id": "one", "question": "Why?"},
            turn_index=1, default_system2_mode="off"))
        summary = incubation._summarise_turns_by_role([row])["turn"]
        report_summary = incubation._summarise_sequences([{"turns": [row], "observation": {}}])
        assert report_summary["latency_scope"] == "process_wall"
    assert row["latency_ms"] == 250.0
    assert row["controller_latency_ms"] == 1.0
    assert row["latency_scope"] == summary["latency_scope"] == "process_wall"
    assert row["latency_schema_version"] == summary["latency_schema_version"] == 2


def test_creativity_does_not_pool_legacy_and_current_latency():
    legacy = {"error": None, "latency_ms": 1.0}
    current = {"error": None, "latency_ms": 100.0, "latency_scope": "process_wall", "latency_schema_version": 2}
    summary = creativity._summarise_cases([legacy, current])
    assert summary["latency_scope"] == "mixed" and summary["avg_latency_ms"] is None


def test_internal_leak_markers_detect_debug_sections_case_insensitively():
    markers = _internal_leak_markers("final answer\n\n[unconscious insight]\n- bridge")

    assert markers == ["[Unconscious Insight]"]


def test_extract_case_signals_combines_unconscious_motif_and_leakage_metrics():
    events = [
        {
            "event": "unconscious_field",
            "summary": {
                "top_k": ["sage", "shadow"],
                "archetype_map": [
                    {"id": "sage", "label": "Sage", "intensity": 0.42},
                    {"id": "shadow", "label": "Shadow", "intensity": 0.4},
                    {"id": "hero", "label": "Hero", "intensity": 0.18},
                ],
                "emergent_ideas": [{"label": "bridge", "archetype": "sage"}],
                "stress_released": 0.2,
                "cache_depth": 4,
                "motifs": ["bridge"],
                "psychoid_signal": {
                    "attention_bias": [{"archetype": "sage", "weight": 0.7}],
                    "signifier_chain": ["sage:book"],
                    "resonance": 0.4,
                    "psychoid_tension": 0.25,
                },
            },
        },
        {
            "event": "default_mode_reflection",
            "reflections": [{"theme": "patient bridge"}],
        },
        {
            "event": "task_positive_network",
            "load": 0.72,
            "mode": "executive_control",
            "suppressed": True,
        },
        {
            "event": "psychoid_attention_projection",
            "projection": {"norm": 0.6, "chain_length": 2},
        },
        {
            "event": "coherence_unconscious_weave",
            "fabric": {"score": 0.55, "emergent_count": 1},
        },
        {
            "event": "coherence_linguistic_motifs",
            "motifs": {"score": 0.42, "repeated_loops": 1},
        },
        {
            "event": "coherence_signal",
            "signal": {
                "combined": 0.62,
                "tension": 0.22,
                "mode": "balanced",
                "linguistic_depth": 0.5,
            },
        },
    ]

    signals = _extract_case_signals(
        events,
        {},
        "clean answer",
        question="A patient mentor teaches debugging.",
        tags=["sage"],
    )

    assert signals["unconscious"]["top_k"] == ["sage", "shadow"]
    assert signals["unconscious"]["emergent_ideas"] == 1
    assert signals["unconscious"]["incubation_pressure"] > 0.0
    assert signals["psychoid"]["present"] is True
    assert signals["psychoid"]["signifier_chain_len"] == 1
    assert signals["default_mode"]["reflection_count"] == 1
    assert signals["default_mode"]["suppressed"] is True
    assert signals["archetype_trace"]["cue_alignment"] == "top_k_aligned"
    assert signals["archetype_trace"]["cue_top_k_hits"] == ["sage"]
    assert math.isclose(signals["archetype_trace"]["top1_margin"], 0.02, rel_tol=1e-9)
    assert signals["coherence"]["motif_score"] == 0.42
    assert signals["leakage"]["has_internal_leak"] is False
    assert signals["unconscious"]["score"] > 0.0


def test_extract_case_signals_marks_motif_only_archetype_cue_alignment():
    events = [
        {
            "event": "unconscious_field",
            "summary": {
                "top_k": ["sage", "hero"],
                "archetype_map": [
                    {"id": "sage", "label": "Sage", "intensity": 0.39},
                    {"id": "hero", "label": "Hero", "intensity": 0.37},
                    {"id": "shadow", "label": "Shadow", "intensity": 0.36},
                ],
                "emergent_ideas": [],
                "stress_released": 0.0,
                "cache_depth": 2,
                "motifs": ["archetype_match_shadow"],
                "psychoid_signal": {
                    "attention_bias": [{"archetype": "sage", "weight": 0.7}],
                    "signifier_chain": ["sage:book", "shadow:fear"],
                    "resonance": 0.2,
                    "psychoid_tension": 0.2,
                },
            },
        },
        {
            "event": "coherence_linguistic_motifs",
            "motifs": {"score": 0.2, "repeated_loops": 2},
        },
    ]

    signals = _extract_case_signals(
        events,
        {},
        "clean answer",
        question="A brittle legacy module creates hidden fear.",
        tags=["shadow"],
    )

    assert signals["archetype_trace"]["cue_alignment"] == "motif_only"
    assert signals["archetype_trace"]["cue_motif_hits"] == ["shadow"]
    assert signals["archetype_trace"]["motif_without_top_k"] == ["shadow"]
    assert signals["archetype_trace"]["motif_top_k_divergent"] is True
    assert signals["unconscious"]["unreleased_cache"] is True
    assert signals["unconscious"]["incubation_pressure"] > 0.0


def test_closest_near_miss_prefers_smallest_gap_then_similarity():
    attempts = [
        {
            "archetype": "sage",
            "emerged": False,
            "threshold_gap": 0.04,
            "trigger_similarity": 0.91,
        },
        {
            "archetype": "shadow",
            "emerged": False,
            "threshold_gap": 0.02,
            "trigger_similarity": 0.62,
        },
        {
            "archetype": "hero",
            "emerged": False,
            "threshold_gap": 0.02,
            "trigger_similarity": 0.72,
        },
        {
            "archetype": "trickster",
            "emerged": True,
            "threshold_gap": 0.001,
            "trigger_similarity": 0.99,
        },
    ]

    closest = _closest_near_miss(attempts)

    assert closest is not None
    assert closest["archetype"] == "hero"


def test_summarise_cases_reports_rates_and_averages():
    cases = [
        {
            "error": None,
            "latency_ms": 100.0,
            "unconscious": {
                "score": 0.5,
                "emergent_ideas": 1,
                "incubation_pressure": 0.3,
                "cache_depth": 3,
                "unreleased_cache": False,
            },
            "archetype_trace": {
                "cue_alignment": "top_k_aligned",
                "top1_margin": 0.1,
                "activation_entropy": 0.8,
                "motif_top_k_divergent": False,
                "ambiguous_activation": False,
            },
            "psychoid": {
                "present": True,
                "resonance": 0.4,
                "tension": 0.2,
                "signifier_chain_len": 2,
                "attention_bias_count": 4,
            },
            "default_mode": {
                "reflection_count": 1,
                "suppressed": True,
                "task_positive_load": 0.7,
            },
            "coherence": {
                "combined": 0.6,
                "tension": 0.2,
                "unconscious_weave_score": 0.5,
                "motif_score": 0.4,
                "motif_repeated_loops": 1,
            },
            "leakage": {"has_internal_leak": False},
        },
        {
            "error": None,
            "latency_ms": 300.0,
            "unconscious": {
                "score": 0.7,
                "emergent_ideas": 0,
                "incubation_pressure": 0.5,
                "cache_depth": 2,
                "unreleased_cache": True,
            },
            "archetype_trace": {
                "cue_alignment": "motif_only",
                "top1_margin": 0.02,
                "activation_entropy": 0.95,
                "motif_top_k_divergent": True,
                "ambiguous_activation": True,
            },
            "psychoid": {
                "present": True,
                "resonance": 0.6,
                "tension": 0.3,
                "signifier_chain_len": 3,
                "attention_bias_count": 4,
            },
            "default_mode": {
                "reflection_count": 0,
                "suppressed": False,
                "task_positive_load": 0.2,
            },
            "coherence": {
                "combined": 0.8,
                "tension": 0.1,
                "unconscious_weave_score": 0.7,
                "motif_score": None,
                "motif_repeated_loops": 2,
            },
            "leakage": {"has_internal_leak": True},
        },
        {"error": "boom"},
    ]

    summary = _summarise_cases(cases)

    assert summary["ok_cases"] == 2
    assert summary["error_cases"] == 1
    assert summary["leak_cases"] == 1
    assert math.isclose(summary["leak_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["avg_unconscious_score"], 0.6, rel_tol=1e-9)
    assert math.isclose(summary["avg_incubation_pressure"], 0.4, rel_tol=1e-9)
    assert math.isclose(summary["unreleased_cache_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["archetype_cue_top_k_alignment_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["archetype_cue_motif_only_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["archetype_motif_top_k_divergence_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["archetype_ambiguous_activation_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["avg_cache_depth"], 2.5, rel_tol=1e-9)
    assert math.isclose(summary["avg_latency_ms"], 200.0, rel_tol=1e-9)
