import math
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_unconscious_creativity import (  # noqa: E402
    _extract_case_signals,
    _internal_leak_markers,
    _summarise_cases,
)


def test_internal_leak_markers_detect_debug_sections_case_insensitively():
    markers = _internal_leak_markers("final answer\n\n[unconscious insight]\n- bridge")

    assert markers == ["[Unconscious Insight]"]


def test_extract_case_signals_combines_unconscious_motif_and_leakage_metrics():
    events = [
        {
            "event": "unconscious_field",
            "summary": {
                "top_k": ["sage", "shadow"],
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

    signals = _extract_case_signals(events, {}, "clean answer")

    assert signals["unconscious"]["top_k"] == ["sage", "shadow"]
    assert signals["unconscious"]["emergent_ideas"] == 1
    assert signals["psychoid"]["present"] is True
    assert signals["psychoid"]["signifier_chain_len"] == 1
    assert signals["default_mode"]["reflection_count"] == 1
    assert signals["default_mode"]["suppressed"] is True
    assert signals["coherence"]["motif_score"] == 0.42
    assert signals["leakage"]["has_internal_leak"] is False
    assert signals["unconscious"]["score"] > 0.0


def test_summarise_cases_reports_rates_and_averages():
    cases = [
        {
            "error": None,
            "latency_ms": 100.0,
            "unconscious": {"score": 0.5, "emergent_ideas": 1},
            "psychoid": {"present": True, "resonance": 0.4, "tension": 0.2},
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
            },
            "leakage": {"has_internal_leak": False},
        },
        {
            "error": None,
            "latency_ms": 300.0,
            "unconscious": {"score": 0.7, "emergent_ideas": 0},
            "psychoid": {"present": True, "resonance": 0.6, "tension": 0.3},
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
    assert math.isclose(summary["avg_latency_ms"], 200.0, rel_tol=1e-9)
