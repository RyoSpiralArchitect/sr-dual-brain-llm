import math
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_system2 import (  # noqa: E402
    _build_system2_diagnostic,
    _normalise_system2_resolved_signal,
    _resolve_health_min_successes,
    _summarise_cases,
)
from benchmark_system2_ab import _build_pairwise  # noqa: E402
from engine_stdio import _extract_metrics  # noqa: E402


def test_summarise_cases_includes_all_case_noop_metrics():
    cases = [
        {
            "id": "c1",
            "error": None,
            "system2_enabled": True,
            "initial_issues": 4,
            "final_issues": 2,
            "resolved": False,
            "rounds": 2,
            "latency_ms": 1000.0,
            "followup_revision": False,
        },
        {
            "id": "c2",
            "error": None,
            "system2_enabled": True,
            "initial_issues": 1,
            "final_issues": 0,
            "resolved": True,
            "rounds": 1,
            "latency_ms": 1500.0,
            "followup_revision": False,
        },
        {
            "id": "c3",
            "error": None,
            "system2_enabled": False,
            "initial_issues": None,
            "final_issues": None,
            "resolved": None,
            "rounds": None,
            "latency_ms": 700.0,
            "followup_revision": False,
        },
    ]

    summary = _summarise_cases(cases)

    assert summary["ok_cases"] == 3
    assert summary["measured_cases"] == 2
    assert summary["no_op_cases"] == 1
    assert math.isclose(summary["system2_activation_rate"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(summary["measured_case_rate"], 2 / 3, rel_tol=1e-9)

    assert math.isclose(summary["avg_rounds"], 1.5, rel_tol=1e-9)
    assert math.isclose(summary["avg_rounds_all_cases"], 1.0, rel_tol=1e-9)
    assert math.isclose(summary["avg_latency_ms"], 1250.0, rel_tol=1e-9)
    assert math.isclose(
        summary["avg_latency_ms_all_cases"], (1000.0 + 1500.0 + 700.0) / 3.0, rel_tol=1e-9
    )

    assert math.isclose(summary["mean_per_case_reduction_rate"], 0.75, rel_tol=1e-9)
    assert math.isclose(summary["mean_per_case_reduction_rate_all_cases"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["resolved_issue_rate"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["resolved_issue_share_all_cases"], 1 / 3, rel_tol=1e-9)


def test_pairwise_contains_all_case_metric_deltas():
    summary_by_mode = {
        "auto": {
            "issue_reduction_rate": 0.2,
            "resolved_issue_rate": 0.1,
            "mean_per_case_reduction_rate_all_cases": 0.12,
            "resolved_issue_share_all_cases": 0.08,
            "system2_activation_rate": 0.75,
            "avg_latency_ms": 9000.0,
            "avg_latency_ms_all_cases": 9500.0,
            "avg_rounds": 1.6,
            "avg_rounds_all_cases": 1.2,
            "avg_phase_latency_ms": {"left_draft": 1000.0},
            "error_cases": 0,
        },
        "on": {
            "issue_reduction_rate": 0.3,
            "resolved_issue_rate": 0.2,
            "mean_per_case_reduction_rate_all_cases": 0.18,
            "resolved_issue_share_all_cases": 0.14,
            "system2_activation_rate": 1.0,
            "avg_latency_ms": 11000.0,
            "avg_latency_ms_all_cases": 11000.0,
            "avg_rounds": 1.9,
            "avg_rounds_all_cases": 1.9,
            "avg_phase_latency_ms": {"left_draft": 1200.0},
            "error_cases": 0,
        },
    }

    pairs = _build_pairwise(summary_by_mode)
    on_vs_auto = pairs["on_vs_auto"]

    assert math.isclose(
        on_vs_auto["mean_per_case_reduction_rate_all_cases_delta"], 0.06, rel_tol=1e-9
    )
    assert math.isclose(
        on_vs_auto["resolved_issue_share_all_cases_delta"], 0.06, rel_tol=1e-9
    )
    assert math.isclose(on_vs_auto["system2_activation_rate_delta"], 0.25, rel_tol=1e-9)
    assert math.isclose(on_vs_auto["avg_latency_ms_all_cases_delta"], 1500.0, rel_tol=1e-9)
    assert math.isclose(on_vs_auto["avg_rounds_all_cases_delta"], 0.7, rel_tol=1e-9)


def test_resolve_health_min_successes_defaults_and_clamps():
    assert _resolve_health_min_successes(attempts=1, min_successes=None) == 1
    assert _resolve_health_min_successes(attempts=3, min_successes=None) == 2
    assert _resolve_health_min_successes(attempts=5, min_successes=None) == 4

    assert _resolve_health_min_successes(attempts=3, min_successes=1) == 1
    assert _resolve_health_min_successes(attempts=3, min_successes=10) == 3
    assert _resolve_health_min_successes(attempts=3, min_successes=0) == 1


def test_clean_issue_counts_normalize_unresolved_signal():
    resolved, normalized = _normalise_system2_resolved_signal(
        resolved=False,
        initial_issues=0,
        final_issues=0,
    )

    assert resolved is True
    assert normalized is True


def test_system2_diagnostic_tracks_carried_over_issue_categories():
    diagnostic = _build_system2_diagnostic(
        policy_state={
            "critic_kind": "external_json",
            "critic_verdict": "issues",
            "critic_issues": [
                "The causal triage plan does not separate correlation from causation.",
                "The answer needs more concrete verification metrics.",
            ],
            "system2_verify_issues": [
                "Still confuses correlation with causation in the incident timeline.",
                "The plan lacks concrete verification metrics for rollback safety.",
            ],
            "critic_sum": "Initial critic says causal and verification coverage is weak.",
            "system2_verify_critic_sum": "Verify critic says the same issues remain.",
            "system2_followup_progress": 0,
            "system2_followup_eligible": False,
            "system2_truncation_signal": True,
        },
        initial_issues=2,
        final_issues=2,
        resolved=False,
        answer="Investigate the incident timeline.",
    )

    assert diagnostic["status"] == "stalled"
    assert len(diagnostic["carried_over_initial_issues"]) == 2
    assert "causal" in diagnostic["critic_sum_preview"]
    assert "same issues remain" in diagnostic["verify_critic_sum_preview"]
    assert diagnostic["category_counts_initial"]["causal_identification"] >= 1
    assert diagnostic["category_counts_final"]["verification"] >= 1
    assert diagnostic["truncation_signal"] is True


def test_summarise_cases_includes_diagnostic_rollup():
    cases = [
        {
            "id": "c1",
            "error": None,
            "system2_enabled": True,
            "initial_issues": 2,
            "final_issues": 2,
            "resolved": False,
            "rounds": 2,
            "latency_ms": 2000.0,
            "followup_revision": False,
            "truncation_signal": True,
            "system2_diagnostic": {
                "status": "stalled",
                "category_counts_initial": {"causal_identification": 1, "verification": 1},
                "category_counts_final": {"causal_identification": 1, "verification": 1},
                "carried_over_initial_issues": [{"issue": "causal issue"}],
                "newly_reported_final_issues": [],
                "truncation_signal": True,
            },
        }
    ]

    summary = _summarise_cases(cases)

    assert summary["diagnostics"]["diagnostic_cases"] == 1
    assert summary["diagnostics"]["status_counts"] == {"stalled": 1}
    assert summary["diagnostics"]["carried_over_initial_issue_count"] == 1
    assert summary["diagnostics"]["initial_issue_category_counts"]["verification"] == 1
    assert summary["diagnostics"]["truncation_signal_count"] == 1
    assert summary["truncation_signal_cases"] == 1


def test_extract_metrics_carries_system2_issue_texts():
    metrics = _extract_metrics(
        [
            {
                "event": "system2_mode",
                "mode": "on",
                "enabled": True,
                "reason": "forced_on",
            },
            {
                "event": "system2_refinement",
                "rounds": 3,
                "round_target": 3,
                "initial_issues": 2,
                "final_issues": 1,
                "resolved": False,
                "critic_issues": ["Initial causal issue"],
                "verify_issues": ["Verify causal issue"],
                "round3_issues": ["Remaining causal issue"],
                "critic_sum": "Initial critic summary",
                "verify_critic_sum": "Verify critic summary",
                "round3_critic_sum": "Round3 critic summary",
                "followup_progress": 1,
                "followup_eligible": True,
                "stalled_followup": True,
                "truncation_signal": True,
            },
        ]
    )

    system2 = metrics["system2"]

    assert system2["critic_issues"] == ["Initial causal issue"]
    assert system2["verify_issues"] == ["Verify causal issue"]
    assert system2["round3_issues"] == ["Remaining causal issue"]
    assert system2["critic_sum"] == "Initial critic summary"
    assert system2["verify_critic_sum"] == "Verify critic summary"
    assert system2["round3_critic_sum"] == "Round3 critic summary"
    assert system2["followup_progress"] == 1
    assert system2["followup_eligible"] is True
    assert system2["stalled_followup"] is True
    assert system2["truncation_signal"] is True
