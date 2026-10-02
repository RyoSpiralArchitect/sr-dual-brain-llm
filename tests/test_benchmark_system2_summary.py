import asyncio
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_system2_ab  # noqa: E402
import prepare_system2_scoring  # noqa: E402
import score_system2_reference  # noqa: E402
from benchmark_system2 import (  # noqa: E402
    _build_system2_diagnostic,
    _run_case,
    _normalise_system2_resolved_signal,
    _resolve_health_min_successes,
    _summarise_cases,
)
from benchmark_system2_ab import _build_pairwise  # noqa: E402
from engine_stdio import _extract_metrics  # noqa: E402


def test_ab_provenance_hashes_effective_questions_and_excludes_secrets(tmp_path):
    source = tmp_path / "questions.json"
    source.write_text('[{"id":"one","question":"Why?"}]', encoding="utf-8")
    questions = [{"id": "one", "question": "Why?"}]
    provenance = benchmark_system2_ab._question_provenance([source], questions)
    assert provenance["sources"][0]["path"] == "<external>/questions.json"
    assert len(provenance["sources"][0]["sha256"]) == 64
    assert provenance["effective_questions_sha256"] != benchmark_system2_ab._question_provenance(
        [source], [{"id": "one", "question": "Changed?"}]
    )["effective_questions_sha256"]

    model = SimpleNamespace(llm_config=SimpleNamespace(
        provider="openai", model="fixture", api_key="DO_NOT_STORE",
        max_output_tokens=512, timeout_seconds=30, auto_continue=False, max_continuations=0,
    ))
    public = benchmark_system2_ab._public_llm_config(model)
    assert public["model"] == "fixture"
    assert "DO_NOT_STORE" not in json.dumps(public)
    assert "api_key" not in public


def _synthetic_scoring_report(modes):
    config = {
        "modes": modes,
        "include_full_answers": True,
        "source_revision": {"commit": "a" * 40, "dirty": False},
        "question_provenance": {"effective_questions_sha256": "b" * 64},
        "effective_llm_by_mode": {"off": {"left": {"model": "fixture"}}, "on": {"left": {"model": "fixture"}}},
    }
    return {
        "run_id": "fixture-" + "-".join(modes),
        "config": config,
        "modes": {
            mode: {"cases": [{
                "id": "q1", "question": "Compute 2+2?", "system2_mode": mode,
                "system2_enabled": mode == "on",
                "answer": "Four." if mode == "off" else "4.", "error": None,
            }]}
            for mode in modes
        },
    }


def test_blind_scoring_packets_require_complete_counterbalanced_pairs(tmp_path):
    ab = _synthetic_scoring_report(["off", "on"])
    ba = _synthetic_scoring_report(["on", "off"])
    ab_path = tmp_path / "ab.json"
    ba_path = tmp_path / "ba.json"
    ab_path.write_text(json.dumps(ab), encoding="utf-8")
    ba_path.write_text(json.dumps(ba), encoding="utf-8")
    loaded_ab = prepare_system2_scoring._load_report(ab_path, ["off", "on"])
    loaded_ba = prepare_system2_scoring._load_report(ba_path, ["on", "off"])
    blind, key = prepare_system2_scoring.build_packets(loaded_ab, loaded_ba, seed=11)
    assert len(blind["packets"]) == 2
    assert len(key["assignments"]) == 2
    assert "off" not in json.dumps(blind["packets"])
    assert {item["block"] for item in key["assignments"]} == {"block_1", "block_2"}

    ba["modes"]["on"]["cases"][0]["answer"] = ""
    with pytest.raises(ValueError, match="missing complete answer"):
        prepare_system2_scoring.build_packets(ab, ba, seed=11)
    ba["modes"]["on"]["cases"][0]["answer"] = "4."
    ba["config"]["source_revision"]["commit"] = "c" * 40
    with pytest.raises(ValueError, match="source_revision differs"):
        prepare_system2_scoring.build_packets(ab, ba, seed=11)


def test_blind_scoring_cli_writes_local_private_files(tmp_path, monkeypatch):
    ab_path = tmp_path / "ab.json"
    ba_path = tmp_path / "ba.json"
    ab_path.write_text(json.dumps(_synthetic_scoring_report(["off", "on"])), encoding="utf-8")
    ba_path.write_text(json.dumps(_synthetic_scoring_report(["on", "off"])), encoding="utf-8")
    output_dir = tmp_path / "packets"
    monkeypatch.setattr(sys, "argv", [
        "prepare_system2_scoring.py", "--ab", str(ab_path), "--ba", str(ba_path),
        "--output-dir", str(output_dir), "--seed", "11",
    ])
    prepare_system2_scoring.main()
    blind_path = output_dir / "blind_packets.json"
    key_path = output_dir / "reveal_key.json"
    assert len(json.loads(blind_path.read_text(encoding="utf-8"))["packets"]) == 2
    assert len(json.loads(key_path.read_text(encoding="utf-8"))["assignments"]) == 2
    assert os.stat(blind_path).st_mode & 0o777 == 0o600
    assert os.stat(key_path).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        prepare_system2_scoring.main()


def test_full_answers_are_opt_in_for_ab_cases():
    async def answer(*_args, **_kwargs):
        return "A complete fixture answer."

    session = SimpleNamespace(
        telemetry=SimpleNamespace(clear=lambda: None, events=[]),
        controller=SimpleNamespace(process=answer),
    )
    params = {
        "session": session,
        "question_entry": {"id": "q1", "question": "Why?"},
        "index": 1,
        "run_id": "fixture",
        "leading_brain": "auto",
        "default_system2_mode": "off",
        "executive_mode": "off",
        "executive_observer_mode": "off",
        "diagnostics_mode": "off",
    }
    preview_only = asyncio.run(_run_case(**params))
    with_answer = asyncio.run(_run_case(**params, include_answer=True))
    assert "answer" not in preview_only
    assert with_answer["answer"] == "A complete fixture answer."


def test_critic_provider_failure_is_not_counted_as_issue_progress():
    cases = [
        {"id": "bad", "error": None, "critic_validity": "invalid", "critic_failure_reasons": ["provider_error"], "initial_issues": 5,
         "final_issues": 0, "resolved": True, "system2_enabled": True},
        {"id": "good", "error": None, "critic_validity": "valid", "initial_issues": 2,
         "final_issues": 1, "resolved": False, "system2_enabled": True},
    ]
    summary = _summarise_cases(cases)
    assert summary["critic_invalid_cases"] == 1
    assert summary["critic_invalid_by_reason"] == {"provider_error": 1}
    assert summary["measured_cases"] == 1
    assert summary["sum_initial_issues"] == 2
    assert summary["sum_final_issues"] == 1
    assert summary["resolved_issue_cases"] == 0


def test_reference_scorer_accepts_latex_and_rejects_wrong_final():
    score = score_system2_reference.score_answer
    assert score("probability_001", r"=\boxed{\frac{1}{11}}") == "correct"
    assert score("probability_001", r"=\boxed{\frac{1}{10}}") == "incorrect"
    assert score("arith_chain_001", "378 - 96 = 282; Answer: 47") == "correct"
    assert score("arith_chain_001", "378 - 96 = 282; Answer: 48") == "incorrect"
    assert score("algebra_001", r"x=12; verify 3(12)+7=43") == "correct"
    assert score("bayes_001", r"P(D\mid +)=0.1624, or 16.24%") == "correct"
    assert score("bayes_001", "Sensitivity is 0.95 and prevalence is 0.02") == "indeterminate"
    assert score("error_analysis_001", "Growth is 20%, not 40%") == "correct"
    assert score("error_analysis_001", "20% appears in the setup, but growth is 40%") == "incorrect"
    assert score("safety_policy_001", "Four rules") == "unscored"


def test_reference_pair_scores_both_orders_without_critic_self_grading():
    ab = _synthetic_scoring_report(["off", "on"])
    ba = _synthetic_scoring_report(["on", "off"])
    for report in (ab, ba):
        for mode in ("off", "on"):
            case = report["modes"][mode]["cases"][0]
            case["id"] = "probability_001"
            case["answer"] = r"\frac{1}{10}" if mode == "off" else r"\frac{1}{11}"
            case["critic_validity"] = "valid" if mode == "on" else "not_applicable"
    result = score_system2_reference.score_pair(ab, ba)
    assert result["summary"]["paired_scored"] == 2
    assert result["summary"]["on_wins"] == 2
    assert result["summary"]["off_wins"] == 0
    assert all("answer" not in row for row in result["rows"])


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


def test_ab_runner_passes_diagnostics_to_case_runner(monkeypatch):
    seen = []

    class FakeSession:
        left = SimpleNamespace(uses_external_llm=False)
        right = SimpleNamespace(uses_external_llm=False)

        async def close(self):
            return None

    async def fake_create(*, session_id):
        return FakeSession()

    async def fake_run_case(**kwargs):
        seen.append(kwargs)
        return {"id": "q1", "error": None, "system2_enabled": False}

    monkeypatch.setattr(benchmark_system2_ab, "EngineSession", SimpleNamespace(create=fake_create))
    monkeypatch.setattr(benchmark_system2_ab, "_run_case", fake_run_case)
    asyncio.run(
        benchmark_system2_ab._run_mode(
            mode="off",
            questions=[{"id": "q1", "question": "test"}],
            run_id="test",
            session_prefix="test",
            leading_brain="auto",
            executive_mode="off",
            executive_observer_mode="off",
            diagnostics_mode="all",
            critic_health_check="off",
            critic_health_attempts=1,
            critic_health_min_successes=1,
            critic_health_retries=0,
            critic_health_timeout=1.0,
            critic_health_rate_limit_backoff=0.0,
            require_critic_health=False,
        )
    )
    assert seen[0]["diagnostics_mode"] == "all"


def test_resolve_health_min_successes_defaults_and_clamps():
    assert _resolve_health_min_successes(attempts=1, min_successes=None) == 1
    assert _resolve_health_min_successes(attempts=3, min_successes=None) == 2
    assert _resolve_health_min_successes(attempts=5, min_successes=None) == 4

    assert _resolve_health_min_successes(attempts=3, min_successes=1) == 1
    assert _resolve_health_min_successes(attempts=3, min_successes=10) == 3
    assert _resolve_health_min_successes(attempts=3, min_successes=0) == 1


def test_clean_issue_counts_do_not_override_unresolved_signal():
    resolved, normalized = _normalise_system2_resolved_signal(
        resolved=False,
        initial_issues=0,
        final_issues=0,
    )

    assert resolved is False
    assert normalized is False


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
