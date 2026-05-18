import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from summarize_benchmark_report import detect_report_kind, summarize_report  # noqa: E402


def test_summarize_system2_ab_report_includes_modes_and_pairwise_delta():
    report = {
        "run_id": "run-ab",
        "timestamp": "2026-05-18T00:00:00Z",
        "config": {"modes": ["auto", "on"], "question_count": 2},
        "summary_by_mode": {
            "auto": {
                "ok_cases": 2,
                "measured_cases": 1,
                "system2_activation_rate": 0.5,
                "mean_per_case_reduction_rate_all_cases": 0.12,
                "resolved_issue_share_all_cases": 0.08,
                "avg_latency_ms_all_cases": 900.0,
                "avg_rounds_all_cases": 0.5,
            },
            "on": {
                "ok_cases": 2,
                "measured_cases": 2,
                "system2_activation_rate": 1.0,
                "mean_per_case_reduction_rate_all_cases": 0.18,
                "resolved_issue_share_all_cases": 0.14,
                "avg_latency_ms_all_cases": 1200.0,
                "avg_rounds_all_cases": 1.0,
            },
        },
        "pairwise": {
            "on_vs_auto": {
                "mean_per_case_reduction_rate_all_cases_delta": 0.06,
                "resolved_issue_share_all_cases_delta": 0.06,
                "system2_activation_rate_delta": 0.5,
                "avg_latency_ms_all_cases_delta": 300.0,
            }
        },
    }

    markdown = summarize_report(report, top=0)

    assert detect_report_kind(report) == "system2_ab"
    assert "System2 A/B" in markdown
    assert "| auto | 2 | 1 | 50.0% | 12.0% | 8.0% | 900.0 ms | 0.500 |" in markdown
    assert "| on_vs_auto | +6.0% | +6.0% | +50.0% | +300.0 ms |" in markdown


def test_summarize_unconscious_creativity_report_surfaces_leak_and_pressure():
    report = {
        "run_id": "creative",
        "timestamp": "2026-05-18T00:00:00Z",
        "config": {"question_count": 2, "answer_mode": "plain"},
        "summary": {
            "total_cases": 2,
            "ok_cases": 2,
            "leak_cases": 1,
            "leak_rate": 0.5,
            "avg_unconscious_score": 0.6,
            "avg_incubation_pressure": 0.4,
            "emergent_idea_rate": 0.5,
        },
        "cases": [
            {
                "id": "q1",
                "unconscious": {"incubation_pressure": 0.4},
                "tags": ["sage"],
            }
        ],
    }

    markdown = summarize_report(report, top=1)

    assert detect_report_kind(report) == "unconscious_creativity"
    assert "Unconscious Creativity" in markdown
    assert "- 1 cases leaked internal markers." in markdown
    assert "| leak_rate | 50.0% |" in markdown
    assert "| q1 | pressure | 0.400 | sage |" in markdown


def test_summarize_unconscious_incubation_report_surfaces_echo_near_miss():
    report = {
        "run_id": "incubation",
        "timestamp": "2026-05-18T00:00:00Z",
        "config": {"sequence_count": 1, "answer_mode": "plain"},
        "summary": {
            "total_sequences": 1,
            "ok_sequences": 1,
            "leak_sequences": 0,
            "emergent_sequence_rate": 1.0,
            "avg_closest_echo_near_miss_gap": 0.04,
            "near_miss_attempts": 1,
        },
        "sequences": [
            {
                "id": "mirror",
                "observation": {
                    "closest_echo_near_miss_gap": 0.04,
                    "target_emergent_hits": ["syzygy"],
                },
            }
        ],
    }

    markdown = summarize_report(report, top=1)

    assert detect_report_kind(report) == "unconscious_incubation"
    assert "Unconscious Incubation" in markdown
    assert "- No sequence-level leaks were reported." in markdown
    assert "| avg_closest_echo_near_miss_gap | 0.040 |" in markdown
    assert "| mirror | echo_gap | 0.040 | syzygy |" in markdown


def test_cli_writes_markdown_output(tmp_path):
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "summary.md"
    report_path.write_text(
        json.dumps(
            {
                "run_id": "system2",
                "summary": {
                    "total_cases": 1,
                    "ok_cases": 1,
                    "system2_activation_rate": 1.0,
                    "issue_reduction_rate": 0.25,
                },
                "cases": [{"id": "q1", "latency_ms": 123.4, "resolved": True}],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "summarize_benchmark_report.py"),
            str(report_path),
            "--output",
            str(output_path),
            "--top",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    markdown = output_path.read_text(encoding="utf-8")
    assert "System2" in markdown
    assert "| system2_activation_rate | 100.0% |" in markdown
