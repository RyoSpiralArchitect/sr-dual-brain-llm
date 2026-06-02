import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_benchmark_reports import compare_reports, summarize_history  # noqa: E402


def test_compare_reports_classifies_improvements_and_regressions():
    before = {
        "run_id": "creative-before",
        "timestamp": "2026-05-18T00:00:00Z",
        "summary": {
            "total_cases": 2,
            "ok_cases": 2,
            "error_cases": 0,
            "leak_rate": 0.1,
            "avg_unconscious_score": 0.4,
            "avg_latency_ms": 800.0,
        },
        "cases": [],
    }
    after = {
        "run_id": "creative-after",
        "timestamp": "2026-05-18T01:00:00Z",
        "summary": {
            "total_cases": 2,
            "ok_cases": 2,
            "error_cases": 0,
            "leak_rate": 0.0,
            "avg_unconscious_score": 0.6,
            "avg_latency_ms": 950.0,
        },
        "cases": [],
    }

    markdown = compare_reports(before, after)

    assert "# Benchmark Comparison" in markdown
    assert "| leak_rate | 10.0% | 0.0% | -10.0% | improved |" in markdown
    assert "| avg_unconscious_score | 0.400 | 0.600 | +0.200 | improved |" in markdown
    assert "| avg_latency_ms | 800.0 ms | 950.0 ms | +150.0 ms | regressed |" in markdown


def test_summarize_history_reports_first_to_latest_delta():
    rows = [
        {
            "run_id": "incubation-1",
            "timestamp": "2026-05-18T00:00:00Z",
            "summary": {
                "ok_sequences": 2,
                "error_sequences": 0,
                "emergent_sequence_rate": 0.5,
                "avg_closest_echo_near_miss_gap": 0.2,
            },
        },
        {
            "run_id": "incubation-2",
            "timestamp": "2026-05-18T01:00:00Z",
            "summary": {
                "ok_sequences": 3,
                "error_sequences": 0,
                "emergent_sequence_rate": 0.75,
                "avg_closest_echo_near_miss_gap": 0.1,
            },
        },
    ]

    markdown = summarize_history(rows, source="history.jsonl")

    assert "# Benchmark History Trend" in markdown
    assert "| incubation-2 | 2026-05-18T01:00:00Z | 3 | 0 | 75.0% | 0.100 |" in markdown
    assert "| emergent_sequence_rate | 50.0% | 75.0% | +25.0% | improved |" in markdown
    assert "| avg_closest_echo_near_miss_gap | 0.200 | 0.100 | -0.100 | improved |" in markdown


def test_cli_writes_history_markdown(tmp_path):
    history_path = tmp_path / "history.jsonl"
    output_path = tmp_path / "trend.md"
    rows = [
        {
            "run_id": "system2-1",
            "timestamp": "2026-05-18T00:00:00Z",
            "summary": {
                "ok_cases": 1,
                "error_cases": 0,
                "system2_activation_rate": 0.5,
                "avg_latency_ms_all_cases": 100.0,
            },
        },
        {
            "run_id": "system2-2",
            "timestamp": "2026-05-18T01:00:00Z",
            "summary": {
                "ok_cases": 2,
                "error_cases": 0,
                "system2_activation_rate": 1.0,
                "avg_latency_ms_all_cases": 90.0,
            },
        },
    ]
    history_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "compare_benchmark_reports.py"),
            "--history",
            str(history_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    markdown = output_path.read_text(encoding="utf-8")
    assert "Benchmark History Trend" in markdown
    assert "| system2_activation_rate | 50.0% | 100.0% | +50.0% | improved |" in markdown
