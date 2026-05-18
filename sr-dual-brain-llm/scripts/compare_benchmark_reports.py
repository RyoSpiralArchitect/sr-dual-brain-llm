#!/usr/bin/env python3
"""Compare benchmark reports or summarize benchmark history as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_benchmark_report import (  # noqa: E402
    REPORT_LABELS,
    _as_dict,
    _fmt_number,
    _fmt_value,
    _safe_float,
    _table,
    detect_report_kind,
)

HIGHER_BETTER = "higher"
LOWER_BETTER = "lower"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    polarity: str = NEUTRAL


SYSTEM2_METRICS = (
    MetricSpec("total_cases", "total_cases"),
    MetricSpec("ok_cases", "ok_cases", HIGHER_BETTER),
    MetricSpec("error_cases", "error_cases", LOWER_BETTER),
    MetricSpec("system2_activation_rate", "system2_activation_rate", HIGHER_BETTER),
    MetricSpec(
        "mean_per_case_reduction_rate_all_cases",
        "mean_per_case_reduction_rate_all_cases",
        HIGHER_BETTER,
    ),
    MetricSpec(
        "resolved_issue_share_all_cases",
        "resolved_issue_share_all_cases",
        HIGHER_BETTER,
    ),
    MetricSpec("avg_rounds_all_cases", "avg_rounds_all_cases"),
    MetricSpec("avg_latency_ms_all_cases", "avg_latency_ms_all_cases", LOWER_BETTER),
    MetricSpec("acc_override_consult_rate", "acc_override_consult_rate"),
    MetricSpec("cerebellum_applied_rate", "cerebellum_applied_rate"),
    MetricSpec("cerebellum_issue_reduction_rate", "cerebellum_issue_reduction_rate", HIGHER_BETTER),
)

CREATIVITY_METRICS = (
    MetricSpec("total_cases", "total_cases"),
    MetricSpec("ok_cases", "ok_cases", HIGHER_BETTER),
    MetricSpec("error_cases", "error_cases", LOWER_BETTER),
    MetricSpec("leak_rate", "leak_rate", LOWER_BETTER),
    MetricSpec("avg_unconscious_score", "avg_unconscious_score", HIGHER_BETTER),
    MetricSpec("avg_incubation_pressure", "avg_incubation_pressure"),
    MetricSpec("emergent_idea_rate", "emergent_idea_rate", HIGHER_BETTER),
    MetricSpec("unreleased_cache_rate", "unreleased_cache_rate"),
    MetricSpec(
        "archetype_cue_top_k_alignment_rate",
        "archetype_cue_top_k_alignment_rate",
        HIGHER_BETTER,
    ),
    MetricSpec(
        "archetype_motif_top_k_divergence_rate",
        "archetype_motif_top_k_divergence_rate",
    ),
    MetricSpec("avg_coherence_combined", "avg_coherence_combined", HIGHER_BETTER),
    MetricSpec("avg_latency_ms", "avg_latency_ms", LOWER_BETTER),
)

INCUBATION_METRICS = (
    MetricSpec("total_sequences", "total_sequences"),
    MetricSpec("ok_sequences", "ok_sequences", HIGHER_BETTER),
    MetricSpec("error_sequences", "error_sequences", LOWER_BETTER),
    MetricSpec("total_turns", "total_turns"),
    MetricSpec("leak_sequences", "leak_sequences", LOWER_BETTER),
    MetricSpec("emergent_sequence_rate", "emergent_sequence_rate", HIGHER_BETTER),
    MetricSpec("target_emergent_sequence_rate", "target_emergent_sequence_rate", HIGHER_BETTER),
    MetricSpec("seed_cached_sequence_rate", "seed_cached_sequence_rate", HIGHER_BETTER),
    MetricSpec("seed_to_emergent_links", "seed_to_emergent_links", HIGHER_BETTER),
    MetricSpec(
        "seed_to_emergent_same_archetype_rate",
        "seed_to_emergent_same_archetype_rate",
        HIGHER_BETTER,
    ),
    MetricSpec(
        "seed_to_emergent_origin_match_rate",
        "seed_to_emergent_origin_match_rate",
        HIGHER_BETTER,
    ),
    MetricSpec("near_miss_attempts", "near_miss_attempts"),
    MetricSpec("echo_near_miss_attempts", "echo_near_miss_attempts"),
    MetricSpec("avg_closest_echo_near_miss_gap", "avg_closest_echo_near_miss_gap", LOWER_BETTER),
    MetricSpec("avg_peak_incubation_pressure", "avg_peak_incubation_pressure"),
    MetricSpec("max_peak_cache_depth", "max_peak_cache_depth"),
)

SYSTEM2_AB_MODE_METRICS = (
    MetricSpec("ok_cases", "ok_cases", HIGHER_BETTER),
    MetricSpec("error_cases", "error_cases", LOWER_BETTER),
    MetricSpec("system2_activation_rate", "activation", HIGHER_BETTER),
    MetricSpec("mean_per_case_reduction_rate_all_cases", "reduction", HIGHER_BETTER),
    MetricSpec("resolved_issue_share_all_cases", "resolved", HIGHER_BETTER),
    MetricSpec("avg_latency_ms_all_cases", "latency", LOWER_BETTER),
    MetricSpec("avg_rounds_all_cases", "rounds"),
)


def infer_report_kind(report: Mapping[str, Any]) -> str:
    kind = detect_report_kind(report)
    if kind != "generic":
        return kind
    summary = _as_dict(report.get("summary"))
    run_id = str(report.get("run_id") or "")
    if "summary_by_mode" in report:
        return "system2_ab"
    if (
        "total_sequences" in summary
        or "emergent_sequence_rate" in summary
        or run_id.startswith("unconscious_incubation")
    ):
        return "unconscious_incubation"
    if "avg_unconscious_score" in summary or run_id.startswith("unconscious_creativity"):
        return "unconscious_creativity"
    if "system2_activation_rate" in summary or run_id.startswith("system2"):
        return "system2"
    return "generic"


def _mode_names(*reports: Mapping[str, Any]) -> list[str]:
    modes: set[str] = set()
    for report in reports:
        config_modes = _as_dict(report.get("config")).get("modes")
        if isinstance(config_modes, list):
            modes.update(str(mode) for mode in config_modes)
        modes.update(str(mode) for mode in _as_dict(report.get("summary_by_mode")))
    return sorted(modes)


def _metric_specs(kind: str, *reports: Mapping[str, Any]) -> list[MetricSpec]:
    if kind == "system2":
        return list(SYSTEM2_METRICS)
    if kind == "unconscious_creativity":
        return list(CREATIVITY_METRICS)
    if kind == "unconscious_incubation":
        return list(INCUBATION_METRICS)
    if kind == "system2_ab":
        specs: list[MetricSpec] = []
        for mode in _mode_names(*reports):
            for spec in SYSTEM2_AB_MODE_METRICS:
                specs.append(
                    MetricSpec(
                        f"summary_by_mode.{mode}.{spec.key}",
                        f"{mode} {spec.label}",
                        spec.polarity,
                    )
                )
        return specs

    keys: set[str] = set()
    for report in reports:
        for key, value in _as_dict(report.get("summary")).items():
            if _safe_float(value) is not None:
                keys.add(str(key))
    return [MetricSpec(key, key) for key in sorted(keys)]


def _value_for(report: Mapping[str, Any], spec: MetricSpec) -> Any:
    if spec.key.startswith("summary_by_mode."):
        _, mode, metric = spec.key.split(".", 2)
        return _as_dict(_as_dict(report.get("summary_by_mode")).get(mode)).get(metric)
    return _as_dict(report.get("summary")).get(spec.key)


def _metric_key(spec: MetricSpec) -> str:
    return spec.key.rsplit(".", 1)[-1]


def _is_rate_key(key: str) -> bool:
    return (
        key.endswith("_rate")
        or "_rate_" in key
        or key.endswith("_share")
        or "_share_" in key
    )


def _is_ms_key(key: str) -> bool:
    return key.endswith("_ms") or "_ms_" in key


def _is_count_key(key: str) -> bool:
    return (
        key in {"count", "rounds", "turns", "cache_depth", "peak_cache_depth"}
        or key.endswith("_index")
        or key.endswith("_cache_depth")
        or key.endswith("_count")
        or key.endswith("_cases")
        or key.endswith("_turns")
        or key.endswith("_sequences")
        or key.endswith("_links")
        or key.endswith("_attempts")
    )


def _fmt_delta(value: float | None, key: str) -> str:
    if value is None:
        return "-"
    prefix = "+" if value > 0 else ""
    if _is_rate_key(key):
        return f"{prefix}{value * 100:.1f}%"
    if _is_ms_key(key):
        return f"{prefix}{value:.1f} ms"
    if _is_count_key(key) and float(value).is_integer():
        return f"{prefix}{int(value)}"
    return f"{prefix}{_fmt_number(value)}"


def _assessment(delta: float | None, polarity: str) -> str:
    if delta is None:
        return "unavailable"
    if abs(delta) < 1e-12:
        return "same"
    if polarity == HIGHER_BETTER:
        return "improved" if delta > 0 else "regressed"
    if polarity == LOWER_BETTER:
        return "improved" if delta < 0 else "regressed"
    return "changed"


def _comparison_rows(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    specs: Sequence[MetricSpec],
) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for spec in specs:
        before_value = _value_for(before, spec)
        after_value = _value_for(after, spec)
        before_number = _safe_float(before_value)
        after_number = _safe_float(after_value)
        if before_number is None and after_number is None:
            continue
        delta = None
        if before_number is not None and after_number is not None:
            delta = after_number - before_number
        key = _metric_key(spec)
        rows.append(
            (
                spec.label,
                _fmt_value(before_value, key=key),
                _fmt_value(after_value, key=key),
                _fmt_delta(delta, key),
                _assessment(delta, spec.polarity),
            )
        )
    return rows


def _input_rows(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    before_source: str | None,
    after_source: str | None,
) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for label, report, source in (
        ("before", before, before_source),
        ("after", after, after_source),
    ):
        kind = infer_report_kind(report)
        rows.append(
            (
                label,
                REPORT_LABELS.get(kind, kind),
                str(report.get("run_id") or "-"),
                str(report.get("timestamp") or "-"),
                Path(source).name if source else "-",
            )
        )
    return rows


def _attention_from_rows(rows: Sequence[Sequence[str]]) -> list[str]:
    regressed = [row for row in rows if row[-1] == "regressed"]
    improved = [row for row in rows if row[-1] == "improved"]
    items: list[str] = []
    if regressed:
        details = ", ".join(f"{row[0]} ({row[3]})" for row in regressed[:5])
        items.append(f"Regressions: {details}.")
    if improved:
        details = ", ".join(f"{row[0]} ({row[3]})" for row in improved[:5])
        items.append(f"Improvements: {details}.")
    if not items:
        items.append("No oriented metric changed enough to classify as improved or regressed.")
    return items


def compare_reports(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_source: str | None = None,
    after_source: str | None = None,
) -> str:
    before_kind = infer_report_kind(before)
    after_kind = infer_report_kind(after)
    kind = before_kind if before_kind == after_kind else "generic"
    specs = _metric_specs(kind, before, after)
    rows = _comparison_rows(before, after, specs)
    sections = [
        "# Benchmark Comparison",
        "",
        "## Inputs",
        "",
        _table(
            ("Side", "Type", "Run ID", "Timestamp", "Source"),
            _input_rows(before, after, before_source, after_source),
        ),
        "",
        "## Attention",
        "",
    ]
    sections.extend(f"- {item}" for item in _attention_from_rows(rows))
    sections.extend(
        [
            "",
            "## Metric Deltas",
            "",
            _table(("Metric", "Before", "After", "Delta", "Assessment"), rows),
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def _history_metric_specs(kind: str, rows: Sequence[Mapping[str, Any]]) -> list[MetricSpec]:
    def has_value(spec: MetricSpec) -> bool:
        return any(_safe_float(_value_for(row, spec)) is not None for row in rows)

    specs = _metric_specs(kind, *rows)
    if kind == "system2_ab":
        return [spec for spec in specs if has_value(spec)][:12]
    preferred = {
        "system2": {
            "ok_cases",
            "error_cases",
            "system2_activation_rate",
            "mean_per_case_reduction_rate_all_cases",
            "resolved_issue_share_all_cases",
            "avg_latency_ms_all_cases",
        },
        "unconscious_creativity": {
            "ok_cases",
            "error_cases",
            "leak_rate",
            "avg_unconscious_score",
            "avg_incubation_pressure",
            "emergent_idea_rate",
            "avg_coherence_combined",
            "avg_latency_ms",
        },
        "unconscious_incubation": {
            "ok_sequences",
            "error_sequences",
            "emergent_sequence_rate",
            "target_emergent_sequence_rate",
            "seed_cached_sequence_rate",
            "seed_to_emergent_links",
            "avg_closest_echo_near_miss_gap",
            "avg_peak_incubation_pressure",
        },
    }.get(kind)
    if preferred:
        return [spec for spec in specs if spec.key in preferred and has_value(spec)]
    return [spec for spec in specs if has_value(spec)][:8]


def _history_run_rows(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[MetricSpec]
) -> list[list[str]]:
    out: list[list[str]] = []
    for row in rows:
        rendered = [
            str(row.get("run_id") or "-"),
            str(row.get("timestamp") or "-"),
        ]
        for spec in specs:
            rendered.append(_fmt_value(_value_for(row, spec), key=_metric_key(spec)))
        out.append(rendered)
    return out


def summarize_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str | None = None,
) -> str:
    if not rows:
        return "# Benchmark History Trend\n\nNo history rows found.\n"
    kind = infer_report_kind(rows[-1])
    specs = _history_metric_specs(kind, rows)
    first = rows[0]
    latest = rows[-1]
    delta_rows = _comparison_rows(first, latest, specs)
    headers = ["Run ID", "Timestamp"] + [spec.label for spec in specs]
    sections = [
        "# Benchmark History Trend",
        "",
        "## Source",
        "",
        _table(
            ("Field", "Value"),
            (
                ("type", REPORT_LABELS.get(kind, kind)),
                ("rows", str(len(rows))),
                ("source", source or "-"),
            ),
        ),
        "",
        "## Runs",
        "",
        _table(headers, _history_run_rows(rows, specs)),
        "",
        "## First To Latest Delta",
        "",
        _table(("Metric", "First", "Latest", "Delta", "Assessment"), delta_rows),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return data


def _load_history(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"History row {line_number} must be a JSON object: {path}")
        rows.append(data)
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", help="Two benchmark JSON reports to compare.")
    parser.add_argument("--history", help="Summarize a benchmark JSONL history file instead.")
    parser.add_argument("--limit", type=int, default=10, help="History rows to include from the tail.")
    parser.add_argument("--output", "-o", help="Write Markdown to this path instead of stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.history:
        if args.reports:
            parser.error("Do not pass report paths when --history is set.")
        history_path = Path(args.history).expanduser().resolve()
        markdown = summarize_history(
            _load_history(history_path, limit=int(args.limit)),
            source=str(history_path),
        )
    else:
        if len(args.reports) != 2:
            parser.error("Pass exactly two benchmark reports, or use --history.")
        before_path = Path(args.reports[0]).expanduser().resolve()
        after_path = Path(args.reports[1]).expanduser().resolve()
        markdown = compare_reports(
            _load_json(before_path),
            _load_json(after_path),
            before_source=str(before_path),
            after_source=str(after_path),
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
