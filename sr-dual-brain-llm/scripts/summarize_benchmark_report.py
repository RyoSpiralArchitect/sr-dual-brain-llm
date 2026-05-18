#!/usr/bin/env python3
"""Render benchmark JSON reports as compact Markdown summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_LABELS = {
    "system2": "System2",
    "system2_ab": "System2 A/B",
    "unconscious_creativity": "Unconscious Creativity",
    "unconscious_incubation": "Unconscious Incubation",
    "generic": "Generic Benchmark",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_value(value: Any, *, key: str = "", signed: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _safe_float(value)
    if number is not None:
        prefix = "+" if signed and number > 0 else ""
        if (
            key.endswith("_rate")
            or "_rate_" in key
            or key.endswith("_share")
            or "_share_" in key
        ):
            return f"{prefix}{number * 100:.1f}%"
        if key.endswith("_ms") or "_ms_" in key:
            return f"{prefix}{number:.1f} ms"
        if (
            not key.startswith(("avg_", "mean_", "min_", "max_"))
            and (
                key.endswith("_count")
                or key.endswith("_cases")
                or key.endswith("_turns")
                or key.endswith("_sequences")
                or key.endswith("_links")
                or key.endswith("_attempts")
            )
        ):
            return f"{int(number)}"
        return f"{prefix}{_fmt_number(number)}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def _metric_rows(summary: Mapping[str, Any], keys: Sequence[str]) -> list[tuple[str, str]]:
    return [(key, _fmt_value(summary.get(key), key=key)) for key in keys if key in summary]


def _config_rows(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys = (
        "question_count",
        "sequence_count",
        "modes",
        "system2_mode",
        "answer_mode",
        "leading_brain",
        "executive_mode",
        "executive_observer_mode",
        "low_signal_filter",
        "critic_health_check",
        "require_critic_health",
        "llm_capable",
    )
    return [(key, _fmt_value(config.get(key), key=key)) for key in keys if key in config]


def detect_report_kind(report: Mapping[str, Any]) -> str:
    summary = _as_dict(report.get("summary"))
    if isinstance(report.get("summary_by_mode"), dict):
        return "system2_ab"
    if isinstance(report.get("sequences"), list):
        return "unconscious_incubation"
    if isinstance(report.get("cases"), list):
        if "avg_unconscious_score" in summary or "leak_rate" in summary:
            return "unconscious_creativity"
        if "system2_activation_rate" in summary or "issue_reduction_rate" in summary:
            return "system2"
    return "generic"


def _run_rows(report: Mapping[str, Any], kind: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("type", REPORT_LABELS.get(kind, kind))]
    for key in ("run_id", "timestamp", "question_set"):
        if report.get(key) is not None:
            rows.append((key, _fmt_value(report.get(key), key=key)))
    sets = report.get("question_sets") or report.get("sequence_sets")
    if isinstance(sets, list) and sets:
        rows.append(("sets", ", ".join(str(Path(str(item)).name) for item in sets)))
    return rows


def _attention_items(report: Mapping[str, Any], kind: str) -> list[str]:
    summary = _as_dict(report.get("summary"))
    items: list[str] = []
    if kind == "system2":
        if (summary.get("error_cases") or 0) > 0:
            items.append(f"{summary.get('error_cases')} cases errored.")
        if (summary.get("no_op_cases") or 0) > 0:
            items.append(f"{summary.get('no_op_cases')} cases did not activate System2.")
        reduction = _safe_float(summary.get("mean_per_case_reduction_rate_all_cases"))
        if reduction is not None:
            items.append(f"All-case issue reduction averaged {_fmt_value(reduction, key='mean_per_case_reduction_rate_all_cases')}.")
    elif kind == "system2_ab":
        pairwise = _as_dict(report.get("pairwise"))
        on_vs_auto = _as_dict(pairwise.get("on_vs_auto"))
        delta = _safe_float(on_vs_auto.get("mean_per_case_reduction_rate_all_cases_delta"))
        if delta is not None:
            items.append(f"`on` vs `auto` all-case issue-reduction delta: {_fmt_value(delta, key='mean_per_case_reduction_rate_all_cases_delta', signed=True)}.")
    elif kind == "unconscious_creativity":
        leak_cases = _safe_int(summary.get("leak_cases")) or 0
        if leak_cases:
            items.append(f"{leak_cases} cases leaked internal markers.")
        else:
            items.append("No internal marker leaks were reported.")
        pressure = _safe_float(summary.get("avg_incubation_pressure"))
        if pressure is not None:
            items.append(f"Average incubation pressure: {_fmt_value(pressure, key='avg_incubation_pressure')}.")
    elif kind == "unconscious_incubation":
        leak_sequences = _safe_int(summary.get("leak_sequences")) or 0
        if leak_sequences:
            items.append(f"{leak_sequences} sequences leaked internal markers.")
        else:
            items.append("No sequence-level leaks were reported.")
        echo_gap = _safe_float(summary.get("avg_closest_echo_near_miss_gap"))
        if echo_gap is not None:
            items.append(f"Average closest echo near-miss gap: {_fmt_value(echo_gap, key='avg_closest_echo_near_miss_gap')}.")
    return items


def _system2_section(summary: Mapping[str, Any]) -> str:
    keys = (
        "total_cases",
        "ok_cases",
        "error_cases",
        "system2_activation_rate",
        "measured_cases",
        "no_op_cases",
        "issue_reduction_rate",
        "mean_per_case_reduction_rate_all_cases",
        "resolved_issue_rate",
        "resolved_issue_share_all_cases",
        "avg_rounds_all_cases",
        "avg_latency_ms_all_cases",
        "acc_override_consult_rate",
        "cerebellum_applied_rate",
        "cerebellum_issue_reduction_rate",
    )
    return "## Summary\n\n" + _table(("Metric", "Value"), _metric_rows(summary, keys))


def _system2_ab_section(report: Mapping[str, Any]) -> str:
    summary_by_mode = _as_dict(report.get("summary_by_mode"))
    modes = list(_as_list(_as_dict(report.get("config")).get("modes"))) or list(summary_by_mode)
    rows = []
    for mode in modes:
        summary = _as_dict(summary_by_mode.get(str(mode)))
        rows.append(
            (
                str(mode),
                _fmt_value(summary.get("ok_cases"), key="ok_cases"),
                _fmt_value(summary.get("measured_cases"), key="measured_cases"),
                _fmt_value(summary.get("system2_activation_rate"), key="system2_activation_rate"),
                _fmt_value(summary.get("mean_per_case_reduction_rate_all_cases"), key="mean_per_case_reduction_rate_all_cases"),
                _fmt_value(summary.get("resolved_issue_share_all_cases"), key="resolved_issue_share_all_cases"),
                _fmt_value(summary.get("avg_latency_ms_all_cases"), key="avg_latency_ms_all_cases"),
                _fmt_value(summary.get("avg_rounds_all_cases"), key="avg_rounds_all_cases"),
            )
        )
    out = [
        "## Mode Summary",
        "",
        _table(
            (
                "Mode",
                "OK",
                "Measured",
                "Activation",
                "Reduction",
                "Resolved",
                "Latency",
                "Rounds",
            ),
            rows,
        ),
    ]
    pairwise = _as_dict(report.get("pairwise"))
    if pairwise:
        pair_rows = []
        for name, payload in sorted(pairwise.items()):
            data = _as_dict(payload)
            pair_rows.append(
                (
                    name,
                    _fmt_value(
                        data.get("mean_per_case_reduction_rate_all_cases_delta"),
                        key="mean_per_case_reduction_rate_all_cases_delta",
                        signed=True,
                    ),
                    _fmt_value(
                        data.get("resolved_issue_share_all_cases_delta"),
                        key="resolved_issue_share_all_cases_delta",
                        signed=True,
                    ),
                    _fmt_value(
                        data.get("system2_activation_rate_delta"),
                        key="system2_activation_rate_delta",
                        signed=True,
                    ),
                    _fmt_value(
                        data.get("avg_latency_ms_all_cases_delta"),
                        key="avg_latency_ms_all_cases_delta",
                        signed=True,
                    ),
                )
            )
        out.extend(
            [
                "",
                "## Pairwise Deltas",
                "",
                _table(
                    ("Pair", "Reduction", "Resolved", "Activation", "Latency"),
                    pair_rows,
                ),
            ]
        )
    return "\n".join(out)


def _unconscious_creativity_section(summary: Mapping[str, Any]) -> str:
    keys = (
        "total_cases",
        "ok_cases",
        "error_cases",
        "leak_cases",
        "leak_rate",
        "avg_unconscious_score",
        "avg_incubation_pressure",
        "emergent_idea_rate",
        "unreleased_cache_rate",
        "archetype_cue_top_k_alignment_rate",
        "archetype_cue_motif_only_rate",
        "archetype_motif_top_k_divergence_rate",
        "avg_coherence_combined",
        "avg_latency_ms",
    )
    return "## Summary\n\n" + _table(("Metric", "Value"), _metric_rows(summary, keys))


def _unconscious_incubation_section(summary: Mapping[str, Any]) -> str:
    keys = (
        "total_sequences",
        "ok_sequences",
        "error_sequences",
        "total_turns",
        "leak_sequences",
        "emergent_sequences",
        "emergent_sequence_rate",
        "target_emergent_sequence_rate",
        "seed_cached_sequence_rate",
        "seed_to_emergent_links",
        "seed_to_emergent_same_archetype_rate",
        "seed_to_emergent_origin_match_rate",
        "near_miss_attempts",
        "echo_near_miss_attempts",
        "avg_closest_echo_near_miss_gap",
        "avg_peak_incubation_pressure",
        "max_peak_cache_depth",
    )
    return "## Summary\n\n" + _table(("Metric", "Value"), _metric_rows(summary, keys))


def _top_cases(report: Mapping[str, Any], kind: str, limit: int) -> str:
    if limit <= 0:
        return ""
    items = _as_list(report.get("sequences" if kind == "unconscious_incubation" else "cases"))
    if not items:
        return ""
    rows: list[tuple[str, str, str, str]] = []
    for item in items:
        data = _as_dict(item)
        if kind == "unconscious_incubation":
            obs = _as_dict(data.get("observation"))
            score = obs.get("closest_echo_near_miss_gap")
            signal = "echo_gap"
            detail = obs.get("target_emergent_hits") or obs.get("seed_to_emergent_transitions")
        elif kind == "unconscious_creativity":
            unconscious = _as_dict(data.get("unconscious"))
            score = unconscious.get("incubation_pressure")
            signal = "pressure"
            detail = _as_dict(unconscious.get("closest_near_miss")).get("archetype") or data.get("tags")
        else:
            score = data.get("latency_ms")
            signal = "latency_ms"
            detail = data.get("error") or data.get("resolved")
        rows.append(
            (
                str(data.get("id") or data.get("qid") or "-"),
                signal,
                _fmt_value(score, key=str(signal)),
                _fmt_value(detail),
            )
        )
    rows = rows[:limit]
    return "## Highlights\n\n" + _table(("Item", "Signal", "Value", "Detail"), rows)


def summarize_report(report: Mapping[str, Any], *, source: str | None = None, top: int = 5) -> str:
    kind = detect_report_kind(report)
    summary = _as_dict(report.get("summary"))
    config = _as_dict(report.get("config"))
    sections = [
        "# Benchmark Summary",
        "",
        "## Run",
        "",
        _table(("Field", "Value"), _run_rows(report, kind)),
    ]
    if source:
        sections.extend(["", f"Source: `{source}`"])
    config_rows = _config_rows(config)
    if config_rows:
        sections.extend(["", "## Config", "", _table(("Field", "Value"), config_rows)])
    attention = _attention_items(report, kind)
    if attention:
        sections.extend(["", "## Attention", ""])
        sections.extend(f"- {item}" for item in attention)
    if kind == "system2_ab":
        sections.extend(["", _system2_ab_section(report)])
    elif kind == "system2":
        sections.extend(["", _system2_section(summary)])
    elif kind == "unconscious_creativity":
        sections.extend(["", _unconscious_creativity_section(summary)])
    elif kind == "unconscious_incubation":
        sections.extend(["", _unconscious_incubation_section(summary)])
    else:
        rows = sorted((str(k), _fmt_value(v, key=str(k))) for k, v in summary.items())
        sections.extend(["", "## Summary", "", _table(("Metric", "Value"), rows)])
    highlights = _top_cases(report, kind, top)
    if highlights:
        sections.extend(["", highlights])
    return "\n".join(sections).rstrip() + "\n"


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to a benchmark JSON report.")
    parser.add_argument("--output", "-o", help="Write Markdown to this path instead of stdout.")
    parser.add_argument("--top", type=int, default=5, help="Number of highlight rows to include.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report_path = Path(args.report).expanduser().resolve()
    report = _load_report(report_path)
    markdown = summarize_report(report, source=str(report_path), top=int(args.top))
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
