#!/usr/bin/env python3
"""Render benchmark JSON reports as compact Markdown summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
            not key.startswith(("avg_", "mean_"))
            and (
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
        ):
            return f"{int(number)}"
        return f"{prefix}{_fmt_number(number)}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        pairs = (f"{name}={count}" for name, count in sorted(value.items()))
        return ", ".join(pairs) or "-"
    return str(value)


def _short_text(value: Any, limit: int = 84) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _join_values(value: Any, *, limit: int = 4) -> str:
    items = _as_list(value)
    if not items:
        return "-"
    rendered = [str(item) for item in items[:limit]]
    if len(items) > limit:
        rendered.append(f"+{len(items) - limit} more")
    return ", ".join(rendered)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("\n", "<br>").replace("|", "\\|")

    out = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(out)


def _tag_counts(
    items: Iterable[Mapping[str, Any]], *, include_turns: bool = False
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        for tag in _as_list(item.get("tags")):
            text = str(tag).strip()
            if text:
                counts[text] += 1
        if include_turns:
            for turn in _as_list(item.get("turns")):
                turn_data = _as_dict(turn)
                for tag in _as_list(turn_data.get("tags")):
                    text = str(tag).strip()
                    if text:
                        counts[text] += 1
    return counts


def _tag_coverage_section(report: Mapping[str, Any], kind: str) -> str:
    items = _as_list(report.get("sequences" if kind == "unconscious_incubation" else "cases"))
    counts = _tag_counts(
        [item for item in items if isinstance(item, dict)],
        include_turns=kind == "unconscious_incubation",
    )
    if not counts:
        return ""
    rows = [(tag, count) for tag, count in counts.most_common(12)]
    return "## Tag Coverage\n\n" + _table(("Tag", "Count"), rows)


def _count_breakdowns_section(summary: Mapping[str, Any]) -> str:
    rows: list[tuple[str, str, str]] = []
    for key, value in sorted(summary.items()):
        if not key.endswith("_counts") or not isinstance(value, dict):
            continue
        label = key.removesuffix("_counts")
        ordered_counts = sorted(
            value.items(),
            key=lambda item: (-(_safe_float(item[1]) or 0.0), str(item[0])),
        )
        for name, count in ordered_counts:
            rows.append((label, str(name), _fmt_value(count, key="count")))
    if not rows:
        return ""
    return "## Count Breakdowns\n\n" + _table(("Breakdown", "Value", "Count"), rows)


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


def _role_summary_section(summary: Mapping[str, Any]) -> str:
    turns_by_role = _as_dict(summary.get("turns_by_role"))
    if not turns_by_role:
        return ""
    rows = []
    for role, payload in sorted(turns_by_role.items()):
        data = _as_dict(payload)
        rows.append(
            (
                str(role),
                _fmt_value(data.get("turns"), key="turns"),
                _fmt_value(data.get("emergent_rate"), key="emergent_rate"),
                _fmt_value(data.get("seed_cached_rate"), key="seed_cached_rate"),
                _fmt_value(data.get("harvest_attempt_turn_rate"), key="harvest_attempt_turn_rate"),
                _fmt_value(data.get("near_miss_turn_rate"), key="near_miss_turn_rate"),
                _fmt_value(data.get("cue_top_k_alignment_rate"), key="cue_top_k_alignment_rate"),
                _fmt_value(data.get("avg_incubation_pressure"), key="avg_incubation_pressure"),
                _fmt_value(data.get("avg_closest_near_miss_gap"), key="avg_closest_near_miss_gap"),
            )
        )
    return "## Role Summary\n\n" + _table(
        (
            "Role",
            "Turns",
            "Emergent",
            "Seed Cached",
            "Harvest",
            "Near Miss",
            "Cue Align",
            "Pressure",
            "Gap",
        ),
        rows,
    )


def _system2_case_rows(
    items: Sequence[Any], limit: int
) -> list[tuple[str, str, str, str, str, str, str, str]]:
    rows = []
    for item in items[:limit]:
        data = _as_dict(item)
        initial = data.get("initial_issues")
        final = data.get("final_issues")
        issues = "-"
        if initial is not None or final is not None:
            before = _fmt_value(initial, key="issue_count")
            after = _fmt_value(final, key="issue_count")
            issues = f"{before} -> {after}"
        rows.append(
            (
                str(data.get("id") or data.get("qid") or "-"),
                _join_values(data.get("tags"), limit=3),
                _short_text(data.get("question"), 68),
                issues,
                _fmt_value(data.get("resolved")),
                _fmt_value(data.get("rounds"), key="rounds"),
                _fmt_value(data.get("latency_ms"), key="latency_ms"),
                str(data.get("system2_reason") or data.get("error") or "-"),
            )
        )
    return rows


def _creativity_case_rows(
    items: Sequence[Any], limit: int
) -> list[tuple[str, str, str, str, str, str, str, str, str, str]]:
    rows = []
    for item in items[:limit]:
        data = _as_dict(item)
        unconscious = _as_dict(data.get("unconscious"))
        coherence = _as_dict(data.get("coherence"))
        leakage = _as_dict(data.get("leakage"))
        rows.append(
            (
                str(data.get("id") or data.get("qid") or "-"),
                _join_values(data.get("tags"), limit=3),
                _short_text(data.get("question"), 58),
                _join_values(unconscious.get("top_k"), limit=3),
                _fmt_value(unconscious.get("score"), key="score"),
                _fmt_value(unconscious.get("incubation_pressure"), key="incubation_pressure"),
                _fmt_value(unconscious.get("cache_depth"), key="cache_depth"),
                _fmt_value(unconscious.get("emergent_ideas"), key="emergent_ideas"),
                _fmt_value(coherence.get("combined"), key="coherence_combined"),
                _fmt_value(leakage.get("has_internal_leak")),
            )
        )
    return rows


def _incubation_case_rows(
    items: Sequence[Any], limit: int
) -> list[tuple[str, str, str, str, str, str, str, str, str, str]]:
    rows = []
    for item in items[:limit]:
        data = _as_dict(item)
        obs = _as_dict(data.get("observation"))
        rows.append(
            (
                str(data.get("id") or "-"),
                _join_values(data.get("tags"), limit=3),
                _short_text(data.get("title"), 48),
                _fmt_value(obs.get("turns"), key="turns"),
                _fmt_value(obs.get("first_emergent_turn_index"), key="first_emergent_turn_index"),
                _join_values(obs.get("target_emergent_hits"), limit=3),
                _join_values(obs.get("seed_to_emergent_transitions"), limit=3),
                _fmt_value(obs.get("closest_echo_near_miss_gap"), key="closest_echo_near_miss_gap"),
                _fmt_value(obs.get("pressure_delta"), key="pressure_delta"),
                _fmt_value(obs.get("peak_cache_depth"), key="peak_cache_depth"),
            )
        )
    return rows


def _case_details(report: Mapping[str, Any], kind: str, limit: int) -> str:
    if limit <= 0:
        return ""
    items = _as_list(report.get("sequences" if kind == "unconscious_incubation" else "cases"))
    if not items:
        return ""
    if kind == "unconscious_incubation":
        return "## Sequence Details\n\n" + _table(
            (
                "Sequence",
                "Tags",
                "Title",
                "Turns",
                "First Emergent",
                "Target Hits",
                "Transitions",
                "Echo Gap",
                "Pressure Delta",
                "Peak Cache",
            ),
            _incubation_case_rows(items, limit),
        )
    if kind == "unconscious_creativity":
        return "## Case Details\n\n" + _table(
            (
                "Item",
                "Tags",
                "Prompt",
                "Top-K",
                "Score",
                "Pressure",
                "Cache",
                "Emergent",
                "Coherence",
                "Leak",
            ),
            _creativity_case_rows(items, limit),
        )
    return "## Case Details\n\n" + _table(
        (
            "Item",
            "Tags",
            "Prompt",
            "Issues",
            "Resolved",
            "Rounds",
            "Latency",
            "Reason",
        ),
        _system2_case_rows(items, limit),
    )


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
        role_summary = _role_summary_section(summary)
        if role_summary:
            sections.extend(["", role_summary])
    else:
        rows = sorted((str(k), _fmt_value(v, key=str(k))) for k, v in summary.items())
        sections.extend(["", "## Summary", "", _table(("Metric", "Value"), rows)])
    count_breakdowns = _count_breakdowns_section(summary)
    if count_breakdowns:
        sections.extend(["", count_breakdowns])
    tag_coverage = _tag_coverage_section(report, kind)
    if tag_coverage:
        sections.extend(["", tag_coverage])
    details = _case_details(report, kind, top)
    if details:
        sections.extend(["", details])
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
    parser.add_argument("--top", type=int, default=5, help="Number of detail rows to include.")
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
