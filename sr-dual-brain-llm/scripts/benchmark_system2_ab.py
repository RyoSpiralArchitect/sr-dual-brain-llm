#!/usr/bin/env python3
"""Run System2 benchmark in A/B mode across multiple system2 settings.

This script reuses the existing single-run benchmark plumbing and executes the
same question set for each requested `system2_mode` (typically off/auto/on),
then writes a comparative report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_system2 import (  # noqa: E402
    _append_history,
    _check_critic_health,
    _history_trend,
    _load_history,
    _load_questions,
    _run_case,
    _summarise_cases,
)
from engine_stdio import EngineSession  # noqa: E402


def _parse_modes(raw: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for token in str(raw or "").split(","):
        mode = token.strip().lower()
        if mode not in {"off", "auto", "on"}:
            continue
        if mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    if not out:
        return ["off", "auto", "on"]
    return out


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _build_pairwise(summary_by_mode: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    def _phase_delta_map(
        left_summary: Dict[str, Any],
        right_summary: Dict[str, Any],
    ) -> Dict[str, float | None]:
        left_phases = (
            left_summary.get("avg_phase_latency_ms")
            if isinstance(left_summary.get("avg_phase_latency_ms"), dict)
            else {}
        )
        right_phases = (
            right_summary.get("avg_phase_latency_ms")
            if isinstance(right_summary.get("avg_phase_latency_ms"), dict)
            else {}
        )
        all_keys = sorted(set(left_phases.keys()) | set(right_phases.keys()))
        return {
            phase: _delta(
                _safe_float(left_phases.get(phase)),
                _safe_float(right_phases.get(phase)),
            )
            for phase in all_keys
        }

    def _cmp(lhs: str, rhs: str) -> Dict[str, Any]:
        left = summary_by_mode.get(lhs, {})
        right = summary_by_mode.get(rhs, {})
        return {
            "from": rhs,
            "to": lhs,
            "issue_reduction_rate_delta": _delta(
                _safe_float(left.get("issue_reduction_rate")),
                _safe_float(right.get("issue_reduction_rate")),
            ),
            "resolved_issue_rate_delta": _delta(
                _safe_float(left.get("resolved_issue_rate")),
                _safe_float(right.get("resolved_issue_rate")),
            ),
            "avg_latency_ms_delta": _delta(
                _safe_float(left.get("avg_latency_ms")),
                _safe_float(right.get("avg_latency_ms")),
            ),
            "avg_rounds_delta": _delta(
                _safe_float(left.get("avg_rounds")),
                _safe_float(right.get("avg_rounds")),
            ),
            "avg_phase_latency_ms_delta": _phase_delta_map(left, right),
            "error_cases_delta": _delta(
                _safe_float(left.get("error_cases")),
                _safe_float(right.get("error_cases")),
            ),
        }

    pairs = [("auto", "off"), ("on", "off"), ("on", "auto")]
    return {
        f"{lhs}_vs_{rhs}": _cmp(lhs, rhs)
        for lhs, rhs in pairs
        if lhs in summary_by_mode and rhs in summary_by_mode
    }


async def _run_mode(
    *,
    mode: str,
    questions: List[Dict[str, Any]],
    run_id: str,
    session_prefix: str,
    leading_brain: str,
    executive_mode: str,
    executive_observer_mode: str,
    critic_health_check: str,
    critic_health_attempts: int,
    require_critic_health: bool,
) -> Dict[str, Any]:
    session_id = f"{session_prefix}-{mode}"
    print(f"[ab] mode={mode} session_id={session_id}")
    session = await EngineSession.create(session_id=session_id)
    try:
        llm_capable = bool(
            getattr(session.left, "uses_external_llm", False)
            and getattr(session.right, "uses_external_llm", False)
        )
        critic_health: Dict[str, Any] = {"checked": False}
        if critic_health_check == "on":
            critic_health = await _check_critic_health(
                session=session,
                attempts=critic_health_attempts,
            )
            print(
                "[ab] mode={mode} critic_health healthy={healthy} attempts={attempts} provider={provider} model={model}".format(
                    mode=mode,
                    healthy=critic_health.get("healthy"),
                    attempts=critic_health.get("attempts"),
                    provider=critic_health.get("provider"),
                    model=critic_health.get("model"),
                )
            )
            if require_critic_health and not bool(critic_health.get("healthy")):
                raise RuntimeError(f"Critic health check failed for mode={mode}")

        cases: List[Dict[str, Any]] = []
        for idx, entry in enumerate(questions, 1):
            case = await _run_case(
                session=session,
                question_entry=entry,
                index=idx,
                run_id=f"{run_id}-{mode}",
                leading_brain=leading_brain,
                default_system2_mode=mode,
                executive_mode=executive_mode,
                executive_observer_mode=executive_observer_mode,
            )
            cases.append(case)
            print(
                "[ab] mode={mode} {idx:03d}/{total} id={id} enabled={enabled} "
                "issues={initial}->{final} rounds={rounds}/{target} resolved={resolved} error={error}".format(
                    mode=mode,
                    idx=idx,
                    total=len(questions),
                    id=case.get("id"),
                    enabled=case.get("system2_enabled"),
                    initial=case.get("initial_issues"),
                    final=case.get("final_issues"),
                    rounds=case.get("rounds"),
                    target=case.get("round_target"),
                    resolved=case.get("resolved"),
                    error=("yes" if case.get("error") else "no"),
                )
            )

        summary = _summarise_cases(cases)
        return {
            "mode": mode,
            "llm_capable": llm_capable,
            "critic_health": critic_health,
            "summary": summary,
            "cases": cases,
        }
    finally:
        await session.close()


async def _run(args: argparse.Namespace) -> int:
    questions_path = Path(args.questions).resolve()
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    questions = _load_questions(questions_path)
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(questions)
    if args.limit is not None and args.limit > 0:
        questions = questions[: int(args.limit)]
    if not questions:
        raise RuntimeError("No benchmark questions to run.")

    modes = _parse_modes(args.modes)
    run_id = time.strftime("system2_ab_%Y%m%d_%H%M%S")
    low_signal_filter = str(args.low_signal_filter or "on").strip().lower()
    if low_signal_filter not in {"on", "off"}:
        low_signal_filter = "on"
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = (
        "1" if low_signal_filter == "on" else "0"
    )
    critic_health_check = str(args.critic_health_check or "on").strip().lower()
    if critic_health_check not in {"on", "off"}:
        critic_health_check = "on"

    print(f"[ab] run_id={run_id}")
    print(f"[ab] questions={len(questions)} source={questions_path}")
    print(f"[ab] modes={modes}")
    print(f"[ab] system2_low_signal_filter={low_signal_filter}")

    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        by_mode[mode] = await _run_mode(
            mode=mode,
            questions=questions,
            run_id=run_id,
            session_prefix=f"{args.session_prefix}-{run_id}",
            leading_brain=args.leading_brain,
            executive_mode=args.executive_mode,
            executive_observer_mode=args.executive_observer_mode,
            critic_health_check=critic_health_check,
            critic_health_attempts=int(args.critic_health_attempts),
            require_critic_health=bool(args.require_critic_health),
        )

    summary_by_mode = {mode: payload.get("summary", {}) for mode, payload in by_mode.items()}
    pairwise = _build_pairwise(summary_by_mode)

    report = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question_set": str(questions_path),
        "config": {
            "modes": modes,
            "session_prefix": args.session_prefix,
            "leading_brain": args.leading_brain,
            "executive_mode": args.executive_mode,
            "executive_observer_mode": args.executive_observer_mode,
            "low_signal_filter": low_signal_filter,
            "critic_health_check": critic_health_check,
            "critic_health_attempts": int(args.critic_health_attempts),
            "require_critic_health": bool(args.require_critic_health),
            "question_count": len(questions),
        },
        "summary_by_mode": summary_by_mode,
        "pairwise": pairwise,
        "modes": by_mode,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ab] wrote report: {output_path}")

    history_path = Path(args.history).resolve() if args.history else None
    if history_path is not None:
        row = {
            "run_id": run_id,
            "timestamp": report["timestamp"],
            "question_set": str(questions_path),
            "summary_by_mode": summary_by_mode,
            "pairwise": pairwise,
            "config": report["config"],
        }
        _append_history(history_path, row)
        history = _load_history(history_path, limit=max(1, int(args.history_limit)))
        trend = _history_trend(history)
        print(
            "[ab] trend runs={runs} mean_issue_reduction_rate={rate} mean_resolved_issue_rate={resolved}".format(
                runs=trend.get("runs"),
                rate=trend.get("mean_issue_reduction_rate"),
                resolved=trend.get("mean_resolved_issue_rate"),
            )
        )
        print(f"[ab] appended history: {history_path}")

    for mode in modes:
        summary = summary_by_mode.get(mode, {})
        print(
            "[ab] summary mode={mode} measured={measured} reduction_rate={reduction} "
            "resolved_rate={resolved} avg_rounds={rounds} avg_latency_ms={latency}".format(
                mode=mode,
                measured=summary.get("measured_cases"),
                reduction=summary.get("issue_reduction_rate"),
                resolved=summary.get("resolved_issue_rate"),
                rounds=summary.get("avg_rounds"),
                latency=summary.get("avg_latency_ms"),
            )
        )

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "examples" / "system2_benchmark_questions_en.json"),
        help="Path to benchmark question set JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "samples" / "system2_ab_last.json"),
        help="Write comparative report JSON to this path.",
    )
    parser.add_argument(
        "--history",
        default=str(PROJECT_ROOT / "samples" / "system2_ab_history.jsonl"),
        help="Append compact run summary to JSONL history (set empty to disable).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="How many recent history runs to load for trend output.",
    )
    parser.add_argument(
        "--modes",
        default="off,auto,on",
        help="Comma separated system2 modes to compare (subset of off,auto,on).",
    )
    parser.add_argument(
        "--session-prefix",
        default="system2-ab",
        help="Session id prefix used for each mode run.",
    )
    parser.add_argument(
        "--leading-brain",
        choices=["auto", "left", "right"],
        default="auto",
    )
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
    parser.add_argument(
        "--low-signal-filter",
        choices=["on", "off"],
        default="on",
        help="Toggle System2 low-signal critic issue filter.",
    )
    parser.add_argument(
        "--critic-health-check",
        choices=["on", "off"],
        default="on",
        help="Run preflight critic health checks before benchmark cases.",
    )
    parser.add_argument(
        "--critic-health-attempts",
        type=int,
        default=3,
        help="Number of preflight critic probes used to judge JSON stability.",
    )
    parser.add_argument(
        "--require-critic-health",
        action="store_true",
        help="Abort benchmark when critic health check fails.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.history is not None and str(args.history).strip() == "":
        args.history = None
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
