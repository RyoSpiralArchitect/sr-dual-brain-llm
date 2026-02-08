#!/usr/bin/env python3
"""Run a fixed System2 benchmark and track issue-decay over time.

The benchmark replays a fixed question set through DualBrainController and
collects per-turn System2 metrics:
  - initial issues
  - final issues
  - rounds / round target
  - resolved flag

Each run writes a full JSON report and appends a compact history row to JSONL
so you can watch improvement trends across repeated experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from engine_stdio import EngineSession, _extract_metrics


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Question file must be a JSON array.")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw, 1):
        if isinstance(item, str):
            q = item.strip()
            if not q:
                continue
            out.append(
                {
                    "id": f"q{idx:03d}",
                    "question": q,
                }
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Question entry #{idx} must be string/object.")
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"Question entry #{idx} missing non-empty 'question'.")
        out.append(
            {
                "id": str(item.get("id") or f"q{idx:03d}"),
                "question": question,
                "system2_mode": (
                    str(item.get("system2_mode")).strip().lower()
                    if item.get("system2_mode") is not None
                    else None
                ),
                "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
            }
        )
    return out


def _last_event(events: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for ev in reversed(events):
        if ev.get("event") == name:
            return ev
    return {}


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


_CRITIC_FALLBACK_MARKERS = (
    "(fallback) external critic model unavailable",
    "(fallback) external critic response was unstructured",
)


def _is_critic_fallback_issue(issue: Any) -> bool:
    text = str(issue or "").strip().lower()
    return any(marker in text for marker in _CRITIC_FALLBACK_MARKERS)


def _evaluate_critic_health_result(result: Dict[str, Any]) -> tuple[bool, str]:
    verdict = str(result.get("verdict") or "").strip().lower()
    if verdict not in {"ok", "issues"}:
        return False, "invalid_verdict"
    issues = result.get("issues")
    if verdict == "issues":
        if not isinstance(issues, list) or not issues:
            return False, "empty_issues"
        if any(_is_critic_fallback_issue(item) for item in issues):
            return False, "fallback_issue"
    return True, "ok"


async def _check_critic_health(
    *,
    session: EngineSession,
    attempts: int,
) -> Dict[str, Any]:
    attempts = max(1, int(attempts))
    failures: List[Dict[str, Any]] = []
    for idx in range(attempts):
        try:
            result = await session.right.criticise_reasoning(
                qid=f"critic-health-{idx+1}",
                question="Compute 2+2 and explain briefly.",
                draft="2+2=5",
                temperature=0.15,
                context="Health check for external critic JSON stability.",
                allow_micro_fallback=False,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            failures.append(
                {
                    "attempt": idx + 1,
                    "reason": f"exception:{exc.__class__.__name__}",
                }
            )
            continue

        healthy, reason = _evaluate_critic_health_result(
            result if isinstance(result, dict) else {}
        )
        if not healthy:
            issues = result.get("issues") if isinstance(result, dict) else []
            failures.append(
                {
                    "attempt": idx + 1,
                    "reason": reason,
                    "verdict": (
                        str(result.get("verdict"))
                        if isinstance(result, dict) and result.get("verdict") is not None
                        else None
                    ),
                    "issues_preview": (
                        [str(item) for item in issues[:2]]
                        if isinstance(issues, list)
                        else []
                    ),
                }
            )

    cfg = getattr(session.right, "llm_config", None)
    provider = getattr(cfg, "provider", None)
    model = getattr(cfg, "model", None)
    return {
        "checked": True,
        "healthy": len(failures) == 0,
        "attempts": attempts,
        "provider": provider,
        "model": model,
        "failures": failures,
    }


def _summarise_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    ok_cases = [c for c in cases if not c.get("error")]
    measured = [
        c
        for c in ok_cases
        if c.get("initial_issues") is not None and c.get("final_issues") is not None
    ]
    measured_count = len(measured)
    system2_enabled_cases = [c for c in ok_cases if c.get("system2_enabled") is True]

    sum_initial = sum(int(c["initial_issues"]) for c in measured)
    sum_final = sum(int(c["final_issues"]) for c in measured)
    net_reduction = sum_initial - sum_final
    reduction_rate = (net_reduction / sum_initial) if sum_initial > 0 else None

    issue_cases = [c for c in measured if int(c["initial_issues"]) > 0]
    issue_cases_count = len(issue_cases)
    resolved_issue_cases = [
        c for c in issue_cases if c.get("resolved") is True and int(c["final_issues"]) == 0
    ]
    resolved_issue_rate = (
        len(resolved_issue_cases) / issue_cases_count if issue_cases_count > 0 else None
    )

    per_case_reduction = []
    rounds_values = []
    latency_values = []
    followup_count = 0
    for case in measured:
        initial = int(case["initial_issues"])
        final = int(case["final_issues"])
        if initial > 0:
            per_case_reduction.append((initial - final) / initial)
        rounds = _safe_float(case.get("rounds"))
        if rounds is not None:
            rounds_values.append(rounds)
        latency = _safe_float(case.get("latency_ms"))
        if latency is not None:
            latency_values.append(latency)
        if case.get("followup_revision") is True:
            followup_count += 1

    return {
        "total_cases": total,
        "ok_cases": len(ok_cases),
        "error_cases": total - len(ok_cases),
        "system2_enabled_cases": len(system2_enabled_cases),
        "measured_cases": measured_count,
        "sum_initial_issues": sum_initial,
        "sum_final_issues": sum_final,
        "net_issue_reduction": net_reduction,
        "issue_reduction_rate": reduction_rate,
        "issue_cases": issue_cases_count,
        "resolved_issue_cases": len(resolved_issue_cases),
        "resolved_issue_rate": resolved_issue_rate,
        "followup_revision_cases": followup_count,
        "followup_revision_rate": (
            followup_count / measured_count if measured_count > 0 else None
        ),
        "avg_rounds": (statistics.mean(rounds_values) if rounds_values else None),
        "avg_latency_ms": (statistics.mean(latency_values) if latency_values else None),
        "mean_per_case_reduction_rate": (
            statistics.mean(per_case_reduction) if per_case_reduction else None
        ),
    }


def _append_history(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_history(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    if limit > 0:
        return out[-limit:]
    return out


def _history_trend(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {"runs": 0}

    def _pick(path: str, row: Dict[str, Any]) -> Any:
        cur: Any = row
        for key in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    rates = []
    resolved_rates = []
    for row in history:
        rate = _safe_float(_pick("summary.issue_reduction_rate", row))
        if rate is not None:
            rates.append(rate)
        rr = _safe_float(_pick("summary.resolved_issue_rate", row))
        if rr is not None:
            resolved_rates.append(rr)
    latest = history[-1]
    return {
        "runs": len(history),
        "latest_run_id": latest.get("run_id"),
        "mean_issue_reduction_rate": (statistics.mean(rates) if rates else None),
        "mean_resolved_issue_rate": (
            statistics.mean(resolved_rates) if resolved_rates else None
        ),
    }


async def _run_case(
    *,
    session: EngineSession,
    question_entry: Dict[str, Any],
    index: int,
    run_id: str,
    leading_brain: str,
    default_system2_mode: str,
    executive_mode: str,
    executive_observer_mode: str,
) -> Dict[str, Any]:
    qid = f"{run_id}-c{index:03d}"
    question = str(question_entry.get("question") or "")
    mode = str(question_entry.get("system2_mode") or default_system2_mode).strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = default_system2_mode

    session.telemetry.clear()
    started = time.perf_counter()
    error_text: Optional[str] = None
    answer = ""
    try:
        answer = await session.controller.process(
            question,
            qid=qid,
            leading_brain=(None if leading_brain == "auto" else leading_brain),
            system2_mode=mode,
            executive_mode=executive_mode,
            executive_observer_mode=executive_observer_mode,
        )
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    events = list(session.telemetry.events)
    metrics = _extract_metrics(events) if events else {}
    system2 = metrics.get("system2") if isinstance(metrics.get("system2"), dict) else {}
    policy_state = (
        _last_event(events, "policy_decision").get("state")
        if isinstance(_last_event(events, "policy_decision").get("state"), dict)
        else {}
    )

    initial_issues = _safe_int(system2.get("initial_issues"))
    if initial_issues is None:
        initial_issues = _safe_int(policy_state.get("system2_issue_count_initial"))
    final_issues = _safe_int(system2.get("final_issues"))
    if final_issues is None:
        final_issues = _safe_int(policy_state.get("system2_issue_count_final"))
    rounds = _safe_int(system2.get("rounds"))
    if rounds is None:
        rounds = _safe_int(policy_state.get("system2_rounds"))
    round_target = _safe_int(system2.get("round_target"))
    if round_target is None:
        round_target = _safe_int(policy_state.get("system2_round_target"))
    resolved = _safe_bool(system2.get("resolved"))
    if resolved is None:
        resolved = _safe_bool(policy_state.get("system2_resolved"))
    followup_revision = _safe_bool(system2.get("followup_revision"))
    if followup_revision is None:
        followup_revision = _safe_bool(policy_state.get("system2_followup_revision"))
    system2_enabled = _safe_bool(system2.get("enabled"))
    if system2_enabled is None:
        system2_enabled = _safe_bool(policy_state.get("system2_enabled"))
    low_signal_filter = _safe_bool(system2.get("low_signal_filter"))
    if low_signal_filter is None:
        low_signal_filter = _safe_bool(policy_state.get("system2_low_signal_filter"))

    reduction = None
    if initial_issues is not None and final_issues is not None:
        reduction = initial_issues - final_issues

    interaction = _last_event(events, "interaction_complete")
    latency_ms = _safe_float(metrics.get("latency_ms"))
    if latency_ms is None:
        latency_ms = _safe_float(interaction.get("latency_ms"))

    return {
        "index": index,
        "id": question_entry.get("id") or f"q{index:03d}",
        "qid": qid,
        "question": question,
        "system2_mode": mode,
        "system2_enabled": system2_enabled,
        "low_signal_filter": low_signal_filter,
        "system2_reason": system2.get("reason") or policy_state.get("system2_reason"),
        "rounds": rounds,
        "round_target": round_target,
        "initial_issues": initial_issues,
        "final_issues": final_issues,
        "issue_reduction": reduction,
        "resolved": resolved,
        "followup_revision": followup_revision,
        "followup_new_issues": (
            system2.get("followup_new_issues")
            if isinstance(system2.get("followup_new_issues"), list)
            else (
                policy_state.get("system2_followup_new_issues")
                if isinstance(policy_state.get("system2_followup_new_issues"), list)
                else []
            )
        ),
        "latency_ms": latency_ms if latency_ms is not None else elapsed_ms,
        "error": error_text,
        "answer_preview": (answer[:240] if answer else ""),
    }


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

    run_id = time.strftime("system2_%Y%m%d_%H%M%S")
    session_id = str(args.session_id or run_id).strip() or run_id
    low_signal_filter = str(args.low_signal_filter or "on").strip().lower()
    if low_signal_filter not in {"on", "off"}:
        low_signal_filter = "on"
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = (
        "1" if low_signal_filter == "on" else "0"
    )

    print(f"[bench] run_id={run_id}")
    print(f"[bench] session_id={session_id}")
    print(f"[bench] questions={len(questions)} source={questions_path}")
    print(
        "[bench] system2_low_signal_filter={mode}".format(
            mode=low_signal_filter
        )
    )

    session = await EngineSession.create(session_id=session_id)
    try:
        llm_capable = bool(
            getattr(session.left, "uses_external_llm", False)
            and getattr(session.right, "uses_external_llm", False)
        )
        if not llm_capable:
            print(
                "[bench] warning: external LLM not configured for both hemispheres; "
                "System2 quality metrics may be pessimistic."
            )
        critic_health: Dict[str, Any] = {"checked": False}
        critic_health_mode = str(args.critic_health_check or "on").strip().lower()
        if critic_health_mode == "on":
            critic_health = await _check_critic_health(
                session=session,
                attempts=args.critic_health_attempts,
            )
            print(
                "[bench] critic_health healthy={healthy} attempts={attempts} provider={provider} model={model}".format(
                    healthy=critic_health.get("healthy"),
                    attempts=critic_health.get("attempts"),
                    provider=critic_health.get("provider"),
                    model=critic_health.get("model"),
                )
            )
            failures = critic_health.get("failures")
            if isinstance(failures, list) and failures:
                for failure in failures[:5]:
                    print(
                        "[bench] critic_health_failure attempt={attempt} reason={reason} verdict={verdict} issues={issues}".format(
                            attempt=failure.get("attempt"),
                            reason=failure.get("reason"),
                            verdict=failure.get("verdict"),
                            issues=failure.get("issues_preview"),
                        )
                    )
            if args.require_critic_health and not bool(critic_health.get("healthy")):
                raise RuntimeError(
                    "Critic health check failed. Resolve provider/model connectivity before benchmark."
                )

        cases: List[Dict[str, Any]] = []
        for idx, entry in enumerate(questions, 1):
            case = await _run_case(
                session=session,
                question_entry=entry,
                index=idx,
                run_id=run_id,
                leading_brain=args.leading_brain,
                default_system2_mode=args.system2_mode,
                executive_mode=args.executive_mode,
                executive_observer_mode=args.executive_observer_mode,
            )
            cases.append(case)
            print(
                "[bench] {idx:03d}/{total} id={id} mode={mode} enabled={enabled} "
                "issues={initial}->{final} rounds={rounds}/{target} resolved={resolved} error={error}".format(
                    idx=idx,
                    total=len(questions),
                    id=case.get("id"),
                    mode=case.get("system2_mode"),
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
        output = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question_set": str(questions_path),
            "config": {
                "session_id": session_id,
                "leading_brain": args.leading_brain,
                "system2_mode": args.system2_mode,
                "executive_mode": args.executive_mode,
                "executive_observer_mode": args.executive_observer_mode,
                "low_signal_filter": low_signal_filter,
                "critic_health_check": critic_health_mode,
                "critic_health_attempts": int(args.critic_health_attempts),
                "require_critic_health": bool(args.require_critic_health),
                "question_count": len(questions),
                "llm_capable": llm_capable,
            },
            "critic_health": critic_health,
            "summary": summary,
            "cases": cases,
        }

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[bench] wrote report: {output_path}")

        history_path = Path(args.history).resolve() if args.history else None
        history_trend: Dict[str, Any] = {"runs": 0}
        if history_path is not None:
            history_row = {
                "run_id": run_id,
                "timestamp": output["timestamp"],
                "question_set": str(questions_path),
                "summary": summary,
                "config": output["config"],
            }
            _append_history(history_path, history_row)
            history = _load_history(history_path, limit=max(1, int(args.history_limit)))
            history_trend = _history_trend(history)
            print(f"[bench] appended history: {history_path}")

        print(
            "[bench] summary measured={measured} reduction_rate={reduction} resolved_rate={resolved} "
            "avg_rounds={rounds} avg_latency_ms={latency}".format(
                measured=summary.get("measured_cases"),
                reduction=summary.get("issue_reduction_rate"),
                resolved=summary.get("resolved_issue_rate"),
                rounds=summary.get("avg_rounds"),
                latency=summary.get("avg_latency_ms"),
            )
        )
        if history_trend.get("runs", 0) > 0:
            print(
                "[bench] trend runs={runs} mean_reduction_rate={rr} mean_resolved_rate={sr}".format(
                    runs=history_trend.get("runs"),
                    rr=history_trend.get("mean_issue_reduction_rate"),
                    sr=history_trend.get("mean_resolved_issue_rate"),
                )
            )

    finally:
        await session.close()

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
        default=str(PROJECT_ROOT / "samples" / "system2_benchmark_last.json"),
        help="Write full benchmark report JSON to this path.",
    )
    parser.add_argument(
        "--history",
        default=str(PROJECT_ROOT / "samples" / "system2_benchmark_history.jsonl"),
        help="Append compact run summary to JSONL history (set empty to disable).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="How many recent history runs to load for trend output.",
    )
    parser.add_argument(
        "--session-id",
        default="system2-benchmark",
        help="Session id used for this benchmark run.",
    )
    parser.add_argument(
        "--system2-mode",
        choices=["auto", "on", "off"],
        default="on",
        help="Default system2 mode for each benchmark case.",
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
