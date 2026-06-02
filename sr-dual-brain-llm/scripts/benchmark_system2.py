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
import math
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


def _parse_question_paths(raw: str) -> List[Path]:
    tokens = [tok.strip() for tok in str(raw or "").split(",") if tok.strip()]
    return [Path(token).expanduser().resolve() for token in tokens]


def _filter_questions(
    questions: List[Dict[str, Any]],
    *,
    only_ids: str | None,
    only_tags: str | None,
) -> List[Dict[str, Any]]:
    ids = {tok.strip() for tok in str(only_ids or "").split(",") if tok.strip()}
    tags = {
        tok.strip().lower()
        for tok in str(only_tags or "").split(",")
        if tok.strip()
    }

    out = list(questions)
    if ids:
        out = [q for q in out if str(q.get("id") or "").strip() in ids]
    if tags:
        filtered: List[Dict[str, Any]] = []
        for q in out:
            raw_tags = q.get("tags")
            if not isinstance(raw_tags, list):
                raw_tags = []
            norm = {str(tag).strip().lower() for tag in raw_tags if str(tag).strip()}
            if norm.intersection(tags):
                filtered.append(q)
        out = filtered
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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    try:
        q = float(percentile)
    except Exception:
        return None
    if not math.isfinite(q):
        return None
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))

    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return float(data[0])

    k = (len(data) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(data[f])
    weight = k - f
    return float(data[f] * (1.0 - weight) + data[c] * weight)


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _normalise_system2_resolved_signal(
    *,
    resolved: Optional[bool],
    initial_issues: Optional[int],
    final_issues: Optional[int],
) -> tuple[Optional[bool], bool]:
    if (
        initial_issues is not None
        and final_issues is not None
        and int(initial_issues) == 0
        and int(final_issues) == 0
    ):
        return True, resolved is not True
    return resolved, False


_CRITIC_FALLBACK_MARKERS = (
    "(fallback) external critic model unavailable",
    "(fallback) external critic model not configured",
    "(fallback) external critic response was unstructured",
)


ISSUE_CATEGORY_KEYWORDS = {
    "causal_identification": (
        "causal",
        "cause",
        "causation",
        "correlation",
        "confound",
        "counterfactual",
        "rollback",
        "deployment",
        "timeline",
    ),
    "safety_privacy": (
        "pii",
        "privacy",
        "personal",
        "sensitive",
        "redact",
        "anonym",
        "policy",
        "consent",
        "retention",
        "access",
    ),
    "specificity": (
        "specific",
        "concrete",
        "example",
        "mechanism",
        "step",
        "actionable",
        "explicit",
    ),
    "verification": (
        "verify",
        "validation",
        "test",
        "measure",
        "metric",
        "evidence",
        "check",
        "confirm",
    ),
    "edge_cases": (
        "edge",
        "exception",
        "failure",
        "risk",
        "missing",
        "omit",
        "does not address",
    ),
    "quantitative_reasoning": (
        "calculate",
        "probability",
        "percentage",
        "rate",
        "unit",
        "numeric",
        "math",
    ),
}


def _is_critic_fallback_issue(issue: Any) -> bool:
    text = str(issue or "").strip().lower()
    if text.startswith("(fallback)"):
        return True
    return any(marker in text for marker in _CRITIC_FALLBACK_MARKERS)


def _text_list(value: Any, *, limit: int = 12, max_len: int = 500) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text[:max_len])
        if len(out) >= limit:
            break
    return out


def _normalise_issue_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def _issue_token_set(value: Any) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
    return {
        tok
        for tok in _normalise_issue_text(value).split()
        if len(tok) >= 4 and tok not in stop
    }


def _issue_similarity(left: Any, right: Any) -> float:
    lhs = _normalise_issue_text(left)
    rhs = _normalise_issue_text(right)
    if not lhs or not rhs:
        return 0.0
    if lhs in rhs or rhs in lhs:
        return 1.0
    left_tokens = _issue_token_set(lhs)
    right_tokens = _issue_token_set(rhs)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(
        left_tokens.union(right_tokens)
    )


def _match_issue(item: str, candidates: List[str], *, threshold: float = 0.2) -> Dict[str, Any]:
    best_text = ""
    best_score = 0.0
    item_categories = set(_issue_categories(item))
    for candidate in candidates:
        score = _issue_similarity(item, candidate)
        candidate_categories = set(_issue_categories(candidate))
        if item_categories.intersection(candidate_categories):
            score = max(score, 0.22)
        if score > best_score:
            best_score = score
            best_text = candidate
    return {
        "matched": bool(best_score >= threshold),
        "score": round(best_score, 4),
        "text": best_text if best_score >= threshold else "",
    }


def _issue_categories(issue: Any) -> List[str]:
    text = _normalise_issue_text(issue)
    categories = [
        category
        for category, keywords in ISSUE_CATEGORY_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    return categories or ["other"]


def _count_categories(issues: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for issue in issues:
        for category in _issue_categories(issue):
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _latest_system2_issues(policy_state: Dict[str, Any]) -> List[str]:
    round3 = _text_list(policy_state.get("system2_round3_issues"))
    if round3:
        return round3
    verify = _text_list(policy_state.get("system2_verify_issues"))
    if verify:
        return verify
    return _text_list(policy_state.get("critic_issues"))


def _build_system2_diagnostic(
    *,
    policy_state: Dict[str, Any],
    initial_issues: Optional[int],
    final_issues: Optional[int],
    resolved: Optional[bool],
    answer: str,
) -> Dict[str, Any]:
    initial_issue_texts = _text_list(policy_state.get("critic_issues"))
    verify_issue_texts = _text_list(policy_state.get("system2_verify_issues"))
    round3_issue_texts = _text_list(policy_state.get("system2_round3_issues"))
    final_issue_texts = _latest_system2_issues(policy_state)
    followup_new = _text_list(policy_state.get("system2_followup_new_issues"))
    round3_new = _text_list(policy_state.get("system2_round3_new_issues"))

    carried_over: List[Dict[str, Any]] = []
    resolved_initial: List[Dict[str, Any]] = []
    for issue in initial_issue_texts:
        match = _match_issue(issue, final_issue_texts)
        payload = {
            "issue": issue,
            "match_score": match["score"],
            "final_match": match["text"],
            "categories": _issue_categories(issue),
        }
        if match["matched"]:
            carried_over.append(payload)
        else:
            resolved_initial.append(payload)

    newly_reported: List[Dict[str, Any]] = []
    for issue in final_issue_texts:
        match = _match_issue(issue, initial_issue_texts)
        if not match["matched"]:
            newly_reported.append(
                {
                    "issue": issue,
                    "nearest_initial_score": match["score"],
                    "categories": _issue_categories(issue),
                }
            )

    progress = None
    if initial_issues is not None and final_issues is not None:
        progress = max(0, int(initial_issues) - int(final_issues))

    if resolved is True:
        status = "resolved"
    elif initial_issues is None or final_issues is None:
        status = "unmeasured"
    elif int(initial_issues or 0) == 0 and int(final_issues or 0) == 0:
        status = "clean_initial"
    elif progress == 0 and int(final_issues or 0) > 0:
        status = "stalled"
    elif progress and int(final_issues or 0) > 0:
        status = "partial_progress"
    else:
        status = "unresolved"

    return {
        "status": status,
        "critic_kind": policy_state.get("critic_kind"),
        "critic_verdict": policy_state.get("critic_verdict"),
        "followup_verdict": policy_state.get("system2_followup_verdict"),
        "round3_verdict": policy_state.get("system2_round3_verdict"),
        "initial_issue_texts": initial_issue_texts,
        "verify_issue_texts": verify_issue_texts,
        "round3_issue_texts": round3_issue_texts,
        "final_issue_texts": final_issue_texts,
        "carried_over_initial_issues": carried_over,
        "resolved_initial_issues": resolved_initial,
        "newly_reported_final_issues": newly_reported,
        "followup_new_issues": followup_new,
        "round3_new_issues": round3_new,
        "critic_sum_preview": str(policy_state.get("critic_sum") or "")[:800],
        "verify_critic_sum_preview": str(
            policy_state.get("system2_verify_critic_sum") or ""
        )[:800],
        "round3_critic_sum_preview": str(
            policy_state.get("system2_round3_critic_sum") or ""
        )[:800],
        "category_counts_initial": _count_categories(initial_issue_texts),
        "category_counts_final": _count_categories(final_issue_texts),
        "followup_progress": _safe_int(policy_state.get("system2_followup_progress")),
        "followup_eligible": _safe_bool(policy_state.get("system2_followup_eligible")),
        "verify_issues_raw": _safe_int(policy_state.get("system2_issue_count_verify_raw")),
        "verify_issues_calibrated": _safe_int(
            policy_state.get("system2_issue_count_verify_calibrated")
        ),
        "round3_issues_raw": _safe_int(policy_state.get("system2_issue_count_round3_raw")),
        "round3_issues_calibrated": _safe_int(
            policy_state.get("system2_issue_count_round3_calibrated")
        ),
        "truncation_signal": _safe_bool(
            policy_state.get("system2_truncation_signal")
        ),
        "pitfall_patterns": _text_list(policy_state.get("system2_pitfall_patterns")),
        "answer_preview": answer[:500] if answer else "",
    }


def _summarise_diagnostics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    diagnostics = [
        c.get("system2_diagnostic")
        for c in cases
        if isinstance(c.get("system2_diagnostic"), dict)
    ]
    status_counts: Dict[str, int] = {}
    initial_category_counts: Dict[str, int] = {}
    final_category_counts: Dict[str, int] = {}
    carried_over = 0
    newly_reported = 0
    truncation_signal_count = 0
    for diagnostic in diagnostics:
        status = str(diagnostic.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if diagnostic.get("truncation_signal") is True:
            truncation_signal_count += 1
        carried = diagnostic.get("carried_over_initial_issues")
        if isinstance(carried, list):
            carried_over += len(carried)
        new_final = diagnostic.get("newly_reported_final_issues")
        if isinstance(new_final, list):
            newly_reported += len(new_final)
        initial_counts = diagnostic.get("category_counts_initial")
        if isinstance(initial_counts, dict):
            for key, value in initial_counts.items():
                count = _safe_int(value)
                if count is not None:
                    initial_category_counts[str(key)] = (
                        initial_category_counts.get(str(key), 0) + count
                    )
        final_counts = diagnostic.get("category_counts_final")
        if isinstance(final_counts, dict):
            for key, value in final_counts.items():
                count = _safe_int(value)
                if count is not None:
                    final_category_counts[str(key)] = (
                        final_category_counts.get(str(key), 0) + count
                    )
    return {
        "diagnostic_cases": len(diagnostics),
        "status_counts": dict(sorted(status_counts.items())),
        "initial_issue_category_counts": dict(sorted(initial_category_counts.items())),
        "final_issue_category_counts": dict(sorted(final_category_counts.items())),
        "carried_over_initial_issue_count": carried_over,
        "newly_reported_final_issue_count": newly_reported,
        "truncation_signal_count": truncation_signal_count,
    }


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


def _resolve_health_min_successes(
    *,
    attempts: int,
    min_successes: Optional[int],
) -> int:
    attempts = max(1, int(attempts))
    if min_successes is None:
        return max(1, attempts - 1)
    try:
        value = int(min_successes)
    except Exception:
        value = attempts - 1
    return max(1, min(attempts, value))


async def _check_critic_health(
    *,
    session: EngineSession,
    attempts: int,
    min_successes: Optional[int] = None,
    retries_per_attempt: int = 1,
    timeout_seconds: Optional[float] = None,
    rate_limit_backoff_seconds: float = 2.5,
) -> Dict[str, Any]:
    attempts = max(1, int(attempts))
    retries_per_attempt = max(0, int(retries_per_attempt))
    rate_limit_backoff_seconds = max(0.0, float(rate_limit_backoff_seconds))
    required_successes = _resolve_health_min_successes(
        attempts=attempts,
        min_successes=min_successes,
    )

    cfg = getattr(session.right, "llm_config", None)
    provider = getattr(cfg, "provider", None)
    model = getattr(cfg, "model", None)
    timeout_override = _safe_float(timeout_seconds)
    original_timeout = None
    if cfg is not None and timeout_override is not None:
        try:
            original_timeout = float(getattr(cfg, "timeout_seconds", 40))
            setattr(cfg, "timeout_seconds", max(original_timeout, timeout_override))
        except Exception:
            original_timeout = None

    probes = [
        {
            "question": "Compute 2+2 and explain briefly.",
            "draft": "2+2=5",
        },
        {
            "question": "If all A are B and some B are C, must some A be C?",
            "draft": "Yes, it always follows.",
        },
        {
            "question": "A test has 95% sensitivity and 90% specificity; prevalence 2%. Is posterior near 95%?",
            "draft": "Yes, because sensitivity is 95%, posterior is around 95%.",
        },
    ]

    successes = 0
    failures: List[Dict[str, Any]] = []
    try:
        for idx in range(attempts):
            probe = probes[idx % len(probes)]
            probe_ok = False
            last_failure: Dict[str, Any] = {}
            for retry in range(retries_per_attempt + 1):
                try:
                    result = await session.right.criticise_reasoning(
                        qid=f"critic-health-{idx+1}-{retry+1}",
                        question=probe["question"],
                        draft=probe["draft"],
                        temperature=0.05,
                        context="Health check for external critic JSON stability.",
                        allow_micro_fallback=False,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    last_failure = {
                        "attempt": idx + 1,
                        "retry": retry + 1,
                        "reason": f"exception:{exc.__class__.__name__}",
                    }
                else:
                    payload = result if isinstance(result, dict) else {}
                    healthy, reason = _evaluate_critic_health_result(payload)
                    critic_kind = str(payload.get("critic_kind") or "").strip().lower()
                    if healthy and not critic_kind.startswith("external"):
                        healthy = False
                        reason = "non_external_kind"
                    if healthy:
                        successes += 1
                        probe_ok = True
                        break
                    issues = payload.get("issues")
                    last_failure = {
                        "attempt": idx + 1,
                        "retry": retry + 1,
                        "reason": reason,
                        "critic_kind": critic_kind or None,
                        "verdict": (
                            str(payload.get("verdict"))
                            if payload.get("verdict") is not None
                            else None
                        ),
                        "issues_preview": (
                            [str(item) for item in issues[:2]]
                            if isinstance(issues, list)
                            else []
                        ),
                        "critic_sum": str(payload.get("critic_sum") or "")[:200],
                    }

                if retry < retries_per_attempt:
                    failure_text = (
                        f"{last_failure.get('reason') or ''} "
                        f"{last_failure.get('critic_sum') or ''}"
                    ).lower()
                    if any(
                        marker in failure_text
                        for marker in ("rate limit", "rate_limited", "429")
                    ):
                        await asyncio.sleep(
                            (rate_limit_backoff_seconds * (retry + 1))
                            + random.random() * 0.25
                        )
                    else:
                        await asyncio.sleep(0.35 * (2**retry) + random.random() * 0.15)

            if not probe_ok:
                failures.append(last_failure or {"attempt": idx + 1, "reason": "unknown"})

            if successes >= required_successes:
                break
            remaining = attempts - (idx + 1)
            if successes + remaining < required_successes:
                break
    finally:
        if cfg is not None and original_timeout is not None:
            try:
                setattr(cfg, "timeout_seconds", original_timeout)
            except Exception:
                pass

    return {
        "checked": True,
        "healthy": successes >= required_successes,
        "attempts": attempts,
        "successes": successes,
        "required_successes": required_successes,
        "retries_per_attempt": retries_per_attempt,
        "timeout_seconds": timeout_override,
        "rate_limit_backoff_seconds": rate_limit_backoff_seconds,
        "provider": provider,
        "model": model,
        "failures": failures,
    }


def _summarise_cases_base(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    ok_cases = [c for c in cases if not c.get("error")]
    measured = [
        c
        for c in ok_cases
        if c.get("initial_issues") is not None and c.get("final_issues") is not None
    ]
    measured_count = len(measured)
    no_op_cases = [
        c
        for c in ok_cases
        if c.get("initial_issues") is None or c.get("final_issues") is None
    ]
    system2_enabled_cases = [c for c in ok_cases if c.get("system2_enabled") is True]
    truncation_signal_cases = [
        c for c in ok_cases if c.get("truncation_signal") is True
    ]

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
    per_case_reduction_all = []
    rounds_values = []
    rounds_values_all = []
    latency_values = []
    latency_values_all = []
    phase_latency_totals: Dict[str, float] = {}
    phase_latency_counts: Dict[str, int] = {}
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
        phase_latency = case.get("phase_latency_ms")
        if isinstance(phase_latency, dict):
            for phase, value in phase_latency.items():
                phase_name = str(phase or "").strip()
                phase_value = _safe_float(value)
                if not phase_name or phase_value is None:
                    continue
                phase_latency_totals[phase_name] = (
                    phase_latency_totals.get(phase_name, 0.0) + phase_value
                )
                phase_latency_counts[phase_name] = (
                    phase_latency_counts.get(phase_name, 0) + 1
                )
        if case.get("followup_revision") is True:
            followup_count += 1

    for case in ok_cases:
        initial_raw = case.get("initial_issues")
        final_raw = case.get("final_issues")
        if initial_raw is not None and final_raw is not None and int(initial_raw) > 0:
            initial = int(initial_raw)
            final = int(final_raw)
            per_case_reduction_all.append((initial - final) / initial)
        else:
            per_case_reduction_all.append(0.0)

        rounds_all = _safe_float(case.get("rounds"))
        rounds_values_all.append(rounds_all if rounds_all is not None else 0.0)

        latency_all = _safe_float(case.get("latency_ms"))
        if latency_all is not None:
            latency_values_all.append(latency_all)

    ok_count = len(ok_cases)
    activation_rate = (
        len(system2_enabled_cases) / ok_count if ok_count > 0 else None
    )
    measured_case_rate = measured_count / ok_count if ok_count > 0 else None
    resolved_issue_share_all = (
        len(resolved_issue_cases) / ok_count if ok_count > 0 else None
    )

    acc_conflict_values = [
        float(c["acc_conflict_level"])
        for c in ok_cases
        if c.get("acc_conflict_level") is not None
    ]
    acc_override_cases = [c for c in ok_cases if c.get("acc_override_consult") is True]
    acc_system2_bump_cases = [c for c in ok_cases if c.get("acc_system2_bump") is True]
    acc_temp_drops = [
        float(c["acc_temperature_drop"])
        for c in ok_cases
        if c.get("acc_temperature_drop") is not None
    ]

    cerebellum_applied_cases = [
        c for c in ok_cases if c.get("cerebellum_applied") is True
    ]
    cerebellum_resolved_cases = [
        c for c in ok_cases if c.get("cerebellum_resolved") is True
    ]
    cerebellum_measured = [
        c
        for c in ok_cases
        if c.get("cerebellum_initial_issues") is not None
        and c.get("cerebellum_final_issues") is not None
    ]
    cerebellum_sum_initial = sum(int(c["cerebellum_initial_issues"]) for c in cerebellum_measured)
    cerebellum_sum_final = sum(int(c["cerebellum_final_issues"]) for c in cerebellum_measured)
    cerebellum_net_reduction = cerebellum_sum_initial - cerebellum_sum_final
    cerebellum_reduction_rate = (
        cerebellum_net_reduction / cerebellum_sum_initial
        if cerebellum_sum_initial > 0
        else None
    )
    cerebellum_issue_cases = [
        c for c in cerebellum_measured if int(c["cerebellum_initial_issues"]) > 0
    ]
    cerebellum_issue_cases_count = len(cerebellum_issue_cases)
    cerebellum_resolved_issue_cases = [
        c for c in cerebellum_issue_cases if int(c["cerebellum_final_issues"]) == 0
    ]
    cerebellum_resolved_issue_rate = (
        len(cerebellum_resolved_issue_cases) / cerebellum_issue_cases_count
        if cerebellum_issue_cases_count > 0
        else None
    )

    return {
        "total_cases": total,
        "ok_cases": len(ok_cases),
        "error_cases": total - len(ok_cases),
        "system2_enabled_cases": len(system2_enabled_cases),
        "system2_activation_rate": activation_rate,
        "truncation_signal_cases": len(truncation_signal_cases),
        "truncation_signal_rate": (
            len(truncation_signal_cases) / len(ok_cases) if ok_cases else None
        ),
        "measured_cases": measured_count,
        "measured_case_rate": measured_case_rate,
        "no_op_cases": len(no_op_cases),
        "sum_initial_issues": sum_initial,
        "sum_final_issues": sum_final,
        "net_issue_reduction": net_reduction,
        "issue_reduction_rate": reduction_rate,
        "issue_cases": issue_cases_count,
        "resolved_issue_cases": len(resolved_issue_cases),
        "resolved_issue_rate": resolved_issue_rate,
        "resolved_issue_share_all_cases": resolved_issue_share_all,
        "followup_revision_cases": followup_count,
        "followup_revision_rate": (
            followup_count / measured_count if measured_count > 0 else None
        ),
        "avg_rounds": (statistics.mean(rounds_values) if rounds_values else None),
        "avg_rounds_all_cases": (
            statistics.mean(rounds_values_all) if rounds_values_all else None
        ),
        "rounds_p50": _percentile(rounds_values, 50),
        "rounds_p90": _percentile(rounds_values, 90),
        "rounds_p95": _percentile(rounds_values, 95),
        "rounds_all_cases_p50": _percentile(rounds_values_all, 50),
        "rounds_all_cases_p90": _percentile(rounds_values_all, 90),
        "rounds_all_cases_p95": _percentile(rounds_values_all, 95),
        "avg_latency_ms": (statistics.mean(latency_values) if latency_values else None),
        "avg_latency_ms_all_cases": (
            statistics.mean(latency_values_all) if latency_values_all else None
        ),
        "latency_ms_p50": _percentile(latency_values, 50),
        "latency_ms_p90": _percentile(latency_values, 90),
        "latency_ms_p95": _percentile(latency_values, 95),
        "latency_ms_all_cases_p50": _percentile(latency_values_all, 50),
        "latency_ms_all_cases_p90": _percentile(latency_values_all, 90),
        "latency_ms_all_cases_p95": _percentile(latency_values_all, 95),
        "avg_phase_latency_ms": {
            phase: (
                phase_latency_totals[phase] / phase_latency_counts[phase]
                if phase_latency_counts.get(phase, 0) > 0
                else None
            )
            for phase in sorted(phase_latency_totals.keys())
        },
        "mean_per_case_reduction_rate": (
            statistics.mean(per_case_reduction) if per_case_reduction else None
        ),
        "mean_per_case_reduction_rate_all_cases": (
            statistics.mean(per_case_reduction_all)
            if per_case_reduction_all
            else None
        ),
        "acc_conflict_signal_cases": len(acc_conflict_values),
        "acc_conflict_level_avg": (
            statistics.mean(acc_conflict_values) if acc_conflict_values else None
        ),
        "acc_override_consult_cases": len(acc_override_cases),
        "acc_override_consult_rate": (
            len(acc_override_cases) / ok_count if ok_count > 0 else None
        ),
        "acc_system2_bump_cases": len(acc_system2_bump_cases),
        "acc_system2_bump_rate": (
            len(acc_system2_bump_cases) / ok_count if ok_count > 0 else None
        ),
        "acc_temperature_drop_avg": (
            statistics.mean(acc_temp_drops) if acc_temp_drops else None
        ),
        "cerebellum_applied_cases": len(cerebellum_applied_cases),
        "cerebellum_applied_rate": (
            len(cerebellum_applied_cases) / ok_count if ok_count > 0 else None
        ),
        "cerebellum_resolved_cases": len(cerebellum_resolved_cases),
        "cerebellum_resolved_rate": (
            len(cerebellum_resolved_cases) / ok_count if ok_count > 0 else None
        ),
        "cerebellum_measured_cases": len(cerebellum_measured),
        "cerebellum_sum_initial_issues": cerebellum_sum_initial,
        "cerebellum_sum_final_issues": cerebellum_sum_final,
        "cerebellum_net_issue_reduction": cerebellum_net_reduction,
        "cerebellum_issue_reduction_rate": cerebellum_reduction_rate,
        "cerebellum_issue_cases": cerebellum_issue_cases_count,
        "cerebellum_resolved_issue_rate": cerebellum_resolved_issue_rate,
        "diagnostics": _summarise_diagnostics(cases),
    }


def _summarise_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _summarise_cases_base(cases)

    tag_map: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        raw_tags = case.get("tags")
        if not isinstance(raw_tags, list):
            continue
        for tag in raw_tags:
            norm = str(tag).strip().lower()
            if not norm:
                continue
            tag_map.setdefault(norm, []).append(case)

    summary["summary_by_tag"] = {
        tag: _summarise_cases_base(tag_cases)
        for tag, tag_cases in sorted(tag_map.items())
    }
    return summary


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


def _history_trend(
    history: List[Dict[str, Any]], *, question_set: str | None = None
) -> Dict[str, Any]:
    if not history:
        return {"runs": 0, "all_runs": 0}

    filtered = history
    qset = str(question_set or "").strip()
    if qset:
        filtered = [
            row for row in history if str(row.get("question_set") or "").strip() == qset
        ]
        if not filtered:
            filtered = history

    def _pick(path: str, row: Dict[str, Any]) -> Any:
        cur: Any = row
        for key in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    rates = []
    resolved_rates = []
    rates_all = []
    resolved_share_all = []
    for row in filtered:
        rate = _safe_float(_pick("summary.issue_reduction_rate", row))
        if rate is not None:
            rates.append(rate)
        rr = _safe_float(_pick("summary.resolved_issue_rate", row))
        if rr is not None:
            resolved_rates.append(rr)
        rate_all = _safe_float(_pick("summary.mean_per_case_reduction_rate_all_cases", row))
        if rate_all is not None:
            rates_all.append(rate_all)
        resolved_all = _safe_float(_pick("summary.resolved_issue_share_all_cases", row))
        if resolved_all is not None:
            resolved_share_all.append(resolved_all)
    latest = filtered[-1]
    return {
        "runs": len(filtered),
        "all_runs": len(history),
        "latest_run_id": latest.get("run_id"),
        "mean_issue_reduction_rate": (statistics.mean(rates) if rates else None),
        "mean_resolved_issue_rate": (
            statistics.mean(resolved_rates) if resolved_rates else None
        ),
        "mean_issue_reduction_rate_all_cases": (
            statistics.mean(rates_all) if rates_all else None
        ),
        "mean_resolved_issue_share_all_cases": (
            statistics.mean(resolved_share_all) if resolved_share_all else None
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
    diagnostics_mode: str,
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
    resolved, resolved_normalized = _normalise_system2_resolved_signal(
        resolved=resolved,
        initial_issues=initial_issues,
        final_issues=final_issues,
    )
    system2_enabled = _safe_bool(system2.get("enabled"))
    if system2_enabled is None:
        system2_enabled = _safe_bool(policy_state.get("system2_enabled"))
    low_signal_filter = _safe_bool(system2.get("low_signal_filter"))
    if low_signal_filter is None:
        low_signal_filter = _safe_bool(policy_state.get("system2_low_signal_filter"))
    truncation_signal = _safe_bool(system2.get("truncation_signal"))
    if truncation_signal is None:
        truncation_signal = _safe_bool(policy_state.get("system2_truncation_signal"))

    reduction = None
    if initial_issues is not None and final_issues is not None:
        reduction = initial_issues - final_issues

    interaction = _last_event(events, "interaction_complete")
    latency_ms = _safe_float(metrics.get("latency_ms"))
    if latency_ms is None:
        latency_ms = _safe_float(interaction.get("latency_ms"))
    latency_payload = metrics.get("latency") if isinstance(metrics.get("latency"), dict) else {}
    phase_latency = (
        latency_payload.get("phases_ms")
        if isinstance(latency_payload.get("phases_ms"), dict)
        else {}
    )

    acc_conflict_level = None
    acc_conflict_payload = policy_state.get("acc_conflict_pre")
    if isinstance(acc_conflict_payload, dict):
        acc_conflict_level = _safe_float(acc_conflict_payload.get("conflict_level"))
    acc_override_consult = _safe_bool(policy_state.get("acc_override_consult"))
    acc_system2_bump = _safe_bool(policy_state.get("acc_system2_bump"))
    acc_temperature_drop = None
    acc_temp_payload = policy_state.get("acc_temperature")
    if isinstance(acc_temp_payload, dict):
        acc_temperature_drop = _safe_float(acc_temp_payload.get("drop"))

    cerebellum_payload = policy_state.get("cerebellum_micro")
    cerebellum_applied = None
    cerebellum_resolved = None
    cerebellum_initial_issues = None
    cerebellum_final_issues = None
    cerebellum_domain = None
    cerebellum_confidence = None
    if isinstance(cerebellum_payload, dict):
        cerebellum_applied = _safe_bool(cerebellum_payload.get("applied"))
        cerebellum_resolved = _safe_bool(cerebellum_payload.get("resolved"))
        cerebellum_initial_issues = _safe_int(cerebellum_payload.get("initial_issues"))
        cerebellum_final_issues = _safe_int(cerebellum_payload.get("final_issues"))
        cerebellum_domain = (
            str(cerebellum_payload.get("domain") or "").strip() or None
        )
        cerebellum_confidence = _safe_float(cerebellum_payload.get("confidence"))

    case = {
        "index": index,
        "id": question_entry.get("id") or f"q{index:03d}",
        "qid": qid,
        "question": question,
        "tags": (
            question_entry.get("tags") if isinstance(question_entry.get("tags"), list) else []
        ),
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
        "resolved_normalized_from_clean_issue_counts": resolved_normalized,
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
        "followup_progress": _safe_int(system2.get("followup_progress")),
        "followup_eligible": _safe_bool(system2.get("followup_eligible")),
        "stalled_followup": _safe_bool(system2.get("stalled_followup")),
        "truncation_signal": truncation_signal,
        "latency_ms": latency_ms if latency_ms is not None else elapsed_ms,
        "phase_latency_ms": phase_latency,
        "error": error_text,
        "answer_preview": (answer[:240] if answer else ""),
        "acc_conflict_level": acc_conflict_level,
        "acc_override_consult": acc_override_consult,
        "acc_system2_bump": acc_system2_bump,
        "acc_temperature_drop": acc_temperature_drop,
        "cerebellum_applied": cerebellum_applied,
        "cerebellum_resolved": cerebellum_resolved,
        "cerebellum_initial_issues": cerebellum_initial_issues,
        "cerebellum_final_issues": cerebellum_final_issues,
        "cerebellum_domain": cerebellum_domain,
        "cerebellum_confidence": cerebellum_confidence,
    }
    diagnostics_norm = str(diagnostics_mode or "off").strip().lower()
    if diagnostics_norm not in {"off", "unresolved", "all"}:
        diagnostics_norm = "off"
    should_emit_diagnostic = bool(
        diagnostics_norm == "all"
        or (
            diagnostics_norm == "unresolved"
            and (
                resolved is False
                or (
                    initial_issues is not None
                    and final_issues is not None
                    and int(final_issues) > 0
                )
            )
        )
    )
    if should_emit_diagnostic:
        diagnostic_state = dict(policy_state)
        if isinstance(system2.get("critic_issues"), list):
            diagnostic_state["critic_issues"] = system2.get("critic_issues")
        if isinstance(system2.get("verify_issues"), list):
            diagnostic_state["system2_verify_issues"] = system2.get("verify_issues")
        if isinstance(system2.get("round3_issues"), list):
            diagnostic_state["system2_round3_issues"] = system2.get("round3_issues")
        if isinstance(system2.get("followup_new_issues"), list):
            diagnostic_state["system2_followup_new_issues"] = system2.get(
                "followup_new_issues"
            )
        if system2.get("followup_verdict") is not None:
            diagnostic_state["system2_followup_verdict"] = system2.get(
                "followup_verdict"
            )
        if system2.get("critic_sum") is not None:
            diagnostic_state["critic_sum"] = system2.get("critic_sum")
        if system2.get("verify_critic_sum") is not None:
            diagnostic_state["system2_verify_critic_sum"] = system2.get(
                "verify_critic_sum"
            )
        if system2.get("round3_critic_sum") is not None:
            diagnostic_state["system2_round3_critic_sum"] = system2.get(
                "round3_critic_sum"
            )
        if system2.get("followup_progress") is not None:
            diagnostic_state["system2_followup_progress"] = system2.get(
                "followup_progress"
            )
        if system2.get("followup_eligible") is not None:
            diagnostic_state["system2_followup_eligible"] = system2.get(
                "followup_eligible"
            )
        if system2.get("stalled_followup") is not None:
            diagnostic_state["system2_stalled_followup"] = system2.get(
                "stalled_followup"
            )
        if system2.get("truncation_signal") is not None:
            diagnostic_state["system2_truncation_signal"] = system2.get(
                "truncation_signal"
            )
        case["system2_diagnostic"] = _build_system2_diagnostic(
            policy_state=diagnostic_state,
            initial_issues=initial_issues,
            final_issues=final_issues,
            resolved=resolved,
            answer=answer,
        )
    return case


async def _run(args: argparse.Namespace) -> int:
    question_paths = _parse_question_paths(args.questions)
    if not question_paths:
        raise ValueError("No --questions paths provided.")
    missing_paths = [path for path in question_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Questions file(s) not found: {paths}".format(
                paths=", ".join(str(p) for p in missing_paths)
            )
        )

    questions: List[Dict[str, Any]] = []
    for path in question_paths:
        questions.extend(_load_questions(path))
    questions = _filter_questions(
        questions,
        only_ids=getattr(args, "only_ids", None),
        only_tags=getattr(args, "only_tags", None),
    )
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for entry in questions:
        qid = str(entry.get("id") or "").strip()
        if not qid:
            continue
        if qid in seen_ids:
            duplicate_ids.add(qid)
        else:
            seen_ids.add(qid)
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:8])
        more = "…" if len(duplicate_ids) > 8 else ""
        print(
            f"[bench] warning: duplicate question ids detected ({len(duplicate_ids)}): {preview}{more}"
        )
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
    question_set_label = ",".join(str(path) for path in question_paths)
    if len(question_paths) == 1:
        print(f"[bench] questions={len(questions)} source={question_paths[0]}")
    else:
        print(f"[bench] questions={len(questions)} sources={len(question_paths)}")
        for path in question_paths:
            print(f"[bench] question_source={path}")
    if getattr(args, "only_ids", None):
        print(f"[bench] filter only_ids={args.only_ids}")
    if getattr(args, "only_tags", None):
        print(f"[bench] filter only_tags={args.only_tags}")
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
            left_external = bool(getattr(session.left, "uses_external_llm", False))
            right_external = bool(getattr(session.right, "uses_external_llm", False))
            provider_env = (
                os.environ.get("LLM_PROVIDER")
                or os.environ.get("LEFT_BRAIN_PROVIDER")
                or os.environ.get("RIGHT_BRAIN_PROVIDER")
            )
            model_env = (
                os.environ.get("LLM_MODEL_ID")
                or os.environ.get("LEFT_BRAIN_MODEL")
                or os.environ.get("RIGHT_BRAIN_MODEL")
            )
            has_api_key = any(
                os.environ.get(name)
                for name in (
                    "LLM_API_KEY",
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "MISTRAL_API_KEY",
                    "XAI_API_KEY",
                    "HUGGINGFACE_API_TOKEN",
                    "HF_TOKEN",
                )
            )
            if has_api_key and (not provider_env or not model_env):
                print(
                    "[bench] hint: API key is set but provider/model is missing. "
                    "Set LLM_PROVIDER + LLM_MODEL_ID (or LEFT_BRAIN_* / RIGHT_BRAIN_*)."
                )
            elif provider_env and model_env and has_api_key:
                print(
                    f"[bench] hint: external LLM active? left={left_external} right={right_external}. "
                    "You can configure hemispheres separately via LEFT_BRAIN_* and RIGHT_BRAIN_* env vars."
                )
        critic_health: Dict[str, Any] = {"checked": False}
        critic_health_mode = str(args.critic_health_check or "on").strip().lower()
        if critic_health_mode == "on":
            critic_health = await _check_critic_health(
                session=session,
                attempts=args.critic_health_attempts,
                min_successes=args.critic_health_min_successes,
                retries_per_attempt=args.critic_health_retries,
                timeout_seconds=args.critic_health_timeout,
                rate_limit_backoff_seconds=args.critic_health_rate_limit_backoff,
            )
            print(
                "[bench] critic_health healthy={healthy} successes={successes}/{required} attempts={attempts} "
                "retries={retries} timeout={timeout} rate_limit_backoff={rate_limit_backoff} provider={provider} model={model}".format(
                    healthy=critic_health.get("healthy"),
                    successes=critic_health.get("successes"),
                    required=critic_health.get("required_successes"),
                    attempts=critic_health.get("attempts"),
                    retries=critic_health.get("retries_per_attempt"),
                    timeout=critic_health.get("timeout_seconds"),
                    rate_limit_backoff=critic_health.get("rate_limit_backoff_seconds"),
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
                provider = critic_health.get("provider")
                model = critic_health.get("model")
                raise RuntimeError(
                    "Critic health check failed (provider={provider} model={model}). "
                    "Configure external critic via LLM_PROVIDER + LLM_MODEL_ID + <PROVIDER>_API_KEY "
                    "(or RIGHT_BRAIN_PROVIDER + RIGHT_BRAIN_MODEL + <PROVIDER>_API_KEY), "
                    "or rerun with --critic-health-check off.".format(
                        provider=provider,
                        model=model,
                    )
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
                diagnostics_mode=args.diagnostics,
            )
            cases.append(case)
            diag = case.get("system2_diagnostic")
            print(
                "[bench] {idx:03d}/{total} id={id} mode={mode} enabled={enabled} "
                "issues={initial}->{final} rounds={rounds}/{target} resolved={resolved} "
                "diag={diag} error={error}".format(
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
                    diag=diag.get("status") if isinstance(diag, dict) else "off",
                    error=("yes" if case.get("error") else "no"),
                )
            )

        summary = _summarise_cases(cases)
        output = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question_set": question_set_label,
            "question_sets": [str(path) for path in question_paths],
            "config": {
                "session_id": session_id,
                "leading_brain": args.leading_brain,
                "system2_mode": args.system2_mode,
                "executive_mode": args.executive_mode,
                "executive_observer_mode": args.executive_observer_mode,
                "low_signal_filter": low_signal_filter,
                "diagnostics": str(args.diagnostics),
                "critic_health_check": critic_health_mode,
                "critic_health_attempts": int(args.critic_health_attempts),
                "critic_health_min_successes": (
                    int(args.critic_health_min_successes)
                    if args.critic_health_min_successes is not None
                    else None
                ),
                "critic_health_retries": int(args.critic_health_retries),
                "critic_health_timeout": float(args.critic_health_timeout),
                "critic_health_rate_limit_backoff": float(
                    args.critic_health_rate_limit_backoff
                ),
                "require_critic_health": bool(args.require_critic_health),
                "question_count": len(questions),
                "question_sets": [str(path) for path in question_paths],
                "llm_capable": llm_capable,
                "callosum_timeout_ms": int(getattr(session.controller, "default_timeout_ms", 0) or 0),
                "timeout_multiplier": _safe_float(os.environ.get("DUALBRAIN_TIMEOUT_MULTIPLIER")),
                "system2_timeout_multiplier": _safe_float(
                    os.environ.get("DUALBRAIN_SYSTEM2_TIMEOUT_MULTIPLIER")
                ),
                "timeout_max_ms": _safe_int(os.environ.get("DUALBRAIN_TIMEOUT_MAX_MS")),
                "system2_round_target_min": _safe_int(
                    os.environ.get("DUALBRAIN_SYSTEM2_ROUND_TARGET_MIN")
                ),
                "system2_stalled_followup": os.environ.get(
                    "DUALBRAIN_SYSTEM2_STALLED_FOLLOWUP",
                    "on",
                ),
                "system2_stalled_min_issues": _safe_int(
                    os.environ.get("DUALBRAIN_SYSTEM2_STALLED_MIN_ISSUES")
                ),
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
                "question_set": question_set_label,
                "summary": summary,
                "config": output["config"],
            }
            _append_history(history_path, history_row)
            history = _load_history(history_path, limit=max(1, int(args.history_limit)))
            history_trend = _history_trend(history, question_set=question_set_label)
            print(f"[bench] appended history: {history_path}")

        print(
            "[bench] summary measured={measured}/{ok} activation={activation} no_op={no_op} "
            "reduction_rate={reduction} resolved_rate={resolved} "
            "reduction_all={reduction_all} resolved_all={resolved_all} "
            "avg_rounds={rounds} avg_rounds_all={rounds_all} "
            "avg_latency_ms={latency} avg_latency_ms_all={latency_all}".format(
                measured=summary.get("measured_cases"),
                ok=summary.get("ok_cases"),
                activation=summary.get("system2_activation_rate"),
                no_op=summary.get("no_op_cases"),
                reduction=summary.get("issue_reduction_rate"),
                resolved=summary.get("resolved_issue_rate"),
                reduction_all=summary.get("mean_per_case_reduction_rate_all_cases"),
                resolved_all=summary.get("resolved_issue_share_all_cases"),
                rounds=summary.get("avg_rounds"),
                rounds_all=summary.get("avg_rounds_all_cases"),
                latency=summary.get("avg_latency_ms"),
                latency_all=summary.get("avg_latency_ms_all_cases"),
            )
        )
        if history_trend.get("runs", 0) > 0:
            print(
                "[bench] trend runs={runs}/{all_runs} mean_reduction_rate={rr} mean_resolved_rate={sr} "
                "mean_reduction_rate_all={rr_all} mean_resolved_share_all={sr_all}".format(
                    runs=history_trend.get("runs"),
                    all_runs=history_trend.get("all_runs"),
                    rr=history_trend.get("mean_issue_reduction_rate"),
                    sr=history_trend.get("mean_resolved_issue_rate"),
                    rr_all=history_trend.get("mean_issue_reduction_rate_all_cases"),
                    sr_all=history_trend.get("mean_resolved_issue_share_all_cases"),
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
        help="Comma separated paths to benchmark question set JSON files.",
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
        "--diagnostics",
        choices=["off", "unresolved", "all"],
        default="unresolved",
        help="Include System2 critic issue diagnostics in the report.",
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
        "--critic-health-min-successes",
        type=int,
        default=None,
        help="Minimum successful probes required; default is attempts-1.",
    )
    parser.add_argument(
        "--critic-health-retries",
        type=int,
        default=1,
        help="Retry count per health probe when critic output is unstable.",
    )
    parser.add_argument(
        "--critic-health-timeout",
        type=float,
        default=32.0,
        help="Timeout seconds used for critic health probes.",
    )
    parser.add_argument(
        "--critic-health-rate-limit-backoff",
        type=float,
        default=2.5,
        help="Extra backoff seconds applied on health-check rate-limit failures.",
    )
    parser.add_argument(
        "--require-critic-health",
        action="store_true",
        help="Abort benchmark when critic health check fails.",
    )
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Comma separated question ids to run (e.g., logic_001,code_review_001).",
    )
    parser.add_argument(
        "--only-tags",
        default=None,
        help="Comma separated tags; run questions that match any tag.",
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
