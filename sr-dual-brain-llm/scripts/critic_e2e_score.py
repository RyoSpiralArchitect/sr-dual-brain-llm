"""Exact scoring for the paired Luna draft-to-final experiment."""

from collections import Counter
import json
from statistics import median


ARMS = ("without_critic", "with_critic")
STAGES = ("draft", "critic", *ARMS)
GRADES = ("exact", "semantic_wrong", "invalid_json", "invalid_shape", "unavailable")


def _object_without_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def _reject_constant(text):
    raise ValueError(f"Non-JSON constant: {text}")


def grade(expected, stage):
    """Score only a terminally complete delivered answer.

    Generation provenance and delivered-answer provenance can differ when the
    native integration retains a previously completed draft.
    """
    output = stage.get("output")
    if stage.get("output_status") != "complete" or not isinstance(output, str) or not output.strip():
        return {"grade": "unavailable", "matched_fields": None, "wrong_fields": []}
    try:
        parsed = json.loads(
            output, object_pairs_hook=_object_without_duplicate_keys, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError):
        return {"grade": "invalid_json", "matched_fields": None, "wrong_fields": []}
    if (not isinstance(parsed, dict) or parsed.keys() != expected.keys()
            or any(type(value) not in (str, int, type(None)) for value in parsed.values())):
        return {"grade": "invalid_shape", "matched_fields": None, "wrong_fields": []}
    wrong_fields = sorted(key for key, value in expected.items()
                          if type(parsed[key]) is not type(value) or parsed[key] != value)
    return {"grade": "semantic_wrong" if wrong_fields else "exact",
            "matched_fields": len(expected) - len(wrong_fields), "wrong_fields": wrong_fields}


def _nonnegative_int(raw):
    return raw if type(raw) is int and raw >= 0 else None


def _stage_metrics(stage):
    admission = stage.get("admission") or {}
    response = admission.get("response")
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "terminal": stage.get("terminal", "not_attempted"),
        "generation_status": stage.get("generation_status"),
        "output_status": stage.get("output_status"),
        "fatal": stage.get("fatal"),
        "latency_ms": stage.get("latency_ms"),
        "admitted": bool(admission),
        "transport": admission.get("transport"),
        "http_status": admission.get("http_status"),
        "response_model": response.get("model"),
        "input_tokens": _nonnegative_int(usage.get("prompt_tokens")),
        "output_tokens": _nonnegative_int(usage.get("completion_tokens")),
        "reasoning_tokens": _nonnegative_int(details.get("reasoning_tokens")),
    }


def _counts(values, labels):
    counts = Counter(values)
    return {key: counts[key] for key in labels}


def _stage_summary(stages):
    admitted = [row for row in stages if row["admitted"]]
    latencies = [row["latency_ms"] for row in stages if row["latency_ms"] is not None]
    usage = {}
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        known = [row[key] for row in admitted if row[key] is not None]
        usage[key] = {
            "known_total": sum(known),
            "missing_admissions": len(admitted) - len(known),
        }
    return {
        "slots": len(stages),
        "admissions": len(admitted),
        "terminal_counts": dict(Counter(row["terminal"] for row in stages)),
        "generation_status_counts": dict(Counter(
            row["generation_status"] or "not_recorded" for row in stages
        )),
        "observed_latency_count": len(latencies),
        "median_latency_ms": round(median(latencies), 2) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "usage": usage,
    }


def _outcomes(rows):
    draft_counts = _counts((row["draft"]["grade"] for row in rows), GRADES)
    arms = {}
    for arm in ARMS:
        transitions = {
            before: dict.fromkeys(GRADES, 0) for before in GRADES
        }
        for row in rows:
            transitions[row["draft"]["grade"]][row[arm]["grade"]] += 1
        by_final = _counts((row[arm]["grade"] for row in rows), GRADES)
        correct_to_nonexact = {
            category: transitions["exact"][category] for category in GRADES if category != "exact"
        }
        arms[arm] = {
            "trials": len(rows),
            "final_counts": by_final,
            "final_exact": by_final["exact"],
            "wrong_to_exact": transitions["semantic_wrong"]["exact"],
            "wrong_drafts": draft_counts["semantic_wrong"],
            "exact_to_wrong": transitions["exact"]["semantic_wrong"],
            "exact_to_nonexact": sum(correct_to_nonexact.values()),
            "exact_to_nonexact_by_grade": correct_to_nonexact,
            "exact_drafts": draft_counts["exact"],
            "invalid_to_exact": sum(transitions[category]["exact"] for category in ("invalid_json", "invalid_shape")),
            "invalid_drafts": draft_counts["invalid_json"] + draft_counts["invalid_shape"],
            "unavailable_to_exact": transitions["unavailable"]["exact"],
            "unavailable_drafts": draft_counts["unavailable"],
            "transitions": transitions,
        }

    def paired_result(row):
        without = row["without_critic"]["grade"] == "exact"
        with_ = row["with_critic"]["grade"] == "exact"
        if with_ and without:
            return "both_exact"
        if with_:
            return "with_only_exact"
        if without:
            return "without_only_exact"
        return "neither_exact"

    pair_labels = ("both_exact", "with_only_exact", "without_only_exact", "neither_exact")
    return {
        "trials": len(rows),
        "draft_counts": draft_counts,
        "arms": arms,
        "paired_exact": _counts((paired_result(row) for row in rows), pair_labels),
        "paired_exact_by_draft_grade": {
            category: _counts((paired_result(row) for row in rows if row["draft"]["grade"] == category),
                              pair_labels)
            for category in GRADES
        },
        "valid_critic_reviews": sum(row["critic"]["valid"] for row in rows),
        "review_delivered": sum(row["critic"]["delivered"] for row in rows),
    }


def score(cases, journal):
    """Regenerate public metrics from recorded outputs and terminal decisions."""
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate fixture case ID")
    rows = []
    seen = set()
    for raw in journal["rows"]:
        case = by_id[raw["case_id"]]
        rep = raw["rep"]
        key = (case["id"], rep)
        if key in seen or type(rep) is not int or rep not in (1, 2):
            raise ValueError(f"Duplicate or invalid case repetition: {key}")
        seen.add(key)
        stages = raw["stages"]
        if journal["complete"] and any(
            not isinstance(stages.get(name), dict) or not stages[name].get("terminal")
            for name in STAGES
        ):
            raise ValueError(f"A complete journal must account for every stage outcome: {key}")
        row = {"case_id": case["id"], "domain": case["domain"], "rep": rep,
               "expected_fields": len(case["expected"])}
        for name in ("draft", *ARMS):
            stage = stages.get(name) or {}
            row[name] = {**_stage_metrics(stage), **grade(case["expected"], stage)}
            if name in ARMS:
                original = (stages.get("draft") or {}).get("output")
                final = stage.get("output")
                row[name]["text_changed_from_draft"] = (
                    final != original if isinstance(original, str) and isinstance(final, str) else None
                )
        critic_stage = stages.get("critic") or {}
        critic = critic_stage.get("output")
        critic = critic if isinstance(critic, dict) else {}
        valid = (critic_stage.get("terminal") == "returned"
                 and critic.get("critic_external_review_valid") is True
                 and critic.get("verdict") in ("ok", "issues"))
        delivered = raw.get("review_delivered") is True
        if delivered and (not valid or critic["verdict"] != "issues"):
            raise ValueError(f"Review was delivered without valid external provenance: {key}")
        issues = critic.get("issues")
        row["critic"] = {
            **_stage_metrics(critic_stage),
            "valid": valid,
            "delivered": delivered,
            "status": critic.get("critic_status"),
            "verdict": critic.get("verdict") if valid else None,
            "issue_count": len(issues) if valid and isinstance(issues, list) else None,
        }
        rows.append(row)
    planned = 2 * len(cases)
    if journal["complete"] and len(rows) != planned:
        raise ValueError("A complete journal must account for every planned trial")
    return {
        "schema": "critic-e2e-metrics-v1",
        "source": journal["source"],
        "fixture_sha256": journal["fixture_sha256"],
        "complete": journal["complete"],
        "planned_trials": planned,
        "planned_admissions": planned * len(STAGES),
        "recorded_trials": len(rows),
        "summary": _outcomes(rows),
        "by_domain": {domain: _outcomes([row for row in rows if row["domain"] == domain])
                      for domain in sorted({case["domain"] for case in cases})},
        "stages": {name: _stage_summary([row[name] for row in rows]) for name in STAGES},
        "all_calls": _stage_summary([row[name] for row in rows for name in STAGES]),
        "rows": rows,
    }
