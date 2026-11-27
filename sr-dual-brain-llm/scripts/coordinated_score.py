"""Score the fixed or natural coordinated critic cohort from private receipts."""

from collections import Counter
import json

from critic_e2e_score import (
    GRADES, _object_without_duplicate_keys, _reject_constant, _stage_metrics, _stage_summary, grade,
)


ARMS = ("unfiltered", "verified_regenerate", "coordinated")
STAGES = ("draft", "critic", "verifier", *ARMS)
PHYSICAL_STAGES = tuple(name for name in STAGES if name != "coordinated")
ROUTES = ("retain_confirmed", "commit_checked", "fallback_regenerate")


def same(left, right):
    return type(left) is type(right) and left == right


def _answer(stage):
    if stage.get("output_status") != "complete" or not isinstance(stage.get("output"), str):
        return None
    try:
        output = json.loads(stage["output"], object_pairs_hook=_object_without_duplicate_keys,
                            parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    if isinstance(output, dict) and all(type(value) in (int, str, type(None)) for value in output.values()):
        return output
    return None


def _fields(values, label):
    if not isinstance(values, list) or not all(isinstance(field, str) and field for field in values):
        raise ValueError(f"Invalid {label} list")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")
    return set(values)


def _field_audit(raw, expected, original):
    claims = raw.get("claims", [])
    claim_fields = _fields([claim["field"] for claim in claims], "claim fields")
    approved = _fields(raw.get("approved_fields", []), "approved fields")
    applied = _fields(raw.get("applied_fields", []), "applied fields")
    if not applied <= approved <= claim_fields:
        raise ValueError("Applied and approved fields must refer to known claims")
    checks = raw.get("checks", [])
    _fields([check["field"] for check in checks], "checked fields")
    audited_claims = []
    for claim in claims:
        field = claim["field"]
        if original is not None and (field not in original or not same(claim["found"], original[field])):
            raise ValueError(f"Claim does not bind the recorded draft field: {field}")
        eligible = field in expected
        good = same(claim["replacement"], expected[field]) if eligible else None
        damaging = same(claim["found"], expected[field]) and not good if eligible else None
        audited_claims.append({
            "field": field if eligible else None,
            "correct_replacement": good,
            "would_damage_correct_field": damaging,
            "approved": field in approved,
            "applied": field in applied,
        })
    audited_checks = []
    for item in checks:
        field = item["field"]
        status = item["status"]
        usable = status in ("accepted", "verify_disagrees") and "verified_value" in item
        value = item.get("verified_value")
        audited_checks.append({
            "field": field if field in expected else None, "status": status, "usable": usable,
            "oracle_agreement": same(value, expected[field]) if usable and field in expected else None,
            "draft_agreement": same(value, original[field]) if usable and original is not None
                              and field in original else None,
        })
    return audited_claims, audited_checks, applied


def _route_invariants(stages, raw, applied, complete, key):
    draft = stages.get("draft") or {}
    reference = stages.get("verified_regenerate") or {}
    candidate = stages.get("coordinated") or {}
    route = candidate.get("terminal", "not_attempted")
    if candidate.get("admission"):
        raise ValueError(f"Coordination cannot make a separate provider call: {key}")
    if route not in ROUTES:
        if complete or route != "not_attempted":
            raise ValueError(f"Missing or invalid coordination route: {key}")
        return False
    if not isinstance(raw.get("route_reason"), str) or not raw["route_reason"]:
        raise ValueError(f"Missing route reason: {key}")
    if route in ("retain_confirmed", "fallback_regenerate"):
        origin = draft if route == "retain_confirmed" else reference
        if (applied or origin.get("output") != candidate.get("output")
                or origin.get("output_status") != candidate.get("output_status")):
            raise ValueError(f"Output differs from recorded {route} route: {key}")
    elif route == "commit_checked":
        initial = _answer(draft)
        final = _answer(reference)
        committed = _answer(candidate)
        if initial is None or committed is None or initial.keys() != committed.keys():
            raise ValueError(f"Commit does not preserve the complete draft shape: {key}")
        actual_edits = {field for field in initial if not same(initial[field], committed[field])}
        replacements = {claim["field"]: claim["replacement"] for claim in raw.get("claims", [])}
        if (applied != actual_edits or applied and (final is None or not all(
            field in final and same(committed[field], final[field])
            and same(committed[field], replacements[field]) for field in applied
        ))):
            raise ValueError(f"Commit contains unauthorized edits: {key}")
        if not applied and candidate.get("output") != draft.get("output"):
            raise ValueError(f"Commit changed draft bytes without applying any edits: {key}")
    return True


def _pair(rows, candidate, baseline):
    counts = dict.fromkeys(("both_exact", "candidate_only_exact", "baseline_only_exact", "neither_exact"), 0)
    for row in rows:
        yes = row[candidate]["grade"] == "exact"
        ref = row[baseline]["grade"] == "exact"
        if yes and ref:
            counts["both_exact"] += 1
        elif yes:
            counts["candidate_only_exact"] += 1
        elif ref:
            counts["baseline_only_exact"] += 1
        else:
            counts["neither_exact"] += 1
    return counts


def _transitions(rows, arm):
    transitions = {state: dict.fromkeys(GRADES, 0) for state in GRADES}
    original_fields_fixed = original_correct_fields_lost = 0
    for row in rows:
        before, after = row["draft"], row[arm]
        transitions[before["grade"]][after["grade"]] += 1
        if before["matched_fields"] is not None and after["matched_fields"] is not None:
            earlier, later = set(before["wrong_fields"]), set(after["wrong_fields"])
            original_fields_fixed += len(earlier - later)
            original_correct_fields_lost += len(later - earlier)
    return {
        "final_exact": sum(ends["exact"] for ends in transitions.values()),
        "wrong_to_exact": transitions["semantic_wrong"]["exact"],
        "wrong_drafts": sum(transitions["semantic_wrong"].values()),
        "exact_to_wrong": transitions["exact"]["semantic_wrong"],
        "exact_to_nonexact": sum(transitions["exact"].values()) - transitions["exact"]["exact"],
        "exact_drafts": sum(transitions["exact"].values()),
        "original_fields_fixed": original_fields_fixed,
        "original_correct_fields_lost": original_correct_fields_lost,
        "transitions": transitions,
    }


def _proposal_counts(rows):
    claims = [claim for row in rows for claim in row["claims"]]
    checks = [check for row in rows for check in row["checks"]]
    wrong_targets = [(row, field) for row in rows for field in row["draft"]["wrong_fields"]]
    known = [claim for claim in claims if claim["correct_replacement"] is not None]
    good = [claim for claim in known if claim["correct_replacement"]]
    bad = [claim for claim in known if not claim["correct_replacement"]]
    harmful = [claim for claim in bad if claim["would_damage_correct_field"]]
    return {
        "proposed": len(claims), "ungradeable": len(claims) - len(known),
        "correct_proposed": len(good), "incorrect_proposed": len(bad),
        "would_damage_correct_field": len(harmful),
        "correct_approved": sum(claim["approved"] for claim in good),
        "incorrect_approved": sum(claim["approved"] for claim in bad),
        "damaging_approved": sum(claim["approved"] for claim in harmful),
        "correct_applied": sum(claim["applied"] for claim in good),
        "incorrect_applied": sum(claim["applied"] for claim in bad),
        "damaging_applied": sum(claim["applied"] for claim in harmful),
        "draft_wrong_fields": len(wrong_targets),
        "draft_wrong_fields_targeted_by_correct_proposal": sum(
            any(c["field"] == field and c["correct_replacement"] for c in row["claims"])
            for row, field in wrong_targets
        ),
        "field_checks": len(checks), "usable_checks": sum(check["usable"] for check in checks),
        "oracle_correct_checks": sum(check["oracle_agreement"] is True for check in checks),
        "oracle_wrong_checks": sum(check["oracle_agreement"] is False for check in checks),
        "check_statuses": dict(Counter(check["status"] for check in checks)),
    }


def _summary(rows):
    saved = [row["verified_regenerate"] for row in rows
             if row["coordinated"]["terminal"] == "retain_confirmed" and row["verified_regenerate"]["admitted"]]
    reference_calls = sum(row["verified_regenerate"]["admitted"] for row in rows)
    routes = {}
    for route in ROUTES:
        routed = [row for row in rows if row["coordinated"]["terminal"] == route]
        routes[route] = {
            "rows": len(routed),
            "initial_exact": sum(row["draft"]["grade"] == "exact" for row in routed),
            "paired_verified": _pair(routed, "coordinated", "verified_regenerate"),
        }
    return {
        "trials": len(rows),
        "draft_counts": dict(Counter(row["draft"]["grade"] for row in rows)),
        "arms": {arm: _transitions(rows, arm) for arm in ARMS},
        "paired_exact": {
            baseline: _pair(rows, "coordinated", baseline)
            for baseline in ("unfiltered", "verified_regenerate")
        },
        "routes": routes,
        "route_reasons": dict(Counter(row["route_reason"] for row in rows if row["route_reason"] is not None)),
        "claim_counts": _proposal_counts(rows),
        "final_calls": {
            "unfiltered": sum(row["unfiltered"]["admitted"] for row in rows),
            "verified_regenerate": reference_calls,
            "coordinated_logical": reference_calls - len(saved),
            "avoided": len(saved),
            "observed_usage_of_avoided_calls": _stage_summary(saved)["usage"],
        },
    }


def _qualification(summary, complete):
    candidate = summary["arms"]["coordinated"]
    baseline = summary["arms"]["verified_regenerate"]
    repaired_delta = candidate["wrong_to_exact"] - baseline["wrong_to_exact"]
    losses_delta = candidate["exact_to_nonexact"] - baseline["exact_to_nonexact"]
    final_exact_delta = candidate["final_exact"] - baseline["final_exact"]
    matched_losses = summary["paired_exact"]["verified_regenerate"]["baseline_only_exact"]
    avoided = summary["final_calls"]["avoided"]
    estimable = complete and bool(baseline["wrong_drafts"] and baseline["exact_drafts"])
    preserves = repaired_delta >= 0 and losses_delta <= 0
    return {
        "estimable": estimable,
        "quality_advantage": (final_exact_delta > 0 and preserves) if estimable else None,
        "efficiency_advantage": (
            final_exact_delta == 0 and matched_losses == 0 and preserves and avoided > 0
        ) if estimable else None,
        "repairs_delta": repaired_delta,
        "regressions_delta": losses_delta,
        "final_exact_delta": final_exact_delta,
        "matched_exact_losses": matched_losses,
        "avoided_calls": avoided,
    }


def score(cases, journal):
    """Publish grades, field and route outcomes, and provider activity without text or values."""
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate oracle case ID")
    cohort = journal["cohort"]
    if cohort not in ("fixed", "natural"):
        raise ValueError(f"Invalid cohort: {cohort}")
    complete = journal["complete"]
    rows, seen = [], set()
    for raw in journal["rows"]:
        case = by_id[raw["case_id"]]
        rep = raw["rep"]
        key = (case["id"], rep)
        if key in seen or type(rep) is not int or rep not in (1, 2):
            raise ValueError(f"Duplicate or invalid repetition: {key}")
        seen.add(key)
        stages = raw["stages"]
        if complete and any(not (stages.get(name) or {}).get("terminal") for name in STAGES):
            raise ValueError(f"Incomplete stage accounting: {key}")
        expected = case["expected"]
        claims, checks, applied = _field_audit(raw, expected, _answer(stages.get("draft") or {}))
        _route_invariants(stages, raw, applied, complete, key)
        row = {
            "case_id": case["id"], "rep": rep, "domain": case["domain"],
            "route_reason": raw.get("route_reason"), "claims": claims, "checks": checks,
            "arm_order": raw.get("arm_order"),
        }
        if "group" in case:
            row["group"] = case["group"]
        for name in STAGES:
            item = stages.get(name) or {}
            row[name] = _stage_metrics(item)
            if name in (*ARMS, "draft"):
                row[name].update(grade(expected, item))
        rows.append(row)
    planned = 2 * len(cases)
    if complete and len(rows) != planned:
        raise ValueError("Complete cohort must contain both repetitions of every frozen case")
    summary = _summary(rows)
    domains = sorted({case["domain"] for case in cases})
    groups = sorted({case["group"] for case in cases if "group" in case})
    return {
        "schema": "coordinated-critic-metrics-v1", "cohort": cohort,
        "source": journal["source"], "fixture_sha256": journal["fixture_sha256"],
        "complete": complete, "planned_trials": planned, "recorded_trials": len(rows),
        "summary": summary, "qualification": _qualification(summary, complete),
        "by_domain": {domain: _summary([row for row in rows if row["domain"] == domain])
                      for domain in domains},
        "by_group": {group: _summary([row for row in rows if row.get("group") == group]) for group in groups},
        "stages": {name: _stage_summary([row[name] for row in rows]) for name in STAGES},
        "all_calls": _stage_summary([row[name] for row in rows for name in PHYSICAL_STAGES]),
        "rows": rows,
    }
