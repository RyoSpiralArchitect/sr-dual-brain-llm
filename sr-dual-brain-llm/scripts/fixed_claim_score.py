"""Grade a fixed mixed-critique cohort and the zero-pass retention policy."""
from collections import Counter
import json

from critic_e2e_score import GRADES, _stage_metrics, _stage_summary, grade


ARMS = ("unfiltered", "verified_regenerate", "retain_on_empty")
STAGES = ("draft", "verifier", *ARMS)
PHYSICAL_STAGES = ("verifier", "unfiltered", "verified_regenerate")
GROUPS = ("mixed", "correct_control")
CLAIM_KINDS = ("correct_correction", "damaging_proposal")
USABLE = {"accepted", "verify_disagrees"}
DECISION_STATUSES = USABLE | {
    "verifier_unavailable", "invalid_verifier_response", "ambiguous_verifier_field",
    "missing_verifier_field", "invalid_verifier_field", "invalid_evidence",
}


def _same(left, right):
    return type(left) is type(right) and left == right


def _grade_claims(case, decisions, complete):
    proposed = case["claims"]
    proposed_fields = {claim["field"] for claim in proposed}
    if not proposed or len(proposed_fields) != len(proposed):
        raise ValueError(f"Missing or duplicate fixture proposals in {case['id']}")
    by_field = {row["field"]: row for row in decisions}
    if len(by_field) != len(decisions):
        raise ValueError(f"Duplicate claim decision in {case['id']}")
    if not set(by_field) <= proposed_fields:
        raise ValueError(f"Unexpected claim decision in {case['id']}")
    if complete and len(by_field) != len(proposed):
        raise ValueError(f"Missing claim decisions in {case['id']}")
    result = []
    for claim in proposed:
        field = claim["field"]
        found = claim["found"]
        replacement = claim["replacement"]
        if not _same(found, case["draft"][field]) or _same(found, replacement):
            raise ValueError(f"Fixture proposal does not change the stated draft value: {case['id']}")
        expected = case["expected"][field]
        if _same(replacement, expected):
            kind = "correct_correction"
        elif _same(found, expected):
            kind = "damaging_proposal"
        else:
            raise ValueError(f"Fixture proposal is neither corrective nor damaging: {case['id']}")
        decision = by_field.get(field)
        status = "not_attempted"
        if decision is not None:
            if not all(_same(decision[key], claim[key]) for key in ("found", "replacement")):
                raise ValueError(f"Decision changes the proposal in {case['id']}")
            status = decision["status"]
            if status not in DECISION_STATUSES:
                raise ValueError(f"Invalid decision status in {case['id']}")
        independent_correct = (
            _same(decision["verified_value"], expected)
            if status in USABLE and "verified_value" in decision else None
        )
        result.append({"field": field, "kind": kind, "status": status,
                       "accepted": status == "accepted",
                       "independent_value_correct": independent_correct})
    return result


def _claim_counts(rows):
    counts = {}
    for kind in CLAIM_KINDS:
        claims = [claim for row in rows for claim in row["claims"] if claim["kind"] == kind]
        statuses = Counter(claim["status"] for claim in claims)
        accepted = statuses["accepted"]
        checked = accepted + statuses["verify_disagrees"]
        counts[kind] = {
            "opportunities": len(claims),
            "accepted": accepted,
            "blocked": len(claims) - accepted,
            "with_usable_derivation": checked,
            "known_independent_values": sum(claim["independent_value_correct"] is not None for claim in claims),
            "correct_independent_values": sum(claim["independent_value_correct"] is True for claim in claims),
            "rejected_by_disagreement": statuses["verify_disagrees"],
            "without_usable_derivation": len(claims) - checked,
            "statuses": dict(statuses),
        }
    return counts


def _pair(rows, first, second):
    summary = {"both_exact": 0, f"{first}_only_exact": 0, f"{second}_only_exact": 0, "neither_exact": 0}
    for row in rows:
        a = row[first]["grade"] == "exact"
        b = row[second]["grade"] == "exact"
        key = "both_exact" if a and b else (
            f"{first}_only_exact" if a else f"{second}_only_exact" if b else "neither_exact"
        )
        summary[key] += 1
    return summary


def _transition_summary(rows, arm):
    transitions = {grade_name: dict.fromkeys(GRADES, 0) for grade_name in GRADES}
    for row in rows:
        transitions[row["draft"]["grade"]][row[arm]["grade"]] += 1
    return {
        "final_exact": sum(t["exact"] for t in transitions.values()),
        "wrong_to_exact": transitions["semantic_wrong"]["exact"],
        "wrong_drafts": sum(transitions["semantic_wrong"].values()),
        "exact_to_wrong": transitions["exact"]["semantic_wrong"],
        "exact_to_nonexact": sum(transitions["exact"].values()) - transitions["exact"]["exact"],
        "exact_drafts": sum(transitions["exact"].values()),
        "transitions": transitions,
    }


def _zero_pass_summary(rows):
    regenerate = "verified_regenerate"
    retain = "retain_on_empty"
    by_draft = {}
    for draft_grade in ("exact", "semantic_wrong"):
        matched = [row for row in rows if row["draft"]["grade"] == draft_grade]
        by_draft[draft_grade] = {
            "trials": len(matched),
            "regenerate_exact": sum(row[regenerate]["grade"] == "exact" for row in matched),
            "regenerate_semantic_wrong": sum(row[regenerate]["grade"] == "semantic_wrong" for row in matched),
            "retain_exact": sum(row[retain]["grade"] == "exact" for row in matched),
            "paired_exact": _pair(matched, regenerate, retain),
        }
    saved = []
    for row in rows:
        if row["retained_draft_bytes"] and row[regenerate]["admitted"]:
            saved.append(row[regenerate])
    return {
        "trials": len(rows),
        "drafts_returned_verbatim": sum(row["retained_draft_bytes"] for row in rows),
        "verified_usable_zero_pass_trials": sum(row["all_derivations_usable"] for row in rows),
        "incomplete_derivation_zero_pass_trials": sum(not row["all_derivations_usable"] for row in rows),
        "regen_calls_avoided_under_retention": len(saved),
        "avoided_regeneration_usage": _stage_summary(saved)["usage"],
        "paired_exact": _pair(rows, regenerate, retain),
        "by_draft_grade": by_draft,
    }


def _summary(rows):
    zero = [row for row in rows if row["zero_pass"]]
    mixed = [row for row in rows if row["group"] == "mixed"]
    joint_success = sum(
        all(claim["accepted"] == (claim["kind"] == "correct_correction")
            for claim in row["claims"])
        for row in mixed
    )
    return {
        "trials": len(rows),
        "claim_counts": _claim_counts(rows),
        "mixed_trials": len(mixed),
        "mixed_trials_both_decisions_correct": joint_success,
        "draft_counts": {name: sum(row["draft"]["grade"] == name for row in rows) for name in GRADES},
        "accepted_claim_count_histogram": dict(sorted(Counter(row["accepted_claims"] for row in rows).items())),
        "arms": {arm: _transition_summary(rows, arm) for arm in ARMS},
        "paired_exact": {
            "unfiltered_vs_verified": _pair(rows, "unfiltered", "verified_regenerate"),
            "verified_vs_retain": _pair(rows, "verified_regenerate", "retain_on_empty"),
        },
        "zero_pass": _zero_pass_summary(zero),
    }


def _retention_invariants(stages, accepted, complete, case_id):
    retained = stages.get("retain_on_empty") or {}
    if retained.get("admission"):
        raise ValueError(f"Retention policy may not make an independent admission: {case_id}")
    if not retained or retained.get("terminal") == "not_attempted":
        if complete:
            raise ValueError(f"Retention policy outcome missing: {case_id}")
        return False
    original = stages["draft"] if not accepted else stages["verified_regenerate"]
    terminal = "draft_retained" if not accepted else "shared_regeneration"
    if (retained.get("terminal") != terminal
            or retained.get("output") != original.get("output")
            or retained.get("output_status") != original.get("output_status")):
        raise ValueError(f"Retention policy violated its recorded route: {case_id}")
    return not accepted


def score(cases, journal):
    """Produce metrics with no answers, proposal values, or derivation prose."""
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate fixture case ID")
    complete = journal["complete"]
    rows = []
    seen = set()
    for raw in journal["rows"]:
        case = by_id[raw["case_id"]]
        group = case["group"]
        if group not in GROUPS:
            raise ValueError(f"Unexpected fixture group {group}")
        rep = raw["rep"]
        key = (case["id"], rep)
        if key in seen or type(rep) is not int or rep not in (1, 2):
            raise ValueError(f"Duplicate or invalid repetition: {key}")
        seen.add(key)
        stages = raw["stages"]
        if complete and any(
                not isinstance(stages.get(name), dict) or not stages[name].get("terminal")
                for name in STAGES):
            raise ValueError(f"Missing terminal decision in complete run: {key}")
        draft = stages["draft"]
        if draft.get("admission") or draft["output_status"] != "complete":
            raise ValueError(f"Draft must be fixture data: {key}")
        row = {"case_id": case["id"], "rep": rep, "domain": case["domain"], "group": group}
        for name in STAGES:
            stage = stages.get(name) or {}
            row[name] = _stage_metrics(stage)
            if name != "verifier":
                row[name].update(grade(case["expected"], stage))
            if name in ARMS:
                output = stage.get("output")
                row[name]["text_changed_from_draft"] = (output != draft["output"]
                                                        if isinstance(output, str) else None)
        expected_grade = "semantic_wrong" if group == "mixed" else "exact"
        if row["draft"]["grade"] != expected_grade:
            raise ValueError(f"Unexpected fixture draft grade in {key}")
        supplied = json.loads(draft["output"])
        if set(supplied) != set(case["draft"]) or any(
                not _same(value, supplied[field]) for field, value in case["draft"].items()):
            raise ValueError(f"Recorded draft differs from frozen fixture in {key}")
        dispositions = raw.get("decisions", [])
        row["claims"] = _grade_claims(case, dispositions, complete)
        row["accepted_claims"] = sum(item["accepted"] for item in row["claims"])
        row["all_derivations_usable"] = bool(row["claims"]) and all(
            item["status"] in USABLE for item in row["claims"]
        )
        # Including no-usable-verifier outcomes is intentional for routing.
        row["zero_pass"] = len(dispositions) == len(row["claims"]) and not row["accepted_claims"]
        row["retained_draft_bytes"] = _retention_invariants(stages, row["accepted_claims"], complete, key)
        rows.append(row)
    planned = 2 * len(cases)
    if complete and len(rows) != planned:
        raise ValueError("Complete run does not contain both repetitions of every case")
    return {
        "schema": "fixed-mixed-claim-metrics-v1",
        "source": journal["source"],
        "fixture_sha256": journal["fixture_sha256"],
        "complete": complete,
        "planned_trials": planned,
        "recorded_trials": len(rows),
        "summary": _summary(rows),
        "by_group": {group: _summary([row for row in rows if row["group"] == group]) for group in GROUPS},
        "by_domain": {domain: _summary([row for row in rows if row["domain"] == domain])
                      for domain in sorted({case["domain"] for case in cases})},
        "stages": {stage: _stage_summary([row[stage] for row in rows]) for stage in STAGES},
        "all_calls": _stage_summary([row[stage] for row in rows for stage in PHYSICAL_STAGES]),
        "rows": rows,
    }
