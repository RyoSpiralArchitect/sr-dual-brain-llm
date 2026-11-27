"""Score the matched three-arm critic verification study offline."""

from collections import Counter

from critic_e2e_score import GRADES, _stage_metrics, _stage_summary, grade


ARMS = ("without_critic", "standard_critic", "verified_critic")
STAGES = ("draft", "critic", "verifier", *ARMS)
CLAIM_COUNTS = (
    "proposed", "accepted", "rejected", "unparseable", "correct_proposed",
    "correct_accepted", "correct_rejected", "incorrect_proposed",
    "incorrect_accepted", "incorrect_rejected", "damaging_proposed",
    "damaging_accepted", "damaging_rejected", "ungradeable",
)


def _same(a, b):
    return type(a) is type(b) and a == b


def _count_grades(values):
    counts = Counter(values)
    return {key: counts[key] for key in GRADES}


def _grade_claims(raw, expected):
    claims = raw.get("claims", [])
    decisions = raw.get("decisions", [])
    by_field = {item["field"]: item for item in decisions}
    if len(by_field) != len(decisions) or len({claim["field"] for claim in claims}) != len(claims):
        raise ValueError("Duplicate field in parsed claims or verification decisions")
    if set(by_field) != {claim["field"] for claim in claims}:
        raise ValueError("Each parsed claim needs one verification disposition")
    audits = []
    for claim in claims:
        field = claim["field"]
        decision = by_field[field]
        if not all(_same(claim[key], decision[key]) for key in ("found", "replacement")):
            raise ValueError("Verification disposition refers to a different proposal")
        correct = _same(claim["replacement"], expected[field]) if field in expected else None
        damages = (
            _same(claim["found"], expected[field]) and not correct if field in expected else None
        )
        audits.append({
            "field": field if field in expected else None,
            "accepted": decision["status"] == "accepted",
            "correct_replacement": correct,
            "would_damage_correct_field": damages,
        })
    return audits


def _claim_summary(rows):
    counts = dict.fromkeys(CLAIM_COUNTS, 0)
    repair_opportunities = 0
    drafts_with_correct_proposal = set()
    drafts_with_correct_acceptance = set()
    for row in rows:
        wrong_fields = set(row["draft"]["wrong_fields"])
        repair_opportunities += len(wrong_fields)
        pair = (row["case_id"], row["rep"])
        counts["unparseable"] += row["parse_rejection_count"]
        for proposal in row["claims"]:
            status = "accepted" if proposal["accepted"] else "rejected"
            counts["proposed"] += 1
            counts[status] += 1
            correctness = proposal["correct_replacement"]
            if correctness is None:
                counts["ungradeable"] += 1
                continue
            prefix = "correct" if correctness else "incorrect"
            counts[prefix + "_proposed"] += 1
            counts[prefix + "_" + status] += 1
            if proposal["would_damage_correct_field"]:
                counts["damaging_proposed"] += 1
                counts["damaging_" + status] += 1
            if correctness and proposal["field"] in wrong_fields:
                drafts_with_correct_proposal.add((pair, proposal["field"]))
                if proposal["accepted"]:
                    drafts_with_correct_acceptance.add((pair, proposal["field"]))
    counts.update(
        wrong_fields_available=repair_opportunities,
        wrong_fields_with_correct_proposal=len(drafts_with_correct_proposal),
        wrong_fields_with_correct_accepted_proposal=len(drafts_with_correct_acceptance),
    )
    return counts


def _paired(rows, first, second):
    counts = dict.fromkeys(("both_exact", f"{first}_only_exact", f"{second}_only_exact", "neither_exact"), 0)
    for row in rows:
        first_exact = row[first]["grade"] == "exact"
        second_exact = row[second]["grade"] == "exact"
        if first_exact and second_exact:
            name = "both_exact"
        elif first_exact:
            name = f"{first}_only_exact"
        elif second_exact:
            name = f"{second}_only_exact"
        else:
            name = "neither_exact"
        counts[name] += 1
    return counts


def _outcomes(rows):
    draft_counts = _count_grades(row["draft"]["grade"] for row in rows)
    arms = {}
    for arm in ARMS:
        transitions = {before: dict.fromkeys(GRADES, 0) for before in GRADES}
        for row in rows:
            transitions[row["draft"]["grade"]][row[arm]["grade"]] += 1
        final_counts = _count_grades(row[arm]["grade"] for row in rows)
        exact_losses = {key: transitions["exact"][key] for key in GRADES if key != "exact"}
        arms[arm] = {
            "trials": len(rows),
            "final_counts": final_counts,
            "final_exact": final_counts["exact"],
            "wrong_to_exact": transitions["semantic_wrong"]["exact"],
            "wrong_drafts": draft_counts["semantic_wrong"],
            "exact_to_wrong": exact_losses["semantic_wrong"],
            "exact_to_nonexact": sum(exact_losses.values()),
            "exact_to_nonexact_by_grade": exact_losses,
            "exact_drafts": draft_counts["exact"],
            "transitions": transitions,
        }

    standard_verified = ("standard_critic", "verified_critic")
    changed = [row for row in rows if row["notes_differ"]]
    unchanged = [row for row in rows if not row["notes_differ"]]
    return {
        "trials": len(rows),
        "draft_counts": draft_counts,
        "arms": arms,
        "paired_exact": {
            "standard_vs_verified": _paired(rows, *standard_verified),
            "without_vs_standard": _paired(rows, "without_critic", "standard_critic"),
            "without_vs_verified": _paired(rows, "without_critic", "verified_critic"),
        },
        "standard_vs_verified_by_draft_grade": {
            category: _paired(
                (row for row in rows if row["draft"]["grade"] == category), *standard_verified
            )
            for category in GRADES
        },
        "changed_notes": {
            "trials": len(changed),
            "paired_exact": _paired(changed, *standard_verified),
        },
        "identical_notes": {
            "trials": len(unchanged),
            "paired_exact": _paired(unchanged, *standard_verified),
        },
        "valid_critic_reviews": sum(row["critic"]["valid"] for row in rows),
        "trials_with_proposals": sum(bool(row["claims"]) for row in rows),
        "trials_with_accepted_proposals": sum(
            any(claim["accepted"] for claim in row["claims"]) for row in rows
        ),
        "claim_counts": _claim_summary(rows),
    }


def _success_gate(summary, complete):
    standard = summary["arms"]["standard_critic"]
    verified = summary["arms"]["verified_critic"]
    estimable = complete and bool(standard["wrong_drafts"] and standard["exact_drafts"])
    return {
        "estimable": estimable,
        "passes": (
            verified["wrong_to_exact"] >= standard["wrong_to_exact"]
            and verified["exact_to_nonexact"] < standard["exact_to_nonexact"]
            if estimable else None
        ),
        "repairs_delta": verified["wrong_to_exact"] - standard["wrong_to_exact"],
        "exact_regressions_delta": verified["exact_to_nonexact"] - standard["exact_to_nonexact"],
    }


def score(cases, journal):
    """Generate public metrics without raw model text or claim values."""
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate fixture case ID")
    complete = journal["complete"]
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
        if complete and any(
            not isinstance(stages.get(stage), dict) or not stages[stage].get("terminal")
            for stage in STAGES
        ):
            raise ValueError(f"Complete journal must account for every stage slot: {key}")
        order = raw.get("arm_order", [])
        if sorted(order) != sorted(ARMS):
            raise ValueError(f"Invalid final arm order: {key}")
        row = {
            "case_id": case["id"], "rep": rep, "domain": case["domain"], "arm_order": order,
            "expected_fields": len(case["expected"]),
        }
        for name in STAGES:
            stage = stages.get(name) or {}
            row[name] = _stage_metrics(stage)
            if name in ("draft", *ARMS):
                row[name].update(grade(case["expected"], stage))
        critic_stage = stages.get("critic") or {}
        review = critic_stage.get("output") or {}
        review = review if isinstance(review, dict) else {}
        verdict = review.get("verdict")
        valid = (critic_stage.get("terminal") == "returned"
                 and review.get("critic_external_review_valid") is True
                 and verdict in ("ok", "issues"))
        issues = review.get("issues")
        row["critic"].update(valid=valid, verdict=verdict if valid else None,
                             issue_count=len(issues) if valid and isinstance(issues, list) else None)
        row["claims"] = _grade_claims(raw, case["expected"])
        if row["claims"] and (not valid or verdict != "issues"):
            raise ValueError(f"Claims without valid external critique: {key}")
        row["parse_rejection_count"] = len(raw.get("parse_rejections", []))
        notes_differ = raw.get("notes_differ")
        row["notes_differ"] = (
            notes_differ if type(notes_differ) is bool
            else any(not claim["accepted"] for claim in row["claims"])
        )
        rows.append(row)

    planned = 2 * len(cases)
    if complete and len(rows) != planned:
        raise ValueError("A complete journal must account for every planned pair")
    summary = _outcomes(rows)
    return {
        "schema": "critic-verified-metrics-v1",
        "source": journal["source"],
        "fixture_sha256": journal["fixture_sha256"],
        "complete": complete,
        "planned_trials": planned,
        "recorded_trials": len(rows),
        "maximum_admissions": planned * len(STAGES),
        "summary": summary,
        "success_gate": _success_gate(summary, complete),
        "by_domain": {domain: _outcomes([row for row in rows if row["domain"] == domain])
                      for domain in sorted({case["domain"] for case in cases})},
        "stages": {name: _stage_summary([row[name] for row in rows]) for name in STAGES},
        "all_calls": _stage_summary([row[name] for row in rows for name in STAGES]),
        "rows": rows,
    }
