"""Research routing for typed drafts with independent field checks."""

from dataclasses import dataclass
import json

from critic_claim_gate import _decode, _equal, _scalar, decisions

CHECKED = {"accepted", "verify_disagrees"}


def answer_fields(text, names):
    data = _decode(text)
    if isinstance(data, dict) and set(data) == set(names) and all(_scalar(v) for v in data.values()):
        return data
    return None


def inspect_draft(question, draft, verifier_text):
    """Check all existing fields, including any omitted from the critic's issue list."""
    probes = [{"field": name, "found": value, "replacement": value}
              for name, value in draft.items()]
    return decisions(question, probes, verifier_text)


def usable_values(checks):
    return {row["field"]: row["verified_value"] for row in checks if row["status"] in CHECKED}


def supported_claims(claims, checks):
    values = usable_values(checks)
    return [claim for claim in claims if claim["field"] in values
            and _equal(values[claim["field"]], claim["replacement"])]


@dataclass(frozen=True)
class Route:
    reason: str
    changes: dict

    @property
    def retain(self):
        return self.reason == "all_fields_confirmed"


def select_route(draft, checks, approved):
    if draft is None:
        return Route("draft_unavailable_or_bad_shape", {})
    values = usable_values(checks)
    if len(checks) != len(draft) or set(values) != set(draft):
        return Route("verification_incomplete", {})
    changes = {field: value for field, value in values.items()
               if not _equal(value, draft[field])}
    if not changes:
        return Route("all_fields_confirmed", {})
    claimed = {claim["field"]: claim["replacement"] for claim in approved}
    if (set(changes) != set(claimed)
            or any(not _equal(value, claimed[field]) for field, value in changes.items())):
        return Route("uncovered_discrepancy", {})
    return Route("verified_claims_cover_all_changes", changes)


def select_answer(route, draft_stage, draft, regenerated):
    """Return the logical coordinated result; use the matched generation where needed."""
    if route.retain:
        return ("retain_confirmed", route.reason, draft_stage["output"],
                draft_stage["output_status"], [])
    if route.changes and regenerated["output_status"] == "complete":
        final = answer_fields(regenerated["output"], draft)
        if final is not None and all(_equal(final[key], value) for key, value in route.changes.items()):
            updated = {**draft, **route.changes}
            text = json.dumps(updated, ensure_ascii=False, separators=(",", ":"))
            return "commit_checked", route.reason, text, "complete", list(route.changes)
    reason = "integration_did_not_confirm_all_changes" if route.changes else route.reason
    return ("fallback_regenerate", reason, regenerated["output"],
            regenerated["output_status"], [])
