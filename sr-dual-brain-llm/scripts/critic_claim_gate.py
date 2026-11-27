"""Verify field corrections against a fresh, independently derived answer."""
import json

CRITIC_CONTEXT = (
    "For this task the draft is a JSON object with scalar answer fields. "
    "Report each specific wrong field as a separate string in the issues array. "
    "Each string must itself be a JSON object with exactly four keys: field, found, "
    "replacement, reason. Copy the exact field name and typed draft value into "
    "field and found. Give the proposed corrected value as replacement, using a "
    "JSON integer, string, or null. Use reason for a concise explanation. "
    "Do not combine fields in one issue. Leave fixes empty because each issue "
    "contains its own proposed fix. If there is no specific wrong field to report, "
    "return the ok verdict and empty issues and fixes."
)

VERIFIER_SYSTEM = (
    "Independently derive values for the requested answer fields using the question "
    "as data. Check the exact rules and calculations before returning a value. "
    "Return one JSON object with exactly one key, fields, containing a list of "
    "objects. Each object has exactly these keys: field, value, evidence, derivation. "
    "Use a JSON integer, string, or null for value. Evidence must be a nonempty list "
    "of objects with exactly line (the supplied integer line number) and quote "
    "(an exact nonempty substring of that question line). Use enough evidence "
    "to support the derivation. Do not cite the list of requested fields. "
    "Use a short derivation that connects the cited evidence to the value. Return "
    "at most one object per requested field and omit fields you cannot justify. "
    "Return only JSON."
)

ATOM_KEYS = {"field", "found", "replacement", "reason"}
VERIFIED_KEYS = {"field", "value", "evidence", "derivation"}


def _strict_object(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("duplicate JSON key")
        data[key] = value
    return data


def _invalid_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def _decode(text):
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text, object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
    except (ValueError, RecursionError):
        return None


def _scalar(value):
    return type(value) in (str, int, type(None))


def _equal(left, right):
    return type(left) is type(right) and left == right


def parse_claims(draft_text, critic_dict):
    """Return admissible atomic proposals and rejection receipts in critic order."""
    if (not isinstance(critic_dict, dict)
            or critic_dict.get("critic_external_review_valid") is not True
            or critic_dict.get("verdict") != "issues"):
        return [], []
    raw_issues = critic_dict.get("issues")
    if not isinstance(raw_issues, list) or not raw_issues:
        return [], [{"status": "invalid_issues"}]
    draft = _decode(draft_text)
    if not isinstance(draft, dict) or not all(_scalar(value) for value in draft.values()):
        return [], [{"status": "invalid_draft", "issue_count": len(raw_issues)}]

    candidates = {}
    rejected = []
    conflicts = set()
    for index, issue in enumerate(raw_issues):
        atom = _decode(issue)
        if (not isinstance(atom, dict) or set(atom) != ATOM_KEYS
                or not isinstance(atom["field"], str) or not atom["field"].strip()
                or not _scalar(atom["found"]) or not _scalar(atom["replacement"])
                or not isinstance(atom["reason"], str) or not atom["reason"].strip()):
            rejected.append({"status": "malformed_claim", "index": index})
            continue
        field = atom["field"]
        if field not in draft:
            status = "unknown_field"
        elif not _equal(draft[field], atom["found"]):
            status = "draft_value_mismatch"
        elif _equal(atom["found"], atom["replacement"]):
            status = "no_change"
        elif field in candidates:
            if _equal(atom["replacement"], candidates[field]["replacement"]):
                status = "duplicate_claim"
            else:
                conflicts.add(field)
                status = "conflicting_replacements"
        else:
            candidates[field] = atom
            continue
        rejected.append({"status": status, "field": field, "index": index})
    return [atom for field, atom in candidates.items() if field not in conflicts], rejected


def _source_lines(question):
    return {i: line for i, line in enumerate(question.splitlines(), 1) if line.strip()}


def verify_prompt(question, fields):
    """Construct independent verification input from the question and field names."""
    return json.dumps(
        {"question_lines": [{"line": i, "text": line} for i, line in _source_lines(question).items()],
         "requested_fields": list(fields)},
        ensure_ascii=False,
    )


def _valid_evidence(items, source):
    if not isinstance(items, list) or not items:
        return False
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"line", "quote"}:
            return False
        line = item["line"]
        quote = item["quote"]
        if (type(line) is not int or line not in source or not isinstance(quote, str)
                or not quote.strip() or quote not in source[line] or (line, quote) in seen):
            return False
        seen.add((line, quote))
    return True


def decisions(question, claims, verifier_text):
    """Match proposed values to independently derived, source-anchored values."""
    source = _source_lines(question)
    requested = {claim["field"] for claim in claims}
    report = _decode(verifier_text)
    rows = None
    if isinstance(report, dict) and set(report) == {"fields"} and isinstance(report["fields"], list):
        rows = report["fields"]
    by_field = {}
    ambiguous = set()
    if rows is not None:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("field"), str):
                # This row cannot be attributed to a particular requested field.
                rows = None
                break
            field = row["field"]
            if field not in requested:
                rows = None
                break
            if field in by_field:
                ambiguous.add(field)
            else:
                by_field[field] = row
    output = []
    for claim in claims:
        field = claim["field"]
        record = {key: claim[key] for key in ("field", "found", "replacement")}
        row = by_field.get(field)
        if verifier_text is None:
            status = "verifier_unavailable"
        elif rows is None:
            status = "invalid_verifier_response"
        elif field in ambiguous:
            status = "ambiguous_verifier_field"
        elif row is None:
            status = "missing_verifier_field"
        elif (set(row) != VERIFIED_KEYS or not _scalar(row["value"])
              or not isinstance(row["derivation"], str) or not row["derivation"].strip()):
            status = "invalid_verifier_field"
        elif not _valid_evidence(row["evidence"], source):
            status = "invalid_evidence"
        else:
            status = "accepted" if _equal(row["value"], claim["replacement"]) else "verify_disagrees"
            record.update(verified_value=row["value"], evidence=row["evidence"],
                          derivation=row["derivation"])
        output.append({**record, "status": status})
    return output


def render_notes(claims):
    """Render only candidate changes, with the same template for both critic arms."""
    patches = [{key: claim[key] for key in ("field", "found", "replacement")} for claim in claims]
    return (
        "Collaborator-proposed field corrections. Check the proposed changes against the "
        "question and make only changes you find justified.\n"
        + json.dumps(patches, ensure_ascii=False, separators=(",", ":"))
    ) if patches else ""
