import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_verified_critic as bench
import critic_claim_gate as gate
from core.llm_client import LLMClient, LLMConfig
from offline_provider_guard import forbid_provider_network

QUESTION = "First answer is 3.\n\nSecond answer is 8."


def atom(field, found, replacement, reason="Check arithmetic."):
    return {"field": field, "found": found, "replacement": replacement, "reason": reason}


def review(*atoms):
    return {"critic_external_review_valid": True, "verdict": "issues",
            "issues": [json.dumps(a) for a in atoms]}


def derived(field, value, line, quote):
    return {"field": field, "value": value, "evidence": [{"line": line, "quote": quote}],
            "derivation": "Read the answer from its source line."}


def test_mixed_accepts_one_rejects_one():
    claims, rejected = gate.parse_claims(
        '{"first":9,"second":8}', review(atom("first", 9, 3), atom("second", 8, 0)))
    assert len(claims) == 2 and not rejected
    prompt = json.loads(gate.verify_prompt(QUESTION, [c["field"] for c in claims]))
    assert prompt == {
        "question_lines": [{"line": 1, "text": "First answer is 3."},
                           {"line": 3, "text": "Second answer is 8."}],
        "requested_fields": ["first", "second"],
    }
    result = gate.decisions(QUESTION, claims, json.dumps({"fields": [
        derived("first", 3, 1, "answer is 3"), derived("second", 8, 3, "answer is 8")
    ]}))
    assert [r["status"] for r in result] == ["accepted", "verify_disagrees"]
    notes = gate.render_notes([r for r in result if r["status"] == "accepted"])
    assert "first" in notes and "second" not in notes
    assert "Check arithmetic" not in notes and "Read the answer" not in notes


def test_bad_quote_returns_no_accepted_claim():
    claims = [atom("first", 9, 3)]
    for line, quote in ((3, "First"), (2, "anything"), (True, "First"), (1, " ")):
        result = gate.decisions(QUESTION, claims, json.dumps({
            "fields": [derived("first", 3, line, quote)]
        }))
        assert result[0]["status"] == "invalid_evidence"


def test_type_checks_and_strict_json():
    claims, rejected = gate.parse_claims(
        '{"first":1}', review(atom("first", True, 3), atom("first", "1", 3), atom("first", 1, 1)))
    assert not claims
    assert [r["status"] for r in rejected] == [
        "malformed_claim", "draft_value_mismatch", "no_change"
    ]
    assert not gate.parse_claims('{"first":1,"first":1}', review(atom("first", 1, 3)))[0]
    result = gate.decisions(QUESTION, [atom("first", 1, 3)], '{"fields":[{"field":"first","field":"first"}]}')
    assert result[0]["status"] == "invalid_verifier_response"
    result = gate.decisions(QUESTION, [atom("first", 1, 3)], json.dumps({
        "fields": [derived("first", True, 1, "First")]
    }))
    assert result[0]["status"] == "invalid_verifier_field"


def test_dedup_and_conflicting_corrections():
    same = review(atom("first", 1, 3), atom("first", 1, 3, "Another reason."))
    claims, rejected = gate.parse_claims('{"first":1}', same)
    assert len(claims) == 1 and rejected[0]["status"] == "duplicate_claim"
    conflicting = review(atom("first", 1, 3), atom("first", 1, 4), atom("second", 1, 8))
    claims, rejected = gate.parse_claims('{"first":1,"second":1}', conflicting)
    assert [c["field"] for c in claims] == ["second"]
    assert rejected[0]["status"] == "conflicting_replacements"
    duplicate = derived("first", 3, 1, "First")
    result = gate.decisions(QUESTION, [atom("first", 1, 3), atom("second", 1, 8)], json.dumps({
        "fields": [duplicate, duplicate, derived("second", 8, 3, "Second")]
    }))
    assert [r["status"] for r in result] == ["ambiguous_verifier_field", "accepted"]


def test_missing_and_unknown_verifier_fields_fail_closed():
    claims = [atom("first", 1, 3)]
    assert gate.decisions(QUESTION, claims, None)[0]["status"] == "verifier_unavailable"
    assert gate.decisions(QUESTION, claims, '{"fields":[]}')[0]["status"] == "missing_verifier_field"
    assert gate.decisions(QUESTION, claims, '{"fields":[{"field":"extra"}]}')[0]["status"] == "invalid_verifier_response"
    assert gate.render_notes([]) == ""


@pytest.mark.asyncio
async def test_three_matched_finals_receive_only_their_intended_notes(monkeypatch):
    draft = '{"first":9,"second":8}'
    proposed = [
        atom("first", 9, 3, "The first source explicitly says 3."),
        atom("second", 8, 0, "The second source does not support 8."),
    ]
    final_notes = []

    async def post(self, url, payload, headers, *, timeout_seconds):
        assert payload["model"] == bench.MODEL and payload["max_completion_tokens"] == 4096
        assert "reasoning_effort" not in payload and 0 < timeout_seconds <= 24
        system, user = payload["messages"]
        prompt = user["content"]
        if prompt == QUESTION:
            content = draft
        elif prompt.startswith("User question:"):
            assert draft in prompt
            content = json.dumps({"verdict": "issues", "issues": [json.dumps(a) for a in proposed],
                                  "fixes": []})
        elif system["content"] == gate.VERIFIER_SYSTEM:
            assert json.loads(prompt) == json.loads(gate.verify_prompt(QUESTION, ["first", "second"]))
            assert draft not in prompt and "replacement" not in prompt
            content = json.dumps({"fields": [derived("first", 3, 1, "answer is 3"),
                                              derived("second", 8, 3, "answer is 8")]})
        else:
            assert draft in prompt
            notes = prompt.split("Internal collaborator notes (do not output directly):\n", 1)[1]
            final_notes.append(notes.removesuffix("\n\nFinal answer:"))
            content = draft
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    for name, value in {**bench.ENV, "DUALBRAIN_ANSWER_REASONING_EFFORT": "provider_default",
                        "DUALBRAIN_ANSWER_TOKEN_BUDGET": str(bench.CAP)}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(LLMClient, "_post_json", post)
    config = LLMConfig("openai", bench.MODEL, "offline", api_base="https://api.openai.com/v1",
                       max_output_tokens=bench.CAP, timeout_seconds=bench.SECONDS, auto_continue=False)
    with forbid_provider_network():
        result = await bench.VerifiedTrial(config, QUESTION).run({"id": "synthetic"}, 1, bench.ARMS)
    assert result["notes_differ"]
    assert all(stage["admission"] and stage["fatal"] is None for stage in result["stages"].values())
    assert final_notes == [bench.NEUTRAL, gate.render_notes(proposed), gate.render_notes(proposed[:1])]
    assert [row["status"] for row in result["decisions"]] == ["accepted", "verify_disagrees"]
