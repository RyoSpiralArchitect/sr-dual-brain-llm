"""Verify that matched outputs cannot bypass the scalar field policy."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_coordinated_critic as bench
from coordinated_policy import answer_fields, select_answer, select_route, supported_claims
from coordinated_score import score
from core.llm_client import LLMClient, LLMConfig
from critic_claim_gate import VERIFIER_SYSTEM, render_notes, verify_prompt
from offline_provider_guard import forbid_provider_network


def check(field, value, status="accepted"):
    return {"field": field, "verified_value": value, "status": status}


def stage(text):
    return {"output": text, "output_status": "complete"}


def test_retain_requires_all_fields_and_uses_no_regeneration_result():
    draft = {"a": 1, "b": 2}
    original = stage('{"a":1, "b":2}')
    route = select_route(draft, [check("a", 1), check("b", 2)], [])
    assert select_answer(route, original, draft, None) == (
        "retain_confirmed", "all_fields_confirmed", original["output"], "complete", [],
    )
    assert select_route(draft, [check("a", 1)], []).reason == "verification_incomplete"
    assert select_route(draft, [check("a", 1), check("b", 3, "verify_disagrees")], []).reason == (
        "uncovered_discrepancy"
    )


def test_projection_requires_current_schema_and_integration_agreement():
    draft = {"a": 1, "b": 2}
    checks = [check("a", None, "verify_disagrees"), check("b", 2)]
    claims = [{"field": "a", "found": 1, "replacement": None}]
    route = select_route(draft, checks, supported_claims(claims, checks))
    original = stage('{"a":1,"b":2}')
    actual = select_answer(route, original, draft, stage('{"a":null,"b":99}'))
    assert actual[0] == "commit_checked" and json.loads(actual[2]) == {"a": None, "b": 2}
    for text in ('{"b":2}', '{"a":false,"b":2}', '{"a":null,"a":null,"b":2}'):
        assert select_answer(route, original, draft, stage(text))[:3] == (
            "fallback_regenerate", "integration_did_not_confirm_all_changes", text,
        )
    assert answer_fields('{"a":true,"b":2}', draft) is None


def test_frozen_fixtures_and_balanced_launch_order():
    for cohort, n in (("fixed", 32), ("natural", 24)):
        cases, _ = bench.load_cases(cohort)
        trials = list(bench.plan(cases))
        assert len(trials) == n
        assert all({o for c, _, o in trials if c == case} == {bench.FINALS, bench.FINALS[::-1]}
                   for case in cases)


@pytest.mark.asyncio
async def test_native_finals_share_draft_and_only_commit_corroborated_fields(monkeypatch):
    question = "Set first to 3 and second to 8.\nReturn fields first and second."
    draft = '{"first":9,"second":8}'
    case = {"id": "probe", "domain": "boolean", "expected": {"first": 3, "second": 8}}
    claims = [{"field": "first", "found": 9, "replacement": 3, "reason": "Read the task."}]
    calls = []

    async def post(self, url, payload, headers, *, timeout_seconds):
        assert payload["max_completion_tokens"] == 4096 and "reasoning_effort" not in payload
        messages = payload["messages"]
        calls.append(messages)
        if messages[0]["content"] == VERIFIER_SYSTEM:
            assert messages[1]["content"] == verify_prompt(question, ["first", "second"])
            output = {"fields": [
                {"field": field, "value": value, "evidence": [{"line": 1, "quote": quote}],
                 "derivation": "Read the value from the first line."}
                for field, value, quote in (("first", 3, "first to 3"), ("second", 8, "second to 8"))
            ]}
        else:
            assert draft in messages[1]["content"] and render_notes(claims) in messages[1]["content"]
            output = {"first": 3, "second": 0}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}

    monkeypatch.setattr(LLMClient, "_post_json", post)
    monkeypatch.setenv("DUALBRAIN_ANSWER_REASONING_EFFORT", "provider_default")
    config = LLMConfig("openai", bench.MODEL, "offline", api_base="https://api.openai.com/v1",
                       max_output_tokens=bench.CAP, timeout_seconds=bench.SECONDS, auto_continue=False)
    with forbid_provider_network():
        row = await bench.CoordinatedTrial(config, question).run(
            case_id=case["id"], fields=list(case["expected"]), rep=1, order=bench.FINALS,
            fixed=(draft, claims),
        )
    assert len(calls) == 3
    assert all(stage["fatal"] is None for stage in row["stages"].values())
    assert row["stages"]["coordinated"]["admission"] == {}
    assert row["applied_fields"] == ["first"]
    result = score([case], {"source": "offline", "fixture_sha256": "offline", "cohort": "fixed",
                            "complete": False, "rows": [row]})
    assert result["rows"][0]["coordinated"]["grade"] == "exact"
    assert result["rows"][0]["verified_regenerate"]["grade"] == "semantic_wrong"
