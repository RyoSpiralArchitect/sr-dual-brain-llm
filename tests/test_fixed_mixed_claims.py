"""The zero-pass policy must preserve the draft and admit no final call."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_fixed_claims as bench
from core.llm_client import LLMClient, LLMConfig
from critic_claim_gate import VERIFIER_SYSTEM, render_notes, verify_prompt
from fixed_claim_score import score
from offline_provider_guard import forbid_provider_network


def test_freeze_has_balanced_fixed_opportunities():
    cases, _ = bench.load_cases()
    trials = list(bench.plan(cases))
    assert len(trials) == 32
    assert all({o for c, _, o in trials if c == case} == {bench.FINALS, bench.FINALS[::-1]}
               for case in cases)


@pytest.mark.asyncio
async def test_zero_pass_returns_each_fixed_draft_without_invoking_generation(monkeypatch):
    cases, _ = bench.load_cases()
    trial = bench.FixedTrial(None, "test")

    async def unexpected(*args, **kwargs):
        pytest.fail("The no-patch route invoked final generation")

    monkeypatch.setattr(trial, "stage", unexpected)
    with forbid_provider_network():
        for group in ("mixed", "correct_control"):
            text = bench.draft_text(next(case for case in cases if case["group"] == group))
            result = await trial.verified_final({"output": text, "output_status": "complete"}, [],
                                                retain_when_empty=True)
            assert result["terminal"] == "draft_retained"
            assert result["output"] == text and result["admission"] == {}


@pytest.mark.asyncio
async def test_fixed_mixed_notes_and_nonzero_policy_share_generation(monkeypatch):
    question = 'Set first to 3 and second to 8.\nReturn fields first and second.'
    case = {"id": "toy", "domain": "boolean", "group": "mixed", "question": question,
            "expected": {"first": 3, "second": 8}, "draft": {"first": 9, "second": 8},
            "claims": [
                {"field": "first", "found": 9, "replacement": 3, "reason": "Read first value."},
                {"field": "second", "found": 8, "replacement": 0, "reason": "False change."},
            ]}
    messages = []

    async def post(self, url, payload, headers, *, timeout_seconds):
        messages.append(payload["messages"])
        system, user = payload["messages"]
        prompt = user["content"]
        if system["content"] == VERIFIER_SYSTEM:
            assert json.loads(prompt) == json.loads(verify_prompt(question, ["first", "second"]))
            output = {"fields": [
                {"field": field, "value": value, "evidence": [{"line": 1, "quote": quote}],
                 "derivation": "Read the stated value."}
                for field, value, quote in (("first", 3, "first to 3"), ("second", 8, "second to 8"))
            ]}
        else:
            assert bench.draft_text(case) in prompt
            output = {"first": 3, "second": 0 if "False change" in prompt
                      or '"replacement":0' in prompt else 8}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}

    monkeypatch.setattr(LLMClient, "_post_json", post)
    monkeypatch.setenv("DUALBRAIN_ANSWER_REASONING_EFFORT", "provider_default")
    config = LLMConfig("openai", bench.MODEL, "offline", api_base="https://api.openai.com/v1",
                       max_output_tokens=bench.CAP, timeout_seconds=bench.SECONDS, auto_continue=False)
    with forbid_provider_network():
        trial = await bench.FixedTrial(config, question).run(case, 1, bench.FINALS)
    assert len(messages) == 3
    assert [item["status"] for item in trial["decisions"]] == ["accepted", "verify_disagrees"]
    assert render_notes(case["claims"]) in messages[1][1]["content"]
    assert render_notes(case["claims"][:1]) in messages[2][1]["content"]
    stages = trial["stages"]
    assert stages["retain_on_empty"]["terminal"] == "shared_regeneration"
    assert stages["retain_on_empty"]["output"] == stages["verified_regenerate"]["output"]
    assert not stages["retain_on_empty"]["admission"]
    report = score([case], {"source": "offline", "fixture_sha256": "offline",
                             "complete": False, "rows": [trial]})
    assert report["summary"]["mixed_trials_both_decisions_correct"] == 1
    assert report["all_calls"]["admissions"] == 3
