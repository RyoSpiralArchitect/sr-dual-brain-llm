"""The paired design must survive the native prompt / generation paths."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_critic_e2e as bench
from core.llm_client import LLMClient, LLMConfig
from critic_e2e_score import grade, score
from offline_provider_guard import forbid_provider_network


def test_frozen_order_and_strict_json_grade():
    cases, _ = bench.load_cases()
    trials = list(bench.plan(cases))
    assert len(trials) == 24
    assert len({(case["id"], rep) for case, rep, _ in trials}) == 24
    assert all({order for c, _, order in trials if c == case} == {bench.ARMS, bench.ARMS[::-1]}
               for case in cases)
    complete = {"output_status": "complete"}
    assert grade({"a": None}, {**complete, "output": '{"a":0}'})["grade"] == "semantic_wrong"
    assert grade({"a": 1}, {**complete, "output": '{"a":true}'})["grade"] == "invalid_shape"
    assert grade({"a": 1}, {**complete, "output": '{"a":1,"a":1}'})["grade"] == "invalid_json"


@pytest.mark.asyncio
async def test_paired_native_prompts_and_draft_retention(monkeypatch):
    cases, digest = bench.load_cases()
    case = cases[0]
    right = case["expected"]
    wrong = {**right, "q1": "mixed"}
    requests = []

    async def post(self, url, payload, headers, *, timeout_seconds):
        requests.append(payload)
        prompt = payload["messages"][-1]["content"]
        assert payload["max_completion_tokens"] == 4096
        assert "reasoning_effort" not in payload and 0 < timeout_seconds <= 24
        assert case["oracle_rationale"] not in str(payload["messages"])
        finish = "stop"
        if prompt == case["question"]:
            out = wrong
        elif prompt.startswith("User question:"):
            assert json.dumps(wrong) in prompt and json.dumps(right) not in prompt
            out = {"verdict": "issues", "issues": ["q1 should be true"],
                   "fixes": ["Update q1 to true."]}
        else:
            assert json.dumps(wrong) in prompt and json.dumps(right) not in prompt
            out = right
            if "Update q1 to true" not in prompt:
                assert bench.NEUTRAL in prompt
                finish = "length"
        return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(out)}}]}

    for key, value in {**bench.ENV, "DUALBRAIN_ANSWER_REASONING_EFFORT": "provider_default",
                       "DUALBRAIN_ANSWER_TOKEN_BUDGET": str(bench.CAP)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(LLMClient, "_post_json", post)
    config = LLMConfig("openai", bench.MODEL, "offline", api_base="https://api.openai.com/v1",
                       max_output_tokens=bench.CAP, timeout_seconds=bench.SECONDS, auto_continue=False)
    with forbid_provider_network():
        trial = await bench.Trial(config, case["question"]).run(case, 1, bench.ARMS)
    assert len(requests) == 4 and trial["review_delivered"]
    assert all(stage["fatal"] is None for stage in trial["stages"].values())
    metrics = score(cases, {"complete": False, "fixture_sha256": digest,
                            "source": "offline", "rows": [trial]})
    assert metrics["all_calls"]["admissions"] == 4
    assert metrics["summary"]["arms"]["with_critic"]["wrong_to_exact"] == 1
    assert metrics["summary"]["arms"]["without_critic"]["wrong_to_exact"] == 0
    assert metrics["rows"][0]["without_critic"]["output_status"] == "complete"
    assert metrics["rows"][0]["without_critic"]["generation_status"] == "partial"
