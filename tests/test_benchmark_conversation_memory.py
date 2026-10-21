import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
import benchmark_conversation_memory as bench
from core.llm_client import LLMConfig


def test_actual_answer_path_exact_replay_and_grade_coverage():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    config = LLMConfig("openai", "test-model", "NEVER_PERSIST_KEY")
    journal = bench.Journal(config, {})

    async def post(url, payload, headers, **kwargs):
        user = payload["messages"][-1]["content"]
        if payload.get("response_format"):
            packet = json.loads(user)
            if "records" in packet:
                text = json.dumps({"decisions": [dict(id=r["id"], supported=True, quote=r["answer"])
                                                   for r in packet["records"]]})
            else:
                text = json.dumps({"scores": {label: {c["id"]: {"verdict":"pass", "quote":"記録"}
                      for c in packet["criteria"]} for label in packet["answers"]}})
        else:
            text = "記録を確認しました。"
        return {"model":"test-model", "choices":[{"finish_reason":"stop", "message":{"content":text}}],
                "usage":{"prompt_tokens":10,"completion_tokens":5}}

    journal.transport._post_json = post
    report = asyncio.run(bench.run(fixture, config, journal, phase="development",
                                  arms=["off", "checked", "conversation_whole_batch", "conversation"], grade=True))
    assert report["grader_failures"] == 0
    assert all(s["graded"] == s["both_pass"] == s["probes"] == 16 for s in report["summaries"].values())
    assert "NEVER_PERSIST_KEY" not in json.dumps(journal.data)
    assert sum(v["calls"] for v in report["usage_by_role"].values()) == report["provider_calls"]
    assert sum(v["input_tokens"] for v in report["usage_by_role"].values()) == report["input_tokens"]
    assert report["served_models"] == ["test-model"]
    replay = bench.Journal(config, {}, frozen=journal.data)
    replayed = asyncio.run(bench.run(fixture, config, replay, phase="development",
                                   arms=["off", "checked", "conversation_whole_batch", "conversation"], grade=True))
    assert replayed["provider_calls"] == 0 and replayed["grades"] == report["grades"]
    broken = deepcopy(journal.data)
    key = next(k for k, rows in broken["responses"].items() if rows[0]["request"]["role"] == "answer")
    broken["responses"][key].append(broken["responses"][key][0])
    incomplete = bench.Journal(config, {}, frozen=broken)
    with pytest.raises(ValueError, match="coverage"):
        asyncio.run(bench.run(fixture, config, incomplete, phase="development",
                             arms=["off", "checked", "conversation_whole_batch", "conversation"], grade=True))


@pytest.mark.parametrize("failure_at", [2, 4])
@pytest.mark.parametrize("failure_mode", ["timeout", "length", "null", "empty"])
@pytest.mark.parametrize("arm", ["off", "conversation"])
def test_failed_source_or_probe_aborts_before_fallback_can_affect_later_turns(failure_at, failure_mode, arm):
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    config = LLMConfig("openai", "test-model", "NOT_A_KEY")
    journal = bench.Journal(config, {})
    saved = []
    calls = []

    async def post(url, payload, headers, **kwargs):
        if payload.get("response_format"):
            packet = json.loads(payload["messages"][-1]["content"])
            text = json.dumps({"decisions":[dict(id=r["id"], supported=False, quote="") for r in packet["records"]]})
            return {"model":"test-model", "choices":[{"finish_reason":"stop", "message":{"content":text}}],
                    "usage":{"prompt_tokens":10, "completion_tokens":5}}
        calls.append(payload)
        fail = len(calls) == failure_at
        if fail and failure_mode == "timeout":
            raise TimeoutError("simulated provider failure")
        reason = "length" if fail and failure_mode == "length" else "stop"
        text = None if fail and failure_mode == "null" else "" if fail and failure_mode == "empty" else "Recorded source"
        return {"model":"test-model", "choices":[{"finish_reason":reason, "message":{"content":text}}],
                "usage":{"prompt_tokens":10, "completion_tokens":5}}

    journal.transport._post_json = post
    with pytest.raises(RuntimeError, match="episode aborted"):
        asyncio.run(bench.run(fixture, config, journal, phase="development", arms=[arm],
                             grade=True, save=lambda report: saved.append(deepcopy(report))))
    assert len(calls) == len(saved[-1]["rows"]) == failure_at
    assert saved[-1]["rows"][-1]["status"] == "unavailable"
    assert saved[-1]["rows"][-1]["answer"] is None and not saved[-1]["complete"]
    assert not saved[-1]["grades"] and not journal.data["complete"]


def test_pretransport_failure_is_unavailable_and_aborts_without_later_turns():
    fixture = bench.read_json(bench.DEFAULT_FIXTURE)
    config = LLMConfig("openai", "test-model", "NOT_A_KEY")
    journal = bench.Journal(config, {})
    journal.transport = None
    saved = []
    with pytest.raises(RuntimeError, match="episode aborted"):
        asyncio.run(bench.run(fixture, config, journal, phase="development", arms=["off"],
                             save=lambda report: saved.append(deepcopy(report))))
    assert journal.calls == 0 and not journal.events
    assert len(saved[-1]["rows"]) == 1 and saved[-1]["rows"][0]["status"] == "unavailable"


def test_live_call_ceiling_stops_before_another_transport_attempt():
    journal = bench.Journal(LLMConfig("openai", "test-model", "NOT_A_KEY"), {}, max_calls=1)
    calls = []
    async def post(*args, **kwargs):
        calls.append(True)
        return {"model":"test-model", "choices":[{"finish_reason":"stop", "message":{"content":"answer"}}]}
    journal.transport._post_json = post
    async def scenario():
        await journal.post("answer", "https://example.invalid", {}, {})
        with pytest.raises(ValueError, match="call budget"):
            await journal.post("answer", "https://example.invalid", {}, {})
    asyncio.run(scenario())
    assert journal.calls == len(calls) == 1
    assert bench.usage_for(journal.events)["input_tokens"] is None
    for invalid in (0, 257, True):
        with pytest.raises(ValueError, match="Call limit"):
            bench.Journal(None, {}, max_calls=invalid)


def test_growth_fixture_grades_against_the_reference_at_each_turn_and_exact_replays():
    fixture = bench.read_json(bench.PROJECT / "examples/conversation_memory_cache_growth.json")
    config = LLMConfig("openai", "test-model", "NOT_A_KEY")
    journal = bench.Journal(config, {})
    references = []
    async def post(url, payload, headers, **kwargs):
        if payload.get("response_format"):
            packet = json.loads(payload["messages"][-1]["content"])
            if "records" in packet:
                text = json.dumps({"decisions":[dict(id=r["id"], supported=True, quote=r["answer"])
                                                   for r in packet["records"]]})
            else:
                references.append(packet["reference"])
                text = json.dumps({"scores":{label:{c["id"]:{"verdict":"pass","quote":"記録"}
                       for c in packet["criteria"]} for label in packet["answers"]}})
        else:
            text = "記録を確認しました。"
        return {"model":"test-model","choices":[{"finish_reason":"stop","message":{"content":text}}],
                "usage":{"prompt_tokens":10,"completion_tokens":5}}
    journal.transport._post_json = post
    arms = ["conversation_whole_batch", "conversation"]
    report = asyncio.run(bench.run(fixture, config, journal, phase="evaluation", arms=arms, grade=True))
    assert references == [e["reference"] for ep in fixture["episodes"] for e in ep["events"] if e["kind"]=="probe"]
    assert all(s["graded"] == s["both_pass"] == 16 for s in report["summaries"].values())
    replay = bench.Journal(config, {}, frozen=journal.data)
    repeated = asyncio.run(bench.run(fixture, config, replay, phase="evaluation", arms=arms, grade=True))
    assert repeated["provider_calls"] == 0 and repeated["grades"] == report["grades"]
    assert all(v["calls"] == v["records_sent"] == 0 for v in repeated["usage_by_role"].values())
    with pytest.raises(ValueError, match="no episodes"):
        asyncio.run(bench.run(fixture, config, replay, phase="development", arms=arms))
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        asyncio.run(bench.run(fixture, config, replay, phase="evaluation", arms=["conversation", "conversation"]))
