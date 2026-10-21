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
                                  arms=["off", "checked", "conversation"], grade=True))
    assert report["grader_failures"] == 0
    assert all(s["graded"] == s["both_pass"] == s["probes"] == 16 for s in report["summaries"].values())
    assert "NEVER_PERSIST_KEY" not in json.dumps(journal.data)
    replay = bench.Journal(config, {}, frozen=journal.data)
    replayed = asyncio.run(bench.run(fixture, config, replay, phase="development",
                                   arms=["off", "checked", "conversation"], grade=True))
    assert replayed["provider_calls"] == 0 and replayed["grades"] == report["grades"]
    broken = deepcopy(journal.data)
    key = next(k for k, rows in broken["responses"].items() if rows[0]["request"]["role"] == "answer")
    broken["responses"][key].append(broken["responses"][key][0])
    incomplete = bench.Journal(config, {}, frozen=broken)
    with pytest.raises(ValueError, match="coverage"):
        asyncio.run(bench.run(fixture, config, incomplete, phase="development",
                             arms=["off", "checked", "conversation"], grade=True))
