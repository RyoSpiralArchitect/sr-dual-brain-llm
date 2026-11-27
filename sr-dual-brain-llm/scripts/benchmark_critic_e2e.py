#!/usr/bin/env python3
"""Run paired native draft-to-final trials with and without a critic's issues."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from benchmark_critic_reviewer_only import (
    CAP, ENV, MODEL, ROOT, SECONDS, SingleCallClient, now, write_json,
)
from critic_e2e_score import score
from core.answer_generation import Generation, current_generation, guard_answer_generation, reasoning_profile
from core.llm_client import LLMConfig
from core.models import LeftBrainModel, RightBrainModel

FIXTURE = ROOT / "examples/critic_e2e_fresh.json"
SEED = 2712
REPS = 2
BATCH_SIZE = 2
LIM = 96
ARMS = ("without_critic", "with_critic")
NEUTRAL = "No collaborator review was supplied."
STAGES = ("draft", "critic", *ARMS)


def load_cases():
    raw = FIXTURE.read_bytes()
    cases = json.loads(raw)["cases"]
    ids = {c["id"] for c in cases}
    if len(cases) != 12 or len(ids) != 12 or Counter(c["domain"] for c in cases) != {
        "boolean": 3, "sql": 3, "event": 3, "evidence": 3,
    }:
        raise ValueError("Expected 12 fresh unique tasks across four domains")
    for case in cases:
        expected = case["expected"]
        if not 3 <= len(expected) <= 6 or not all(
            type(v) in (str, int, type(None)) for v in expected.values()
        ):
            raise ValueError(f"Invalid scalar oracle: {case['id']}")
    return cases, hashlib.sha256(raw).hexdigest()


def plan(cases):
    rng = random.Random(SEED)
    # Each task sees both arm launch orders, one in each repetition.
    reverse_first = {case["id"]: rng.choice((True, False)) for case in cases}
    for rep in range(1, REPS + 1):
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            reverse = reverse_first[case["id"]] if rep == 1 else not reverse_first[case["id"]]
            yield case, rep, ARMS[::-1] if reverse else ARMS


class CompleteOnly:
    """Select the nonstreaming native generation path with the same prompts."""
    def __init__(self, client):
        self.complete = client.complete


def unavailable():
    return {"output": "", "output_status": "not_attempted", "terminal": "not_attempted",
            "generation_status": "not_attempted", "latency_ms": None, "admission": {}, "fatal": None}


class Trial:
    def __init__(self, config, question):
        self.config = config
        self.question = question
        self.telemetry = self  # Used by the native guard on exceptional exit.

    def log(self, *_args, **_kwargs):
        pass

    @guard_answer_generation
    async def ordinary(self, client, *, draft=None, note=None):
        model = LeftBrainModel(self.config)
        model._llm_client = CompleteOnly(client)
        state = current_generation()
        if draft is None:
            answer = await model.generate_answer(self.question, context="")
        else:
            # Each final gets an independent copy of draft provenance.
            initial = Generation(draft["output"], draft["output_status"])
            state.record("draft", initial, initial.text, 0)
            answer = await model.integrate_info_async(
                question=self.question, draft=initial.text, info=note,
            )
        return {"output": answer, "output_status": state.status(answer),
                "generation_status": state.records[-1]["status"]}

    async def critic(self, client, draft):
        model = RightBrainModel(self.config)
        model._llm_client = client
        with reasoning_profile("provider_default"):
            result = await model.criticise_reasoning(
                "critic-e2e", self.question, draft["output"], allow_micro_fallback=False,
            )
        return {"output": result, "output_status": result["critic_status"], "generation_status": None}

    async def stage(self, name, *, draft=None, note=None):
        client = SingleCallClient(self.config, "provider_default")
        clock = asyncio.get_running_loop().time
        started = clock()
        call = self.critic(client, draft) if name == "critic" else self.ordinary(
            client, draft=draft, note=note,
        )
        task = asyncio.create_task(call)
        done, _ = await asyncio.wait({task}, timeout=SECONDS)
        elapsed = clock() - started
        if task not in done or elapsed >= SECONDS:
            terminal = "owner_deadline"
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            result = {"output": None if name == "critic" else "",
                      "output_status": terminal, "generation_status": terminal}
        else:
            terminal = "returned"
            result = task.result()
        return {**result, "terminal": terminal, "latency_ms": round(elapsed * 1000, 2),
                "admission": client.receipt, "fatal": client.fatal}

    async def run(self, case, rep, arm_order):
        stages = {name: unavailable() for name in STAGES}
        draft = stages["draft"] = await self.stage("draft")
        delivered = False
        if not draft["fatal"]:
            critic = stages["critic"] = await self.stage("critic", draft=draft)
            review = critic["output"] or {}
            note = review.get("critic_sum")
            delivered = bool(review.get("critic_external_review_valid") and
                             review.get("verdict") == "issues" and note)
            if not critic["fatal"]:
                notes = {"without_critic": NEUTRAL, "with_critic": note if delivered else NEUTRAL}
                finals = await asyncio.gather(*(self.stage(arm, draft=draft, note=notes[arm])
                                                for arm in arm_order))
                stages.update(zip(arm_order, finals))
                self.verify_matched(stages, draft, notes)
        return {"case_id": case["id"], "rep": rep, "arm_order": arm_order,
                "review_delivered": delivered and bool(stages["with_critic"]["admission"]),
                "stages": stages}

    def verify_matched(self, stages, draft, notes):
        systems = []
        for arm in ARMS:
            stage = stages[arm]
            payload = stage["admission"].get("request")
            if not payload:
                stage["fatal"] = stage["fatal"] or "missing_final_admission"
                continue
            system, user = payload["messages"]
            expected = (
                f"User message:\n{self.question}\n\nDraft answer:\n{draft['output']}\n\n"
                f"Internal collaborator notes (do not output directly):\n{notes[arm]}\n\nFinal answer:"
            )
            if user["content"] != expected:
                stage["fatal"] = "unmatched_final_prompt"
            systems.append(system["content"])
        if len(set(systems)) > 1:
            stages["with_critic"]["fatal"] = "unmatched_final_system"


async def live(cases, digest, source, path):
    config = LLMConfig("openai", MODEL, os.environ["OPENAI_API_KEY"],
                       api_base="https://api.openai.com/v1", max_output_tokens=CAP,
                       timeout_seconds=SECONDS, auto_continue=False, max_continuations=0)
    os.environ.update(ENV, DUALBRAIN_ANSWER_REASONING_EFFORT="provider_default",
                      DUALBRAIN_ANSWER_TOKEN_BUDGET=str(CAP), DUALBRAIN_COMPLETION_WORK_PRODUCTS="0")
    journal = {"source": source, "fixture_sha256": digest, "started_at": now(),
               "complete": False, "rows": []}
    write_json(path / "journal.json", journal)
    trials = list(plan(cases))
    if len(trials) * len(STAGES) != LIM:
        raise ValueError("Unexpected trial budget")
    for i in range(0, len(trials), BATCH_SIZE):
        wave = await asyncio.gather(*(Trial(config, case["question"]).run(case, rep, order)
                                      for case, rep, order in trials[i:i + BATCH_SIZE]))
        journal["rows"].extend(wave)
        fatals = [stage["fatal"] for row in wave for stage in row["stages"].values() if stage["fatal"]]
        if fatals:
            journal["stop_reason"] = fatals[0]
        write_json(path / "journal.json", journal)
        print(f"{len(journal['rows'])}/{len(trials)} paired trials recorded", flush=True)
        if fatals:
            break
    journal["complete"] = len(journal["rows"]) == len(trials) and "stop_reason" not in journal
    journal["finished_at"] = now()
    write_json(path / "journal.json", journal)
    write_json(path / "metrics.json", score(cases, journal))
    print("complete" if journal["complete"] else f"stopped: {journal['stop_reason']}", flush=True)
    return journal["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", type=Path, metavar="NEW_PRIVATE_DIRECTORY")
    modes.add_argument("--score", type=Path, metavar="JOURNAL")
    args = parser.parse_args()
    cases, digest = load_cases()
    if args.score:
        journal = json.loads(args.score.read_bytes())
        if journal["fixture_sha256"] != digest:
            raise SystemExit("Journal does not match frozen input")
        print(json.dumps(score(cases, journal), ensure_ascii=False, indent=2))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("Freeze experiment files in a clean commit before running")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    os.umask(0o077)
    args.live.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not asyncio.run(live(cases, digest, source, args.live)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
