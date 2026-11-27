#!/usr/bin/env python3
"""Compare neutral revision, unfiltered critique, and independently checked critique."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import suppress
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import random
import subprocess

from benchmark_critic_e2e import NEUTRAL, Trial, unavailable
from benchmark_critic_reviewer_only import (
    CAP, ENV, MODEL, ROOT, SECONDS, SingleCallClient, now, write_json,
)
from core.answer_generation import reasoning_profile
from core.llm_client import LLMConfig
from core.models import RightBrainModel
from critic_claim_gate import (
    CRITIC_CONTEXT, VERIFIER_SYSTEM, decisions, parse_claims, render_notes, verify_prompt,
)
from verified_critic_score import score

FIXTURE = ROOT / "examples/verified_critic_fresh.json"
SEED = 2713
REPS = 2
BATCH_SIZE = 2
ARMS = ("without_critic", "standard_critic", "verified_critic")
STAGES = ("draft", "critic", "verifier", *ARMS)
MAX_ADMISSIONS = 144


def load_cases():
    raw = FIXTURE.read_bytes()
    cases = json.loads(raw)["cases"]
    if len(cases) != 12 or len({c["id"] for c in cases}) != 12 or Counter(
        c["domain"] for c in cases
    ) != {"boolean": 3, "sql": 3, "event": 3, "evidence": 3}:
        raise ValueError("Expected 12 new unique tasks, three per domain")
    for case in cases:
        expected = case["expected"]
        if not 4 <= len(expected) <= 6 or not all(
            isinstance(k, str) and k and type(v) in (str, int, type(None))
            for k, v in expected.items()
        ):
            raise ValueError(f"Invalid scalar oracle: {case['id']}")
    return cases, hashlib.sha256(raw).hexdigest()


def plan(cases):
    rng = random.Random(SEED)
    for rep in range(1, REPS + 1):
        cases_in_block = list(cases)
        rng.shuffle(cases_in_block)
        orders = list(permutations(ARMS)) * 2
        rng.shuffle(orders)
        for case, order in zip(cases_in_block, orders, strict=True):
            yield case, rep, order


class VerifiedTrial(Trial):
    async def critic(self, client, draft):
        model = RightBrainModel(self.config)
        model._llm_client = client
        with reasoning_profile("provider_default"):
            result = await model.criticise_reasoning(
                "verified-critic", self.question, draft["output"], context=CRITIC_CONTEXT,
                allow_micro_fallback=False,
            )
        return {"output": result, "output_status": result["critic_status"], "generation_status": None}

    async def verify(self, fields):
        client = SingleCallClient(self.config, "provider_default")
        clock = asyncio.get_running_loop().time
        prompt = verify_prompt(self.question, fields)
        started = clock()
        with reasoning_profile("provider_default"):
            task = asyncio.create_task(client.complete(
                prompt, system=VERIFIER_SYSTEM, max_output_tokens=CAP, timeout_seconds=SECONDS,
                require_complete=True, allow_continuation=False,
            ))
        done, _ = await asyncio.wait({task}, timeout=SECONDS)
        elapsed = clock() - started
        output = ""
        if task not in done or elapsed >= SECONDS:
            terminal, status = "owner_deadline", "owner_deadline"
            task.cancel()
            with suppress(asyncio.CancelledError, RuntimeError, ValueError):
                await task
        else:
            terminal = "returned"
            try:
                output = task.result()
            except Exception as exc:
                status = getattr(exc, "reason", None) or "provider_error"
            else:
                status = "complete"
        return {"output": output, "output_status": status, "generation_status": status,
                "terminal": terminal, "latency_ms": round(elapsed * 1000, 2),
                "admission": client.receipt, "fatal": client.fatal}

    async def run(self, case, rep, arm_order):
        stages = {name: unavailable() for name in STAGES}
        claims, rejections, evaluations = [], [], []
        notes_differ = False
        draft = stages["draft"] = await self.stage("draft")
        if not draft["fatal"]:
            critic = stages["critic"] = await self.stage("critic", draft=draft)
            if draft["output_status"] == "complete" and isinstance(critic["output"], dict):
                claims, rejections = parse_claims(draft["output"], critic["output"])
            if not critic["fatal"]:
                verifier = stages["verifier"]
                if claims:
                    verifier = stages["verifier"] = await self.verify(sorted({c["field"] for c in claims}))
                    if verifier["output_status"] == "complete":
                        evaluations = decisions(self.question, claims, verifier["output"])
                    else:
                        evaluations = decisions(self.question, claims, None)
                if not verifier["fatal"]:
                    keep = {item["field"] for item in evaluations if item["status"] == "accepted"}
                    notes = {"without_critic": NEUTRAL,
                             "standard_critic": render_notes(claims) if claims else NEUTRAL,
                             "verified_critic": render_notes([c for c in claims if c["field"] in keep])
                             if keep else NEUTRAL}
                    notes_differ = notes["standard_critic"] != notes["verified_critic"]
                    results = await asyncio.gather(*(self.stage(
                        arm, draft=draft, note=notes[arm],
                    ) for arm in arm_order))
                    stages.update(zip(arm_order, results))
                    self.verify_matched(stages, draft, notes)
        return {"case_id": case["id"], "rep": rep, "arm_order": arm_order, "stages": stages,
                "claims": claims, "parse_rejections": rejections, "decisions": evaluations,
                "notes_differ": notes_differ}

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
            stages["verified_critic"]["fatal"] = "unmatched_final_system"


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
    if len(trials) * len(STAGES) != MAX_ADMISSIONS:
        raise ValueError("Unexpected trial budget")
    for i in range(0, len(trials), BATCH_SIZE):
        wave = await asyncio.gather(*(VerifiedTrial(config, case["question"]).run(case, rep, order)
                                      for case, rep, order in trials[i:i + BATCH_SIZE]))
        journal["rows"].extend(wave)
        fatal = next((st["fatal"] for trial in wave for st in trial["stages"].values() if st["fatal"]), None)
        if fatal:
            journal["stop_reason"] = fatal
        write_json(path / "journal.json", journal)
        print(f"{len(journal['rows'])}/{len(trials)} triplets recorded", flush=True)
        if fatal:
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
