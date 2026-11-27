#!/usr/bin/env python3
"""Measure claim verification and direct draft return on fixed mixed critiques."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from benchmark_critic_e2e import NEUTRAL, unavailable
from benchmark_critic_reviewer_only import CAP, ENV, MODEL, ROOT, SECONDS, now, write_json
from benchmark_verified_critic import VerifiedTrial
from core.llm_client import LLMConfig
from critic_claim_gate import decisions, parse_claims, render_notes
from fixed_claim_score import score

FIXTURE = ROOT / "examples/critic_mixed_fixed.json"
SEED = 2714
REPS = 2
BATCH_SIZE = 2
FINALS = ("unfiltered", "verified_regenerate")
STAGES = ("draft", "verifier", *FINALS, "retain_on_empty")
MAX_ADMISSIONS = 96


def same(value, other):
    return type(value) is type(other) and value == other


def draft_text(case):
    return json.dumps(case["draft"], ensure_ascii=False, separators=(",", ":"))


def frozen_claims(case):
    critique = {
        "critic_external_review_valid": True,
        "verdict": "issues",
        "issues": [json.dumps(item, ensure_ascii=False) for item in case["claims"]],
    }
    parsed, rejected = parse_claims(draft_text(case), critique)
    if rejected or parsed != case["claims"]:
        raise ValueError(f"Fixture has nonbinding or ambiguous claims: {case['id']}")
    return parsed


def load_cases():
    raw = FIXTURE.read_bytes()
    fixture = json.loads(raw)
    mixed = [{**case, "group": "mixed"} for case in fixture["cases"]]
    controls = [{**case, "group": "correct_control"} for case in fixture["correct_controls"]]
    domains = ("boolean", "sql", "event", "evidence")
    cases = mixed + controls
    if (Counter(c["domain"] for c in mixed) != dict.fromkeys(domains, 3)
            or Counter(c["domain"] for c in controls) != dict.fromkeys(domains, 1)
            or len({c["id"] for c in cases}) != 16):
        raise ValueError("Expected three mixed and one correct-draft control per domain")
    for case in cases:
        expected, draft = case["expected"], case["draft"]
        if (set(expected) != set(draft) or not expected
                or any(type(value) not in (str, int, type(None))
                       for value in (*expected.values(), *draft.values()))):
            raise ValueError(f"Invalid scalar draft or oracle: {case['id']}")
        exact = all(same(draft[key], value) for key, value in expected.items())
        claims = frozen_claims(case)
        good = {c["field"] for c in claims if same(expected[c["field"]], c["replacement"])}
        bad = {c["field"] for c in claims if same(expected[c["field"]], c["found"])}
        wrong_fields = {key for key in expected if not same(draft[key], expected[key])}
        if not bad or len(good) + len(bad) != len(claims) or good != wrong_fields:
            raise ValueError(f"Claims must fix all wrong fields or damage a correct one: {case['id']}")
        if (case["group"] == "mixed" and (exact or not good)
                or case["group"] == "correct_control" and not exact):
            raise ValueError(f"Draft and claims must match fixed group: {case['id']}")
    return cases, hashlib.sha256(raw).hexdigest()


def plan(cases):
    rng = random.Random(SEED)
    first = list(cases)
    rng.shuffle(first)
    standard_first = {case["id"] for case in first[::2]}
    for rep in range(1, REPS + 1):
        subjects = list(cases)
        rng.shuffle(subjects)
        for case in subjects:
            in_order = (case["id"] in standard_first) == (rep == 1)
            yield case, rep, FINALS if in_order else FINALS[::-1]


class FixedTrial(VerifiedTrial):
    async def verified_final(self, draft, accepted, *, retain_when_empty=False, reuse=None):
        if retain_when_empty and not accepted:
            return {**unavailable(), "output": draft["output"],
                    "output_status": draft["output_status"], "terminal": "draft_retained"}
        if reuse is not None:
            return {**unavailable(), "output": reuse["output"],
                    "output_status": reuse["output_status"], "terminal": "shared_regeneration",
                    "generation_status": reuse["generation_status"]}
        return await self.stage(
            "verified_regenerate", draft=draft, note=render_notes(accepted) or NEUTRAL,
        )

    async def run(self, case, rep, order):
        stages = {name: unavailable() for name in STAGES}
        draft = stages["draft"] = {**unavailable(), "output": draft_text(case),
                                  "output_status": "complete", "terminal": "fixture"}
        claims = frozen_claims(case)
        verifier = stages["verifier"] = await self.verify(sorted(c["field"] for c in claims))
        text = verifier["output"] if verifier["output_status"] == "complete" else None
        evaluated = decisions(self.question, claims, text)
        accepted = [item for item in evaluated if item["status"] == "accepted"]
        if not verifier["fatal"]:
            notes = {"unfiltered": render_notes(claims),
                     "verified_regenerate": render_notes(accepted) or NEUTRAL}
            final_tasks = [
                self.stage(arm, draft=draft, note=notes[arm]) if arm == "unfiltered"
                else self.verified_final(draft, accepted) for arm in order
            ]
            stages.update(zip(order, await asyncio.gather(*final_tasks), strict=True))
            stages["retain_on_empty"] = await self.verified_final(
                draft, accepted, retain_when_empty=True, reuse=stages["verified_regenerate"],
            )
            self.check_finals(stages, notes)
        return {"case_id": case["id"], "rep": rep, "arm_order": order,
                "stages": stages, "decisions": evaluated}

    def check_finals(self, stages, notes):
        systems = []
        for arm in FINALS:
            stage = stages[arm]
            request = stage["admission"].get("request")
            if not request:
                stage["fatal"] = stage["fatal"] or "missing_final_admission"
                continue
            system, user = request["messages"]
            systems.append(system)
            expected = (
                f"User message:\n{self.question}\n\nDraft answer:\n{stages['draft']['output']}\n\n"
                f"Internal collaborator notes (do not output directly):\n{notes[arm]}\n\nFinal answer:"
            )
            if user["content"] != expected:
                stage["fatal"] = "unmatched_final_prompt"
        if systems and any(item != systems[0] for item in systems):
            stages["verified_regenerate"]["fatal"] = "unmatched_final_system"


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
    if len(trials) * (1 + len(FINALS)) != MAX_ADMISSIONS:
        raise ValueError("Unexpected fixed trial budget")
    for i in range(0, len(trials), BATCH_SIZE):
        wave = await asyncio.gather(*(FixedTrial(config, case["question"]).run(case, rep, order)
                                      for case, rep, order in trials[i:i + BATCH_SIZE]))
        journal["rows"].extend(wave)
        fatal = next((st["fatal"] for trial in wave for st in trial["stages"].values() if st["fatal"]), None)
        if fatal:
            journal["stop_reason"] = fatal
        write_json(path / "journal.json", journal)
        print(f"{len(journal['rows'])}/{len(trials)} fixed rows recorded", flush=True)
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
