#!/usr/bin/env python3
"""Measure matched field confirmation, revision, and coordinated selection."""
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
from benchmark_fixed_claims import draft_text, frozen_claims
from benchmark_verified_critic import VerifiedTrial
from coordinated_policy import answer_fields, inspect_draft, select_answer, select_route, supported_claims
from core.llm_client import LLMConfig
from critic_claim_gate import _equal, _scalar, parse_claims, render_notes
from coordinated_score import score

FIXTURES = {
    "fixed": ROOT / "examples/coordinated_critic_fixed.json",
    "natural": ROOT / "examples/coordinated_critic_natural.json",
}
SEED = 2811
REPS = 2
BATCH_SIZE = 2
FINALS = ("unfiltered", "verified_regenerate")
STAGES = ("draft", "critic", "verifier", *FINALS, "coordinated")
MAX_ADMISSIONS = {"fixed": 96, "natural": 120}
DOMAINS = ("boolean", "sql", "event", "evidence")


def load_cases(cohort):
    raw = FIXTURES[cohort].read_bytes()
    data = json.loads(raw)
    cases = [{**case, "group": "natural" if cohort == "natural" else "mixed"} for case in data["cases"]]
    if Counter(c["domain"] for c in cases) != dict.fromkeys(DOMAINS, 3):
        raise ValueError("Expected three authored cases per domain")
    if cohort == "fixed":
        controls = data["correct_controls"]
        if Counter(c["domain"] for c in controls) != dict.fromkeys(DOMAINS, 1):
            raise ValueError("Expected one exact fixed control per domain")
        cases.extend({**case, "group": "correct_control"} for case in controls)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        expected = case["expected"]
        if not 4 <= len(expected) <= 6 or not all(_scalar(value) for value in expected.values()):
            raise ValueError(f"Invalid known answer fields: {case['id']}")
        if cohort == "fixed":
            draft = answer_fields(draft_text(case), expected)
            if draft is None:
                raise ValueError(f"Invalid fixed draft shape: {case['id']}")
            claims = frozen_claims(case)
            wrong = {k for k,v in expected.items() if not _equal(draft[k], v)}
            repairs = {a["field"] for a in claims if _equal(a["replacement"], expected[a["field"]])}
            harmful = {a["field"] for a in claims if _equal(a["found"], expected[a["field"]])}
            if (not harmful or len(repairs) + len(harmful) != len(claims)
                    or repairs != wrong or bool(wrong) != (case["group"] == "mixed")):
                raise ValueError(f"Fixed draft/proposals disagree with oracle: {case['id']}")
    return cases, hashlib.sha256(raw).hexdigest()


def plan(cases):
    rng = random.Random(SEED)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    standard_first = {case["id"] for case in shuffled[::2]}
    for rep in range(1, REPS + 1):
        subjects = list(cases)
        rng.shuffle(subjects)
        for case in subjects:
            standard = (case["id"] in standard_first) == (rep == 1)
            yield case, rep, FINALS if standard else FINALS[::-1]


class CoordinatedTrial(VerifiedTrial):
    async def run(self, *, case_id, fields, rep, order, fixed=None):
        stages = {name: unavailable() for name in STAGES}
        claims, approved, checks, applied = [], [], [], []
        parse_rejections = []
        route_reason = "trial_stopped"
        if fixed is None:
            draft = stages["draft"] = await self.stage("draft")
            if not draft["fatal"]:
                critic = stages["critic"] = await self.stage("critic", draft=draft)
                if draft["output_status"] == "complete" and isinstance(critic["output"], dict):
                    claims, parse_rejections = parse_claims(draft["output"], critic["output"])
        else:
            draft = stages["draft"] = {**unavailable(), "output": fixed[0],
                                      "output_status": "complete", "terminal": "fixture"}
            claims = fixed[1]
        if not any(stage["fatal"] for stage in stages.values()):
            original = answer_fields(draft["output"], fields) if draft["output_status"] == "complete" else None
            if original is not None:
                verifier = stages["verifier"] = await self.verify(sorted(fields))
                text = verifier["output"] if verifier["output_status"] == "complete" else None
                checks = inspect_draft(self.question, original, text)
                approved = supported_claims(claims, checks)
            route = select_route(original, checks, approved)
            if route.retain:
                stages["coordinated"], route_reason, applied = self.policy_output(route, draft, original, None)
            if not stages["verifier"]["fatal"]:
                notes = {"unfiltered": render_notes(claims) or NEUTRAL,
                         "verified_regenerate": render_notes(approved) or NEUTRAL}
                # Both live outputs are measured counterfactuals. The retained route
                # above returns before reading either of these generation results.
                results = await asyncio.gather(*(self.stage(arm, draft=draft, note=notes[arm])
                                                 for arm in order))
                stages.update(zip(order, results, strict=True))
                self.check_finals(stages, notes)
                if not route.retain:
                    stages["coordinated"], route_reason, applied = self.policy_output(
                        route, draft, original, stages["verified_regenerate"],
                    )
        return {"case_id": case_id, "rep": rep, "arm_order": order, "stages": stages,
                "claims": claims, "parse_rejections": parse_rejections, "checks": checks,
                "approved_fields": [item["field"] for item in approved],
                "applied_fields": applied, "route_reason": route_reason}

    @staticmethod
    def policy_output(route, draft, original, regenerated):
        terminal, reason, text, status, fields = select_answer(route, draft, original, regenerated)
        return {**unavailable(), "output": text, "output_status": status,
                "terminal": terminal}, reason, fields

    def check_finals(self, stages, notes):
        system_messages = []
        draft = stages["draft"]["output"]
        for arm in FINALS:
            stage = stages[arm]
            request = stage["admission"].get("request")
            if not request:
                stage["fatal"] = stage["fatal"] or "missing_final_admission"
                continue
            system, user = request["messages"]
            system_messages.append(system)
            expected = (f"User message:\n{self.question}\n\nDraft answer:\n{draft}\n\n"
                        f"Internal collaborator notes (do not output directly):\n{notes[arm]}\n\nFinal answer:")
            if user["content"] != expected:
                stage["fatal"] = "unmatched_final_prompt"
        if system_messages and any(s != system_messages[0] for s in system_messages):
            stages["verified_regenerate"]["fatal"] = "unmatched_final_system"


async def live(cases, cohort, digest, source, path):
    config = LLMConfig("openai", MODEL, os.environ["OPENAI_API_KEY"],
                       api_base="https://api.openai.com/v1", max_output_tokens=CAP,
                       timeout_seconds=SECONDS, auto_continue=False, max_continuations=0)
    os.environ.update(ENV, DUALBRAIN_ANSWER_REASONING_EFFORT="provider_default",
                      DUALBRAIN_ANSWER_TOKEN_BUDGET=str(CAP), DUALBRAIN_COMPLETION_WORK_PRODUCTS="0")
    journal = {"source": source, "fixture_sha256": digest, "cohort": cohort, "started_at": now(),
               "complete": False, "rows": []}
    write_json(path / "journal.json", journal)
    trials = list(plan(cases))
    if len(trials) * (3 if cohort == "fixed" else 5) != MAX_ADMISSIONS[cohort]:
        raise ValueError("Unexpected frozen admission budget")
    for i in range(0, len(trials), BATCH_SIZE):
        wave = await asyncio.gather(*(CoordinatedTrial(config, case["question"]).run(
            case_id=case["id"], fields=list(case["expected"]), rep=rep, order=order,
            fixed=(draft_text(case), frozen_claims(case)) if cohort == "fixed" else None,
        ) for case, rep, order in trials[i:i + BATCH_SIZE]))
        journal["rows"].extend(wave)
        fatal = next((st["fatal"] for trial in wave for st in trial["stages"].values() if st["fatal"]), None)
        if fatal:
            journal["stop_reason"] = fatal
        write_json(path / "journal.json", journal)
        print(f"{cohort}: {len(journal['rows'])}/{len(trials)} matched rows recorded", flush=True)
        if fatal:
            break
    journal["complete"] = len(journal["rows"]) == len(trials) and "stop_reason" not in journal
    journal["finished_at"] = now()
    write_json(path / "journal.json", journal)
    write_json(path / "metrics.json", score(cases, journal))
    print(f"{cohort}: complete" if journal["complete"] else f"{cohort}: stopped: {journal['stop_reason']}", flush=True)
    return journal["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=FIXTURES)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", type=Path, metavar="NEW_PRIVATE_DIRECTORY")
    modes.add_argument("--score", type=Path, metavar="JOURNAL")
    args = parser.parse_args()
    cases, digest = load_cases(args.cohort)
    if args.score:
        journal = json.loads(args.score.read_bytes())
        if journal["fixture_sha256"] != digest or journal["cohort"] != args.cohort:
            raise SystemExit("Journal does not match frozen input")
        print(json.dumps(score(cases, journal), ensure_ascii=False, indent=2))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("Freeze both fixtures and protocol in a clean commit before running")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    os.umask(0o077)
    args.live.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not asyncio.run(live(cases, args.cohort, digest, source, args.live)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
