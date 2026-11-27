#!/usr/bin/env python3
"""Measure the native critic on frozen paired drafts with a single call per cell."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.answer_generation import reasoning_profile
from core.llm_client import LLMClient, LLMConfig
from core.models import RightBrainModel

MODEL = "gpt-6-luna"
EFFORTS = ("provider_default", "low", "medium")
CAP = 4096
SECONDS = 24
SEED = 2711
FIXTURE = ROOT / "examples" / "critic_reviewer_only.json"
ENV = {
    "DUALBRAIN_CRITIC_TIMEOUT_SECONDS": str(SECONDS),
    "DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS": str(CAP),
    "DUALBRAIN_CRITIC_JSON_REPAIR_CALL": "0",
}
CONTEXT = (
    "For this review, check each labelled field against the supplied task. "
    "The requested draft is just a list of field values and does not need explanations. "
    "For every erroneous field, use one issue string with this exact format: "
    "FIELD=<label>; FOUND=<draft value>; EXPECTED=<correct value>. "
    "Copy the label and draft value exactly and give the corrected value without commentary in "
    "that issue string. You may explain the reason or correction in fixes. "
    "If all fields are correct, use the ok verdict with empty issues and fixes."
)
WITNESS = re.compile(r"FIELD\s*=\s*([^;]+);\s*FOUND\s*=\s*([^;]+);\s*EXPECTED\s*=\s*([^;]+)", re.I)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def fields(text):
    rows = [line.split(": ", 1) for line in text.splitlines()]
    if len(rows) != 3 or any(len(row) != 2 or not all(row) for row in rows):
        raise ValueError("Candidate must have exactly three labelled fields")
    values = dict(rows)
    if len(values) != len(rows):
        raise ValueError("Duplicate candidate field")
    return values


def fixture():
    raw = FIXTURE.read_bytes()
    cases = json.loads(raw)["cases"]
    if (len(cases) != 12 or len({case["id"] for case in cases}) != 12
            or Counter(case["domain"] for case in cases)
            != {"boolean": 3, "sql": 3, "event": 3, "evidence": 3}):
        raise ValueError("Expected 12 unique cases, three in each frozen domain")
    for case in cases:
        clean = fields(case["correct_candidate"])
        wrong = fields(case["wrong_candidate"])
        defect = case["defect"]
        field = defect["field"]
        if (list(clean) != list(wrong) or
                [key for key in clean if clean[key] != wrong[key]] != [field] or
                clean[field] != defect["correct"] or wrong[field] != defect["wrong"]):
            raise ValueError(f"Defect is absent or not isolated: {case['id']}")
    return cases, sha(raw)


def schedule(cases):
    batches = [(case, variant) for case in cases for variant in ("correct", "wrong")]
    random.Random(SEED).shuffle(batches)
    for index, (case, variant) in enumerate(batches):
        offset = index % len(EFFORTS)
        yield case, variant, EFFORTS[offset:] + EFFORTS[:offset]


class SingleCallClient(LLMClient):
    def __init__(self, config, effort):
        super().__init__(config)
        self.effort = effort
        self.receipt = {}
        self.fatal = None

    async def _post_json(self, url, payload, headers, *, timeout_seconds=None):
        if self.receipt:
            self.fatal = "unexpected_repeat_admission"
            raise RuntimeError("Single critic request already used")
        expected = None if self.effort == "provider_default" else self.effort
        expected_keys = {"model", "messages", "max_completion_tokens"}
        if expected:
            expected_keys.add("reasoning_effort")
        if (set(payload) != expected_keys or payload["model"] != MODEL
                or payload["max_completion_tokens"] != CAP
                or payload.get("reasoning_effort") != expected
                or [m["role"] for m in payload["messages"]] != ["system", "user"]
                or not 0 < float(timeout_seconds or 0) <= SECONDS
                or url != "https://api.openai.com/v1/chat/completions"):
            self.fatal = "unexpected_request"
            raise RuntimeError("Critic request does not match the frozen contract")
        self.receipt.update(request=payload, requested_at=now(),
                            http_timeout_seconds=timeout_seconds)
        try:
            response = await super()._post_json(url, payload, headers, timeout_seconds=timeout_seconds)
        except asyncio.CancelledError:
            self.receipt["transport"] = "cancelled"
            raise
        except Exception as exc:
            self.receipt["transport"] = type(exc).__name__
            status = re.match(r"openai returned (\d{3}):", str(exc))
            if status:
                self.receipt["http_status"] = int(status[1])
            if self.receipt.get("http_status") in {400, 401, 403, 404}:
                self.fatal = f"http_{self.receipt['http_status']}"
            # Keep the transport type in the receipt. The runner permits only
            # one admission, including rate-limit or network failures.
            raise RuntimeError("Single critic request failed") from None
        self.receipt.update(transport="returned", response=response)
        return response


async def review(config, case, variant, effort):
    model = RightBrainModel(config)
    client = SingleCallClient(config, effort)
    model._llm_client = client
    clock = asyncio.get_running_loop().time
    start = clock()
    with reasoning_profile(effort):
        task = asyncio.create_task(model.criticise_reasoning(
            "critic-reviewer-only", case["question"], case[f"{variant}_candidate"],
            context=CONTEXT, allow_micro_fallback=False))
    done, _ = await asyncio.wait({task}, timeout=SECONDS)
    elapsed = clock() - start
    if task not in done or elapsed >= SECONDS:
        terminal, result = "owner_deadline", None
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    else:
        terminal, result = "returned", task.result()
    return {"case_id": case["id"], "variant": variant, "effort": effort,
            "terminal": terminal, "latency_ms": round(elapsed * 1000, 2),
            "admission": client.receipt, "verdict": result, "fatal": client.fatal}


def grade(case, row):
    result = row["verdict"]
    valid = row["terminal"] == "returned" and bool(result and result["critic_external_review_valid"])
    status = result["critic_status"] if result is not None else row["terminal"]
    verdict = result["verdict"] if valid else None
    if not valid:
        return status, verdict, "unavailable", None
    if row["variant"] == "correct":
        return status, verdict, "false_issue" if verdict == "issues" else "accepted_correct", None
    if verdict == "ok":
        return status, verdict, "false_ok", None
    defect = case["defect"]
    expected = (defect["field"].casefold(), defect["wrong"], defect["correct"])
    matches = []
    for issue in result["issues"]:
        match = WITNESS.fullmatch(issue.strip())
        if match:
            field, found, correction = (part.strip() for part in match.groups())
            matches.append((field.casefold(), found, correction) == expected)
        else:
            matches.append(False)
    return status, verdict, "detected" if any(matches) else "off_target_issue", sum(not m for m in matches)


def score(cases, receipt):
    by_id = {case["id"]: case for case in cases}
    rows = []
    seen = set()
    for raw in receipt["rows"]:
        case = by_id[raw["case_id"]]
        cell = (case["id"], raw["variant"], raw["effort"])
        if cell in seen or cell[1] not in {"correct", "wrong"} or cell[2] not in EFFORTS:
            raise ValueError(f"Invalid or duplicate review: {cell}")
        seen.add(cell)
        status, verdict, outcome, unmatched = grade(case, raw)
        admission = raw["admission"]
        response = admission.get("response", {})
        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        token_keys = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                      "reasoning_tokens": details.get("reasoning_tokens")}
        tokens = {key: value if type(value) is int and value >= 0 else None
                  for key, value in token_keys.items()}
        rows.append({"case_id": case["id"], "domain": case["domain"], "variant": cell[1], "effort": cell[2],
                     "outcome": outcome, "critic_status": status, "verdict": verdict,
                     "unmatched_issues": unmatched, "terminal": raw["terminal"],
                     "latency_ms": raw["latency_ms"], "admitted": bool(admission),
                     "transport": admission.get("transport"), "http_status": admission.get("http_status"),
                     "response_model": response.get("model") if isinstance(response, dict) else None,
                     **tokens})
    summary = {}
    for effort in EFFORTS:
        group = [row for row in rows if row["effort"] == effort]
        if not group:
            continue
        summary[effort] = {
            variant: dict(Counter(row["outcome"] for row in group if row["variant"] == variant))
            for variant in ("correct", "wrong")
        }
        summary[effort].update(
            median_latency_ms=round(statistics.median(row["latency_ms"] for row in group), 2),
            valid=sum(row["verdict"] is not None for row in group),
            calls=sum(row["admitted"] for row in group),
            missing_usage=sum(row["admitted"] and (row["input_tokens"] is None or row["output_tokens"] is None)
                              for row in group),
            known_input_tokens=sum(row["input_tokens"] or 0 for row in group),
            known_output_tokens=sum(row["output_tokens"] or 0 for row in group),
        )
    return {"schema": "critic-reviewer-only-metrics-v1", "source": receipt["source"],
            "fixture_sha256": receipt["fixture_sha256"], "complete": receipt["complete"],
            "model": MODEL, "deadline_seconds": SECONDS, "completion_cap": CAP,
            "seed": SEED, "planned_cells": len(cases) * 2 * len(EFFORTS),
            "admissions": sum(row["admitted"] for row in rows),
            "summary": summary, "rows": rows}


async def live(cases, fixture_hash, source, path):
    config = LLMConfig("openai", MODEL, os.environ["OPENAI_API_KEY"],
                       api_base="https://api.openai.com/v1", max_output_tokens=CAP,
                       timeout_seconds=SECONDS, auto_continue=False, max_continuations=0)
    os.environ.update(ENV)
    journal = {"source": source, "fixture_sha256": fixture_hash, "started_at": now(),
               "complete": False, "rows": []}
    write_json(path / "journal.json", journal)
    for case, variant, efforts in schedule(cases):
        batch = await asyncio.gather(*(review(config, case, variant, effort) for effort in efforts))
        journal["rows"].extend(batch)
        write_json(path / "journal.json", journal)
        print(f"{len(journal['rows'])}/72 cells recorded", flush=True)
        fatals = [item["fatal"] for item in batch if item["fatal"]]
        if fatals:
            journal["stop_reason"] = fatals[0]
            break
        # Byte-identical inputs and cap across arms, differing only in effort.
        payloads = [item["admission"].get("request") for item in batch]
        if not all(payloads) or len({canonical({k: v for k, v in p.items() if k != "reasoning_effort"})
                                     for p in payloads}) != 1:
            journal["stop_reason"] = "unmatched_payload"
            break
    journal["complete"] = len(journal["rows"]) == 72 and "stop_reason" not in journal
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
    cases, fixture_hash = fixture()
    if args.score:
        journal = json.loads(args.score.read_bytes())
        if journal["fixture_sha256"] != fixture_hash:
            raise SystemExit("Journal does not match the frozen fixture")
        print(json.dumps(score(cases, journal), ensure_ascii=False, indent=2))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not configured")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("Freeze experiment files in a clean commit before running")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    os.umask(0o077)
    args.live.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not asyncio.run(live(cases, fixture_hash, source, args.live)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
