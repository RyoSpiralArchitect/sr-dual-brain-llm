#!/usr/bin/env python3
"""Prepare local, blinded System2 answer pairs from counterbalanced A/B reports.

This script never calls a model. The reveal key and complete answers stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_report(path: Path, expected_modes: list[str]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), f"{path}: report must be a JSON object")
    config = report.get("config")
    _require(isinstance(config, dict), f"{path}: missing config")
    _require(config.get("modes") == expected_modes, f"{path}: expected mode order {expected_modes}")
    _require(config.get("include_full_answers") is True, f"{path}: complete answers were not saved")
    revision = config.get("source_revision")
    _require(
        isinstance(revision, dict) and bool(revision.get("commit")) and revision.get("dirty") is False,
        f"{path}: a clean, recorded source revision is required",
    )
    provenance = config.get("question_provenance")
    _require(
        isinstance(provenance, dict) and bool(provenance.get("effective_questions_sha256")),
        f"{path}: missing effective question hash",
    )
    _require(isinstance(config.get("effective_llm_by_mode"), dict), f"{path}: missing model configuration")
    modes = report.get("modes")
    _require(isinstance(modes, dict), f"{path}: missing mode records")
    for mode in ("off", "on"):
        _require(isinstance(modes.get(mode), dict), f"{path}: missing {mode} mode")
        _require(isinstance(modes[mode].get("cases"), list), f"{path}: missing {mode} cases")
    return report


def _case_map(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    cases = report["modes"][mode]["cases"]
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        _require(isinstance(case, dict), f"{mode}: case must be an object")
        case_id = str(case.get("id") or "").strip()
        _require(bool(case_id) and case_id not in result, f"{mode}: missing or duplicate case id")
        _require(case.get("system2_mode") == mode, f"{mode}/{case_id}: unexpected effective mode")
        _require(not case.get("error"), f"{mode}/{case_id}: errored case cannot be scored")
        _require(isinstance(case.get("answer"), str) and bool(case["answer"].strip()),
                 f"{mode}/{case_id}: missing complete answer")
        _require(isinstance(case.get("question"), str) and bool(case["question"].strip()),
                 f"{mode}/{case_id}: missing question")
        result[case_id] = case
    return result


def _paired_cases(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    off_cases = _case_map(report, "off")
    on_cases = _case_map(report, "on")
    _require(off_cases.keys() == on_cases.keys(), "off/on case IDs differ")
    pairs = []
    for case_id, off in off_cases.items():
        on = on_cases[case_id]
        _require(off["question"] == on["question"], f"{case_id}: off/on questions differ")
        pairs.append((case_id, off["question"], off, on))
    _require(bool(pairs), "no matched cases")
    return pairs


def build_packets(ab: dict[str, Any], ba: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    ab_config = ab["config"]
    ba_config = ba["config"]
    for key in ("source_revision", "question_provenance", "effective_llm_by_mode"):
        _require(ab_config[key] == ba_config[key], f"AB/BA {key} differs")
    ab_pairs = _paired_cases(ab)
    ba_pairs = _paired_cases(ba)
    _require(
        [(case_id, question) for case_id, question, _, _ in ab_pairs]
        == [(case_id, question) for case_id, question, _, _ in ba_pairs],
        "AB/BA question sets or order differ",
    )

    rng = random.Random(seed)
    raw = [("block_1", row) for row in ab_pairs] + [("block_2", row) for row in ba_pairs]
    rng.shuffle(raw)
    packets = []
    assignments = []
    for index, (block, (case_id, question, off_case, on_case)) in enumerate(raw, 1):
        off_first = bool(rng.getrandbits(1))
        off_answer = off_case["answer"]
        on_answer = on_case["answer"]
        answer_a, answer_b = (off_answer, on_answer) if off_first else (on_answer, off_answer)
        packet_id = f"P{index:04d}"
        packets.append({
            "packet_id": packet_id,
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
        })
        assignments.append({
            "packet_id": packet_id,
            "block": block,
            "case_id": case_id,
            "answer_a_mode": "off" if off_first else "on",
            "answer_b_mode": "on" if off_first else "off",
            "off_system2_enabled": off_case.get("system2_enabled"),
            "on_system2_enabled": on_case.get("system2_enabled"),
        })
    blind = {
        "schema": "system2-blind-pairs-v1",
        "rubric": {
            "correctness": "0=incorrect, 1=partly correct, 2=correct",
            "completeness": "0=misses core request, 1=partial, 2=complete",
            "unsupported_claims": "0=none, 1=minor, 2=material",
            "tie_or_uncertain": "Record ties and uncertainty explicitly; do not force a winner.",
        },
        "packets": packets,
    }
    key = {
        "schema": "system2-blind-key-v1",
        "source_revision": ab_config["source_revision"],
        "question_provenance": ab_config["question_provenance"],
        "run_ids": {"block_1": ab.get("run_id"), "block_2": ba.get("run_id")},
        "seed": seed,
        "assignments": assignments,
    }
    return blind, key


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab", type=Path, required=True, help="Clean off,on report with complete answers")
    parser.add_argument("--ba", type=Path, required=True, help="Clean on,off report with complete answers")
    parser.add_argument("--output-dir", type=Path, required=True, help="New local directory for blind packets and reveal key")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ab_path = args.ab.resolve()
    ba_path = args.ba.resolve()
    _require(ab_path != ba_path, "AB and BA reports must be distinct")
    ab = _load_report(ab_path, ["off", "on"])
    ba = _load_report(ba_path, ["on", "off"])
    blind, key = build_packets(ab, ba, seed=args.seed)
    key["report_sha256"] = {
        "block_1": hashlib.sha256(ab_path.read_bytes()).hexdigest(),
        "block_2": hashlib.sha256(ba_path.read_bytes()).hexdigest(),
    }

    args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_private_json(args.output_dir / "blind_packets.json", blind)
    _write_private_json(args.output_dir / "reveal_key.json", key)
    print(f"Prepared {len(blind['packets'])} local blinded pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
