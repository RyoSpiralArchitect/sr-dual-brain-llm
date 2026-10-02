#!/usr/bin/env python3
"""Score fixed-answer System2 cases without asking the evaluated model to judge itself.

This intentionally covers only seven questions with unambiguous reference
answers. Unknown wording is indeterminate, never silently counted as wrong.
Complete answers remain in the input reports; the output contains no answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCORED_IDS = (
    "arith_chain_001", "algebra_001", "probability_001", "bayes_001",
    "unit_consistency_001", "table_inference_001", "error_analysis_001",
)


def _explicit_number(text: str, pattern: str, expected: float, tolerance: float) -> str:
    matches = list(re.finditer(pattern, text, flags=re.I | re.S))
    if not matches:
        return "indeterminate"
    value = float(matches[-1].group(1))
    return "correct" if abs(value - expected) <= tolerance else "incorrect"


def score_answer(case_id: str, answer: str) -> str:
    """Return correct/incorrect/indeterminate/unscored for an explicit final answer."""
    if case_id not in SCORED_IDS:
        return "unscored"
    text = str(answer or "").replace(r"\%", "%")
    if not text.strip():
        return "indeterminate"
    # The last boxed expression usually carries the final value; retain the
    # following braces so LaTeX fractions can still be parsed.
    boxed = text.rfind(r"\boxed{")
    answer_marker = list(re.finditer(r"\b(?:final answer|answer|result)\s*[:：]", text, re.I))
    focus_at = max(boxed, answer_marker[-1].end() if answer_marker else -1)
    focus = text[focus_at:] if focus_at >= 0 else text
    if case_id == "arith_chain_001":
        return _explicit_number(focus, r"(?:^|=|:|\{)\s*(-?\d+(?:\.\d+)?)\b", 47, 0.001)
    if case_id == "algebra_001":
        return _explicit_number(text, r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\b", 12, 0.001)
    if case_id == "probability_001":
        fraction = list(re.finditer(
            r"(?:\\(?:dfrac|tfrac|frac)\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}|(?<!\d)(\d+)\s*/\s*(\d+))",
            focus,
        ))
        if fraction:
            match = fraction[-1]
            numerator = int(match.group(1) or match.group(3))
            denominator = int(match.group(2) or match.group(4))
            if denominator == 0:
                return "indeterminate"
            return "correct" if abs(numerator / denominator - 1 / 11) < 0.001 else "incorrect"
        decimal = list(re.finditer(r"\b(0\.\d+|\d+(?:\.\d+)?)\s*(%)?", focus))
        if decimal:
            match = decimal[-1]
            value = float(match.group(1)) / (100 if match.group(2) else 1)
            return "correct" if abs(value - 1 / 11) < 0.002 else "incorrect"
        return "indeterminate"
    if case_id == "bayes_001":
        if boxed >= 0:
            result = list(re.finditer(r"\b(0\.\d+|\d+(?:\.\d+)?)\s*(%)?", focus))
            if not result:
                return "indeterminate"
            match = result[-1]
            number, percent = match.group(1), match.group(2)
        else:
            normalized = text.replace(r"\mid", "|").replace(r"\approx", "≈")
            direct = list(re.finditer(
                r"(?:posterior(?: probability)?|P\s*\(\s*(?:D|disease)\s*\|\s*\+\s*\))\s*(?:is|=|≈|:)\s*(0\.\d+|\d+(?:\.\d+)?)\s*(%)?",
                normalized, re.I,
            ))
            if not direct:
                return "indeterminate"
            number, percent = direct[-1].group(1), direct[-1].group(2)
        value = float(number) / (100 if percent else 1)
        return "correct" if abs(value - (0.95 * 0.02 / (0.95 * 0.02 + 0.1 * 0.98))) < 0.003 else "incorrect"
    if case_id == "unit_consistency_001":
        matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(?:\\text\{\s*)?km\s*/\s*h", focus, re.I))
        if matches:
            return "correct" if abs(float(matches[0].group(1)) - 200 / 3.5) < 0.5 else "incorrect"
        return "indeterminate"
    if case_id == "table_inference_001":
        matches = list(re.finditer(
            r"(?:variant\s*)?([ABC])\s+(?:yields|has|produces|gives|is)\s+(?:the\s+)?(?:most|largest|highest)",
            text, re.I,
        ))
        if not matches:
            return "indeterminate"
        return "correct" if matches[-1].group(1).upper() == "A" else "incorrect"
    if case_id == "error_analysis_001":
        # The prompt itself contains the wrong 40%; a response may explicitly
        # say "not 40%" after giving the corrected 20%.
        final_claim = list(re.finditer(
            r"(?:growth\s+is|grew\s+by|correct(?:ed)?\s+(?:growth|percentage)\s+is)\s*(\d+(?:\.\d+)?)\s*%",
            text, re.I,
        ))
        if final_claim:
            return "correct" if abs(float(final_claim[-1].group(1)) - 20) < 0.1 else "incorrect"
        if re.search(r"(?<!\d)20(?:\.0+)?\s*(?:\\)?%", text):
            return "correct"
        if re.search(r"(?<!\d)40(?:\.0+)?\s*(?:\\)?%", text):
            return "incorrect"
        return "indeterminate"
    raise AssertionError(case_id)


def _read(path: Path, order: list[str]) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("config", {}).get("modes") != order:
        raise ValueError(f"{path}: expected mode order {order}")
    if report["config"].get("include_full_answers") is not True:
        raise ValueError(f"{path}: complete answers are required")
    if report["config"].get("source_revision", {}).get("dirty") is not False:
        raise ValueError(f"{path}: source revision must be clean")
    return report


def score_pair(ab: dict[str, Any], ba: dict[str, Any]) -> dict[str, Any]:
    config_ab, config_ba = ab["config"], ba["config"]
    if {key: value for key, value in config_ab.items() if key != "modes"} != {
        key: value for key, value in config_ba.items() if key != "modes"
    }:
        raise ValueError("benchmark configuration differs between blocks")
    rows: list[dict[str, Any]] = []
    block_questions: dict[str, str] = {}
    for block, report in (("ab", ab), ("ba", ba)):
        by_mode = {}
        for mode in ("off", "on"):
            cases = report["modes"][mode]["cases"]
            by_mode[mode] = {str(case["id"]): case for case in cases}
            if len(by_mode[mode]) != len(cases):
                raise ValueError(f"{block}/{mode}: duplicate case IDs")
        if list(by_mode["off"]) != list(by_mode["on"]):
            raise ValueError(f"{block}: case IDs differ between modes")
        if block == "ba" and list(by_mode["off"]) != block_ids:
            raise ValueError("case IDs differ between blocks")
        if block == "ab":
            block_ids = list(by_mode["off"])
        for case_id in sorted(by_mode["off"]):
            row = {"block": block, "id": case_id}
            if by_mode["off"][case_id].get("question") != by_mode["on"][case_id].get("question"):
                raise ValueError(f"{block}/{case_id}: questions differ between modes")
            if block == "ba" and by_mode["off"][case_id].get("question") != block_questions[case_id]:
                raise ValueError(f"{case_id}: questions differ between blocks")
            if block == "ab":
                block_questions[case_id] = by_mode["off"][case_id].get("question")
            for mode in ("off", "on"):
                case = by_mode[mode][case_id]
                if case.get("system2_mode") != mode:
                    raise ValueError(f"{block}/{mode}/{case_id}: mode override")
                if case.get("error") or not case.get("answer"):
                    status = "unavailable"
                else:
                    status = score_answer(case_id, case["answer"])
                row[mode] = status
                row[f"{mode}_critic_validity"] = case.get("critic_validity")
            rows.append(row)
    scored = [r for r in rows if r["off"] in {"correct", "incorrect"} and r["on"] in {"correct", "incorrect"}]
    return {
        "method": "fixed_reference_v1",
        "scope_ids": list(SCORED_IDS),
        "source_revision": config_ab["source_revision"],
        "effective_questions_sha256": config_ab["question_provenance"]["effective_questions_sha256"],
        "rows": rows,
        "summary": {
            "paired_scored": len(scored),
            "on_wins": sum(r["on"] == "correct" and r["off"] == "incorrect" for r in scored),
            "off_wins": sum(r["off"] == "correct" and r["on"] == "incorrect" for r in scored),
            "ties_correct": sum(r["off"] == r["on"] == "correct" for r in scored),
            "ties_incorrect": sum(r["off"] == r["on"] == "incorrect" for r in scored),
            "indeterminate_or_unavailable": sum(
                r["id"] in SCORED_IDS and r not in scored for r in rows
            ),
            "unscored_open_ended": sum(r["id"] not in SCORED_IDS for r in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", type=Path, required=True)
    parser.add_argument("--ba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_pair(_read(args.ab, ["off", "on"]), _read(args.ba, ["on", "off"]))
    report["inputs_sha256"] = {
        "ab": hashlib.sha256(args.ab.read_bytes()).hexdigest(),
        "ba": hashlib.sha256(args.ba.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
