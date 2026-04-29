#!/usr/bin/env python3
"""Benchmark latent motif handling and plain-answer leakage for the unconscious field."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_system2 import _filter_questions, _load_questions, _parse_question_paths  # noqa: E402
from engine_stdio import EngineSession, _extract_metrics  # noqa: E402

INTERNAL_SECTION_MARKERS = (
    "[Unconscious Insight]",
    "[Stress Released]",
    "[Psychoid Attention Bias]",
    "[Psychoid Field Alignment]",
    "[Psychoid Signifiers]",
    "[Default Mode Reflection]",
    "[Coherence Integration]",
    "[Unconscious Linguistic Fabric]",
    "[Linguistic Motifs]",
    "[Hemisphere Routing]",
    "[Hemisphere Semantic Tilt]",
    "[Collaboration Profile]",
    "[Neural Impulse Activity]",
    "[Architecture Path]",
)


def _last_event(events: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for ev in reversed(events or []):
        if ev.get("event") == name:
            return ev
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return float(count) / float(total)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _internal_leak_markers(answer: str) -> List[str]:
    lowered = str(answer or "").lower()
    return [marker for marker in INTERNAL_SECTION_MARKERS if marker.lower() in lowered]


def _latent_grounding_score(
    *,
    top_k_count: int,
    emergent_count: int,
    psychoid_resonance: Optional[float],
    weave_score: Optional[float],
    motif_score: Optional[float],
) -> float:
    top_component = min(1.0, max(0.0, float(top_k_count) / 3.0))
    emergent_component = min(1.0, max(0.0, float(emergent_count) / 2.0))
    resonance_component = max(0.0, min(1.0, psychoid_resonance or 0.0))
    weave_component = max(0.0, min(1.0, weave_score or 0.0))
    motif_component = max(0.0, min(1.0, motif_score or 0.0))
    return float(
        round(
            0.2 * top_component
            + 0.2 * emergent_component
            + 0.2 * resonance_component
            + 0.25 * weave_component
            + 0.15 * motif_component,
            6,
        )
    )


def _extract_case_signals(
    events: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    answer: str,
) -> Dict[str, Any]:
    unconscious_summary = _as_dict(_last_event(events, "unconscious_field").get("summary"))
    psychoid_event_signal = _as_dict(_last_event(events, "psychoid_signal").get("signal"))
    psychoid_signal = psychoid_event_signal or _as_dict(unconscious_summary.get("psychoid_signal"))
    projection = _as_dict(_last_event(events, "psychoid_attention_projection").get("projection"))
    dmn = _last_event(events, "default_mode_reflection")
    task_positive = _last_event(events, "task_positive_network")
    coherence_event_signal = _as_dict(_last_event(events, "coherence_signal").get("signal"))
    coherence_metrics = _as_dict(metrics.get("coherence"))
    coherence = coherence_event_signal or coherence_metrics
    weave = _as_dict(_last_event(events, "coherence_unconscious_weave").get("fabric"))
    if not weave:
        weave = _as_dict(coherence.get("unconscious"))
    motifs = _as_dict(_last_event(events, "coherence_linguistic_motifs").get("motifs"))
    if not motifs:
        motifs = _as_dict(coherence.get("motifs"))

    top_k = [str(item) for item in _as_list(unconscious_summary.get("top_k"))]
    emergent_ideas = _as_list(unconscious_summary.get("emergent_ideas"))
    unconscious_motifs = [str(item) for item in _as_list(unconscious_summary.get("motifs"))]
    attention_bias = _as_list(psychoid_signal.get("attention_bias"))
    signifier_chain = [str(item) for item in _as_list(psychoid_signal.get("signifier_chain"))]
    reflections = _as_list(dmn.get("reflections"))

    psychoid_resonance = _safe_float(psychoid_signal.get("resonance"))
    psychoid_tension = _safe_float(psychoid_signal.get("psychoid_tension"))
    weave_score = _safe_float(weave.get("score"))
    motif_score = _safe_float(motifs.get("score"))
    leak_markers = _internal_leak_markers(answer)
    grounding_score = _latent_grounding_score(
        top_k_count=len(top_k),
        emergent_count=len(emergent_ideas),
        psychoid_resonance=psychoid_resonance,
        weave_score=weave_score,
        motif_score=motif_score,
    )

    return {
        "unconscious": {
            "top_k": top_k,
            "top_count": len(top_k),
            "emergent_ideas": len(emergent_ideas),
            "stress_released": _safe_float(unconscious_summary.get("stress_released")),
            "cache_depth": _safe_int(unconscious_summary.get("cache_depth")),
            "motifs": unconscious_motifs,
            "score": grounding_score,
        },
        "psychoid": {
            "present": bool(psychoid_signal),
            "attention_bias_count": len(attention_bias),
            "signifier_chain_len": len(signifier_chain),
            "resonance": psychoid_resonance,
            "tension": psychoid_tension,
            "projection_norm": _safe_float(projection.get("norm")),
            "projection_chain_length": _safe_int(projection.get("chain_length")),
        },
        "default_mode": {
            "reflection_count": len(reflections),
            "suppressed": bool(task_positive.get("suppressed")),
            "task_positive_load": _safe_float(task_positive.get("load")),
            "task_positive_mode": task_positive.get("mode"),
        },
        "coherence": {
            "combined": _safe_float(coherence.get("combined")),
            "tension": _safe_float(coherence.get("tension")),
            "mode": coherence.get("mode"),
            "linguistic_depth": _safe_float(coherence.get("linguistic_depth")),
            "unconscious_weave_score": weave_score,
            "unconscious_weave_emergent_count": _safe_int(weave.get("emergent_count")),
            "motif_score": motif_score,
            "motif_repeated_loops": _safe_int(motifs.get("repeated_loops")),
        },
        "leakage": {
            "has_internal_leak": bool(leak_markers),
            "markers": leak_markers,
        },
    }


def _summarise_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_cases = [case for case in cases if not case.get("error")]
    total = len(cases)
    ok_total = len(ok_cases)

    def floats(path: tuple[str, ...]) -> List[float]:
        out: List[float] = []
        for case in ok_cases:
            value: Any = case
            for part in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
            number = _safe_float(value)
            if number is not None:
                out.append(number)
        return out

    leak_cases = sum(1 for case in ok_cases if case.get("leakage", {}).get("has_internal_leak"))
    psychoid_cases = sum(1 for case in ok_cases if case.get("psychoid", {}).get("present"))
    default_mode_cases = sum(
        1 for case in ok_cases if (case.get("default_mode", {}).get("reflection_count") or 0) > 0
    )
    suppressed_cases = sum(1 for case in ok_cases if case.get("default_mode", {}).get("suppressed"))
    emergent_cases = sum(
        1 for case in ok_cases if (case.get("unconscious", {}).get("emergent_ideas") or 0) > 0
    )
    weave_cases = sum(
        1
        for case in ok_cases
        if case.get("coherence", {}).get("unconscious_weave_score") is not None
    )
    motif_cases = sum(
        1 for case in ok_cases if case.get("coherence", {}).get("motif_score") is not None
    )

    return {
        "total_cases": total,
        "ok_cases": ok_total,
        "error_cases": total - ok_total,
        "leak_cases": leak_cases,
        "leak_rate": _rate(leak_cases, ok_total),
        "psychoid_signal_rate": _rate(psychoid_cases, ok_total),
        "default_mode_reflection_rate": _rate(default_mode_cases, ok_total),
        "default_mode_suppressed_rate": _rate(suppressed_cases, ok_total),
        "emergent_idea_rate": _rate(emergent_cases, ok_total),
        "unconscious_weave_rate": _rate(weave_cases, ok_total),
        "linguistic_motif_rate": _rate(motif_cases, ok_total),
        "avg_unconscious_score": _mean(floats(("unconscious", "score"))),
        "avg_coherence_combined": _mean(floats(("coherence", "combined"))),
        "avg_coherence_tension": _mean(floats(("coherence", "tension"))),
        "avg_weave_score": _mean(floats(("coherence", "unconscious_weave_score"))),
        "avg_motif_score": _mean(floats(("coherence", "motif_score"))),
        "avg_psychoid_resonance": _mean(floats(("psychoid", "resonance"))),
        "avg_psychoid_tension": _mean(floats(("psychoid", "tension"))),
        "avg_task_positive_load": _mean(floats(("default_mode", "task_positive_load"))),
        "avg_latency_ms": _mean(floats(("latency_ms",))),
    }


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


async def _run_case(
    *,
    session: EngineSession,
    question_entry: Dict[str, Any],
    index: int,
    run_id: str,
    answer_mode: str,
    leading_brain: str,
    system2_mode: str,
    executive_mode: str,
    executive_observer_mode: str,
) -> Dict[str, Any]:
    qid = f"{run_id}-c{index:03d}"
    question = str(question_entry.get("question") or "")
    session.telemetry.clear()
    started = time.perf_counter()
    error_text: Optional[str] = None
    answer = ""
    try:
        answer = await session.controller.process(
            question,
            qid=qid,
            answer_mode=answer_mode,
            leading_brain=(None if leading_brain == "auto" else leading_brain),
            system2_mode=system2_mode,
            executive_mode=executive_mode,
            executive_observer_mode=executive_observer_mode,
        )
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    events = list(session.telemetry.events)
    metrics = _extract_metrics(events) if events else {}
    signals = _extract_case_signals(events, metrics, answer)
    return {
        "index": index,
        "id": question_entry.get("id") or f"q{index:03d}",
        "qid": qid,
        "question": question,
        "tags": question_entry.get("tags") if isinstance(question_entry.get("tags"), list) else [],
        "answer_mode": answer_mode,
        "system2_mode": system2_mode,
        "latency_ms": _safe_float(metrics.get("latency_ms")) or elapsed_ms,
        "telemetry_events": len(events),
        "error": error_text,
        "answer_preview": answer[:280] if answer else "",
        **signals,
    }


async def _run(args: argparse.Namespace) -> int:
    question_paths = _parse_question_paths(args.questions)
    if not question_paths:
        raise ValueError("No --questions paths provided.")
    missing_paths = [path for path in question_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Questions file(s) not found: {paths}".format(
                paths=", ".join(str(path) for path in missing_paths)
            )
        )

    questions: List[Dict[str, Any]] = []
    for path in question_paths:
        questions.extend(_load_questions(path))
    questions = _filter_questions(
        questions,
        only_ids=getattr(args, "only_ids", None),
        only_tags=getattr(args, "only_tags", None),
    )
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(questions)
    if args.limit is not None and args.limit > 0:
        questions = questions[: int(args.limit)]
    if not questions:
        raise RuntimeError("No benchmark questions to run.")

    low_signal_filter = str(args.low_signal_filter or "on").strip().lower()
    os.environ["DUALBRAIN_SYSTEM2_LOW_SIGNAL_FILTER"] = "1" if low_signal_filter == "on" else "0"

    run_id = time.strftime("unconscious_creativity_%Y%m%d_%H%M%S")
    session_id = str(args.session_id or run_id).strip() or run_id
    question_set_label = ",".join(str(path) for path in question_paths)
    print(f"[unconscious-bench] run_id={run_id}")
    print(f"[unconscious-bench] session_id={session_id}")
    print(f"[unconscious-bench] questions={len(questions)}")

    session = await EngineSession.create(session_id=session_id)
    try:
        llm_capable = bool(
            getattr(session.left, "uses_external_llm", False)
            and getattr(session.right, "uses_external_llm", False)
        )
        if not llm_capable:
            print("[unconscious-bench] warning: external LLM not configured for both hemispheres.")

        cases: List[Dict[str, Any]] = []
        for idx, entry in enumerate(questions, 1):
            case = await _run_case(
                session=session,
                question_entry=entry,
                index=idx,
                run_id=run_id,
                answer_mode=args.answer_mode,
                leading_brain=args.leading_brain,
                system2_mode=args.system2_mode,
                executive_mode=args.executive_mode,
                executive_observer_mode=args.executive_observer_mode,
            )
            cases.append(case)
            print(
                "[unconscious-bench] {idx:03d}/{total} id={id} latent={latent} "
                "coh={coh} weave={weave} motif={motif} leak={leak} error={error}".format(
                    idx=idx,
                    total=len(questions),
                    id=case.get("id"),
                    latent=case.get("unconscious", {}).get("score"),
                    coh=case.get("coherence", {}).get("combined"),
                    weave=case.get("coherence", {}).get("unconscious_weave_score"),
                    motif=case.get("coherence", {}).get("motif_score"),
                    leak=case.get("leakage", {}).get("has_internal_leak"),
                    error=("yes" if case.get("error") else "no"),
                )
            )

        summary = _summarise_cases(cases)
        report = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question_set": question_set_label,
            "question_sets": [str(path) for path in question_paths],
            "config": {
                "session_id": session_id,
                "answer_mode": args.answer_mode,
                "leading_brain": args.leading_brain,
                "system2_mode": args.system2_mode,
                "executive_mode": args.executive_mode,
                "executive_observer_mode": args.executive_observer_mode,
                "low_signal_filter": low_signal_filter,
                "question_count": len(questions),
                "only_ids": getattr(args, "only_ids", None),
                "only_tags": getattr(args, "only_tags", None),
                "llm_capable": llm_capable,
            },
            "summary": summary,
            "cases": cases,
        }

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[unconscious-bench] wrote report: {output_path}")

        history_path = Path(args.history).expanduser().resolve() if args.history else None
        if history_path is not None:
            _append_history(
                history_path,
                {
                    "run_id": run_id,
                    "timestamp": report["timestamp"],
                    "question_set": question_set_label,
                    "summary": summary,
                    "config": report["config"],
                },
            )
            print(f"[unconscious-bench] appended history: {history_path}")

        print(
            "[unconscious-bench] summary ok={ok}/{total} leaks={leaks} "
            "latent={latent} coherence={coherence} weave={weave} motif={motif}".format(
                ok=summary.get("ok_cases"),
                total=summary.get("total_cases"),
                leaks=summary.get("leak_cases"),
                latent=summary.get("avg_unconscious_score"),
                coherence=summary.get("avg_coherence_combined"),
                weave=summary.get("avg_weave_score"),
                motif=summary.get("avg_motif_score"),
            )
        )
        if args.expect_no_leaks and summary.get("leak_cases"):
            print("[unconscious-bench] internal debug markers leaked into answers.")
            return 3
    finally:
        await session.close()
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "examples" / "unconscious_creativity_benchmark_questions.json"),
        help="Comma separated paths to unconscious creativity benchmark question sets.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_creativity_last.json"),
    )
    parser.add_argument(
        "--history",
        default=str(REPO_ROOT / "target" / "benchmarks" / "unconscious_creativity_history.jsonl"),
        help="Append compact run summary to JSONL history (set empty to disable).",
    )
    parser.add_argument("--session-id", default="")
    parser.add_argument("--answer-mode", choices=["plain", "debug", "annotated", "meta"], default="plain")
    parser.add_argument("--system2-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--leading-brain", choices=["auto", "left", "right"], default="auto")
    parser.add_argument(
        "--executive-mode",
        choices=["off", "observe", "assist", "polish"],
        default="off",
    )
    parser.add_argument(
        "--executive-observer-mode",
        choices=["off", "metrics", "director", "both"],
        default="off",
    )
    parser.add_argument("--low-signal-filter", choices=["on", "off"], default="on")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--only-tags", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expect-no-leaks", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.history is not None and str(args.history).strip() == "":
        args.history = None
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
