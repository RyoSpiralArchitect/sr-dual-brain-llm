#!/usr/bin/env python3
"""Run the System2 A/B benchmark with live LLM preflight guardrails.

This is a thin wrapper around ``benchmark_system2_ab.py``.  It keeps the
expensive/networked path explicit by checking the LLM environment first, while
still exposing a dry-run mode that is safe in local/offline CI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

API_KEY_FALLBACKS = (
    "LLM_API_KEY",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
)


def _truthy_env(env: Mapping[str, str], name: str) -> bool:
    return bool(str(env.get(name) or "").strip())


def _provider_api_key_name(provider: str | None) -> str | None:
    provider_norm = str(provider or "").strip().upper()
    if not provider_norm:
        return None
    return f"{provider_norm}_API_KEY"


def _first_present(env: Mapping[str, str], names: list[str]) -> str | None:
    for name in names:
        if _truthy_env(env, name):
            return name
    return None


def _scope_env_status(env: Mapping[str, str], scope: str) -> dict[str, Any]:
    prefix = scope.upper()
    provider = str(env.get(f"{prefix}_PROVIDER") or env.get("LLM_PROVIDER") or "").strip()
    model = str(env.get(f"{prefix}_MODEL") or env.get("LLM_MODEL_ID") or "").strip()

    key_candidates: list[str] = []
    scoped_key = f"{prefix}_API_KEY"
    key_candidates.append(scoped_key)
    provider_key = _provider_api_key_name(provider)
    if provider_key:
        key_candidates.append(provider_key)
    key_candidates.extend(API_KEY_FALLBACKS)
    key_source = _first_present(env, list(dict.fromkeys(key_candidates)))

    missing: list[str] = []
    if not provider:
        missing.append("LLM_PROVIDER" if prefix == "LLM" else f"{prefix}_PROVIDER or LLM_PROVIDER")
    if not model:
        missing.append(
            "LLM_MODEL_ID"
            if prefix == "LLM"
            else f"{prefix}_MODEL or LLM_MODEL_ID"
        )
    if not key_source:
        provider_hint = provider_key or "<PROVIDER>_API_KEY"
        if prefix == "LLM":
            missing.append(f"{provider_hint} or LLM_API_KEY")
        else:
            missing.append(f"{scoped_key} or {provider_hint} or LLM_API_KEY")

    return {
        "scope": prefix,
        "configured": bool(provider and model and key_source),
        "provider": provider or None,
        "model": model or None,
        "api_key_present": bool(key_source),
        "api_key_source": key_source,
        "missing": missing,
    }


def _llm_env_status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env_map = env or os.environ
    left = _scope_env_status(env_map, "LEFT_BRAIN")
    right = _scope_env_status(env_map, "RIGHT_BRAIN")
    executive = _scope_env_status(env_map, "EXECUTIVE")
    director = _scope_env_status(env_map, "DIRECTOR")
    shared = _scope_env_status(env_map, "LLM")
    ready = bool(left["configured"] and right["configured"])
    return {
        "ready": ready,
        "reason": (
            "left/right hemispheres configured"
            if ready
            else "missing live LLM config for one or both hemispheres"
        ),
        "left": left,
        "right": right,
        "executive": executive,
        "director": director,
        "shared": shared,
    }


def _default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "target" / "benchmarks" / f"live_llm_system2_ab_{stamp}.json"


def _default_history_path() -> Path:
    return REPO_ROOT / "target" / "benchmarks" / "live_llm_system2_ab_history.jsonl"


def _build_ab_command(
    args: argparse.Namespace,
    *,
    output_path: Path,
    history_path: Path | None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "benchmark_system2_ab.py"),
        "--questions",
        str(args.questions),
        "--output",
        str(output_path),
        "--history",
        str(history_path) if history_path is not None else "",
        "--history-limit",
        str(args.history_limit),
        "--modes",
        str(args.modes),
        "--session-prefix",
        str(args.session_prefix),
        "--leading-brain",
        str(args.leading_brain),
        "--executive-mode",
        str(args.executive_mode),
        "--executive-observer-mode",
        str(args.executive_observer_mode),
        "--low-signal-filter",
        str(args.low_signal_filter),
        "--critic-health-check",
        str(args.critic_health_check),
        "--critic-health-attempts",
        str(args.critic_health_attempts),
        "--critic-health-retries",
        str(args.critic_health_retries),
        "--critic-health-timeout",
        str(args.critic_health_timeout),
        "--critic-health-rate-limit-backoff",
        str(args.critic_health_rate_limit_backoff),
        "--seed",
        str(args.seed),
    ]
    if args.critic_health_min_successes is not None:
        cmd.extend(["--critic-health-min-successes", str(args.critic_health_min_successes)])
    if args.require_critic_health:
        cmd.append("--require-critic-health")
    if args.only_ids:
        cmd.extend(["--only-ids", str(args.only_ids)])
    if args.only_tags:
        cmd.extend(["--only-tags", str(args.only_tags)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.shuffle:
        cmd.append("--shuffle")
    return cmd


def _build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.timeout_multiplier is not None:
        env["DUALBRAIN_TIMEOUT_MULTIPLIER"] = str(args.timeout_multiplier)
    if args.system2_timeout_multiplier is not None:
        env["DUALBRAIN_SYSTEM2_TIMEOUT_MULTIPLIER"] = str(args.system2_timeout_multiplier)
    if args.callosum_timeout_ms is not None:
        env["DUALBRAIN_CALLOSUM_TIMEOUT_MS"] = str(args.callosum_timeout_ms)
    return env


def _print_status(status: dict[str, Any]) -> None:
    print(json.dumps(status, ensure_ascii=False, indent=2))


def _run(args: argparse.Namespace) -> int:
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path()
    history_path = (
        Path(args.history).expanduser().resolve()
        if args.history is not None and str(args.history).strip()
        else _default_history_path()
    )
    if args.no_history:
        history_path = None

    status = _llm_env_status(os.environ)
    cmd = _build_ab_command(args, output_path=output_path, history_path=history_path)
    plan = {
        "status": status,
        "command": cmd,
        "output": str(output_path),
        "history": str(history_path) if history_path is not None else None,
        "cwd": str(REPO_ROOT),
    }

    if args.check_env_only:
        _print_status(status)
        return 0 if status.get("ready") else 2

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if not status.get("ready") and not args.allow_local_fallback:
        print("[live-bench] live LLM config is incomplete; refusing to spend a benchmark run.")
        print("[live-bench] set LLM_PROVIDER + LLM_MODEL_ID + <PROVIDER>_API_KEY,")
        print("[live-bench] or configure LEFT_BRAIN_* and RIGHT_BRAIN_* separately.")
        print("[live-bench] use --allow-local-fallback only when you intentionally want stub/local models.")
        _print_status(status)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[live-bench] output={output_path}")
    if history_path is not None:
        print(f"[live-bench] history={history_path}")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_build_subprocess_env(args), check=False)
    return int(completed.returncode)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "examples" / "system2_benchmark_questions_en.json"),
        help="Comma separated paths to benchmark question set JSON files.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument(
        "--modes",
        default="off,auto,on",
        help="Comma separated system2 modes to compare.",
    )
    parser.add_argument("--session-prefix", default="live-llm-system2-ab")
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
    parser.add_argument("--critic-health-check", choices=["on", "off"], default="on")
    parser.add_argument("--critic-health-attempts", type=int, default=3)
    parser.add_argument("--critic-health-min-successes", type=int, default=None)
    parser.add_argument("--critic-health-retries", type=int, default=1)
    parser.add_argument("--critic-health-timeout", type=float, default=32.0)
    parser.add_argument("--critic-health-rate-limit-backoff", type=float, default=2.5)
    parser.add_argument("--require-critic-health", action="store_true")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--only-tags", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--system2-timeout-multiplier", type=float, default=None)
    parser.add_argument("--callosum-timeout-ms", type=int, default=None)
    parser.add_argument("--allow-local-fallback", action="store_true")
    parser.add_argument("--check-env-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    raise SystemExit(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
