#!/usr/bin/env python3
"""Run a check command without inherited LLM/PostgreSQL configuration.

Only the child environment changes. Proxy, TLS and GitHub authentication remain
available; the caller's credentials are neither displayed nor modified.
"""
import os
import subprocess
import sys


def offline_environment(source):
    scopes = ("LLM_", "LEFT_BRAIN_", "RIGHT_BRAIN_", "EXECUTIVE_", "DIRECTOR_")
    provider_secrets = {"OPENAI_API_KEY", "OPENAI_ORGANIZATION", "ANTHROPIC_API_KEY",
                        "GOOGLE_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY",
                        "HUGGINGFACE_API_TOKEN", "HF_TOKEN", "DUALBRAIN_PG_DSN"}
    return {name: value for name, value in source.items()
            if not name.startswith(scopes) and name not in provider_secrets}


def main():
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Usage: run_offline_checks.py -- COMMAND [ARGS...]")
    raise SystemExit(subprocess.run(command, env=offline_environment(os.environ)).returncode)


if __name__ == "__main__":
    main()
