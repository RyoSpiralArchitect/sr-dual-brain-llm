import argparse
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "sr-dual-brain-llm" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_live_llm import _build_ab_command, _llm_env_status, _scope_env_status  # noqa: E402


def test_llm_env_status_accepts_shared_config_for_both_hemispheres():
    status = _llm_env_status(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL_ID": "gpt-test",
            "OPENAI_API_KEY": "sk-test",
        }
    )

    assert status["ready"] is True
    assert status["left"]["configured"] is True
    assert status["right"]["configured"] is True
    assert status["left"]["api_key_source"] == "OPENAI_API_KEY"


def test_llm_env_status_reports_missing_right_scope():
    status = _llm_env_status(
        {
            "LEFT_BRAIN_PROVIDER": "mistral",
            "LEFT_BRAIN_MODEL": "mistral-large",
            "MISTRAL_API_KEY": "key",
        }
    )

    assert status["ready"] is False
    assert status["left"]["configured"] is True
    assert status["right"]["configured"] is False
    assert status["right"]["missing"]


def test_scope_env_status_shared_model_hint_uses_model_id_only():
    status = _scope_env_status(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
        },
        "LLM",
    )

    assert "LLM_MODEL_ID" in status["missing"]
    assert all("LLM_MODEL or" not in item for item in status["missing"])


def test_build_ab_command_threads_live_defaults():
    args = argparse.Namespace(
        questions="questions.json",
        history_limit=20,
        modes="off,on",
        session_prefix="live-test",
        leading_brain="auto",
        executive_mode="off",
        executive_observer_mode="off",
        low_signal_filter="on",
        critic_health_check="on",
        critic_health_attempts=2,
        critic_health_min_successes=1,
        critic_health_retries=1,
        critic_health_timeout=10.0,
        critic_health_rate_limit_backoff=1.0,
        require_critic_health=True,
        only_ids=None,
        only_tags="logic",
        limit=3,
        shuffle=False,
        seed=9,
    )

    cmd = _build_ab_command(
        args,
        output_path=Path("/tmp/out.json"),
        history_path=Path("/tmp/history.jsonl"),
    )

    assert cmd[0] == sys.executable
    assert any(part.endswith("benchmark_system2_ab.py") for part in cmd)
    assert "--modes" in cmd
    assert "off,on" in cmd
    assert "--require-critic-health" in cmd
    assert cmd[cmd.index("--only-tags") + 1] == "logic"
    assert cmd[cmd.index("--limit") + 1] == "3"
