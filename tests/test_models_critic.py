import asyncio
from types import SimpleNamespace

from core.models import RightBrainModel


class _FailingLLMClient:
    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("simulated provider failure")


class _UnstructuredLLMClient:
    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "I have concerns, but this is not JSON."


class _PythonDictLLMClient:
    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "{'verdict': 'ok', 'issues': [], 'fixes': []}"


def test_criticise_reasoning_returns_fallback_issue_when_provider_fails():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning("qid-1", "Question", "Draft")
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert "unavailable" in str(issues[0]).lower()


def test_criticise_reasoning_returns_parse_fallback_for_unstructured_output():
    model = RightBrainModel()
    model._llm_client = _UnstructuredLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning("qid-2", "Question", "Draft")
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert "unstructured" in str(issues[0]).lower()


def test_criticise_reasoning_parses_python_dict_style_json():
    model = RightBrainModel()
    model._llm_client = _PythonDictLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning("qid-3", "Question", "Draft")
    )

    assert result.get("verdict") == "ok"
    assert result.get("issues") == []
    assert result.get("fixes") == []


def test_criticise_reasoning_uses_micro_critic_on_provider_failure_for_supported_numeric_tasks():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-4",
            "Compute ((27 * 14) - 96) / 6 and briefly show each arithmetic step.",
            "Result is 100.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("47" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_short_backtick_arithmetic():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-5",
            "Compute `2+2`.",
            "Result is 5.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("4" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_can_disable_micro_critic_fallback():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-5",
            "Compute ((27 * 14) - 96) / 6.",
            "Result is 100.",
            allow_micro_fallback=False,
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert "unavailable" in str(issues[0]).lower()
