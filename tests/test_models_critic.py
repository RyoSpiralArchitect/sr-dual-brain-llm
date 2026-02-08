import asyncio
from types import SimpleNamespace

from core.models import RightBrainModel


class _FailingLLMClient:
    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("simulated provider failure")


class _UnstructuredLLMClient:
    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "I have concerns, but this is not JSON."


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
