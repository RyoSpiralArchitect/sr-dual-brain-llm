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


def test_criticise_reasoning_micro_critic_handles_percent_of():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-6",
            "What is 20% of 50?",
            "It is 15.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("10" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_percent_of_japanese():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-7",
            "50の20%は？",
            "答えは12です。",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("10" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_percent_off_final_price():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-8",
            "What is 20% off 50?",
            "It is 45.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("40" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_percent_off_japanese():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-9",
            "50の20%引きは？",
            "45円です。",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("40" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_add_percent_total_tax():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-10",
            "Add 8% tax to 50. What is the total?",
            "Total is 52.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("54" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_percent_change_increase():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-11",
            "What is the percentage increase from 50 to 60?",
            "It is 10%.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("20" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_temperature_conversion_c_to_f():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-12",
            "Convert 20°C to Fahrenheit.",
            "It is 70°F.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("68" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_time_unit_minutes_to_seconds():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-13",
            "How many seconds are in 3 minutes?",
            "160 seconds.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("180" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_fraction_to_percent():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-14",
            "What is 3/4 as a percent?",
            "0.75.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("75" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_what_percent_of():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-15",
            "10 is what percent of 50?",
            "It is 10%.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("20" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_percent_adjust_increase():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-16",
            "Increase 50 by 20%.",
            "Result is 55.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("60" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_length_unit_miles_to_km():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-17",
            "Convert 10 miles to km.",
            "It is 10 km.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("16.093" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_mass_unit_kg_to_lb():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-18",
            "Convert 10 kg to lb.",
            "It is 10 lb.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("22.046" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_mass_unit_lb_to_kg():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-19",
            "Convert 10 lb to kg.",
            "It is 10 kg.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("4.5359" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_length_unit_cm_to_inches():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-20",
            "Convert 10 cm to inches.",
            "It is 10 inches.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("3.937" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_length_unit_inches_to_cm():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-21",
            "Convert 10 inches to cm.",
            "It is 10 cm.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("25.4" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_speed_unit_mps_to_kmph():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-22",
            "Convert 10 m/s to km/h.",
            "It is 10 km/h.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("36" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_speed_unit_kmph_to_mps():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-23",
            "Convert 36 km/h to m/s.",
            "It is 36 m/s.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("10" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_mass_metric_kg_to_g():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-24",
            "Convert 2 kg to g.",
            "It is 200 g.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("2000" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_mass_metric_g_to_kg():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-25",
            "Convert 500 g to kg.",
            "It is 5 kg.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("0.5" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_length_metric_cm_to_m():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-26",
            "Convert 250 cm to m.",
            "It is 25 m.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("2.5" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_length_metric_m_to_cm():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-27",
            "Convert 1.2 m to cm.",
            "It is 1.2 cm.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("120" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_speed_unit_mph_to_kmph():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-28",
            "Convert 10 mph to km/h.",
            "It is 10 km/h.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("16.093" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_speed_unit_kmph_to_mph():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-29",
            "Convert 100 km/h to mph.",
            "It is 100 mph.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("62.137" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")


def test_criticise_reasoning_micro_critic_handles_speed_unit_miles_per_hour_phrase():
    model = RightBrainModel()
    model._llm_client = _FailingLLMClient()
    model.llm_config = SimpleNamespace(timeout_seconds=40)

    result = asyncio.run(
        model.criticise_reasoning(
            "qid-30",
            "Convert 60 miles per hour to km/h.",
            "It is 60 km/h.",
        )
    )

    assert result.get("verdict") == "issues"
    issues = result.get("issues")
    assert isinstance(issues, list) and issues
    assert any("96.5606" in str(item) for item in issues)
    assert str(result.get("critic_kind") or "").startswith("micro")
