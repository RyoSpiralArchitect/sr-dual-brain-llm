from core.anterior_cingulate import ConflictSignal
from core.cerebellum import Cerebellum
from core.micro_critic import MicroCriticResult


def _micro(*, issues: list[str], domain: str = "arithmetic", confidence: float = 0.95) -> MicroCriticResult:
    return MicroCriticResult(
        verdict="issues" if issues else "ok",
        issues=list(issues),
        fixes=["fix it"] if issues else [],
        critic_sum=f"[micro:{domain}] Issues:\n- " + "\n- ".join(issues) if issues else f"[micro:{domain}] ok",
        confidence_r=confidence,
        domain=domain,
    )


def test_cerebellum_forecast_prefers_micro_correction_for_deterministic_issue():
    cerebellum = Cerebellum()
    micro = _micro(issues=["Arithmetic result looks incorrect."])
    conflict = ConflictSignal(
        conflict_level=0.34,
        effort_level=0.20,
        uncertainty=0.08,
        adaptation_signal=0.28,
        recommended_control="monitor",
    )

    forecast = cerebellum.forecast(micro, conflict=conflict)

    assert forecast.recommended_path == "micro_correct"
    assert forecast.predicted_gain >= 0.55
    notes = cerebellum.build_internal_notes(micro, forecast=forecast)
    assert "Cerebellar micro-correction" in notes
    assert "[forward-model]" in notes


def test_cerebellum_forecast_defers_when_consult_already_planned():
    cerebellum = Cerebellum()
    micro = _micro(issues=["Arithmetic result looks incorrect."])
    conflict = ConflictSignal(
        conflict_level=0.76,
        effort_level=0.48,
        uncertainty=0.22,
        adaptation_signal=0.68,
        recommended_control="consult",
    )

    forecast = cerebellum.forecast(
        micro,
        conflict=conflict,
        consult_planned=True,
    )

    assert forecast.recommended_path == "defer"
    assert forecast.residual_risk >= 0.0
