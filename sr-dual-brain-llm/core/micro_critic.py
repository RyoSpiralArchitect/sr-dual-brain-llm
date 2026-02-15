# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

"""Deterministic micro-critic for basic reasoning sanity checks.

This module is intentionally conservative: it only emits issues when it can
compute a high-confidence expected result (e.g., arithmetic, simple probability).
It is designed as a safety net when the external critic is unavailable or
unstructured.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Optional


_FLOAT_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def _close(a: float, b: float, *, tol_abs: float = 1e-3, tol_rel: float = 8e-3) -> bool:
    return abs(a - b) <= max(tol_abs, tol_rel * max(1.0, abs(b)))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _numeric_candidates(text: str) -> list[float]:
    raw = str(text or "")
    out: list[float] = []
    for m in _FRACTION_RE.finditer(raw):
        num = _safe_float(m.group(1))
        den = _safe_float(m.group(2))
        if num is None or den is None or den == 0:
            continue
        out.append(num / den)
    for m in _PERCENT_RE.finditer(raw):
        pct = _safe_float(m.group(1))
        if pct is None:
            continue
        out.append(pct)
        out.append(pct / 100.0)
    for m in _FLOAT_RE.finditer(raw):
        val = _safe_float(m.group(0))
        if val is None:
            continue
        out.append(val)
    # De-dupe with a small epsilon bucket.
    uniq: list[float] = []
    for v in out:
        if any(_close(v, u, tol_abs=1e-9, tol_rel=1e-9) for u in uniq):
            continue
        uniq.append(v)
    return uniq


def _draft_mentions_value(draft: str, expected: float) -> bool:
    for candidate in _numeric_candidates(draft):
        if _close(candidate, expected):
            return True
    return False


def _eval_arith_expr(expr: str) -> Optional[float]:
    expr = str(expr or "").strip()
    if not expr:
        return None
    # LLMs sometimes use caret for exponentiation.
    expr = expr.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError
                return left // right
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError
                return left % right
            if isinstance(node.op, ast.Pow):
                # Avoid extreme exponents.
                if abs(right) > 12:
                    raise ValueError("Exponent too large")
                return left**right
        raise ValueError(f"Unsupported expression node: {node.__class__.__name__}")

    try:
        value = _eval(tree.body)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _linear_coeff(expr: str) -> Optional[tuple[float, float]]:
    """Parse a simple linear expression in x into (a, b) for a*x + b.

    Supports forms like: 3x + 7, 2*x-19, -x+4, x-1.
    """

    raw = str(expr or "")
    if not raw.strip():
        return None
    text = raw.strip().lower()
    text = text.replace(" ", "")
    if "x" not in text:
        # Constant-only expression.
        val = _eval_arith_expr(text)
        return (0.0, float(val)) if val is not None else None

    # Normalise implicit multiplication: 3x -> 3*x, -2x -> -2*x.
    text = re.sub(r"(\d)(x)", r"\1*\2", text)
    text = re.sub(r"(x)(\d)", r"\1*\2", text)

    # Split into terms by '+' and '-' while keeping signs.
    if not text.startswith(("+", "-")):
        text = "+" + text
    terms = re.findall(r"[+\-][^+\-]+", text)
    if not terms:
        return None
    a = 0.0
    b = 0.0
    for term in terms:
        sign = -1.0 if term.startswith("-") else 1.0
        body = term[1:]
        if not body:
            continue
        if "x" in body:
            coeff_part = body.replace("x", "")
            if coeff_part.endswith("*"):
                coeff_part = coeff_part[:-1]
            if coeff_part in ("", "*"):
                coeff = 1.0
            else:
                coeff_val = _eval_arith_expr(coeff_part)
                if coeff_val is None:
                    return None
                coeff = float(coeff_val)
            a += sign * coeff
        else:
            val = _eval_arith_expr(body)
            if val is None:
                return None
            b += sign * float(val)
    return a, b


@dataclass(frozen=True)
class MicroCriticResult:
    verdict: str
    issues: list[str]
    fixes: list[str]
    critic_sum: str
    confidence_r: float
    domain: str


def _critic_payload(*, domain: str, issues: list[str], fixes: list[str], confidence: float) -> MicroCriticResult:
    lines: list[str] = []
    if issues:
        lines.append(f"[micro:{domain}] Issues:")
        lines.extend([f"- {item}" for item in issues[:8]])
    if fixes:
        lines.append(f"[micro:{domain}] Fixes:")
        lines.extend([f"- {item}" for item in fixes[:8]])
    return MicroCriticResult(
        verdict="issues" if issues else "ok",
        issues=issues,
        fixes=fixes,
        critic_sum="\n".join(lines).strip() or f"[micro:{domain}] ok",
        confidence_r=float(max(0.0, min(1.0, confidence))),
        domain=domain,
    )


def _micro_arithmetic(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    # Prefer expressions in backticks.
    expr = None
    m = re.search(r"`([^`]{3,160})`", q)
    if m:
        expr = m.group(1)
    if expr is None:
        # Find a parenthesized/operator-heavy substring.
        candidates = re.findall(r"[\d\(\)\s\+\-\*\/\.\^]{3,160}", q)
        for cand in candidates:
            if re.search(r"\d", cand) and re.search(r"[+\-\*\/\^]", cand):
                expr = cand.strip()
                break
    if expr is None:
        return None
    expected = _eval_arith_expr(expr)
    if expected is None:
        return None
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="arithmetic", issues=[], fixes=[], confidence=0.85)
    issues = [
        f"Arithmetic result looks incorrect or missing; expected approximately {expected:g}."
    ]
    fixes = [
        f"Recompute {expr.strip()} carefully; final result should be {expected:g}."
    ]
    return _critic_payload(domain="arithmetic", issues=issues, fixes=fixes, confidence=0.95)


def _micro_linear_equation(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    if "solve for x" not in q.lower():
        return None
    m = re.search(r"solve for x\s*[:：]\s*([^,\n]+)", q, flags=re.IGNORECASE)
    equation = m.group(1).strip() if m else ""
    if "=" not in equation:
        # Fall back: take the first '=' equation-looking fragment.
        m2 = re.search(r"([^\n]+=[^\n]+)", q)
        equation = m2.group(1).strip() if m2 else ""
    if "=" not in equation:
        return None
    left, right = equation.split("=", 1)
    lc = _linear_coeff(left)
    rc = _linear_coeff(right)
    if lc is None or rc is None:
        return None
    a1, b1 = lc
    a2, b2 = rc
    denom = a1 - a2
    if denom == 0:
        return None
    expected = (b2 - b1) / denom
    x_vals: list[float] = []
    for m3 in re.finditer(r"\bx\s*=\s*(-?\d+(?:\.\d+)?)", str(draft or ""), flags=re.IGNORECASE):
        val = _safe_float(m3.group(1))
        if val is not None:
            x_vals.append(val)
    if any(_close(val, expected) for val in x_vals):
        return _critic_payload(domain="linear_equation", issues=[], fixes=[], confidence=0.82)
    issues = [f"Solution for x seems incorrect or missing; expected x = {expected:g}."]
    fixes = [
        "Collect x terms on one side and constants on the other, then solve for x.",
        f"Verify by substitution; x = {expected:g} satisfies the equation.",
    ]
    return _critic_payload(domain="linear_equation", issues=issues, fixes=fixes, confidence=0.9)


def _micro_average_speed(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "").lower()
    if "average speed" not in q and "平均速度" not in q:
        return None
    dists = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*km", q)]
    times = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", q)]
    if len(dists) < 2 or len(times) < 2:
        return None
    total_dist = sum(dists[:2])
    total_time = sum(times[:2])
    if total_time <= 0:
        return None
    expected = total_dist / total_time
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="avg_speed", issues=[], fixes=[], confidence=0.82)
    issues = [
        f"Average speed looks incorrect or missing; total distance/time gives about {expected:.4g} km/h."
    ]
    fixes = [
        f"Compute total distance ({total_dist:g} km) divided by total time ({total_time:g} h) = {expected:.4g} km/h.",
        "Explain briefly why averaging segment speeds directly is wrong (different time weights).",
    ]
    return _critic_payload(domain="avg_speed", issues=issues, fixes=fixes, confidence=0.9)


def _micro_revenue_growth(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "").lower()
    if "revenue" not in q and "growth" not in q:
        return None
    m = re.search(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", q)
    if not m:
        return None
    start = _safe_float(m.group(1))
    end = _safe_float(m.group(2))
    if start is None or end is None or start == 0:
        return None
    expected = (end - start) / start
    if _draft_mentions_value(draft, expected) or _draft_mentions_value(draft, expected * 100.0):
        return _critic_payload(domain="growth", issues=[], fixes=[], confidence=0.78)
    issues = [
        f"Growth percentage seems wrong; it should be (Δ/{start:g}) = {(expected*100.0):.4g}%."
    ]
    fixes = [f"Compute ( {end:g} - {start:g} ) / {start:g} = {expected:.4g} (={expected*100.0:.4g}%)."]
    return _critic_payload(domain="growth", issues=issues, fixes=fixes, confidence=0.9)


def _micro_probability_without_replacement(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "").lower()
    if "probability" not in q:
        return None
    m_red = re.search(r"(\d+)\s+red", q)
    m_blue = re.search(r"(\d+)\s+blue", q)
    m_green = re.search(r"(\d+)\s+green", q)
    if not (m_red and m_blue and m_green):
        return None
    red = int(m_red.group(1))
    blue = int(m_blue.group(1))
    green = int(m_green.group(1))
    total = red + blue + green
    if total <= 1 or blue <= 1:
        return None
    expected = (blue / total) * ((blue - 1) / (total - 1))
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="probability", issues=[], fixes=[], confidence=0.78)
    issues = [f"Probability looks incorrect or missing; expected about {expected:.6g} (={blue}/{total} * {blue-1}/{total-1})."]
    fixes = [
        "Use without-replacement multiplication: P(first blue) * P(second blue | first blue).",
        f"Compute ({blue}/{total}) * ({blue-1}/{total-1}) = {expected:.6g}.",
    ]
    return _critic_payload(domain="probability", issues=issues, fixes=fixes, confidence=0.9)


def _micro_bayes(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "").lower()
    if "sensitivity" not in q or "specificity" not in q or "prevalence" not in q:
        return None
    m_sens = re.search(r"(\d+(?:\.\d+)?)%\s*sensitivity", q)
    m_spec = re.search(r"(\d+(?:\.\d+)?)%\s*specificity", q)
    m_prev = re.search(r"(\d+(?:\.\d+)?)%\s*\.?\s*what is|(\d+(?:\.\d+)?)%\s*prevalence", q)
    sens = _safe_float(m_sens.group(1)) / 100.0 if m_sens else None
    spec = _safe_float(m_spec.group(1)) / 100.0 if m_spec else None
    prev = None
    if m_prev:
        prev_raw = m_prev.group(1) or m_prev.group(2)
        prev = _safe_float(prev_raw) / 100.0 if prev_raw is not None else None
    if sens is None or spec is None or prev is None:
        return None
    fpr = 1.0 - spec
    denom = sens * prev + fpr * (1.0 - prev)
    if denom <= 0:
        return None
    expected = (sens * prev) / denom
    if _draft_mentions_value(draft, expected) or _draft_mentions_value(draft, expected * 100.0):
        return _critic_payload(domain="bayes", issues=[], fixes=[], confidence=0.76)
    issues = [f"Posterior seems incorrect or missing; Bayes gives about {expected:.4g} (={expected*100.0:.4g}%)."]
    fixes = [
        "Compute P(+)=sens*prev + (1-spec)*(1-prev), then posterior = sens*prev / P(+).",
        f"Numerically: {sens:.3g}*{prev:.3g} / ({sens:.3g}*{prev:.3g} + {fpr:.3g}*{1.0-prev:.3g}) = {expected:.4g}.",
    ]
    return _critic_payload(domain="bayes", issues=issues, fixes=fixes, confidence=0.86)


def _micro_conversions(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    if "conversion" not in q.lower() or "traffic" not in q.lower():
        return None
    rates = {}
    for name in ("A", "B", "C"):
        m = re.search(rf"\b{name}\s*=\s*(\d+(?:\.\d+)?)\s*%", q)
        if m:
            rates[name] = float(m.group(1)) / 100.0
    traffic = {}
    for name in ("A", "B", "C"):
        m = re.search(rf"\b{name}\s*=\s*(\d+(?:\.\d+)?)\s*k", q, flags=re.IGNORECASE)
        if m:
            traffic[name] = float(m.group(1)) * 1000.0
    if len(rates) < 3 or len(traffic) < 3:
        return None
    conversions = {k: rates[k] * traffic[k] for k in ("A", "B", "C")}
    winner = max(conversions.items(), key=lambda kv: kv[1])[0]
    if re.search(rf"\b{winner}\b", str(draft or "")):
        return _critic_payload(domain="conversions", issues=[], fixes=[], confidence=0.72)
    issues = [f"Winner variant seems wrong or missing; expected {winner} (highest conversions)."]
    fixes = [
        "Compute conversions = rate * traffic for each variant, then compare totals.",
        "A: {a:.0f}, B: {b:.0f}, C: {c:.0f} conversions".format(
            a=conversions["A"], b=conversions["B"], c=conversions["C"]
        ),
    ]
    return _critic_payload(domain="conversions", issues=issues, fixes=fixes, confidence=0.84)


def _micro_code_review_avg(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    if "def avg" not in q or "len(nums)" not in q:
        return None
    d = str(draft or "").lower()
    mentions_empty = any(tok in d for tok in ("empty", "len==0", "len == 0", "zero", "zerodivision"))
    mentions_iterable_contract = any(
        tok in d
        for tok in (
            "iterable",
            "generator",
            "iterator",
            "__len__",
            "sequence",
            "convert to list",
            "list(",
        )
    )
    issues: list[str] = []
    fixes: list[str] = []
    if not mentions_empty:
        issues.append("Edge case missing: empty input causes ZeroDivisionError (len(nums)==0).")
        fixes.append("Handle empty input (raise ValueError or return 0/None) before dividing by len(nums).")
    if not mentions_iterable_contract:
        issues.append("Edge case missing: iterators/generators (no len) will fail; the input contract should be clarified.")
        fixes.append("Clarify the contract (requires a sized sequence), or convert once (nums=list(nums)) / implement a one-pass mean.")
    if not issues:
        return _critic_payload(domain="code_review", issues=[], fixes=[], confidence=0.7)
    fixes.append("Consider validating numeric types if this is a public utility.")
    return _critic_payload(domain="code_review", issues=issues[:6], fixes=fixes[:8], confidence=0.85)

def _micro_causal_triage_plan(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    if not ("latency" in q_lower and "error" in q_lower and "deployment" in q_lower):
        return None

    d_lower = str(draft or "").lower()
    step_hits = len(
        re.findall(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+", str(draft or ""))
    )
    has_steps = step_hits >= 3 or ("step" in d_lower and step_hits >= 2)
    has_baseline = any(
        tok in d_lower
        for tok in (
            "baseline",
            "before",
            "after",
            "pre-",
            "post",
            "regression",
            "compare",
        )
    )
    has_isolation = any(
        tok in d_lower
        for tok in (
            "isolate",
            "one at a time",
            "experiment",
            "canary",
            "rollback",
            "revert",
            "bisect",
            "disable",
        )
    )
    has_confounders = any(
        tok in d_lower
        for tok in (
            "traffic",
            "load",
            "upstream",
            "dependency",
            "network",
            "third-party",
            "cdn",
        )
    )

    issues: list[str] = []
    fixes: list[str] = []
    if not has_steps:
        issues.append("Plan is not clearly step-by-step; missing ordered triage steps.")
        fixes.append("Use a numbered sequence: observe/baseline → isolate change → test hypotheses → confirm/mitigate.")
    if not has_baseline:
        issues.append("Missing a before/after baseline comparison to anchor causality claims.")
        fixes.append("Start by comparing metrics pre vs post deployment (latency percentiles + error signatures) with a clear timeframe.")
    if not has_isolation:
        issues.append("Missing an isolation/experiment step (change one factor at a time) to avoid correlation→causation leaps.")
        fixes.append("Add a controlled isolation step: rollback/canary/bisect/config toggle (if available) to test candidate causes.")
    if not has_confounders and (not has_baseline or not has_isolation):
        issues.append("Missing explicit confounder checks (traffic spikes, upstream dependencies, etc.).")
        fixes.append("Check external factors (traffic mix, upstream latency/errors, infra/network changes) before attributing causation.")

    if not issues:
        return _critic_payload(domain="causal_triage", issues=[], fixes=[], confidence=0.7)
    return _critic_payload(
        domain="causal_triage",
        issues=issues[:5],
        fixes=fixes[:8],
        confidence=0.82,
    )


def micro_criticise_reasoning(question: str, draft: str) -> Optional[MicroCriticResult]:
    """Return MicroCriticResult when a supported, high-confidence check applies."""

    detectors = (
        _micro_code_review_avg,
        _micro_causal_triage_plan,
        _micro_linear_equation,
        _micro_average_speed,
        _micro_revenue_growth,
        _micro_probability_without_replacement,
        _micro_bayes,
        _micro_conversions,
        _micro_arithmetic,
    )
    for detector in detectors:
        try:
            result = detector(question, draft)
        except Exception:
            result = None
        if result is not None:
            return result
    return None
