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


def _micro_percent_of(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_percent = "%" in q or "％" in q or "percent" in q_lower
    has_connector = "of" in q_lower or "の" in q
    if not (has_percent and has_connector):
        return None
    # Avoid misclassifying "what percent of" prompts.
    if any(token in q_lower for token in ("what percent", "what percentage")) or any(
        token in q for token in ("何%", "何％", "何パーセント")
    ):
        return None
    # Avoid misclassifying "percent off/discount" prompts as "percent of".
    if any(
        token in q_lower
        for token in (
            " off ",
            "off ",
            "discount",
            "sale",
            "引き",
            "割引",
            "オフ",
            "値引",
        )
    ):
        return None

    patterns: tuple[tuple[str, str], ...] = (
        # English: "20% of 50"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*of\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # English: "20 percent of 50", "20 percentage of 50" (rare, but seen)
        (r"(\d+(?:\.\d+)?)\s*percent(?:age)?\s*of\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # Japanese: "50の20%"
        (r"(-?\d+(?:\.\d+)?)\s*の\s*(\d+(?:\.\d+)?)\s*[%％]", "base_pct"),
        # Japanese-ish: "20%の50"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*の\s*(-?\d+(?:\.\d+)?)", "pct_base"),
    )

    pct: Optional[float] = None
    base: Optional[float] = None
    for pattern, kind in patterns:
        m = re.search(pattern, q_lower)
        if not m:
            continue
        first = _safe_float(m.group(1))
        second = _safe_float(m.group(2))
        if first is None or second is None:
            continue
        if kind == "pct_base":
            pct, base = first, second
        else:
            base, pct = first, second
        break

    if pct is None or base is None:
        return None

    expected = base * (pct / 100.0)
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="percent_of", issues=[], fixes=[], confidence=0.8)

    issues = [
        f"Percentage calculation seems incorrect or missing; expected about {expected:g}."
    ]
    fixes = [
        f"Compute {pct:g}% of {base:g} as {base:g} * ({pct:g}/100) = {expected:g}."
    ]
    return _critic_payload(domain="percent_of", issues=issues, fixes=fixes, confidence=0.92)


def _micro_what_percent_of(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_prompt = any(
        token in q_lower for token in ("what percent", "what percentage")
    ) or any(token in q for token in ("何%", "何％", "何パーセント"))
    if not has_prompt:
        return None

    patterns: tuple[tuple[str, str], ...] = (
        # English: "10 is what percent of 50"
        (r"(-?\d+(?:\.\d+)?)\s+is\s+what\s+percent(?:age)?\s+of\s+(-?\d+(?:\.\d+)?)", "part_whole"),
        # English: "What percent of 50 is 10"
        (r"what\s+percent(?:age)?\s+of\s+(-?\d+(?:\.\d+)?)\s+is\s+(-?\d+(?:\.\d+)?)", "whole_part"),
        # Japanese: "10は50の何%"
        (r"(-?\d+(?:\.\d+)?)\s*は\s*(-?\d+(?:\.\d+)?)\s*の\s*何\s*[%％]", "part_whole"),
        (r"(-?\d+(?:\.\d+)?)\s*は\s*(-?\d+(?:\.\d+)?)\s*の\s*何\s*パーセント", "part_whole"),
    )

    part: Optional[float] = None
    whole: Optional[float] = None
    for pattern, kind in patterns:
        m = re.search(pattern, q_lower)
        if not m:
            continue
        first = _safe_float(m.group(1))
        second = _safe_float(m.group(2))
        if first is None or second is None:
            continue
        if kind == "part_whole":
            part, whole = first, second
        else:
            whole, part = first, second
        break

    if part is None or whole is None or whole == 0:
        return None

    expected = (part / whole) * 100.0
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="what_percent_of", issues=[], fixes=[], confidence=0.76)
    issues = [f"Percent-of relationship seems incorrect or missing; expected about {expected:.6g}%."]
    fixes = [f"Compute ({part:g}/{whole:g})*100 = {expected:.6g}%."]
    return _critic_payload(domain="what_percent_of", issues=issues, fixes=fixes, confidence=0.9)


def _micro_percent_off(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_percent = "%" in q or "％" in q or "percent" in q_lower
    has_discount = any(
        token in q_lower
        for token in (
            " off ",
            "off ",
            "discount",
            "sale",
            "引き",
            "割引",
            "オフ",
            "値引",
        )
    )
    if not (has_percent and has_discount):
        return None

    patterns: tuple[tuple[str, str], ...] = (
        # English: "20% off 50"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*off\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # English: "20 percent off 50", "20% discount on 50"
        (r"(\d+(?:\.\d+)?)\s*percent\s*off\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*discount\s*(?:on|off)?\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # Japanese: "50の20%引き"
        (r"(-?\d+(?:\.\d+)?)\s*の\s*(\d+(?:\.\d+)?)\s*[%％]\s*(?:引き|オフ|割引)", "base_pct"),
        # Japanese-ish: "20%引きの50"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*(?:引き|オフ|割引)\s*の\s*(-?\d+(?:\.\d+)?)", "pct_base"),
    )

    pct: Optional[float] = None
    base: Optional[float] = None
    for pattern, kind in patterns:
        m = re.search(pattern, q_lower)
        if not m:
            continue
        first = _safe_float(m.group(1))
        second = _safe_float(m.group(2))
        if first is None or second is None:
            continue
        if kind == "pct_base":
            pct, base = first, second
        else:
            base, pct = first, second
        break

    if pct is None or base is None:
        return None

    wants_amount = any(
        token in q_lower for token in ("discount amount", "値引き額", "割引額", "off amount")
    )
    if wants_amount:
        expected = base * (pct / 100.0)
        label = "discount amount"
    else:
        expected = base * (1.0 - pct / 100.0)
        label = "final price"

    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="percent_off", issues=[], fixes=[], confidence=0.78)

    issues = [
        f"Percent-off calculation seems incorrect or missing; expected {label} ≈ {expected:g}."
    ]
    if wants_amount:
        fixes = [f"Compute discount = {base:g} * ({pct:g}/100) = {expected:g}."]
    else:
        fixes = [
            f"Compute final price = {base:g} * (1 - {pct:g}/100) = {expected:g}."
        ]
    return _critic_payload(domain="percent_off", issues=issues, fixes=fixes, confidence=0.92)


def _micro_add_percent(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_percent = "%" in q or "％" in q or "percent" in q_lower
    has_add_keyword = any(
        token in q_lower
        for token in (
            "tax",
            "vat",
            "tip",
            "gratuity",
            "税込",
            "税",
            "消費税",
            "チップ",
        )
    )
    if not (has_percent and has_add_keyword):
        return None

    wants_total = any(
        token in q_lower
        for token in (
            "total",
            "final",
            "including",
            "with",
            "after",
            "add",
            "pay",
            "税込",
            "合計",
            "支払",
        )
    )
    wants_amount = any(
        token in q_lower
        for token in (
            "tax amount",
            "tip amount",
            "amount of tax",
            "amount of tip",
            "how much tax",
            "how much tip",
            "税額",
            "チップはいくら",
        )
    )
    if wants_total and wants_amount:
        return None
    if not wants_total and not wants_amount:
        return None

    patterns: tuple[tuple[str, str], ...] = (
        # English: "8% tax on 50", "15% tip on 80"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*(?:tax|vat|tip|gratuity)\s*(?:on|for|to)?\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # English: "50 with 8% tax"
        (r"(-?\d+(?:\.\d+)?)\s*(?:with|plus|including|after)\s*(\d+(?:\.\d+)?)\s*[%％]\s*(?:tax|vat|tip|gratuity)", "base_pct"),
        # Japanese: "50に8%の税"
        (r"(-?\d+(?:\.\d+)?)\s*に\s*(\d+(?:\.\d+)?)\s*[%％]\s*の?\s*(?:税|消費税|チップ)", "base_pct"),
    )

    pct: Optional[float] = None
    base: Optional[float] = None
    for pattern, kind in patterns:
        m = re.search(pattern, q_lower)
        if not m:
            continue
        first = _safe_float(m.group(1))
        second = _safe_float(m.group(2))
        if first is None or second is None:
            continue
        if kind == "pct_base":
            pct, base = first, second
        else:
            base, pct = first, second
        break

    if pct is None or base is None:
        return None

    add_amount = base * (pct / 100.0)
    expected = base + add_amount if wants_total else add_amount
    label = "total" if wants_total else "added amount"
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="add_percent", issues=[], fixes=[], confidence=0.76)

    issues = [f"Add-percent calculation seems incorrect or missing; expected {label} ≈ {expected:g}."]
    if wants_total:
        fixes = [
            f"Compute added amount = {base:g} * ({pct:g}/100) = {add_amount:g}, then total = {base:g} + {add_amount:g} = {expected:g}."
        ]
    else:
        fixes = [f"Compute amount = {base:g} * ({pct:g}/100) = {expected:g}."]
    return _critic_payload(domain="add_percent", issues=issues, fixes=fixes, confidence=0.9)


def _micro_percent_change(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    if not any(
        token in q_lower
        for token in (
            "percent change",
            "percentage change",
            "percent increase",
            "percentage increase",
            "percent decrease",
            "percentage decrease",
            "増加率",
            "減少率",
            "変化率",
            "増減率",
        )
    ):
        return None

    m = re.search(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", q_lower)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*から\s*(\d+(?:\.\d+)?)\s*に", q)
    if not m:
        return None

    start = _safe_float(m.group(1))
    end = _safe_float(m.group(2))
    if start is None or end is None or start == 0:
        return None

    signed = (end - start) / start * 100.0
    wants_increase = any(token in q_lower for token in ("increase", "increased", "増加", "上昇"))
    wants_decrease = any(token in q_lower for token in ("decrease", "decreased", "減少", "低下"))

    expected_candidates: list[float] = []
    if wants_increase and signed < 0:
        return None
    if wants_decrease and signed > 0:
        return None
    if wants_decrease:
        expected_candidates.append(abs(signed))
        label = "percent decrease"
    elif wants_increase:
        expected_candidates.append(signed)
        label = "percent increase"
    else:
        # For generic "percent change", accept either signed change or magnitude.
        expected_candidates.extend([signed, abs(signed)])
        label = "percent change"

    if any(_draft_mentions_value(draft, expected) for expected in expected_candidates):
        return _critic_payload(domain="percent_change", issues=[], fixes=[], confidence=0.74)

    expected_display = expected_candidates[0]
    if wants_decrease and signed < 0:
        expected_display = abs(signed)
    issues = [
        f"{label.capitalize()} seems incorrect or missing; expected about {expected_display:.6g}%."
    ]
    fixes = [
        f"Compute (({end:g} - {start:g}) / {start:g}) * 100 = {signed:.6g}%.",
        "For decreases, report the magnitude as a positive percent decrease (abs)."
        if wants_decrease or (not wants_increase and signed < 0)
        else "For increases, report the result as a percent increase.",
    ]
    return _critic_payload(domain="percent_change", issues=issues, fixes=fixes, confidence=0.9)


def _micro_percent_adjust(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_percent = "%" in q or "％" in q or "percent" in q_lower
    if not has_percent:
        return None

    wants_increase = any(
        token in q_lower
        for token in ("increase", "increased", "raise", "raised")
    ) or any(token in q for token in ("増や", "増", "上げ", "アップ"))
    wants_decrease = any(
        token in q_lower
        for token in ("decrease", "decreased", "reduce", "reduced", "lower", "lowered")
    ) or any(token in q for token in ("減ら", "減", "下げ", "ダウン"))
    if wants_increase and wants_decrease:
        return None
    if not (wants_increase or wants_decrease):
        return None

    wants_delta = any(
        token in q_lower
        for token in ("increase amount", "decrease amount", "difference", "delta")
    ) or any(token in q for token in ("増加分", "減少分", "差分", "どれだけ増", "どれだけ減"))

    patterns: tuple[tuple[str, str], ...] = (
        # English: "Increase 50 by 20%"
        (r"(?:increase|decrease|reduce|lower|raise)\s+(-?\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*[%％]", "base_pct"),
        # English: "50 increased by 20%"
        (r"(-?\d+(?:\.\d+)?)\s*(?:increased|decreased|reduced|lowered|raised)\s+by\s+(\d+(?:\.\d+)?)\s*[%％]", "base_pct"),
        # English: "20% increase on 50"
        (r"(\d+(?:\.\d+)?)\s*[%％]\s*(?:increase|decrease)\s*(?:on|of)?\s*(-?\d+(?:\.\d+)?)", "pct_base"),
        # Japanese: "50を20%増やす / 減らす"
        (r"(-?\d+(?:\.\d+)?)\s*を\s*(\d+(?:\.\d+)?)\s*[%％]\s*(?:増やす|上げる|アップ|減らす|下げる|ダウン)", "base_pct"),
    )

    pct: Optional[float] = None
    base: Optional[float] = None
    for pattern, kind in patterns:
        m = re.search(pattern, q_lower)
        if not m:
            continue
        first = _safe_float(m.group(1))
        second = _safe_float(m.group(2))
        if first is None or second is None:
            continue
        if kind == "base_pct":
            base, pct = first, second
        else:
            pct, base = first, second
        break

    if pct is None or base is None:
        return None

    delta = base * (pct / 100.0)
    if wants_decrease:
        new_value = base - delta
        label = "decrease"
    else:
        new_value = base + delta
        label = "increase"

    expected = delta if wants_delta else new_value
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="percent_adjust", issues=[], fixes=[], confidence=0.74)
    if wants_delta:
        issues = [f"Percent {label} amount seems incorrect or missing; expected about {delta:.6g}."]
        fixes = [f"Compute amount = {base:g} * ({pct:g}/100) = {delta:.6g}."]
    else:
        issues = [f"Percent {label} result seems incorrect or missing; expected about {new_value:.6g}."]
        fixes = [
            f"Compute delta = {base:g} * ({pct:g}/100) = {delta:.6g}, then apply: {base:g} {'+' if not wants_decrease else '-'} {delta:.6g} = {new_value:.6g}."
        ]
    return _critic_payload(domain="percent_adjust", issues=issues, fixes=fixes, confidence=0.9)


def _micro_temp_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_c = "celsius" in q_lower or "°c" in q_lower or "摂氏" in q
    has_f = "fahrenheit" in q_lower or "°f" in q_lower or "華氏" in q
    if not (has_c and has_f):
        return None

    c_val: Optional[float] = None
    f_val: Optional[float] = None
    m_c = re.search(r"(-?\d+(?:\.\d+)?)\s*°\s*c\b", q_lower) or re.search(
        r"(-?\d+(?:\.\d+)?)\s*°c\b", q_lower
    )
    if m_c:
        c_val = _safe_float(m_c.group(1))
    if c_val is None:
        m_c2 = re.search(r"(-?\d+(?:\.\d+)?)\s*celsius\b", q_lower)
        if m_c2:
            c_val = _safe_float(m_c2.group(1))
    if c_val is None:
        m_c3 = re.search(r"摂氏\s*(-?\d+(?:\.\d+)?)", q)
        if m_c3:
            c_val = _safe_float(m_c3.group(1))

    m_f = re.search(r"(-?\d+(?:\.\d+)?)\s*°\s*f\b", q_lower) or re.search(
        r"(-?\d+(?:\.\d+)?)\s*°f\b", q_lower
    )
    if m_f:
        f_val = _safe_float(m_f.group(1))
    if f_val is None:
        m_f2 = re.search(r"(-?\d+(?:\.\d+)?)\s*fahrenheit\b", q_lower)
        if m_f2:
            f_val = _safe_float(m_f2.group(1))
    if f_val is None:
        m_f3 = re.search(r"華氏\s*(-?\d+(?:\.\d+)?)", q)
        if m_f3:
            f_val = _safe_float(m_f3.group(1))

    expected: Optional[float] = None
    direction = ""
    if c_val is not None and ("fahrenheit" in q_lower or "°f" in q_lower or "華氏" in q):
        expected = c_val * 9.0 / 5.0 + 32.0
        direction = "C→F"
    elif f_val is not None and ("celsius" in q_lower or "°c" in q_lower or "摂氏" in q):
        expected = (f_val - 32.0) * 5.0 / 9.0
        direction = "F→C"
    else:
        return None

    if expected is None:
        return None
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="temp_convert", issues=[], fixes=[], confidence=0.76)

    issues = [f"Temperature conversion ({direction}) seems incorrect or missing; expected about {expected:.6g}."]
    if direction == "C→F" and c_val is not None:
        fixes = [f"Use F = C*(9/5)+32 = {c_val:g}*(9/5)+32 = {expected:.6g}."]
    elif direction == "F→C" and f_val is not None:
        fixes = [f"Use C = (F-32)*(5/9) = ({f_val:g}-32)*(5/9) = {expected:.6g}."]
    else:
        fixes = [f"Recompute using the standard C/F conversion formulas (expected {expected:.6g})."]
    return _critic_payload(domain="temp_convert", issues=issues, fixes=fixes, confidence=0.9)


def _micro_time_unit_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    units = {
        "second": {"second", "seconds", "sec", "secs", "秒"},
        "minute": {"minute", "minutes", "min", "mins", "分"},
        "hour": {"hour", "hours", "hr", "hrs", "時間"},
        "day": {"day", "days", "日"},
    }
    multipliers = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}

    from_matches: list[tuple[float, str]] = []
    for name, tokens in units.items():
        for tok in tokens:
            if tok in {"sec", "secs", "min", "mins", "hr", "hrs"}:
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}\b"
            elif tok.isascii():
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}\b"
            else:
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}"
            m = re.search(pattern, q_lower if tok.isascii() else q)
            if m:
                val = _safe_float(m.group(1))
                if val is not None:
                    from_matches.append((val, name))
                    break
        if from_matches:
            break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    target_patterns = (
        (
            "second",
            (
                r"\b(in|to)\s+seconds?\b",
                r"\bhow many\s+seconds?\b",
                r"\bnumber of\s+seconds?\b",
                r"何秒",
                r"秒に",
            ),
        ),
        (
            "minute",
            (
                r"\b(in|to)\s+minutes?\b",
                r"\bhow many\s+minutes?\b",
                r"\bnumber of\s+minutes?\b",
                r"何分",
                r"分に",
            ),
        ),
        (
            "hour",
            (
                r"\b(in|to)\s+hours?\b",
                r"\bhow many\s+hours?\b",
                r"\bnumber of\s+hours?\b",
                r"何時間",
                r"時間に",
            ),
        ),
        (
            "day",
            (
                r"\b(in|to)\s+days?\b",
                r"\bhow many\s+days?\b",
                r"\bnumber of\s+days?\b",
                r"何日",
                r"日に",
            ),
        ),
    )
    for unit_name, patterns in target_patterns:
        for pat in patterns:
            if re.search(pat, q_lower if pat.isascii() else q):
                target_unit = unit_name
                break
        if target_unit is not None:
            break

    if target_unit is None or target_unit == from_unit:
        return None

    expected_seconds = float(value) * multipliers[from_unit]
    expected = expected_seconds / multipliers[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="time_units", issues=[], fixes=[], confidence=0.74)

    issues = [
        f"Time-unit conversion seems incorrect or missing; expected about {expected:.6g} {target_unit}(s)."
    ]
    fixes = [
        f"Convert via seconds: {value:g} {from_unit}(s) = {expected_seconds:.6g} seconds, then divide by {multipliers[target_unit]:g} to get {expected:.6g} {target_unit}(s)."
    ]
    return _critic_payload(domain="time_units", issues=issues, fixes=fixes, confidence=0.88)


def _micro_length_unit_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    # Avoid misclassifying speed prompts like "miles per hour" → use the speed detector.
    if any(token in q_lower for token in ("per hour", "km/h", "kph", "kmh", "mph", "mi/h")):
        return None
    units = {
        "mile": {"mile", "miles", "mi", "マイル"},
        "km": {"km", "kilometer", "kilometers", "kilometre", "kilometres", "キロメートル"},
    }
    multipliers_m = {"mile": 1609.344, "km": 1000.0}

    from_matches: list[tuple[float, str]] = []
    for name, tokens in units.items():
        for tok in tokens:
            if tok.isascii():
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}\b"
                m = re.search(pattern, q_lower)
            else:
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}"
                m = re.search(pattern, q)
            if m:
                val = _safe_float(m.group(1))
                if val is not None:
                    from_matches.append((val, name))
                    break
        if from_matches:
            break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    target_patterns = (
        (
            "km",
            (
                r"\b(in|to)\s*(?:km|kilometers?|kilometres?)\b",
                r"\bhow many\s*(?:km|kilometers?|kilometres?)\b",
                r"何\s*km",
                r"何\s*キロメートル",
            ),
        ),
        (
            "mile",
            (
                r"\b(in|to)\s*(?:miles?|mi)\b",
                r"\bhow many\s*(?:miles?|mi)\b",
                r"何\s*マイル",
            ),
        ),
    )
    for unit_name, patterns in target_patterns:
        for pat in patterns:
            if re.search(pat, q_lower if pat.isascii() else q):
                target_unit = unit_name
                break
        if target_unit is not None:
            break

    if target_unit is None or target_unit == from_unit:
        return None

    meters = float(value) * multipliers_m[from_unit]
    expected = meters / multipliers_m[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="length_units", issues=[], fixes=[], confidence=0.72)

    issues = [
        f"Length-unit conversion seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    fixes = [
        f"Convert via meters: {value:g} {from_unit} = {meters:.6g} m, then divide by {multipliers_m[target_unit]:g} to get {expected:.6g} {target_unit}."
    ]
    return _critic_payload(domain="length_units", issues=issues, fixes=fixes, confidence=0.88)


def _micro_mass_unit_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    units = {
        "kg": {"kg", "kilogram", "kilograms", "キログラム"},
        "lb": {"lb", "lbs", "pound", "pounds", "ポンド", "パウンド"},
    }
    multipliers_kg = {"kg": 1.0, "lb": 0.45359237}

    has_kg = "kg" in q_lower or "kilogram" in q_lower or "キログラム" in q
    has_lb = any(tok in q_lower for tok in ("lb", "lbs", "pound")) or any(tok in q for tok in ("ポンド", "パウンド"))
    if not (has_kg and has_lb):
        return None

    from_matches: list[tuple[float, str]] = []
    for name, tokens in units.items():
        for tok in tokens:
            if tok.isascii():
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}\b"
                m = re.search(pattern, q_lower)
            else:
                pattern = rf"(-?\d+(?:\.\d+)?)\s*{re.escape(tok)}"
                m = re.search(pattern, q)
            if m:
                val = _safe_float(m.group(1))
                if val is not None:
                    from_matches.append((val, name))
                    break
        if from_matches:
            break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    target_patterns = (
        (
            "kg",
            (
                r"\b(in|to)\s*(?:kg|kilograms?)\b",
                r"\bhow many\s*(?:kg|kilograms?)\b",
                r"何\s*kg",
                r"キログラム",
            ),
        ),
        (
            "lb",
            (
                r"\b(in|to)\s*(?:lb|lbs|pounds?)\b",
                r"\bhow many\s*(?:lb|lbs|pounds?)\b",
                r"何\s*(?:ポンド|パウンド)",
                r"ポンド",
                r"パウンド",
            ),
        ),
    )
    for unit_name, patterns in target_patterns:
        for pat in patterns:
            if re.search(pat, q_lower if pat.isascii() else q):
                target_unit = unit_name
                break
        if target_unit is not None:
            break

    if target_unit is None:
        target_unit = "lb" if from_unit == "kg" else "kg"
    if target_unit == from_unit:
        return None

    base_kg = float(value) * multipliers_kg[from_unit]
    expected = base_kg / multipliers_kg[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="mass_units", issues=[], fixes=[], confidence=0.72)

    issues = [
        f"Mass-unit conversion seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    fixes = [
        f"Convert via kilograms: {value:g} {from_unit} = {base_kg:.6g} kg, then divide by {multipliers_kg[target_unit]:g} to get {expected:.6g} {target_unit}."
    ]
    return _critic_payload(domain="mass_units", issues=issues, fixes=fixes, confidence=0.88)


def _micro_mass_metric_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()

    has_kg = "kg" in q_lower or "kilogram" in q_lower or "キログラム" in q
    has_g = bool(
        re.search(r"(-?\d+(?:\.\d+)?)\s*g\b", q_lower)
        or re.search(r"\b(in|to)\s*g\b", q_lower)
        or re.search(r"\b(in|to)\s*grams?\b", q_lower)
        or "gram" in q_lower
        or "グラム" in q
    )
    if not (has_kg and has_g):
        return None

    from_matches: list[tuple[float, str]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"(-?\d+(?:\.\d+)?)\s*kg\b", "kg"),
        (r"(-?\d+(?:\.\d+)?)\s*kilograms?\b", "kg"),
        (r"(-?\d+(?:\.\d+)?)\s*キログラム", "kg"),
        (r"(-?\d+(?:\.\d+)?)\s*g\b", "g"),
        (r"(-?\d+(?:\.\d+)?)\s*grams?\b", "g"),
        (r"(-?\d+(?:\.\d+)?)\s*グラム", "g"),
    )
    for pat, unit in patterns:
        m = re.search(pat, q_lower if pat.isascii() else q)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                from_matches.append((val, unit))
                break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    target_patterns = (
        (
            "g",
            (
                r"\b(in|to)\s*g\b",
                r"\b(in|to)\s*grams?\b",
                r"\bhow many\s*grams?\b",
                r"何\s*g",
                r"グラム",
                r"gに",
            ),
        ),
        (
            "kg",
            (
                r"\b(in|to)\s*kg\b",
                r"\b(in|to)\s*kilograms?\b",
                r"\bhow many\s*kg\b",
                r"何\s*kg",
                r"キログラム",
                r"kgに",
            ),
        ),
    )
    for unit_name, patterns_target in target_patterns:
        for pat in patterns_target:
            if re.search(pat, q_lower if pat.isascii() else q):
                target_unit = unit_name
                break
        if target_unit is not None:
            break

    if target_unit is None or target_unit == from_unit:
        return None

    multiplier = {"kg": 1000.0, "g": 1.0}
    grams = float(value) * multiplier[from_unit]
    expected = grams / multiplier[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="mass_metric", issues=[], fixes=[], confidence=0.74)

    issues = [
        f"Mass conversion (kg↔g) seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    fixes = [
        f"Convert via grams: {value:g} {from_unit} = {grams:.6g} g, then convert to {target_unit} = {expected:.6g}."
    ]
    return _critic_payload(domain="mass_metric", issues=issues, fixes=fixes, confidence=0.9)


def _micro_inch_cm_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_cm = "cm" in q_lower or "centimeter" in q_lower or "centimetre" in q_lower or "センチ" in q or "センチメートル" in q
    has_in = "inch" in q_lower or "inches" in q_lower or "インチ" in q or "″" in q or bool(re.search(r"\b\d+(?:\.\d+)?\s*in\b", q_lower))
    if not (has_cm and has_in):
        return None

    from_matches: list[tuple[float, str]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"(-?\d+(?:\.\d+)?)\s*cm\b", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*centimet(?:er|re)s?\b", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:センチ|センチメートル)", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:inches?|inch)\b", "inch"),
        (r"(-?\d+(?:\.\d+)?)\s*in\b", "inch"),
        (r"(-?\d+(?:\.\d+)?)\s*インチ", "inch"),
        (r"(-?\d+(?:\.\d+)?)\s*″", "inch"),
    )
    for pat, unit in patterns:
        m = re.search(pat, q_lower if pat.isascii() else q)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                from_matches.append((val, unit))
                break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit = "inch" if from_unit == "cm" else "cm"
    multipliers_m = {"cm": 0.01, "inch": 0.0254}
    meters = float(value) * multipliers_m[from_unit]
    expected = meters / multipliers_m[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="length_units_inch_cm", issues=[], fixes=[], confidence=0.72)

    issues = [
        f"Length-unit conversion seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    fixes = [
        f"Use 1 inch = 2.54 cm. Convert via meters: {value:g} {from_unit} = {meters:.6g} m, then convert to {target_unit} = {expected:.6g}."
    ]
    return _critic_payload(domain="length_units_inch_cm", issues=issues, fixes=fixes, confidence=0.9)


def _micro_length_metric_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()

    has_cm = "cm" in q_lower or "centimeter" in q_lower or "centimetre" in q_lower or "センチ" in q or "センチメートル" in q
    has_m = bool(
        re.search(r"(-?\d+(?:\.\d+)?)\s*m\b(?!/)", q_lower)
        or re.search(r"\b(in|to)\s*m\b(?!/)", q_lower)
        or re.search(r"\bmeters?\b", q_lower)
        or re.search(r"\bmetres?\b", q_lower)
        or "メートル" in q
    )
    if not (has_cm and has_m):
        return None

    from_matches: list[tuple[float, str]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"(-?\d+(?:\.\d+)?)\s*cm\b", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*centimet(?:er|re)s?\b", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:センチ|センチメートル)", "cm"),
        (r"(-?\d+(?:\.\d+)?)\s*m\b(?!/)", "m"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:meters?|metres?)\b", "m"),
        (r"(-?\d+(?:\.\d+)?)\s*メートル", "m"),
    )
    for pat, unit in patterns:
        m = re.search(pat, q_lower if pat.isascii() else q)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                from_matches.append((val, unit))
                break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    target_patterns = (
        (
            "m",
            (
                r"\b(in|to)\s*m\b(?!/)",
                r"\b(in|to)\s*(?:meters?|metres?)\b",
                r"\bhow many\s*(?:meters?|metres?)\b",
                r"何\s*m",
                r"メートル",
                r"mに",
            ),
        ),
        (
            "cm",
            (
                r"\b(in|to)\s*cm\b",
                r"\b(in|to)\s*centimet(?:er|re)s?\b",
                r"\bhow many\s*cm\b",
                r"何\s*cm",
                r"センチ",
                r"cmに",
            ),
        ),
    )
    for unit_name, patterns_target in target_patterns:
        for pat in patterns_target:
            if re.search(pat, q_lower if pat.isascii() else q):
                target_unit = unit_name
                break
        if target_unit is not None:
            break

    if target_unit is None or target_unit == from_unit:
        return None

    multipliers_m = {"m": 1.0, "cm": 0.01}
    meters = float(value) * multipliers_m[from_unit]
    expected = meters / multipliers_m[target_unit]
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="length_metric", issues=[], fixes=[], confidence=0.74)

    issues = [
        f"Length conversion (cm↔m) seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    fixes = [
        f"Convert via meters: {value:g} {from_unit} = {meters:.6g} m, then convert to {target_unit} = {expected:.6g}."
    ]
    return _critic_payload(domain="length_metric", issues=issues, fixes=fixes, confidence=0.9)


def _micro_speed_unit_convert(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_mps = "m/s" in q_lower or "meters per second" in q_lower or "metres per second" in q_lower or "メートル毎秒" in q or "mps" in q_lower
    has_kmph = "km/h" in q_lower or "kph" in q_lower or "kmh" in q_lower or "kilometers per hour" in q_lower or "kilometres per hour" in q_lower or "キロ" in q
    if not (has_mps and has_kmph):
        return None

    from_matches: list[tuple[float, str]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"(-?\d+(?:\.\d+)?)\s*m/s\b", "m/s"),
        (r"(-?\d+(?:\.\d+)?)\s*mps\b", "m/s"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:meters?|metres?)\s+per\s+second\b", "m/s"),
        (r"(-?\d+(?:\.\d+)?)\s*km/h\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*kph\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*kmh\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:kilometers?|kilometres?)\s+per\s+hour\b", "km/h"),
    )
    for pat, unit in patterns:
        m = re.search(pat, q_lower)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                from_matches.append((val, unit))
                break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit = "km/h" if from_unit == "m/s" else "m/s"
    expected = float(value) * 3.6 if from_unit == "m/s" else float(value) / 3.6
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="speed_units", issues=[], fixes=[], confidence=0.72)

    issues = [
        f"Speed-unit conversion seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    if from_unit == "m/s":
        fixes = [f"Convert m/s to km/h by multiplying by 3.6: {value:g} * 3.6 = {expected:.6g}."]
    else:
        fixes = [f"Convert km/h to m/s by dividing by 3.6: {value:g} / 3.6 = {expected:.6g}."]
    return _critic_payload(domain="speed_units", issues=issues, fixes=fixes, confidence=0.9)


def _micro_speed_unit_convert_mph_kmph(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()

    has_mph = bool(
        "mph" in q_lower
        or "mi/h" in q_lower
        or "miles per hour" in q_lower
        or "mile per hour" in q_lower
    )
    has_kmph = "km/h" in q_lower or "kph" in q_lower or "kmh" in q_lower or "kilometers per hour" in q_lower or "kilometres per hour" in q_lower or "キロ" in q
    if not (has_mph and has_kmph):
        return None

    from_matches: list[tuple[float, str]] = []
    patterns: tuple[tuple[str, str], ...] = (
        (r"(-?\d+(?:\.\d+)?)\s*mph\b", "mph"),
        (r"(-?\d+(?:\.\d+)?)\s*mi/h\b", "mph"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:miles?|mile)\s+per\s+hour\b", "mph"),
        (r"(-?\d+(?:\.\d+)?)\s*km/h\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*kph\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*kmh\b", "km/h"),
        (r"(-?\d+(?:\.\d+)?)\s*(?:kilometers?|kilometres?)\s+per\s+hour\b", "km/h"),
    )
    for pat, unit in patterns:
        m = re.search(pat, q_lower)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                from_matches.append((val, unit))
                break

    if len(from_matches) != 1:
        return None
    value, from_unit = from_matches[0]

    target_unit: Optional[str] = None
    if re.search(r"\b(in|to)\s*mph\b", q_lower) or re.search(r"\b(in|to)\s*(?:miles?|mile)\s+per\s+hour\b", q_lower):
        target_unit = "mph"
    elif re.search(r"\b(in|to)\s*(?:km/h|kph|kmh)\b", q_lower) or re.search(r"\b(in|to)\s*(?:kilometers?|kilometres?)\s+per\s+hour\b", q_lower):
        target_unit = "km/h"

    if target_unit is None or target_unit == from_unit:
        return None

    factor = 1.609344
    expected = float(value) * factor if from_unit == "mph" else float(value) / factor
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="speed_units_mph", issues=[], fixes=[], confidence=0.74)

    issues = [
        f"Speed-unit conversion (mph↔km/h) seems incorrect or missing; expected about {expected:.6g} {target_unit}."
    ]
    if from_unit == "mph":
        fixes = [f"Use 1 mph = {factor:g} km/h. Compute {value:g} * {factor:g} = {expected:.6g} km/h."]
    else:
        fixes = [f"Use 1 mph = {factor:g} km/h. Compute {value:g} / {factor:g} = {expected:.6g} mph."]
    return _critic_payload(domain="speed_units_mph", issues=issues, fixes=fixes, confidence=0.9)


def _micro_fraction_to_percent(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    q_lower = q.lower()
    has_percent = "%" in q or "％" in q or "percent" in q_lower or "percentage" in q_lower or "パーセント" in q
    if not has_percent:
        return None
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", q)
    if not m:
        return None
    num = _safe_float(m.group(1))
    den = _safe_float(m.group(2))
    if num is None or den is None or den == 0:
        return None
    expected = (num / den) * 100.0
    if _draft_mentions_value(draft, expected):
        return _critic_payload(domain="fraction_percent", issues=[], fixes=[], confidence=0.76)
    issues = [f"Fraction-to-percent conversion seems incorrect or missing; expected about {expected:.6g}%."]
    fixes = [f"Compute ({num:g}/{den:g})*100 = {expected:.6g}%."]
    return _critic_payload(domain="fraction_percent", issues=issues, fixes=fixes, confidence=0.9)


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
    fixes.append("Validate numeric types if this is a public utility.")
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
        fixes.append("Include a controlled isolation step: rollback/canary/bisect/config toggle (if available) to test candidate causes.")
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

def _micro_logic_all_some_inference(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    all_pairs = re.findall(r"\ball\s+([a-z])\s+are\s+([a-z])\b", q_lower)
    some_pairs = re.findall(r"\bsome\s+([a-z])\s+are\s+([a-z])\b", q_lower)
    if len(all_pairs) != 1 or len(some_pairs) < 2:
        return None
    x, y = all_pairs[0]
    premise = next((pair for pair in some_pairs if pair[0] == y), None)
    if not premise:
        return None
    _, z = premise
    if not any(pair[0] == x and pair[1] == z for pair in some_pairs):
        return None

    d_lower = str(draft or "").lower()
    invalid_markers = (
        "does not follow",
        "doesn't follow",
        "not necessarily",
        "cannot conclude",
        "can't conclude",
        "cannot infer",
        "can't infer",
        "not implied",
        "not entailed",
        "does not entail",
        "doesn't entail",
        "counterexample",
        "反例",
        "必ずしも",
        "導けない",
        "導出できない",
        "成り立たない",
    )
    valid_markers = (
        "it follows",
        "therefore some a are c",
        "must be",
        "necessarily",
        "valid inference",
        "entailed",
        "導ける",
        "導出できる",
        "成り立つ",
    )

    mentions_invalid = any(marker in d_lower for marker in invalid_markers)
    mentions_valid = any(marker in d_lower for marker in valid_markers)
    issues: list[str] = []
    fixes: list[str] = []
    if mentions_valid and not mentions_invalid:
        issues.append(f"Inference is invalid: from ∀({x}→{y}) and ∃({y}∧{z}) you cannot conclude ∃({x}∧{z}).")
        fixes.append(
            f"State the correct conclusion: it does NOT follow; {x} could be a subset of {y} disjoint from {z}."
        )
        fixes.append(f"Give a counterexample (e.g., {x}={{1}}, {y}={{1,2}}, {z}={{2}}).")
    elif not mentions_invalid:
        issues.append(f"Missing a clear yes/no conclusion about whether some {x} are {z} follows.")
        fixes.append("Answer explicitly: it does NOT follow; provide a short counterexample and explain quantifiers.")

    if not issues:
        return _critic_payload(domain="logic_syllogism", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="logic_syllogism", issues=issues[:4], fixes=fixes[:8], confidence=0.86)


def _micro_logic_affirming_consequent(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    m = re.search(r"\bif\s+([a-z])\s+then\s+([a-z])\b", q_lower)
    if not m:
        return None
    p = m.group(1)
    q = m.group(2)
    if f"therefore {p}" not in q_lower:
        return None
    q_hits = len(re.findall(rf"(?<![a-z]){re.escape(q)}(?![a-z])", q_lower))
    if q_hits < 2:
        return None

    d_lower = str(draft or "").lower()
    invalid_markers = (
        "affirming the consequent",
        "fallacy",
        "invalid",
        "does not follow",
        "doesn't follow",
        "not necessarily",
        "counterexample",
        "後件肯定",
        "誤謬",
        "反例",
    )
    valid_markers = ("valid", "modus ponens", "therefore p is true", "q implies p")

    mentions_invalid = any(marker in d_lower for marker in invalid_markers)
    mentions_valid = any(marker in d_lower for marker in valid_markers)
    issues: list[str] = []
    fixes: list[str] = []
    if mentions_valid and not mentions_invalid:
        issues.append(f"This is not a valid inference: it affirms the consequent (If {p}→{q}, {q}, therefore {p}).")
        fixes.append("Label it: affirming the consequent (invalid).")
        fixes.append("Provide a counterexample: If it rains then ground is wet; ground is wet; it may still not be raining.")
    elif not mentions_invalid:
        issues.append("Missing the key point: the inference is invalid (affirming the consequent) and needs a counterexample.")
        fixes.append("State it's invalid (affirming the consequent) and include a short counterexample.")

    if not issues:
        return _critic_payload(domain="logic_fallacy", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="logic_fallacy", issues=issues[:4], fixes=fixes[:8], confidence=0.88)


def _micro_logic_demorgan_not_and(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    m = re.search(r"\bnot\s*\(\s*([a-z])\s*(?:and|&&|∧)\s*([a-z])\s*\)", q_lower)
    if not m:
        if "de morgan" not in q_lower:
            return None
        x, y = "a", "b"
    else:
        x, y = m.group(1), m.group(2)

    raw = str(draft or "")
    d_lower = raw.lower()
    has_not_x = bool(re.search(rf"(?:\bnot\s*{re.escape(x)}\b|[!¬~]\s*{re.escape(x)}\b)", d_lower))
    has_not_y = bool(re.search(rf"(?:\bnot\s*{re.escape(y)}\b|[!¬~]\s*{re.escape(y)}\b)", d_lower))
    has_or = (" or " in d_lower) or ("||" in raw) or ("∨" in raw)
    has_and = (" and " in d_lower) or ("&&" in raw) or ("∧" in raw)

    issues: list[str] = []
    fixes: list[str] = []
    if not (has_not_x and has_not_y and has_or) or (has_and and not has_or):
        issues.append(f"De Morgan rewrite is incorrect or missing: ¬({x} ∧ {y}) should become (¬{x}) ∨ (¬{y}).")
        fixes.append(f"Rewrite as: NOT {x} OR NOT {y} (¬{x} ∨ ¬{y}).")

    if not issues:
        return _critic_payload(domain="logic_demorgan", issues=[], fixes=[], confidence=0.8)
    return _critic_payload(domain="logic_demorgan", issues=issues[:3], fixes=fixes[:6], confidence=0.9)


def _micro_algo_sort_complexity(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    if not ("quicksort" in q_lower and "mergesort" in q_lower):
        return None

    d_lower = str(draft or "").lower()
    has_nlogn = bool(re.search(r"(?:n\s*log\s*n|nlogn|o\(\s*n\s*log\s*n\s*\))", d_lower))
    has_n2 = bool(re.search(r"(?:n\s*\^\s*2|n\*\s*n|n2|n²|o\(\s*n\s*\^\s*2\s*\))", d_lower))
    quick_mentions = ("quicksort" in d_lower) or ("quick sort" in d_lower)
    merge_mentions = ("mergesort" in d_lower) or ("merge sort" in d_lower)
    practice_markers = (
        "in-place",
        "in place",
        "extra memory",
        "stable",
        "unstable",
        "cache",
        "linked list",
        "external",
    )
    mentions_practice = any(marker in d_lower for marker in practice_markers)

    issues: list[str] = []
    fixes: list[str] = []
    if not (quick_mentions and merge_mentions and has_nlogn):
        issues.append("Missing core comparison details (name both sorts and include the O(n log n) baseline).")
        fixes.append("Summarize: Quicksort avg O(n log n), worst O(n^2); Mergesort avg/worst O(n log n).")
    if not has_n2:
        issues.append("Missing quicksort worst-case: O(n^2) (e.g., bad pivots / adversarial order).")
        fixes.append("State quicksort worst-case is O(n^2) (mitigate with randomized/median-of-three pivots).")
    if not mentions_practice:
        issues.append("Missing practical trade-offs (memory/stability/in-place/external sorting).")
        fixes.append("Mention trade-offs: quicksort often in-place + cache-friendly; mergesort stable but needs extra memory (arrays) / good for linked lists & external sort.")

    if not issues:
        return _critic_payload(domain="algo_complexity", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="algo_complexity", issues=issues[:5], fixes=fixes[:8], confidence=0.84)


def _micro_constraints_single_worker_schedule(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    if "one worker" not in q_lower:
        return None
    t1_m = re.search(r"\bt1\s*=\s*(\d+(?:\.\d+)?)\s*h\b", q_lower)
    t2_m = re.search(r"\bt2\s*=\s*(\d+(?:\.\d+)?)\s*h\b", q_lower)
    t3_m = re.search(r"\bt3\s*=\s*(\d+(?:\.\d+)?)\s*h\b", q_lower)
    if not (t1_m and t2_m and t3_m):
        return None
    t1 = _safe_float(t1_m.group(1))
    t2 = _safe_float(t2_m.group(1))
    t3 = _safe_float(t3_m.group(1))
    if t1 is None or t2 is None or t3 is None:
        return None
    total = t1 + t2 + t3
    if total <= 0:
        return None

    d_lower = str(draft or "").lower()
    mentions_total = _draft_mentions_value(draft, total)
    mentions_order = bool(re.search(r"t1.*(?:->|→|then|before).*t2", d_lower)) or ("t2 after t1" in d_lower)
    has_tasks = all(tok in d_lower for tok in ("t1", "t2", "t3"))

    issues: list[str] = []
    fixes: list[str] = []
    if not mentions_total:
        issues.append(f"With one worker, makespan is fixed at T1+T2+T3 = {total:g}h; state the completion time.")
        fixes.append(f"State makespan: {total:g}h total (single worker, no parallelism).")
    if not mentions_order or not has_tasks:
        issues.append("Schedule must respect the precedence (T2 after T1) and include all tasks.")
        fixes.append(f"Provide a valid order, e.g., T3 (0–{t3:g}h) → T1 ({t3:g}–{t3+t1:g}h) → T2 ({t3+t1:g}–{total:g}h).")

    if not issues:
        return _critic_payload(domain="constraints_schedule", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="constraints_schedule", issues=issues[:4], fixes=fixes[:8], confidence=0.88)


def _micro_policy_pii_log_sharing(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    if not ("log-sharing policy" in q_lower and "pii" in q_lower and "4" in q_lower and "rule" in q_lower):
        return None

    d_lower = str(draft or "").lower()
    bullet_hits = len(re.findall(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+", str(draft or "")))
    bullet_hits += len(re.findall(r"(?im)^\s*rule\s*\d+\b", str(draft or "")))
    has_4_rules = bullet_hits >= 4
    privacy_markers = ("pii", "redact", "mask", "hash", "anonym", "email", "phone", "ssn", "ip")
    debug_markers = ("request id", "correlation", "trace", "timestamp", "error", "stack", "endpoint")
    has_privacy = any(marker in d_lower for marker in privacy_markers)
    has_debug = any(marker in d_lower for marker in debug_markers)

    issues: list[str] = []
    fixes: list[str] = []
    if not has_4_rules:
        issues.append("Policy must include 4 concrete rules (bullet/numbered).")
        fixes.append("Provide 4 rules as bullets or numbered items.")
    if not has_privacy:
        issues.append("Missing concrete PII handling (redaction/masking/hashing of identifiers).")
        fixes.append("Include explicit PII redaction rules (emails/phones/IPs/usernames → mask/hash/remove).")
    if not has_debug:
        issues.append("Missing debugging-preserving fields (timestamps/request IDs/trace or correlation IDs).")
        fixes.append("Retain non-PII debugging fields: timestamp, request_id/correlation_id, endpoint, error codes, sanitized stack traces.")

    if not issues:
        return _critic_payload(domain="policy_pii_logs", issues=[], fixes=[], confidence=0.7)
    return _critic_payload(domain="policy_pii_logs", issues=issues[:5], fixes=fixes[:8], confidence=0.82)


def _micro_evidence_strength_study_quality(question: str, draft: str) -> Optional[MicroCriticResult]:
    q_lower = str(question or "").lower()
    if not ("study a" in q_lower and "study b" in q_lower):
        return None

    def _parse_n(segment: str) -> Optional[int]:
        m = re.search(r"\bn\s*=\s*([0-9][0-9,]*)\b", segment)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None

    a_match = re.search(r"study a(.*?)(?:study b|$)", q_lower, flags=re.DOTALL)
    seg_a = a_match.group(1) if a_match else ""
    seg_b = q_lower.split("study b", 1)[1] if "study b" in q_lower else ""
    na = _parse_n(seg_a)
    nb = _parse_n(seg_b)
    if na is None or nb is None:
        return None

    a_has_rct = any(tok in seg_a for tok in ("random", "randomized", "rct")) and ("control" in seg_a)
    b_has_rct = any(tok in seg_b for tok in ("random", "randomized", "rct")) and ("control" in seg_b)
    a_no_control = ("no control" in seg_a) or ("without control" in seg_a)
    b_no_control = ("no control" in seg_b) or ("without control" in seg_b)

    prefer = None
    if b_has_rct and a_no_control:
        prefer = "b"
    elif a_has_rct and b_no_control:
        prefer = "a"
    else:
        return None

    d_lower = str(draft or "").lower()
    mentions_a = bool(re.search(r"\bstudy\s*a\b", d_lower))
    mentions_b = bool(re.search(r"\bstudy\s*b\b", d_lower))
    quality_markers = (
        "random",
        "randomized",
        "rct",
        "control group",
        "control",
        "sample size",
        "n=",
        "bias",
        "confound",
    )
    mentions_quality = any(marker in d_lower for marker in quality_markers)

    issues: list[str] = []
    fixes: list[str] = []
    if prefer == "b":
        if not mentions_b:
            issues.append("Conclusion should weigh Study B more than Study A (RCT/control + reduced bias).")
            fixes.append("Prefer Study B: randomized control reduces bias/confounding; larger n improves power.")
        elif not mentions_quality:
            issues.append("Missing justification (randomization/control group + sample size/bias).")
            fixes.append("Justify: RCT + large n increases internal validity and statistical power; no-control small-n is highly biased.")
    else:
        if not mentions_a:
            issues.append("Conclusion should weigh Study A more than Study B (RCT/control + reduced bias).")
            fixes.append("Prefer Study A: randomized control reduces bias/confounding; larger n improves power.")
        elif not mentions_quality:
            issues.append("Missing justification (randomization/control group + sample size/bias).")
            fixes.append("Justify: RCT + large n increases internal validity and statistical power; no-control small-n is highly biased.")

    if not issues:
        return _critic_payload(domain="evidence_strength", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="evidence_strength", issues=issues[:4], fixes=fixes[:8], confidence=0.9)


def _micro_japanese_implication_transitivity(question: str, draft: str) -> Optional[MicroCriticResult]:
    q = str(question or "")
    pairs = re.findall(r"([A-Za-z])なら([A-Za-z])", q)
    if len(pairs) != 3:
        return None
    (a, b), (b2, c), (a2, c2) = pairs
    if not (a == a2 and b == b2 and c == c2):
        return None

    d_lower = str(draft or "").lower()
    valid_markers = (
        "valid",
        "hypothetical syllogism",
        "transit",
        "推移",
        "推移律",
        "妥当",
        "仮言三段論法",
    )
    invalid_markers = ("invalid", "fallacy", "不適", "誤り", "成り立たない", "導けない")
    mentions_valid = any(marker in d_lower for marker in valid_markers)
    mentions_invalid = any(marker in d_lower for marker in invalid_markers)

    issues: list[str] = []
    fixes: list[str] = []
    if mentions_invalid and not mentions_valid:
        issues.append("The implication chain A→B, B→C ⇒ A→C is logically valid (hypothetical syllogism / transitivity).")
        fixes.append(f"State it's valid (推移律 / 仮言三段論法): {a}なら{b} かつ {b}なら{c} なら {a}なら{c}.")
        fixes.append("Note caution: meanings/conditions for B must stay consistent in natural language.")
    elif not mentions_valid:
        issues.append("Missing a clear validity judgment (it is valid as a logical implication chain).")
        fixes.append("State validity: 妥当（推移律/仮言三段論法）。注意: 自然言語ではBの意味のズレ等に注意。")

    if not issues:
        return _critic_payload(domain="japanese_logic", issues=[], fixes=[], confidence=0.72)
    return _critic_payload(domain="japanese_logic", issues=issues[:4], fixes=fixes[:8], confidence=0.9)


def micro_criticise_reasoning(question: str, draft: str) -> Optional[MicroCriticResult]:
    """Return MicroCriticResult when a supported, high-confidence check applies."""

    detectors = (
        _micro_code_review_avg,
        _micro_causal_triage_plan,
        _micro_logic_all_some_inference,
        _micro_logic_affirming_consequent,
        _micro_logic_demorgan_not_and,
        _micro_algo_sort_complexity,
        _micro_constraints_single_worker_schedule,
        _micro_policy_pii_log_sharing,
        _micro_evidence_strength_study_quality,
        _micro_japanese_implication_transitivity,
        _micro_linear_equation,
        _micro_average_speed,
        _micro_revenue_growth,
        _micro_percent_change,
        _micro_what_percent_of,
        _micro_percent_adjust,
        _micro_percent_off,
        _micro_add_percent,
        _micro_fraction_to_percent,
        _micro_percent_of,
        _micro_temp_convert,
        _micro_time_unit_convert,
        _micro_length_unit_convert,
        _micro_inch_cm_convert,
        _micro_mass_unit_convert,
        _micro_mass_metric_convert,
        _micro_speed_unit_convert,
        _micro_speed_unit_convert_mph_kmph,
        _micro_length_metric_convert,
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
