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

import asyncio, random, difflib, json, re, ast, os
from typing import Any, Callable, Dict, Optional, Sequence

from .llm_client import CompletionValidationError, LLMClient, LLMConfig, load_llm_config
from .micro_critic import micro_criticise_reasoning
from .detail_contract import detail_result
from .task_goals import current_task, goal_review

_SYSTEM_GUARDRAILS = (
    "Answer the user's question directly.\n"
    "Output only the user-facing reply (no writing advice, no self-critique, no meta suggestions like 'Add...', 'Consider...', 'You should...').\n"
    "Do not claim real-world sensing or personal experiences (e.g., weather, your day) unless the user explicitly asked and you were given that data.\n"
    "Do not reference or speculate about hidden UI elements, logs, telemetry, metrics, or traces unless the user explicitly provided them in the chat.\n"
    "Do not introduce unrelated topics; if the user's message is ambiguous or very short, ask one brief clarifying question.\n"
    "Do not mention internal orchestration (e.g., left/right brain, corpus callosum, telemetry, architecture paths, qid).\n"
    "Do not add bracketed debug headings or meta commentary about the system.\n"
    "Do not mention system prompts, hidden instructions, or that you are a language model unless the user explicitly asks."
)

_RIGHT_DEEPEN_GUARDRAILS = (
    "When given a draft, return only additional content addressed to the user.\n"
    "Your output must be directly insertable into the final assistant message (no writing advice).\n"
    "Do not restate or paraphrase the draft.\n"
    "Do not say 'Add', 'Consider', 'You should', or similar coaching.\n"
    "Do not mention internal telemetry/metrics/traces or any system internals.\n"
    "Write in the user's language and tone.\n"
    "Prefer 1-5 concise bullet points of extra substance, or a short follow-up question."
)

_RIGHT_CRITIC_GUARDRAILS = (
    "You are a rigorous reviewer (critic) for reasoning tasks.\n"
    "Your job is to find concrete flaws in the draft's reasoning and propose fixes.\n"
    "Return a SINGLE JSON object with these keys:\n"
    '  - "verdict": "ok" | "issues"\n'
    '  - "issues": array of short strings (each a specific problem)\n'
    '  - "fixes": array of short strings (each a concrete fix)\n'
    "Rules:\n"
    "- Do NOT rewrite the full answer.\n"
    "- Do NOT add writing advice, tone advice, or coaching.\n"
    "- Do NOT introduce unrelated topics.\n"
    "- Be specific: point to missing definitions, invalid steps, unstated assumptions, boundary cases, and calculation errors.\n"
    "- Distinguish an attributed report of supplied records from a claim of independent verification. "
    "If a draft uses supported but unverified records, request source attribution and any necessary caveat "
    "rather than deleting their values solely for lack of external verification. "
    "Still flag unsupported values, lost conditions, contradictions and unjustified certainty.\n"
    "- Keep issues canonical: merge paraphrases of the same root cause into one issue.\n"
    "- Prefer <= 6 high-signal issues; avoid micro-fragmentation.\n"
    "- If there are no issues, set verdict='ok' and issues=[], fixes=[].\n"
    "- If verdict='issues', include at least one concrete issue; never return verdict='ok' with a non-empty issues list.\n"
    "- Output JSON only (no code fences, no preamble).\n"
)

_INTEGRATION_GUARDRAILS = (
    "You are producing the final assistant message for the user.\n"
    "You will be given a draft answer and internal notes from a collaborator.\n"
    "Incorporate any helpful content from the notes into a single coherent final answer.\n"
    "Prefer minimal necessary edits: fix correctness issues without broad rewrites, and keep length/voice consistent unless the user requested depth.\n"
    "Do not output the internal notes verbatim.\n"
    "Do not output coaching/critique about your own writing.\n"
    "Ignore any internal metrics/telemetry/traces if present in the notes.\n"
    "Do not mention internal orchestration.\n"
    "Match the user's language and tone unless they requested otherwise."
)


class _CriticSchemaError(ValueError):
    pass


def _unique_critic_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _CriticSchemaError("Duplicate critic field")
        result[key] = value
    return result


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        # Try to strip code fences.
        raw = raw.strip("`").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = raw[start : end + 1].strip()
    try:
        obj = json.loads(blob, object_pairs_hook=_unique_critic_fields)
    except _CriticSchemaError:
        raise
    except Exception:
        # Common LLM failure modes: trailing commas, Python dict syntax.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", blob)
        try:
            obj = json.loads(cleaned, object_pairs_hook=_unique_critic_fields)
        except _CriticSchemaError:
            raise
        except Exception:
            try:
                tree = ast.parse(cleaned, mode="eval")
                # Python-dict syntax is supported, but it must not erase an
                # earlier issue by overwriting a duplicate field either.
                for node in ast.walk(tree):
                    if isinstance(node, ast.Dict):
                        _unique_critic_fields((ast.literal_eval(key), None) for key in node.keys)
                obj = ast.literal_eval(tree)
            except _CriticSchemaError:
                raise
            except Exception:
                return None
    return obj if isinstance(obj, dict) else None


def _normalise_critic_item(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    value = re.sub(r"[\"'`“”‘’]", "", value)
    value = re.sub(r"[，、。．,;:.!?]+$", "", value)
    return value


def _dedupe_critic_items(items: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen_norm: list[str] = []
    for item in items:
        text = str(item or "").strip()
        norm = _normalise_critic_item(text)
        if not norm:
            continue
        duplicated = False
        for prior in seen_norm:
            if norm == prior:
                duplicated = True
                break
            # Near-duplicate phrasing from the same critic pass should collapse.
            if difflib.SequenceMatcher(None, norm, prior).ratio() >= 0.9:
                duplicated = True
                break
        if duplicated:
            continue
        out.append(text)
        seen_norm.append(norm)
        if len(out) >= limit:
            break
    return out


def _validated_critic_response(text):
    """Validate shape before normalization can discard malformed issue fields."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response" if isinstance(text, str) else "invalid_schema"
    try:
        parsed = _extract_json_object(text)
    except _CriticSchemaError:
        return None, "invalid_schema"
    if parsed is None:
        return None, "unstructured"
    if (not isinstance(parsed.get("verdict"), str)
            or parsed["verdict"].strip().lower() not in {"ok", "issues"}
            or any(not isinstance(parsed.get(key), list)
                   or any(not isinstance(item, str) or not item.strip() for item in parsed[key])
                   for key in ("issues", "fixes"))):
        return None, "invalid_schema"
    verdict = parsed["verdict"].strip().lower()
    issues = _dedupe_critic_items(parsed["issues"], limit=12)
    fixes = _dedupe_critic_items(parsed["fixes"], limit=12)
    if ((verdict == "ok" and (parsed["issues"] or parsed["fixes"]))
            or (verdict == "issues" and not issues)):
        return None, "inconsistent"
    return (verdict, issues, fixes), "ok"


def _critic_summary(issues, fixes):
    lines = []
    for heading, items in (("Issues:", issues), ("Fixes:", fixes)):
        if items:
            lines.append(heading)
            lines.extend(f"- {item}" for item in items[:10])
    return "\n".join(lines).strip()

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)

def _openai_vision_prompt(
    text: str,
    vision_images: Sequence[Dict[str, Any]],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": str(text)}]
    for image in vision_images or ():
        if not isinstance(image, dict):
            continue
        url = str(image.get("data_url") or image.get("url") or "").strip()
        if not url:
            continue
        image_url: dict[str, Any] = {"url": url}
        detail = image.get("detail")
        if detail:
            image_url["detail"] = str(detail)
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


class LeftBrainModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or load_llm_config("LEFT_BRAIN") or load_llm_config()
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def generate_source_answer(self, input_text, context, **kwargs):
        """Primary source writes require a nonempty, terminally complete answer."""
        return await self.generate_answer(input_text, context, require_success=True, **kwargs)

    async def generate_answer(
        self,
        input_text: str,
        context: str,
        *,
        vision_images: Optional[Sequence[Dict[str, Any]]] = None,
        on_delta: Optional[Callable[[str], Any]] = None,
        require_success: bool = False,
    ) -> str:
        """Produce a first-pass draft that reflects retrieved memory snippets."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            task = current_task(input_text) if not require_success else None
            if task is not None:
                system_parts.append(task.prompt('left'))
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            try:
                prompt: Any = input_text
                if vision_images:
                    provider = str(self.llm_config.provider if self.llm_config else "").lower()
                    if provider and provider not in {"openai", "mistral", "xai"}:
                        raise ValueError(
                            f"Vision inputs are only supported for openai-style providers; got '{provider}'."
                        )
                    prompt = _openai_vision_prompt(input_text, vision_images)
                completion_kwargs = {"require_complete": True} if require_success else {}
                if hasattr(self._llm_client, "complete_stream"):
                    completion = await self._llm_client.complete_stream(
                        prompt,
                        system=system,
                        temperature=0.4,
                        on_delta=on_delta,
                        **completion_kwargs,
                    )
                else:
                    completion = await self._llm_client.complete(
                        prompt,
                        system=system,
                        temperature=0.4,
                        **completion_kwargs,
                    )
                    if on_delta:
                        maybe = on_delta(completion)
                        if asyncio.iscoroutine(maybe):
                            await maybe
                if require_success and (not isinstance(completion, str) or not completion.strip()):
                    raise ValueError("Source generation returned no text")
                return completion
            except Exception:
                if require_success:
                    raise RuntimeError("Primary source generation failed") from None
                pass

        if require_success:
            raise RuntimeError("Primary source generation unavailable")

        base = f"Draft answer for: {input_text[:80]}"
        if context:
            summary = context.splitlines()[0][:80]
            base += f"\nContext hint: {summary}"
        if any(k in input_text for k in ["詳しく","分析","計算","証拠"]):
            base += " ... (brief, looks complex)"
        if on_delta:
            maybe = on_delta(base)
            if asyncio.iscoroutine(maybe):
                await maybe
        return base

    def estimate_confidence(self, draft: str) -> float:
        return 0.45 if "..." in draft else 0.9

    def integrate_info(self, draft: str, info: str) -> str:
        if not info:
            return draft
        text = str(info or "").strip()
        if not text:
            return draft

        lower = text.lower()
        coaching_markers = (
            "add a ",
            "consider ",
            "if the user",
            "you should",
            "try to",
            "make sure to",
            "もう少し",
            "〜したほうが",
            "した方が",
        )
        if any(marker in lower for marker in coaching_markers):
            # Avoid leaking internal "advisor" notes when no integration LLM is available.
            return draft

        # Avoid "split voice" by refusing to append near-duplicate full answers.
        def _norm(value: str) -> str:
            return "".join(str(value or "").split()).lower()

        try:
            similarity = difflib.SequenceMatcher(None, _norm(draft), _norm(text)).ratio()
        except Exception:
            similarity = 0.0
        if similarity >= 0.62 and len(text) <= int(len(draft) * 2.2):
            return draft

        return f"{draft}\n\n{text}"

    async def integrate_info_async(
        self,
        *,
        question: str,
        draft: str,
        info: str,
        temperature: float = 0.3,
        on_delta: Optional[Callable[[str], Any]] = None,
    ) -> str:
        if not info:
            return draft

        if not self._llm_client:
            merged = self.integrate_info(draft, info)
            if on_delta:
                maybe = on_delta(merged)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return merged

        system_parts = [_SYSTEM_GUARDRAILS, _INTEGRATION_GUARDRAILS]
        task = current_task(question)
        if task is not None:
            system_parts.append(task.prompt('integration', draft=draft))
        system = "\n\n".join(system_parts)
        prompt = (
            f"User message:\n{question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Internal collaborator notes (do not output directly):\n"
            f"{info}\n\n"
            "Final answer:"
        )
        try:
            if hasattr(self._llm_client, "complete_stream"):
                completion = await self._llm_client.complete_stream(
                    prompt,
                    system=system,
                    temperature=temperature,
                    on_delta=on_delta,
                )
            else:
                completion = await self._llm_client.complete(
                    prompt,
                    system=system,
                    temperature=temperature,
                )
                if on_delta:
                    maybe = on_delta(completion)
                    if asyncio.iscoroutine(maybe):
                        await maybe
        except Exception:
            return self.integrate_info(draft, info)
        return completion.strip() or draft

class RightBrainModel:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or load_llm_config("RIGHT_BRAIN") or load_llm_config()
        self._llm_client = LLMClient(self.llm_config) if self.llm_config else None
        self.uses_external_llm = bool(self._llm_client)

    async def generate_lead(
        self,
        question: str,
        context: Optional[str] = None,
        *,
        temperature: float = 0.8,
        on_delta: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """Produce an imagistic first impression before the left brain speaks."""
        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS]
            task = current_task(question)
            if task is not None:
                system_parts.append(task.prompt('right'))
            if context:
                system_parts.append(f"Context:\n{context}")
            system = "\n\n".join(system_parts)
            max_tokens = 220
            timeout_seconds = min(float(self.llm_config.timeout_seconds if self.llm_config else 40), 18.0)
            try:
                if hasattr(self._llm_client, "complete_stream"):
                    return await self._llm_client.complete_stream(
                        question,
                        system=system,
                        temperature=temperature,
                        on_delta=on_delta,
                        max_output_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                completion = await self._llm_client.complete(
                    question,
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                if on_delta:
                    maybe = on_delta(completion)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                return completion
            except Exception:
                pass

        await asyncio.sleep(0.25 + random.random() * 0.2)
        base = f"First impression: {question[:80]}"
        if context:
            snippet = context.splitlines()[0][:80]
            base += f"\nContext echo: {snippet}"
        base += "\nKey images: connection, contrast, hidden assumptions."
        if on_delta:
            maybe = on_delta(base)
            if asyncio.iscoroutine(maybe):
                await maybe
        return base

    async def deepen(
        self,
        qid: str,
        question: str,
        partial_answer: str,
        shared_memory,
        *,
        temperature: float = 0.7,
        budget: str = "small",
        context: Optional[str] = None,
        psychoid_projection: Optional[Dict[str, object]] = None,
    ) -> Dict[str, Any]:
        if self._llm_client:
            budget_norm = str(budget or "small").strip().lower()
            max_tokens = 450 if budget_norm != "large" else 900
            timeout_seconds = min(
                float(self.llm_config.timeout_seconds if self.llm_config else 40),
                22.0 if budget_norm != "large" else 32.0,
            )
            system_parts = []
            system_parts.append(_SYSTEM_GUARDRAILS)
            system_parts.append(_RIGHT_DEEPEN_GUARDRAILS)
            task = current_task(question)
            if task is not None:
                system_parts.append(task.prompt('right'))
            if context:
                system_parts.append(f"Context:\n{context}")
            if psychoid_projection:
                system_parts.append(f"Psychoid projection: {psychoid_projection}")
            system = "\n\n".join(system_parts) if system_parts else None
            try:
                completion = await self._llm_client.complete(
                    (
                        f"{question}\n\n"
                        "Draft (for reference only):\n"
                        f"{partial_answer}\n\n"
                        "Return only user-facing additive content (no coaching, no critique)."
                    ),
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    require_complete=True,
                    allow_continuation=False,
                )
                return detail_result(qid, completion)
            except CompletionValidationError as exc:
                return detail_result(qid, status=exc.reason, error=exc)
            except Exception as exc:
                return detail_result(qid, status='failed', error=exc)

        await asyncio.sleep(0.35 + random.random()*0.3)
        context_text = context if context is not None else shared_memory.retrieve_related(question)
        snippet = context_text[:120]
        detail = (
            f"Deeper take: {question}\n"
            f"- What matters: clarify definitions and the user's goal.\n"
            f"- Hidden assumptions: check what's being held constant.\n"
            f"- Context hint: {snippet}"
        )
        return detail_result(qid, detail, kind='heuristic')

    async def criticise_reasoning(
        self,
        qid: str,
        question: str,
        draft: str,
        *,
        temperature: float = 0.2,
        context: Optional[str] = None,
        psychoid_projection: Optional[Dict[str, object]] = None,
        allow_micro_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Return a structured critique for System2-style reasoning corrections.

        Review validity describes a completed, well-formed review or an explicit
        local micro result, not factual correctness of the draft. External
        validity and each validation stage remain separate from that local result.
        """

        allow_micro = bool(
            allow_micro_fallback and _env_flag("DUALBRAIN_CRITIC_MICRO_FALLBACK", True)
        )
        micro = None
        if allow_micro:
            micro = micro_criticise_reasoning(question, draft)

        if self._llm_client:
            system_parts = [_SYSTEM_GUARDRAILS, _RIGHT_CRITIC_GUARDRAILS]
            task = current_task(question)
            if task is not None:
                system_parts.append(task.prompt('critic'))
            if context:
                system_parts.append(f"Context:\n{context}")
            if psychoid_projection:
                system_parts.append(f"Psychoid projection: {psychoid_projection}")
            system = "\n\n".join(system_parts)
            prompt = (
                "User question:\n"
                f"{question}\n\n"
                "Draft answer (review for correctness):\n"
                f"{draft}\n\n"
                "Return JSON only."
            )
            # Reasoning models also spend this budget on hidden reasoning tokens.
            # Keep the ordinary default while allowing a measured run to reserve
            # enough room for the structured JSON response.
            try:
                max_tokens = int(os.environ.get("DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS", "520"))
            except ValueError:
                max_tokens = 520
            max_tokens = max(256, min(max_tokens, 16384))
            try:
                critic_timeout = float(os.environ.get("DUALBRAIN_CRITIC_TIMEOUT_SECONDS", "24"))
            except ValueError:
                critic_timeout = 24.0
            timeout_seconds = min(
                float(self.llm_config.timeout_seconds if self.llm_config else 40),
                max(1.0, min(critic_timeout, 120.0)),
            )
            completion = ""
            parsed = None
            status = "provider_error"
            error_type = None
            for attempt in range(3):
                try:
                    completion = await self._llm_client.complete(
                        prompt,
                        system=system,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                        require_complete=True,
                    )
                except CompletionValidationError as exc:
                    # A returned response with invalid completion is not a
                    # transport retry candidate, even if its text looks clean.
                    status, error_type = exc.reason, type(exc).__name__
                    break
                except Exception as exc:
                    status, error_type = "provider_error", type(exc).__name__
                    retryable = any(marker in str(exc).lower() for marker in
                                    ("rate limit", "rate_limited", "429", "timeout"))
                    if retryable and attempt < 2:
                        await asyncio.sleep(0.35 * (2 ** attempt) + random.random() * 0.2)
                        continue
                    break
                else:
                    parsed, status = _validated_critic_response(completion)
                    error_type = None
                    break

            validation = {"initial": {"status": status, "error_type": error_type}, "repair": None}
            goal_completion = completion
            critic_kind = "external"
            if (status == "unstructured"
                    and not (micro is not None and micro.verdict == "issues")
                    and _env_flag("DUALBRAIN_CRITIC_JSON_REPAIR_CALL", True)):
                # Preserve the existing single reformat attempt for unstructured
                # text. New shape/completion failures never add a repair call.
                repair_blob = completion.strip()
                if len(repair_blob) > 1400:
                    repair_blob = repair_blob[:1400] + "..."
                repair_prompt = (
                    "Your previous response was not valid JSON.\n"
                    "Convert it into a SINGLE JSON object with keys "
                    '\"verdict\", \"issues\", \"fixes\".\n'
                    "Output JSON only.\n\n"
                    "Unstructured response:\n"
                    f"{repair_blob}\n"
                )
                try:
                    repaired = await self._llm_client.complete(
                        repair_prompt,
                        system=system,
                        temperature=max(0.0, float(temperature) * 0.6),
                        max_output_tokens=max_tokens,
                        timeout_seconds=min(timeout_seconds, 18.0),
                        require_complete=True,
                    )
                except CompletionValidationError as exc:
                    status, error_type = exc.reason, type(exc).__name__
                except Exception as exc:
                    status, error_type = "provider_error", type(exc).__name__
                else:
                    parsed, status = _validated_critic_response(repaired)
                    error_type = None
                    if parsed is not None:
                        critic_kind = "external_repaired"
                        goal_completion = repaired
                validation["repair"] = {"status": status, "error_type": error_type}

            external_valid = parsed is not None and status == "ok"
            review_valid = external_valid
            if external_valid:
                verdict, issues, fixes = parsed
                critic_sum = _critic_summary(issues, fixes) or "No issues detected."
            else:
                verdict = "issues"
                descriptions = {
                    "provider_error": "External critic model unavailable",
                    "empty_response": "External critic returned an empty response",
                    "incomplete_response": "External critic completion was incomplete",
                    "invalid_response": "External critic response envelope was invalid",
                    "refused_response": "External critic refused the review",
                    "invalid_schema": "External critic response schema was invalid",
                    "inconsistent": "External critic response was inconsistent",
                    "unstructured": "External critic response unstructured",
                }
                # Keep the existing downstream unhealthy marker, and never
                # place provider exception text in a fallback summary.
                detail = descriptions.get(status, "External critic response was invalid")
                critic_sum = f"{detail}; unable to verify reasoning."
                issues = [f"(fallback) {critic_sum}"]
                fixes = []
                if micro is not None and micro.verdict == "issues" and status != "empty_response":
                    verdict, issues, fixes = micro.verdict, list(micro.issues), list(micro.fixes)
                    critic_sum = micro.critic_sum
                    critic_kind = "micro"
                    review_valid = True

            # High-confidence sanity override: the explicit local result keeps
            # its own provenance even when a valid external critic missed it.
            if (external_valid and verdict == "ok" and micro is not None
                    and micro.verdict == "issues"
                    and _env_flag("DUALBRAIN_CRITIC_MICRO_SANITY_OVERRIDE", True)):
                verdict, issues, fixes = micro.verdict, list(micro.issues), list(micro.fixes)
                critic_sum = micro.critic_sum
                critic_kind = "micro_sanity_override"
            result = {
                "qid": qid,
                "verdict": verdict,
                "issues": issues,
                "fixes": fixes,
                "critic_sum": critic_sum,
                "confidence_r": (float(micro.confidence_r) if critic_kind.startswith("micro")
                                 else 0.9 if external_valid else 0.0),
                "critic_kind": critic_kind,
                "critic_status": status,
                "critic_review_valid": review_valid,
                "critic_external_review_valid": external_valid,
                "critic_validation": validation,
            }
            if task is not None:
                try:
                    goal_data = _extract_json_object(goal_completion) if external_valid else None
                except (_CriticSchemaError, TypeError, ValueError):
                    goal_data = None
                result['goal_review'] = goal_review(task.plan, draft,
                    goal_data.get('goal_findings') if isinstance(goal_data, dict) else None,
                    external_valid=external_valid,
                    verdict=str(goal_data.get('verdict') or '').strip().lower() if isinstance(goal_data, dict) else None)
            return result

        # Deterministic fallback when no external critic is configured.
        if micro is not None:
            return {
                "qid": qid,
                "verdict": micro.verdict,
                "issues": list(micro.issues),
                "fixes": list(micro.fixes),
                "critic_sum": micro.critic_sum,
                "confidence_r": float(micro.confidence_r),
                "critic_kind": "micro_offline",
                "critic_status": "offline",
                "critic_review_valid": True,
                "critic_external_review_valid": False,
                "critic_validation": {"initial": {"status": "not_requested", "error_type": None}, "repair": None},
            }
        return {
            "qid": qid,
            "verdict": "issues",
            "issues": [
                "(fallback) External critic model not configured; unable to verify reasoning."
            ],
            "fixes": ["Configure a RIGHT_BRAIN LLM provider/model or retry later."],
            "critic_sum": "External critic not configured.",
            "confidence_r": 0.2,
            "critic_kind": "disabled",
            "critic_status": "not_configured",
            "critic_review_valid": False,
            "critic_external_review_valid": False,
            "critic_validation": {"initial": {"status": "not_requested", "error_type": None}, "repair": None},
        }

        # Heuristic fallback (no external LLM available).
        await asyncio.sleep(0.15 + random.random() * 0.15)
        return {
            "qid": qid,
            "verdict": "issues",
            "issues": ["(fallback) External critic model unavailable; unable to verify reasoning."],
            "fixes": ["Consider enabling an external LLM for System2 critic mode."],
            "critic_sum": "External critic model unavailable.",
            "confidence_r": 0.35,
        }
