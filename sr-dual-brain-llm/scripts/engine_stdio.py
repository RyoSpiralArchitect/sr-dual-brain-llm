#!/usr/bin/env python3
"""STDIO JSONL bridge for the dual-brain engine.

This keeps all orchestration logic inside the Python codebase while allowing a
stable external REST gateway (e.g., C# Minimal API) to drive experiments.

Protocol (1 JSON object per line):
Request:  {"id": "...", "method": "process|reset|health", "params": {...}}
Response: {"id": "...", "ok": true, "result": {...}}  or  {"id": "...", "ok": false, "error": {...}}
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.auditor import Auditor
from core.callosum import Callosum
from core.default_mode_network import DefaultModeNetwork
from core.dual_brain import DualBrainController
from core.executive_reasoner import ExecutiveReasonerModel
from core.hypothalamus import Hypothalamus
from core.llm_client import LLMConfig
from core.models import LeftBrainModel, RightBrainModel
from core.orchestrator import Orchestrator
from core.policy import RightBrainPolicy
from core.policy_modes import ReasoningDial
from core.prefrontal_cortex import PrefrontalCortex
from core.psychoid_attention import PsychoidAttentionAdapter
from core.shared_memory import SharedMemory
from core.temporal_hippocampal_indexing import TemporalHippocampalIndexing
from core.unconscious_field import UnconsciousField


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _last_event(events: list[dict[str, Any]], name: str) -> Optional[dict[str, Any]]:
    for ev in reversed(events or []):
        if ev.get("event") == name:
            return ev
    return None


def _extract_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    coherence_ev = _last_event(events, "coherence_signal") or {}
    policy_ev = _last_event(events, "policy_decision") or {}
    lead_ev = _last_event(events, "leading_brain") or {}
    complete_ev = _last_event(events, "interaction_complete") or {}
    arch_ev = _last_event(events, "architecture_path") or {}
    exec_ev = _last_event(events, "executive_reasoner") or {}

    signal = coherence_ev.get("signal") if isinstance(coherence_ev.get("signal"), dict) else {}

    active_modules: list[str] = []
    path = arch_ev.get("path") if isinstance(arch_ev, dict) else None
    if isinstance(path, list):
        mods = set()
        for stage in path:
            if not isinstance(stage, dict):
                continue
            modules = stage.get("modules")
            if not isinstance(modules, list):
                continue
            for item in modules:
                if item:
                    mods.add(str(item))
        active_modules = sorted(mods)

    metrics: dict[str, Any] = {
        "coherence": {
            "combined": signal.get("combined"),
            "tension": signal.get("tension"),
            "mode": signal.get("mode"),
        },
        "policy": {
            "action": policy_ev.get("action"),
            "temperature": policy_ev.get("temperature"),
            "slot_ms": policy_ev.get("slot_ms"),
        },
        "leading": lead_ev.get("leading"),
        "latency_ms": complete_ev.get("latency_ms"),
        "reward": complete_ev.get("reward"),
        "modules": {
            "active": active_modules,
            "count": len(active_modules),
        },
        "executive": {
            "confidence": exec_ev.get("confidence"),
            "latency_ms": exec_ev.get("latency_ms"),
            "source": exec_ev.get("source"),
        },
        "telemetry_events": len(events or []),
    }
    return metrics


class InMemoryTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.events.clear()

    def log(self, event: str, **payload: Any) -> None:
        self.events.append(
            {
                "event": str(event),
                "ts": time.time(),
                **_jsonable(payload),
            }
        )


async def _right_worker(callosum: Any, memory: SharedMemory, right_model: RightBrainModel) -> None:
    while True:
        req = await callosum.recv_request()
        if req.get("type") == "ASK_DETAIL":
            qid = req["qid"]
            try:
                detail = await right_model.deepen(
                    qid,
                    req["question"],
                    req.get("draft_sum", ""),
                    memory,
                    temperature=req.get("temperature", 0.7),
                    budget=req.get("budget", "small"),
                    context=req.get("context"),
                    psychoid_projection=req.get("psychoid_attention_bias"),
                )
                await callosum.publish_response(
                    qid,
                    {
                        "qid": qid,
                        "notes_sum": detail["notes_sum"],
                        "confidence_r": detail["confidence_r"],
                    },
                )
            except Exception as exc:
                await callosum.publish_response(qid, {"qid": qid, "error": str(exc)})
        elif req.get("type") == "ASK_LEAD":
            qid = req.get("qid")
            try:
                impression = await right_model.generate_lead(
                    req.get("question", ""),
                    req.get("context", ""),
                    temperature=float(req.get("temperature", 0.85)),
                )
            except Exception as exc:
                await callosum.publish_response(qid, {"qid": qid, "error": str(exc)})
            else:
                await callosum.publish_response(qid, {"qid": qid, "lead_notes": impression})


@dataclass
class EngineSession:
    session_id: str
    state_store: Any | None
    memory: SharedMemory
    hippocampus: TemporalHippocampalIndexing
    callosum: Callosum
    left: LeftBrainModel
    right: RightBrainModel
    executive: ExecutiveReasonerModel
    policy: RightBrainPolicy
    orchestrator: Orchestrator
    auditor: Auditor
    hypothalamus: Hypothalamus
    dial: ReasoningDial
    prefrontal: PrefrontalCortex
    unconscious: UnconsciousField
    default_mode: DefaultModeNetwork
    psychoid_adapter: PsychoidAttentionAdapter
    telemetry: InMemoryTelemetry
    controller: DualBrainController
    worker_task: asyncio.Task[None]
    persisted_memory: int = 0
    persisted_episodes: int = 0
    trace_cache: "OrderedDict[str, Dict[str, Any]]" = dataclasses.field(default_factory=OrderedDict)
    trace_cache_limit: int = 64

    def remember_trace(
        self,
        qid: str,
        *,
        metrics: Dict[str, Any],
        dialogue_flow: Any,
        executive: Any,
        telemetry: list[dict[str, Any]],
    ) -> None:
        self.trace_cache[qid] = {
            "qid": qid,
            "session_id": self.session_id,
            "metrics": metrics,
            "dialogue_flow": dialogue_flow,
            "executive": executive,
            "telemetry": telemetry,
            "ts": time.time(),
        }
        self.trace_cache.move_to_end(qid)
        while len(self.trace_cache) > self.trace_cache_limit:
            self.trace_cache.popitem(last=False)

    @classmethod
    async def create(
        cls,
        session_id: str,
        *,
        state_store: Any | None = None,
        left_llm_config: Optional[LLMConfig] = None,
        right_llm_config: Optional[LLMConfig] = None,
        executive_llm_config: Optional[LLMConfig] = None,
    ) -> "EngineSession":
        memory = SharedMemory()
        hippocampus = TemporalHippocampalIndexing()
        persisted_memory = 0
        persisted_episodes = 0
        if state_store is not None:
            loaded_memory, loaded_episodes = await state_store.load_session(
                session_id,
                memory_limit=memory.max_items,
            )
            memory.past_qas = list(loaded_memory)[-memory.max_items :]
            hippocampus.episodes = list(loaded_episodes)
            if hasattr(state_store, "load_schema_memories"):
                schema_limit = int(os.environ.get("DUALBRAIN_SCHEMA_MEMORY_LIMIT", "16") or 16)
                schema_memories = await state_store.load_schema_memories(
                    session_id, limit=schema_limit
                )
                memory.put_kv("schema_memories", schema_memories)
            persisted_memory = len(memory.past_qas)
            persisted_episodes = len(hippocampus.episodes)
        callosum = Callosum()
        left = LeftBrainModel(llm_config=left_llm_config)
        right = RightBrainModel(llm_config=right_llm_config)
        executive = ExecutiveReasonerModel(llm_config=executive_llm_config)
        policy = RightBrainPolicy()
        orchestrator = Orchestrator(2)
        auditor = Auditor()
        hypothalamus = Hypothalamus()
        dial_mode = os.environ.get("REASONING_DIAL", "evaluative")
        try:
            dial = ReasoningDial(mode=dial_mode)
        except AssertionError:
            dial = ReasoningDial(mode="evaluative")
        prefrontal = PrefrontalCortex()
        unconscious = UnconsciousField()
        default_mode = DefaultModeNetwork()
        psychoid_adapter = PsychoidAttentionAdapter()
        telemetry = InMemoryTelemetry()
        trace_cache_limit = int(os.environ.get("DUALBRAIN_TRACE_CACHE_SIZE", "64") or 64)
        trace_cache_limit = max(4, min(512, trace_cache_limit))

        controller = DualBrainController(
            callosum=callosum,
            memory=memory,
            left_model=left,
            right_model=right,
            executive_model=executive,
            policy=policy,
            hypothalamus=hypothalamus,
            reasoning_dial=dial,
            auditor=auditor,
            orchestrator=orchestrator,
            telemetry=telemetry,
            hippocampus=hippocampus,
            unconscious_field=unconscious,
            prefrontal_cortex=prefrontal,
            default_mode_network=default_mode,
            psychoid_attention_adapter=psychoid_adapter,
        )

        loop = asyncio.get_running_loop()
        worker_task = loop.create_task(_right_worker(callosum, memory, right))

        return cls(
            session_id=session_id,
            state_store=state_store,
            memory=memory,
            hippocampus=hippocampus,
            callosum=callosum,
            left=left,
            right=right,
            executive=executive,
            policy=policy,
            orchestrator=orchestrator,
            auditor=auditor,
            hypothalamus=hypothalamus,
            dial=dial,
            prefrontal=prefrontal,
            unconscious=unconscious,
            default_mode=default_mode,
            psychoid_adapter=psychoid_adapter,
            telemetry=telemetry,
            controller=controller,
            worker_task=worker_task,
            persisted_memory=persisted_memory,
            persisted_episodes=persisted_episodes,
            trace_cache_limit=trace_cache_limit,
        )

    async def close(self) -> None:
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass


async def _handle_process(session: EngineSession, params: Dict[str, Any]) -> Dict[str, Any]:
    question = str(params.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")

    leading = str(params.get("leading_brain", "auto") or "auto").strip().lower()
    leading_brain: Optional[str]
    if leading in {"auto", "", "null"}:
        leading_brain = None
    elif leading in {"left", "right"}:
        leading_brain = leading
    else:
        raise ValueError("leading_brain must be one of: auto, left, right")

    answer_mode = str(params.get("answer_mode", "plain") or "plain").strip().lower()
    if answer_mode not in {"plain", "debug", "annotated", "meta"}:
        raise ValueError("answer_mode must be one of: plain, debug, annotated, meta")

    executive_mode = str(params.get("executive_mode", "off") or "off").strip().lower()
    if executive_mode not in {"off", "observe", "polish"}:
        raise ValueError("executive_mode must be one of: off, observe, polish")

    return_telemetry = bool(params.get("return_telemetry", False))
    return_dialogue_flow = bool(params.get("return_dialogue_flow", True))

    qid = str(params.get("qid") or uuid.uuid4())

    session.telemetry.clear()

    answer = await session.controller.process(
        question,
        leading_brain=leading_brain,
        qid=qid,
        answer_mode=answer_mode,
        executive_mode=executive_mode,
    )
    after_memory = len(session.memory.past_qas)
    after_episodes = len(session.hippocampus.episodes)

    state_store = session.state_store
    if state_store is not None and (after_memory > session.persisted_memory or after_episodes > session.persisted_episodes):
        new_traces = session.memory.past_qas[session.persisted_memory:]
        new_episodes = session.hippocampus.episodes[session.persisted_episodes:]
        await state_store.append_memory_traces(session.session_id, new_traces, qid=qid)
        await state_store.upsert_episodes(session.session_id, new_episodes)
        session.persisted_memory = len(session.memory.past_qas)
        session.persisted_episodes = len(session.hippocampus.episodes)
    if state_store is not None and hasattr(state_store, "append_telemetry_events"):
        persist_telemetry = str(os.environ.get("DUALBRAIN_PG_PERSIST_TELEMETRY", "1")).strip().lower()
        if persist_telemetry not in {"0", "false", "no", "off"}:
            await state_store.append_telemetry_events(
                session.session_id,
                list(session.telemetry.events),
                qid=qid,
            )

    result: Dict[str, Any] = {
        "qid": qid,
        "answer": answer,
        "session_id": session.session_id,
        "metrics": _extract_metrics(list(session.telemetry.events)),
    }

    telemetry_events = list(session.telemetry.events)
    metrics = result["metrics"]
    dialogue_flow = session.memory.dialogue_flow(qid)
    executive = None
    if isinstance(dialogue_flow, dict):
        executive = dialogue_flow.get("executive")
    session.remember_trace(
        qid,
        metrics=metrics,
        dialogue_flow=dialogue_flow,
        executive=executive,
        telemetry=telemetry_events,
    )

    if return_dialogue_flow:
        result["dialogue_flow"] = dialogue_flow
    if return_telemetry:
        result["telemetry"] = telemetry_events
    return result


async def _handle_process_stream(
    session: EngineSession,
    params: Dict[str, Any],
    *,
    emit_event,
) -> Dict[str, Any]:
    question = str(params.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")

    leading = str(params.get("leading_brain", "auto") or "auto").strip().lower()
    leading_brain: Optional[str]
    if leading in {"auto", "", "null"}:
        leading_brain = None
    elif leading in {"left", "right"}:
        leading_brain = leading
    else:
        raise ValueError("leading_brain must be one of: auto, left, right")

    answer_mode = str(params.get("answer_mode", "plain") or "plain").strip().lower()
    if answer_mode not in {"plain", "debug", "annotated", "meta"}:
        raise ValueError("answer_mode must be one of: plain, debug, annotated, meta")

    executive_mode = str(params.get("executive_mode", "off") or "off").strip().lower()
    if executive_mode not in {"off", "observe", "polish"}:
        raise ValueError("executive_mode must be one of: off, observe, polish")

    return_telemetry = bool(params.get("return_telemetry", False))
    return_dialogue_flow = bool(params.get("return_dialogue_flow", True))

    qid = str(params.get("qid") or uuid.uuid4())

    session.telemetry.clear()

    def on_delta(text: str) -> None:
        if not text:
            return
        emit_event("delta", {"text": str(text)})

    def on_reset() -> None:
        emit_event("reset", {})

    answer = await session.controller.process(
        question,
        leading_brain=leading_brain,
        qid=qid,
        answer_mode=answer_mode,
        on_delta=on_delta,
        on_reset=on_reset,
        executive_mode=executive_mode,
    )

    after_memory = len(session.memory.past_qas)
    after_episodes = len(session.hippocampus.episodes)

    state_store = session.state_store
    if state_store is not None and (after_memory > session.persisted_memory or after_episodes > session.persisted_episodes):
        new_traces = session.memory.past_qas[session.persisted_memory:]
        new_episodes = session.hippocampus.episodes[session.persisted_episodes:]
        await state_store.append_memory_traces(session.session_id, new_traces, qid=qid)
        await state_store.upsert_episodes(session.session_id, new_episodes)
        session.persisted_memory = len(session.memory.past_qas)
        session.persisted_episodes = len(session.hippocampus.episodes)
    if state_store is not None and hasattr(state_store, "append_telemetry_events"):
        persist_telemetry = str(os.environ.get("DUALBRAIN_PG_PERSIST_TELEMETRY", "1")).strip().lower()
        if persist_telemetry not in {"0", "false", "no", "off"}:
            await state_store.append_telemetry_events(
                session.session_id,
                list(session.telemetry.events),
                qid=qid,
            )

    result: Dict[str, Any] = {
        "qid": qid,
        "answer": answer,
        "session_id": session.session_id,
        "metrics": _extract_metrics(list(session.telemetry.events)),
    }

    telemetry_events = list(session.telemetry.events)
    metrics = result["metrics"]
    dialogue_flow = session.memory.dialogue_flow(qid)
    executive = None
    if isinstance(dialogue_flow, dict):
        executive = dialogue_flow.get("executive")
    session.remember_trace(
        qid,
        metrics=metrics,
        dialogue_flow=dialogue_flow,
        executive=executive,
        telemetry=telemetry_events,
    )

    if return_dialogue_flow:
        result["dialogue_flow"] = dialogue_flow
    if return_telemetry:
        result["telemetry"] = telemetry_events
    return result


async def _handle_get_trace(
    sessions: Dict[str, EngineSession],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    session_id = str(params.get("session_id", "default") or "default")
    qid = str(params.get("qid") or "").strip()
    if not qid:
        raise ValueError("qid is required")

    include_telemetry = bool(params.get("include_telemetry", True))
    include_dialogue_flow = bool(params.get("include_dialogue_flow", True))
    include_executive = bool(params.get("include_executive", True))

    session = sessions.get(session_id)
    if session is None:
        return {"session_id": session_id, "qid": qid, "found": False}

    cached = session.trace_cache.get(qid)
    if cached is None:
        return {"session_id": session_id, "qid": qid, "found": False}

    result: Dict[str, Any] = {
        "session_id": session_id,
        "qid": qid,
        "found": True,
        "metrics": cached.get("metrics") or {},
        "ts": cached.get("ts"),
    }
    if include_dialogue_flow:
        result["dialogue_flow"] = cached.get("dialogue_flow")
    if include_telemetry:
        result["telemetry"] = cached.get("telemetry") or []
    if include_executive:
        result["executive"] = cached.get("executive")
    return result


async def _handle_search_episodes(
    sessions: Dict[str, EngineSession],
    params: Dict[str, Any],
    *,
    state_store: Any | None,
) -> Dict[str, Any]:
    session_id = str(params.get("session_id", "default") or "default")
    query = str(params.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")

    topk = int(params.get("topk") or 5)
    candidate_limit = int(params.get("candidate_limit") or 500)

    backend = "none"
    hits: list[tuple[float, Any]] = []

    if state_store is not None and hasattr(state_store, "search_episodes"):
        embedder = TemporalHippocampalIndexing()
        query_vector = embedder.embed_text(query)
        hits = await state_store.search_episodes(
            session_id,
            query_vector.tolist(),
            limit=topk,
            candidate_limit=candidate_limit,
        )
        backend = "pgvector" if getattr(state_store, "_has_pgvector", False) else "postgres"
    else:
        session = sessions.get(session_id)
        if session is not None:
            hits = session.hippocampus.retrieve(query, topk=topk)
            backend = "memory"

    results: list[dict[str, Any]] = []
    for similarity, trace in hits:
        results.append(
            {
                "similarity": float(similarity),
                "qid": getattr(trace, "qid", None),
                "question": getattr(trace, "question", None),
                "answer": getattr(trace, "answer", None),
                "ts": float(getattr(trace, "timestamp", 0.0) or 0.0),
                "leading": getattr(trace, "leading", None),
                "collaboration_strength": getattr(trace, "collaboration_strength", None),
                "selection_reason": getattr(trace, "selection_reason", None),
                "tags": list(getattr(trace, "tags", ()) or ()),
                "annotations": dict(getattr(trace, "annotations", {}) or {}),
                "embedding_version": getattr(trace, "embedding_version", None),
            }
        )

    return {
        "session_id": session_id,
        "query": query,
        "backend": backend,
        "results": results,
    }


async def _handle_query_telemetry(
    params: Dict[str, Any],
    *,
    state_store: Any | None,
) -> Dict[str, Any]:
    session_id = str(params.get("session_id", "default") or "default")
    limit = int(params.get("limit") or 250)
    qid = params.get("qid")
    event = params.get("event")
    since_ts = params.get("since_ts")
    until_ts = params.get("until_ts")

    if state_store is None or not hasattr(state_store, "query_telemetry_events"):
        return {"session_id": session_id, "events": [], "backend": "none"}

    events = await state_store.query_telemetry_events(
        session_id,
        limit=limit,
        qid=str(qid) if qid else None,
        event=str(event) if event else None,
        since_ts=float(since_ts) if since_ts is not None else None,
        until_ts=float(until_ts) if until_ts is not None else None,
    )
    return {"session_id": session_id, "events": events, "backend": "postgres"}


async def _handle_list_schema_memories(
    params: Dict[str, Any],
    *,
    state_store: Any | None,
) -> Dict[str, Any]:
    session_id = str(params.get("session_id", "default") or "default")
    limit = int(params.get("limit") or 16)

    if state_store is None or not hasattr(state_store, "load_schema_memories"):
        return {"session_id": session_id, "schema_memories": [], "backend": "none"}

    rows = await state_store.load_schema_memories(session_id, limit=limit)
    return {"session_id": session_id, "schema_memories": rows, "backend": "postgres"}


_SUPPORTED_PROVIDERS = {"openai", "google", "anthropic", "mistral", "xai", "huggingface"}


def _resolve_api_key(provider: str) -> str:
    provider_upper = provider.upper()
    candidates = [
        f"{provider_upper}_API_KEY",
        "LLM_API_KEY",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
    ]
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    raise ValueError(
        f"Missing API key for provider '{provider}'. Set {provider_upper}_API_KEY (or LLM_API_KEY)."
    )


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _build_llm_config(provider: str, model: str, params: Dict[str, Any], *, scope: str) -> LLMConfig:
    provider_norm = str(provider).strip().lower()
    if provider_norm not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider_norm}'")

    llm_overrides = params.get("llm") if isinstance(params.get("llm"), dict) else {}
    if "api_key" in llm_overrides:
        raise ValueError("Passing api_key in-request is not supported; use environment variables.")

    api_base = llm_overrides.get("api_base")
    organization = llm_overrides.get("organization") or os.environ.get("OPENAI_ORGANIZATION")
    scope_prefix = str(scope or "LLM").strip().upper()
    default_max_tokens = _coerce_int(
        os.environ.get(f"{scope_prefix}_MAX_TOKENS")
        or os.environ.get("LLM_MAX_OUTPUT_TOKENS"),
        default=1024,
    )
    max_output_tokens = _coerce_int(
        llm_overrides.get("max_output_tokens"),
        default=default_max_tokens,
    )
    default_timeout_seconds = _coerce_int(
        os.environ.get(f"{scope_prefix}_TIMEOUT") or os.environ.get("LLM_TIMEOUT"),
        default=40,
    )
    timeout_seconds = _coerce_int(
        llm_overrides.get("timeout_seconds"),
        default=default_timeout_seconds,
    )
    auto_continue = _coerce_bool(
        llm_overrides.get("auto_continue"),
        default=_coerce_bool(
            os.environ.get(f"{scope_prefix}_AUTO_CONTINUE") or os.environ.get("LLM_AUTO_CONTINUE"),
            default=True,
        ),
    )
    default_max_continuations = _coerce_int(
        os.environ.get(f"{scope_prefix}_MAX_CONTINUATIONS")
        or os.environ.get("LLM_MAX_CONTINUATIONS"),
        default=2,
    )
    max_continuations = _coerce_int(
        llm_overrides.get("max_continuations"),
        default=default_max_continuations,
    )

    extra_headers: Dict[str, str] = {}
    if provider_norm == "anthropic":
        extra_headers["anthropic-version"] = str(
            llm_overrides.get("anthropic_version")
            or os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
        )

    return LLMConfig(
        provider=provider_norm,
        model=str(model),
        api_key=_resolve_api_key(provider_norm),
        api_base=str(api_base) if api_base else None,
        organization=str(organization) if organization else None,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        auto_continue=auto_continue,
        max_continuations=max_continuations,
        extra_headers=extra_headers,
    )


def _maybe_extract_llm_configs(
    params: Dict[str, Any],
) -> tuple[Optional[LLMConfig], Optional[LLMConfig], Optional[LLMConfig]]:
    llm = params.get("llm")
    if llm is None:
        return None, None, None
    if not isinstance(llm, dict):
        raise ValueError("llm must be an object")

    provider = llm.get("provider")
    model = llm.get("model")
    if not provider or not model:
        raise ValueError("llm.provider and llm.model are required when llm is provided")

    left_model = llm.get("left_model") or model
    right_model = llm.get("right_model") or model
    executive_model = llm.get("executive_model") or llm.get("reasoner_model") or model

    left_cfg = _build_llm_config(str(provider), str(left_model), params, scope="LEFT_BRAIN")
    right_cfg = _build_llm_config(str(provider), str(right_model), params, scope="RIGHT_BRAIN")
    exec_cfg = _build_llm_config(str(provider), str(executive_model), params, scope="EXECUTIVE")
    return left_cfg, right_cfg, exec_cfg


def _llm_identity(config: Optional[LLMConfig]) -> tuple[Optional[str], Optional[str]]:
    if config is None:
        return None, None
    return config.provider, config.model


async def _handle_reset(
    sessions: Dict[str, EngineSession],
    params: Dict[str, Any],
    *,
    state_store: Any | None,
) -> Dict[str, Any]:
    session_id = str(params.get("session_id", "default") or "default")
    existing = sessions.pop(session_id, None)
    if existing is not None:
        await existing.close()
    if state_store is not None:
        await state_store.reset_session(session_id)
    # Do not eagerly recreate the session: the next /process call can choose the LLM config.
    return {"session_id": session_id, "reset": True, "session_created": False}


def _handle_health(sessions: Dict[str, EngineSession]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "sessions": len(sessions),
        "pid": os.getpid(),
    }


async def main() -> None:
    state_store = None
    pg_error: str | None = None
    pg_dsn = os.environ.get("DUALBRAIN_PG_DSN")
    if pg_dsn:
        try:
            from core.postgres_state_store import PostgresStateStore

            state_store = PostgresStateStore(
                pg_dsn,
                table_prefix=os.environ.get("DUALBRAIN_PG_TABLE_PREFIX", "srdb"),
            )
            await state_store.ensure_schema()
        except Exception as exc:  # pragma: no cover - best-effort optional backend
            pg_error = f"{exc.__class__.__name__}: {exc}"
            state_store = None
            print(
                f"[engine_stdio] Postgres disabled: {pg_error}",
                file=sys.stderr,
                flush=True,
            )

    sessions: Dict[str, EngineSession] = {}
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                req_id = str(req.get("id") or "")
                method = str(req.get("method") or "")
                params = req.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
            except Exception as exc:
                # Cannot reliably respond without a request id; emit to stderr.
                print(f"[engine_stdio] bad request: {exc}", file=sys.stderr, flush=True)
                continue

            try:
                if method == "process":
                    session_id = str(params.get("session_id", "default") or "default")
                    left_cfg, right_cfg, exec_cfg = _maybe_extract_llm_configs(params)
                    session = sessions.get(session_id)
                    if session is None:
                        session = await EngineSession.create(
                            session_id,
                            state_store=state_store,
                            left_llm_config=left_cfg,
                            right_llm_config=right_cfg,
                            executive_llm_config=exec_cfg,
                        )
                        sessions[session_id] = session
                    else:
                        if left_cfg is not None or right_cfg is not None or exec_cfg is not None:
                            requested = (
                                _llm_identity(left_cfg),
                                _llm_identity(right_cfg),
                                _llm_identity(exec_cfg),
                            )
                            current = (
                                _llm_identity(session.left.llm_config),
                                _llm_identity(session.right.llm_config),
                                _llm_identity(session.executive.llm_config),
                            )
                            if requested != current:
                                raise ValueError(
                                    "Session already exists with different LLM config; use a new session_id or call reset."
                                )
                    payload = await _handle_process(session, params)
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "process_stream":
                    session_id = str(params.get("session_id", "default") or "default")
                    left_cfg, right_cfg, exec_cfg = _maybe_extract_llm_configs(params)
                    session = sessions.get(session_id)
                    if session is None:
                        session = await EngineSession.create(
                            session_id,
                            state_store=state_store,
                            left_llm_config=left_cfg,
                            right_llm_config=right_cfg,
                            executive_llm_config=exec_cfg,
                        )
                        sessions[session_id] = session
                    else:
                        if left_cfg is not None or right_cfg is not None or exec_cfg is not None:
                            requested = (
                                _llm_identity(left_cfg),
                                _llm_identity(right_cfg),
                                _llm_identity(exec_cfg),
                            )
                            current = (
                                _llm_identity(session.left.llm_config),
                                _llm_identity(session.right.llm_config),
                                _llm_identity(session.executive.llm_config),
                            )
                            if requested != current:
                                raise ValueError(
                                    "Session already exists with different LLM config; use a new session_id or call reset."
                                )

                    def emit_event(event: str, payload: Dict[str, Any] | None = None) -> None:
                        msg: Dict[str, Any] = {"id": req_id, "event": event}
                        if payload:
                            msg.update(_jsonable(payload))
                        print(json.dumps(msg, ensure_ascii=False), flush=True)

                    payload = await _handle_process_stream(
                        session,
                        params,
                        emit_event=emit_event,
                    )
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "get_trace":
                    payload = await _handle_get_trace(sessions, params)
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "search_episodes":
                    payload = await _handle_search_episodes(
                        sessions, params, state_store=state_store
                    )
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "query_telemetry":
                    payload = await _handle_query_telemetry(params, state_store=state_store)
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "list_schema_memories":
                    payload = await _handle_list_schema_memories(
                        params, state_store=state_store
                    )
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "reset":
                    payload = await _handle_reset(sessions, params, state_store=state_store)
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                elif method == "health":
                    payload = _handle_health(sessions)
                    payload["postgres"] = {
                        "enabled": state_store is not None,
                        "error": pg_error,
                    }
                    resp = {"id": req_id, "ok": True, "result": _jsonable(payload)}
                else:
                    raise ValueError(f"unknown method: {method}")
            except Exception as exc:
                resp = {
                    "id": req_id,
                    "ok": False,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }

            print(json.dumps(resp, ensure_ascii=False), flush=True)
    finally:
        for session in list(sessions.values()):
            try:
                await session.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
