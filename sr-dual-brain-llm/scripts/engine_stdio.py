#!/usr/bin/env python3
"""JSON-RPC bridge for the dual-brain engine.

This keeps all orchestration logic inside the Python codebase while allowing a
stable external REST gateway (e.g., C# Minimal API) to drive experiments.

Protocol (1 JSON object per line):
Request:  {"id": "...", "method": "process|reset|health", "params": {...}}
Response: {"id": "...", "ok": true, "result": {...}}  or  {"id": "...", "ok": false, "error": {...}}

When DUALBRAIN_PIPE_NAME is set, the engine switches to a length-prefixed JSON
protocol over a duplex Unix domain socket created by .NET Named Pipes.
Frame format: [u32 little-endian length][UTF-8 JSON payload]
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import os
import sys
import time
import tempfile
import struct
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
from core.director_reasoner import DirectorReasonerModel
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
    affect_ev = _last_event(events, "affective_state") or {}
    focus_ev = _last_event(events, "prefrontal_focus") or {}
    basal_ev = _last_event(events, "basal_ganglia") or {}
    unconscious_ev = _last_event(events, "unconscious_field") or {}
    projection_ev = _last_event(events, "psychoid_attention_projection") or {}
    neural_ev = _last_event(events, "neural_impulse_activity") or {}
    hemisphere_ev = _last_event(events, "hemisphere_routing") or {}
    hippo_rollup_ev = _last_event(events, "hippocampal_collaboration") or {}

    signal = coherence_ev.get("signal") if isinstance(coherence_ev.get("signal"), dict) else {}

    active_modules: list[str] = []
    stage_modules: list[dict[str, Any]] = []
    path = arch_ev.get("path") if isinstance(arch_ev, dict) else None
    if isinstance(path, list):
        mods = set()
        for stage in path:
            if not isinstance(stage, dict):
                continue
            stage_name = stage.get("stage")
            stage_mods = stage.get("modules")
            if not isinstance(stage_mods, list):
                stage_mods_list: list[str] = []
            else:
                stage_mods_list = [str(item) for item in stage_mods if item]
            if stage_name:
                stage_modules.append(
                    {
                        "stage": str(stage_name),
                        "modules": stage_mods_list,
                    }
                )
            modules = stage.get("modules")
            if not isinstance(modules, list):
                continue
            for item in modules:
                if item:
                    mods.add(str(item))
        active_modules = sorted(mods)

    affect_payload = {
        "valence": affect_ev.get("valence"),
        "arousal": affect_ev.get("arousal"),
        "risk": affect_ev.get("risk"),
        "novelty": affect_ev.get("novelty"),
        "hippocampal_density": affect_ev.get("hippocampal_density"),
    }
    focus_payload = {}
    raw_focus = focus_ev.get("focus") if isinstance(focus_ev.get("focus"), dict) else {}
    if isinstance(raw_focus, dict):
        focus_payload = {
            "keywords": raw_focus.get("keywords") if isinstance(raw_focus.get("keywords"), list) else [],
            "relevance": raw_focus.get("relevance"),
            "hippocampal_overlap": raw_focus.get("hippocampal_overlap"),
            "metric": focus_ev.get("metric"),
        }
    basal_payload = {}
    raw_basal = basal_ev.get("signal") if isinstance(basal_ev.get("signal"), dict) else {}
    if isinstance(raw_basal, dict):
        basal_payload = {
            "go_probability": raw_basal.get("go_probability"),
            "inhibition": raw_basal.get("inhibition"),
            "dopamine_level": raw_basal.get("dopamine_level"),
            "recommended_action": raw_basal.get("recommended_action"),
            "note": raw_basal.get("note"),
        }
    unconscious_payload = {}
    raw_unconscious = unconscious_ev.get("summary") if isinstance(unconscious_ev.get("summary"), dict) else {}
    if isinstance(raw_unconscious, dict):
        psychoid = raw_unconscious.get("psychoid_signal") if isinstance(raw_unconscious.get("psychoid_signal"), dict) else {}
        unconscious_payload = {
            "top_k": raw_unconscious.get("top_k") if isinstance(raw_unconscious.get("top_k"), list) else [],
            "stress_released": raw_unconscious.get("stress_released"),
            "cache_depth": raw_unconscious.get("cache_depth"),
            "psychoid_tension": psychoid.get("psychoid_tension") if isinstance(psychoid, dict) else None,
        }
    projection_payload = {}
    raw_projection = projection_ev.get("projection") if isinstance(projection_ev.get("projection"), dict) else {}
    if isinstance(raw_projection, dict):
        projection_payload = {
            "norm": raw_projection.get("norm"),
            "psychoid_tension": raw_projection.get("psychoid_tension"),
            "chain_length": raw_projection.get("chain_length"),
        }
    neural_payload = {}
    raw_neural = neural_ev.get("activity") if isinstance(neural_ev.get("activity"), dict) else {}
    if isinstance(raw_neural, dict):
        network = raw_neural.get("network_activity") if isinstance(raw_neural.get("network_activity"), dict) else {}
        neural_payload = {
            "hemisphere": raw_neural.get("hemisphere"),
            "total_impulses": raw_neural.get("total_impulses"),
            "stimulus_strength": raw_neural.get("stimulus_strength"),
            "active_ratio": network.get("active_ratio") if isinstance(network, dict) else None,
            "avg_membrane_potential": network.get("avg_membrane_potential") if isinstance(network, dict) else None,
        }
    hippo_rollup = hippo_rollup_ev.get("rollup") if isinstance(hippo_rollup_ev.get("rollup"), dict) else None
    hemisphere_payload = {
        "mode": hemisphere_ev.get("mode"),
        "intensity": hemisphere_ev.get("intensity"),
    }

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
            "stages": stage_modules,
        },
        "brain": {
            "amygdala": affect_payload,
            "prefrontal": focus_payload,
            "basal_ganglia": basal_payload,
            "unconscious": unconscious_payload,
            "psychoid": projection_payload,
            "neural": neural_payload,
            "hippocampus": {
                "total": complete_ev.get("hippocampal_total"),
                "rollup": hippo_rollup,
            },
            "hemisphere": hemisphere_payload,
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


class _EngineTransport:
    async def read(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def write(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class _StdioTransport(_EngineTransport):
    async def read(self) -> Optional[Dict[str, Any]]:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return None
        line = line.strip()
        if not line:
            return {}
        return json.loads(line)

    async def write(self, payload: Dict[str, Any]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


class _PipeTransport(_EngineTransport):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    @staticmethod
    def _path_for_pipe_name(pipe_name: str) -> str:
        base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        base = str(base)
        return os.path.join(base, f"CoreFxPipe_{pipe_name}")

    @classmethod
    async def connect(cls, *, pipe_name: str, timeout_s: float = 5.0) -> "_PipeTransport":
        explicit_path = os.environ.get("DUALBRAIN_PIPE_PATH")
        path = explicit_path or cls._path_for_pipe_name(pipe_name)

        deadline = time.monotonic() + max(0.2, float(timeout_s))
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                reader, writer = await asyncio.open_unix_connection(path)
                return cls(reader, writer)
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_err = exc
                await asyncio.sleep(0.05)
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(0.1)
        raise RuntimeError(
            f"Failed to connect to .NET named pipe '{pipe_name}' at '{path}': {last_err}"
        )

    async def read(self) -> Optional[Dict[str, Any]]:
        try:
            header = await self._reader.readexactly(4)
        except asyncio.IncompleteReadError:
            return None
        length = struct.unpack("<I", header)[0]
        if length == 0:
            return {}
        data = await self._reader.readexactly(int(length))
        return json.loads(data.decode("utf-8"))

    async def write(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = struct.pack("<I", len(data))
        self._writer.write(header)
        self._writer.write(data)
        await self._writer.drain()

    async def close(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass


class _BlobPipe:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    @staticmethod
    def _path_for_pipe_name(pipe_name: str) -> str:
        base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        base = str(base)
        return os.path.join(base, f"CoreFxPipe_{pipe_name}")

    @classmethod
    async def connect(cls, *, pipe_name: str, timeout_s: float = 5.0) -> "_BlobPipe":
        explicit_path = os.environ.get("DUALBRAIN_BLOB_PIPE_PATH")
        path = explicit_path or cls._path_for_pipe_name(pipe_name)

        deadline = time.monotonic() + max(0.2, float(timeout_s))
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                reader, writer = await asyncio.open_unix_connection(path)
                return cls(reader, writer)
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_err = exc
                await asyncio.sleep(0.05)
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(0.1)
        raise RuntimeError(
            f"Failed to connect to blob pipe '{pipe_name}' at '{path}': {last_err}"
        )

    async def close(self) -> None:
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass


async def _read_blob_frame(
    blob: _BlobPipe,
    *,
    expected_session_id: str | None = None,
    expected_blob_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    header_len_raw = await blob.reader.readexactly(4)
    header_len = struct.unpack("<I", header_len_raw)[0]
    if header_len <= 0 or header_len > 1024 * 64:
        raise ValueError(f"Invalid blob header length: {header_len}")
    header_bytes = await blob.reader.readexactly(int(header_len))
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid blob header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("Blob header must be a JSON object")

    data_len_raw = await blob.reader.readexactly(4)
    data_len = struct.unpack("<I", data_len_raw)[0]
    max_bytes = int(os.environ.get("DUALBRAIN_BLOB_MAX_BYTES", str(64 * 1024 * 1024)) or 0)
    if max_bytes > 0 and int(data_len) > max_bytes:
        # Drain and discard to keep the stream aligned.
        remaining = int(data_len)
        while remaining > 0:
            chunk = await blob.reader.readexactly(min(65536, remaining))
            remaining -= len(chunk)
        raise ValueError(f"Blob too large: {data_len} bytes (max {max_bytes})")

    session_id = str(header.get("session_id") or "").strip()
    blob_id = str(header.get("blob_id") or "").strip()
    if expected_session_id and session_id and session_id != expected_session_id:
        raise ValueError(f"Blob session_id mismatch: {session_id} != {expected_session_id}")
    if expected_blob_id and blob_id and blob_id != expected_blob_id:
        raise ValueError(f"Blob blob_id mismatch: {blob_id} != {expected_blob_id}")

    return header, {"data_len": int(data_len)}

def _safe_blob_id(blob_id: str) -> str:
    blob_id = str(blob_id or "").strip()
    if not blob_id:
        raise ValueError("blob_id is empty")
    lowered = blob_id.lower()
    allowed = "0123456789abcdef"
    if not (8 <= len(lowered) <= 64) or any(ch not in allowed for ch in lowered):
        raise ValueError(f"Invalid blob_id: {blob_id!r}")
    return blob_id


def _encode_blob_data_url(path: Path, *, content_type: str) -> tuple[str, str, int]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    ct = str(content_type or "").strip() or "application/octet-stream"
    return f"data:{ct};base64,{b64}", digest, len(data)


async def _resolve_vision_images(
    session_id: str,
    attachments: object,
    *,
    blob_root: Path,
    blob_index: Dict[str, Dict[str, Any]],
    vision_cache: "OrderedDict[str, str]",
    cache_limit: int,
) -> list[dict[str, Any]]:
    if not attachments:
        return []
    if not isinstance(attachments, list):
        raise ValueError("attachments must be an array")

    session_id = str(session_id or "default").strip() or "default"
    resolved: list[dict[str, Any]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        blob_id_raw = item.get("blob_id")
        if not blob_id_raw:
            continue
        blob_id = _safe_blob_id(str(blob_id_raw))
        meta = (blob_index.get(session_id) or {}).get(blob_id) if blob_index else None
        content_type = str(item.get("content_type") or (meta or {}).get("content_type") or "").strip()
        file_name = str(item.get("file_name") or (meta or {}).get("file_name") or "").strip() or None
        sha256 = str((meta or {}).get("sha256") or "").strip() or None

        path = blob_root / session_id / f"{blob_id}.blob"
        if not path.exists():
            raise ValueError(f"Attachment not found: session_id={session_id} blob_id={blob_id}")

        data_url = None
        if sha256 and sha256 in vision_cache:
            data_url = vision_cache[sha256]
            # Refresh LRU ordering.
            vision_cache.move_to_end(sha256)
        else:
            data_url, digest, size_bytes = await asyncio.to_thread(
                _encode_blob_data_url, path, content_type=content_type
            )
            sha256 = sha256 or digest
            if sha256:
                vision_cache[sha256] = data_url
                vision_cache.move_to_end(sha256)
                while len(vision_cache) > max(0, int(cache_limit or 0)):
                    vision_cache.popitem(last=False)
        resolved.append(
            {
                "blob_id": blob_id,
                "content_type": content_type or None,
                "file_name": file_name,
                "sha256": sha256,
                "data_url": data_url,
            }
        )
    return resolved


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
    director: DirectorReasonerModel
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
        executive_observer: Any,
        telemetry: list[dict[str, Any]],
    ) -> None:
        self.trace_cache[qid] = {
            "qid": qid,
            "session_id": self.session_id,
            "metrics": metrics,
            "dialogue_flow": dialogue_flow,
            "executive": executive,
            "executive_observer": executive_observer,
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
        director = DirectorReasonerModel(llm_config=executive_llm_config)
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
            director_model=director,
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
            director=director,
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


async def _handle_process(
    session: EngineSession,
    params: Dict[str, Any],
    *,
    blob_root: Path,
    blob_index: Dict[str, Dict[str, Any]],
    vision_cache: "OrderedDict[str, str]",
    vision_cache_limit: int,
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
    if executive_mode not in {"off", "observe", "assist", "polish"}:
        raise ValueError("executive_mode must be one of: off, observe, assist, polish")

    executive_observer_mode = str(params.get("executive_observer_mode", "off") or "off").strip().lower()
    if executive_observer_mode not in {"off", "metrics", "director", "both"}:
        raise ValueError("executive_observer_mode must be one of: off, metrics, director, both")

    return_telemetry = bool(params.get("return_telemetry", False))
    return_dialogue_flow = bool(params.get("return_dialogue_flow", True))
    return_executive = bool(params.get("return_executive", False))

    qid = str(params.get("qid") or uuid.uuid4())

    session.telemetry.clear()

    vision_images = await _resolve_vision_images(
        session.session_id,
        params.get("attachments"),
        blob_root=blob_root,
        blob_index=blob_index,
        vision_cache=vision_cache,
        cache_limit=vision_cache_limit,
    )
    answer = await session.controller.process(
        question,
        leading_brain=leading_brain,
        qid=qid,
        answer_mode=answer_mode,
        executive_mode=executive_mode,
        executive_observer_mode=executive_observer_mode,
        vision_images=vision_images or None,
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
    executive_observer = None
    if isinstance(dialogue_flow, dict):
        executive = dialogue_flow.get("executive")
        executive_observer = dialogue_flow.get("executive_observer")
    session.remember_trace(
        qid,
        metrics=metrics,
        dialogue_flow=dialogue_flow,
        executive=executive,
        executive_observer=executive_observer,
        telemetry=telemetry_events,
    )

    if return_dialogue_flow:
        result["dialogue_flow"] = dialogue_flow
    if return_telemetry:
        result["telemetry"] = telemetry_events
    if return_executive:
        result["executive"] = executive
        result["executive_observer"] = executive_observer
    return result


async def _handle_process_stream(
    session: EngineSession,
    params: Dict[str, Any],
    *,
    emit_event,
    blob_root: Path,
    blob_index: Dict[str, Dict[str, Any]],
    vision_cache: "OrderedDict[str, str]",
    vision_cache_limit: int,
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
    if executive_mode not in {"off", "observe", "assist", "polish"}:
        raise ValueError("executive_mode must be one of: off, observe, assist, polish")

    executive_observer_mode = str(params.get("executive_observer_mode", "off") or "off").strip().lower()
    if executive_observer_mode not in {"off", "metrics", "director", "both"}:
        raise ValueError("executive_observer_mode must be one of: off, metrics, director, both")

    return_telemetry = bool(params.get("return_telemetry", False))
    return_dialogue_flow = bool(params.get("return_dialogue_flow", True))
    return_executive = bool(params.get("return_executive", False))

    qid = str(params.get("qid") or uuid.uuid4())

    session.telemetry.clear()

    def on_delta(text: str) -> None:
        if not text:
            return
        emit_event("delta", {"text": str(text)})

    def on_reset() -> None:
        emit_event("reset", {})

    vision_images = await _resolve_vision_images(
        session.session_id,
        params.get("attachments"),
        blob_root=blob_root,
        blob_index=blob_index,
        vision_cache=vision_cache,
        cache_limit=vision_cache_limit,
    )
    answer = await session.controller.process(
        question,
        leading_brain=leading_brain,
        qid=qid,
        answer_mode=answer_mode,
        on_delta=on_delta,
        on_reset=on_reset,
        executive_mode=executive_mode,
        executive_observer_mode=executive_observer_mode,
        vision_images=vision_images or None,
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
    executive_observer = None
    if isinstance(dialogue_flow, dict):
        executive = dialogue_flow.get("executive")
        executive_observer = dialogue_flow.get("executive_observer")
    session.remember_trace(
        qid,
        metrics=metrics,
        dialogue_flow=dialogue_flow,
        executive=executive,
        executive_observer=executive_observer,
        telemetry=telemetry_events,
    )

    if return_dialogue_flow:
        result["dialogue_flow"] = dialogue_flow
    if return_telemetry:
        result["telemetry"] = telemetry_events
    if return_executive:
        result["executive"] = executive
        result["executive_observer"] = executive_observer
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
        result["executive_observer"] = cached.get("executive_observer")
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


async def _writer_loop(
    queue: "asyncio.Queue[Optional[Dict[str, Any]]]",
    transport: _EngineTransport,
) -> None:
    while True:
        msg = await queue.get()
        if msg is None:
            break
        try:
            await transport.write(msg)
        except Exception as exc:
            print(f"[engine_stdio] write failed: {exc}", file=sys.stderr, flush=True)

async def _handle_blob_put(
    params: Dict[str, Any],
    *,
    blob_pipe: _BlobPipe | None,
    blob_root: Path,
    blob_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if blob_pipe is None:
        raise ValueError("Blob pipe not available (enable gateway pipes transport).")

    session_id = str(params.get("session_id", "default") or "default").strip() or "default"
    expected_blob_id = str(params.get("blob_id") or "").strip()
    if not expected_blob_id:
        raise ValueError("blob_id is required")

    header, meta = await _read_blob_frame(
        blob_pipe,
        expected_session_id=session_id,
        expected_blob_id=expected_blob_id,
    )
    data_len = int(meta["data_len"])
    content_type = str(header.get("content_type") or "").strip() or None
    file_name = str(header.get("file_name") or "").strip() or None
    blob_id = str(header.get("blob_id") or expected_blob_id).strip() or expected_blob_id

    session_dir = blob_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{blob_id}.blob"

    hasher = hashlib.sha256()
    remaining = data_len
    with open(path, "wb") as f:
        while remaining > 0:
            chunk = await blob_pipe.reader.readexactly(min(65536, remaining))
            f.write(chunk)
            hasher.update(chunk)
            remaining -= len(chunk)

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "blob_id": blob_id,
        "content_type": content_type,
        "file_name": file_name,
        "size_bytes": data_len,
        "sha256": hasher.hexdigest(),
        "stored": True,
    }
    blob_index.setdefault(session_id, {})[blob_id] = payload
    return payload


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
    pipe_name = str(os.environ.get("DUALBRAIN_PIPE_NAME", "") or "").strip()
    transport: _EngineTransport
    if pipe_name:
        timeout_ms = int(os.environ.get("DUALBRAIN_PIPE_CONNECT_TIMEOUT_MS", "5000") or 5000)
        transport = await _PipeTransport.connect(
            pipe_name=pipe_name,
            timeout_s=max(0.2, float(timeout_ms) / 1000.0),
        )
        print(
            f"[engine_stdio] connected named pipe '{pipe_name}'",
            file=sys.stderr,
            flush=True,
        )
    else:
        transport = _StdioTransport()

    blob_pipe: _BlobPipe | None = None
    blob_pipe_name = str(os.environ.get("DUALBRAIN_BLOB_PIPE_NAME", "") or "").strip()
    if blob_pipe_name:
        timeout_ms = int(os.environ.get("DUALBRAIN_BLOB_PIPE_CONNECT_TIMEOUT_MS", "5000") or 5000)
        blob_pipe = await _BlobPipe.connect(
            pipe_name=blob_pipe_name,
            timeout_s=max(0.2, float(timeout_ms) / 1000.0),
        )
        print(
            f"[engine_stdio] connected blob pipe '{blob_pipe_name}'",
            file=sys.stderr,
            flush=True,
        )

    blob_root = Path(
        str(
            os.environ.get("DUALBRAIN_BLOB_DIR")
            or (Path(tempfile.gettempdir()) / "srdb_blobs")
        )
    )
    blob_root.mkdir(parents=True, exist_ok=True)
    blob_index: Dict[str, Dict[str, Any]] = {}
    vision_cache: "OrderedDict[str, str]" = OrderedDict()
    vision_cache_limit = int(os.environ.get("DUALBRAIN_VISION_CACHE_ITEMS", "4") or 4)
    vision_cache_limit = max(0, min(32, vision_cache_limit))

    out_queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
    writer_task = asyncio.create_task(_writer_loop(out_queue, transport))

    def _send(obj: Dict[str, Any]) -> None:
        out_queue.put_nowait(obj)

    try:
        while True:
            try:
                req = await transport.read()
                if req is None:
                    break
                if not req:
                    continue
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
                    payload = await _handle_process(
                        session,
                        params,
                        blob_root=blob_root,
                        blob_index=blob_index,
                        vision_cache=vision_cache,
                        vision_cache_limit=vision_cache_limit,
                    )
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
                        _send(msg)

                    payload = await _handle_process_stream(
                        session,
                        params,
                        emit_event=emit_event,
                        blob_root=blob_root,
                        blob_index=blob_index,
                        vision_cache=vision_cache,
                        vision_cache_limit=vision_cache_limit,
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
                elif method == "blob_put":
                    payload = await _handle_blob_put(
                        params,
                        blob_pipe=blob_pipe,
                        blob_root=blob_root,
                        blob_index=blob_index,
                    )
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

            _send(resp)
    finally:
        try:
            out_queue.put_nowait(None)
        except Exception:
            pass
        try:
            await writer_task
        except Exception:
            pass
        try:
            await transport.close()
        except Exception:
            pass
        try:
            if blob_pipe is not None:
                await blob_pipe.close()
        except Exception:
            pass

        for session in list(sessions.values()):
            try:
                await session.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
