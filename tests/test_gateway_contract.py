import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PROJECT = REPO_ROOT / "csharp" / "SrDualBrain.Gateway" / "SrDualBrain.Gateway.csproj"
GATEWAY_LISTEN_URL = "http://127.0.0.1:0"

_CLEARED_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL_ID",
    "LLM_API_KEY",
    "LLM_API_BASE",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_TIMEOUT",
    "LLM_AUTO_CONTINUE",
    "LLM_MAX_CONTINUATIONS",
    "LEFT_BRAIN_PROVIDER",
    "LEFT_BRAIN_MODEL",
    "LEFT_BRAIN_API_KEY",
    "LEFT_BRAIN_API_BASE",
    "RIGHT_BRAIN_PROVIDER",
    "RIGHT_BRAIN_MODEL",
    "RIGHT_BRAIN_API_KEY",
    "RIGHT_BRAIN_API_BASE",
    "EXECUTIVE_PROVIDER",
    "EXECUTIVE_MODEL",
    "EXECUTIVE_API_KEY",
    "EXECUTIVE_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_ORGANIZATION",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_VERSION",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "MISTRAL_API_BASE",
    "XAI_API_KEY",
    "XAI_API_BASE",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
    "DUALBRAIN_PG_DSN",
    "DUALBRAIN_PG_TABLE_PREFIX",
    "DUALBRAIN_PG_PERSIST_TELEMETRY",
)


def _read_gateway_base_url(log_path: Path) -> str | None:
    logs = log_path.read_text(encoding="utf-8", errors="replace")
    for line in logs.splitlines():
        if "Now listening on:" not in line:
            continue
        raw_url = line.rsplit("Now listening on:", 1)[1].strip().rstrip("/")
        parsed = parse.urlparse(raw_url)
        if parsed.scheme in {"http", "https"} and parsed.hostname == "127.0.0.1" and parsed.port is not None:
            return f"{parsed.scheme}://127.0.0.1:{parsed.port}"
    return None


def _build_gateway() -> None:
    proc = subprocess.run(
        ["dotnet", "build", str(GATEWAY_PROJECT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "dotnet build failed for gateway contract tests.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw or "{}")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return int(exc.code), json.loads(raw or "{}")


def _read_sse(url: str, payload: dict[str, Any], *, timeout: float = 30.0) -> list[tuple[str, dict[str, Any]]]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    events: list[tuple[str, dict[str, Any]]] = []
    with request.urlopen(req, timeout=timeout) as resp:
        current_event: str | None = None
        data_lines: list[str] = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if current_event is not None:
                    data = json.loads("\n".join(data_lines) or "{}")
                    events.append((current_event, data))
                    if current_event == "done":
                        break
                current_event = None
                data_lines = []
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
    return events


def _wait_for_gateway(log_path: Path, proc: subprocess.Popen[str], *, timeout_s: float = 45.0) -> str:
    deadline = time.time() + timeout_s
    last_error = "gateway did not report its listening URL"
    while time.time() < deadline:
        if proc.poll() is not None:
            logs = log_path.read_text(encoding="utf-8", errors="replace")
            raise AssertionError(
                "Gateway process exited before becoming ready.\n"
                f"exit_code={proc.returncode}\n"
                f"logs:\n{logs}"
            )
        base_url = _read_gateway_base_url(log_path)
        if base_url is None:
            time.sleep(0.25)
            continue
        try:
            status, body = _request_json("GET", f"{base_url}/v1/health", timeout=2.0)
            if status == 200 and body.get("gateway") == "ok" and isinstance(body.get("engine"), dict):
                return base_url
            last_error = f"unexpected health payload: status={status} body={body}"
        except Exception as exc:  # pragma: no cover - polling noise is transient
            last_error = str(exc)
        time.sleep(0.25)

    logs = log_path.read_text(encoding="utf-8", errors="replace")
    raise AssertionError(f"{last_error}\nlogs:\n{logs}")


@pytest.fixture(scope="module")
def gateway_base_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    if shutil.which("dotnet") is None:
        pytest.skip("dotnet is required for gateway contract tests")

    _build_gateway()

    log_dir = tmp_path_factory.mktemp("gateway-contract")
    log_path = log_dir / "gateway.log"

    env = os.environ.copy()
    env["DUALBRAIN_REPO_ROOT"] = str(REPO_ROOT)
    env["DUALBRAIN_PYTHON"] = sys.executable
    env["DUALBRAIN_ENGINE_TRANSPORT"] = "stdio"
    env["DUALBRAIN_ENGINE_AUTOSTART"] = "0"
    env["DUALBRAIN_ENGINE_HEALTH_TTL_MS"] = "100"
    env["PYTHONUNBUFFERED"] = "1"
    for key in _CLEARED_ENV_VARS:
        env.pop(key, None)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                "dotnet",
                "run",
                "--no-build",
                "--no-launch-profile",
                "--project",
                str(GATEWAY_PROJECT),
                "--urls",
                GATEWAY_LISTEN_URL,
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            base_url = _wait_for_gateway(log_path, proc)
            yield base_url
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def test_health_endpoint_reports_gateway_and_engine(gateway_base_url: str) -> None:
    status, body = _request_json("GET", f"{gateway_base_url}/v1/health")

    assert status == 200
    assert body["gateway"] == "ok"
    assert body["engine"]["status"] == "ok"
    assert isinstance(body["engine"]["pid"], int)
    assert "postgres" in body["engine"]


def test_process_response_can_be_followed_by_trace_fetch(gateway_base_url: str) -> None:
    session_id = f"contract-process-{uuid.uuid4().hex[:8]}"
    status, body = _request_json(
        "POST",
        f"{gateway_base_url}/v1/process",
        payload={
            "session_id": session_id,
            "question": "Say hello in one short sentence.",
            "leading_brain": "auto",
            "return_telemetry": False,
        },
    )

    assert status == 200
    assert body["session_id"] == session_id
    assert isinstance(body["qid"], str) and body["qid"]
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert isinstance(body["metrics"], dict)
    assert isinstance(body["dialogue_flow"], dict)
    assert "telemetry" not in body

    qid = body["qid"]
    query = parse.urlencode(
        {
            "session_id": session_id,
            "include_telemetry": "true",
            "include_dialogue_flow": "true",
            "include_executive": "true",
        }
    )
    trace_status, trace = _request_json(
        "GET",
        f"{gateway_base_url}/v1/trace/{parse.quote(qid)}?{query}",
    )

    assert trace_status == 200
    assert trace["found"] is True
    assert trace["session_id"] == session_id
    assert trace["qid"] == qid
    assert isinstance(trace["metrics"], dict)
    assert isinstance(trace["telemetry"], list)
    assert isinstance(trace["dialogue_flow"], dict)


def test_stream_endpoint_emits_final_event_and_reset_clears_trace(gateway_base_url: str) -> None:
    session_id = f"contract-stream-{uuid.uuid4().hex[:8]}"
    events = _read_sse(
        f"{gateway_base_url}/v1/process/stream",
        {
            "session_id": session_id,
            "question": "Reply with a short greeting.",
            "leading_brain": "auto",
        },
    )

    event_names = [name for name, _ in events]
    assert event_names[0] == "start"
    assert "final" in event_names
    assert event_names[-1] == "done"
    assert "error" not in event_names

    final_payload = next(payload for name, payload in events if name == "final")
    assert final_payload["session_id"] == session_id
    assert isinstance(final_payload["qid"], str) and final_payload["qid"]
    assert isinstance(final_payload["answer"], str) and final_payload["answer"].strip()
    assert isinstance(final_payload["metrics"], dict)

    qid = final_payload["qid"]
    query = parse.urlencode({"session_id": session_id})
    trace_status, trace = _request_json(
        "GET",
        f"{gateway_base_url}/v1/trace/{parse.quote(qid)}?{query}",
    )
    assert trace_status == 200
    assert trace["found"] is True

    reset_status, reset = _request_json(
        "POST",
        f"{gateway_base_url}/v1/reset",
        payload={"session_id": session_id},
    )
    assert reset_status == 200
    assert reset["reset"] is True
    assert reset["session_id"] == session_id

    missing_status, missing_trace = _request_json(
        "GET",
        f"{gateway_base_url}/v1/trace/{parse.quote(qid)}?{query}",
    )
    assert missing_status == 404
    assert missing_trace["found"] is False
    assert missing_trace["session_id"] == session_id
    assert missing_trace["qid"] == qid
