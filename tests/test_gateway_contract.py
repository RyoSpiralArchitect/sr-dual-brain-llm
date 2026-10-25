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


def test_read_gateway_base_url_extracts_loopback_ephemeral_port(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text(
        "\n".join(
            [
                "info: Microsoft.Hosting.Lifetime[14]",
                "      Now listening on: http://127.0.0.1:43123",
                "info: Microsoft.Hosting.Lifetime[0]",
            ]
        ),
        encoding="utf-8",
    )

    assert _read_gateway_base_url(log_path) == "http://127.0.0.1:43123"


def test_read_gateway_base_url_ignores_non_loopback_or_unbound_logs(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"
    log_path.write_text(
        "\n".join(
            [
                "Now listening on: http://0.0.0.0:43123",
                "Now listening on: http://localhost:43124",
                "Application started. Press Ctrl+C to shut down.",
            ]
        ),
        encoding="utf-8",
    )

    assert _read_gateway_base_url(log_path) is None


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


@pytest.fixture(scope="module", params=["stdio", "pipes"])
def cancellation_gateway(request, tmp_path_factory):
    """Real HTTP/.NET and real framed transport; no model/provider calls."""
    if shutil.which("dotnet") is None:
        pytest.skip("dotnet is required for gateway contract tests")
    _build_gateway()
    directory = tmp_path_factory.mktemp(f"gateway-cancel-{request.param}")
    rpc_log, log_path, script = directory / "rpc.jsonl", directory / "gateway.log", directory / "engine.py"
    script.write_text('''import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["DUALBRAIN_REPO_ROOT"]) / "sr-dual-brain-llm" / "scripts"))
from engine_stdio import _StdioTransport, _PipeTransport, _BlobPipe, _read_blob_frame

async def main():
    pipe_name = os.environ.get("DUALBRAIN_PIPE_NAME")
    transport = await _PipeTransport.connect(pipe_name=pipe_name) if pipe_name else _StdioTransport()
    blob = await _BlobPipe.connect(pipe_name=os.environ["DUALBRAIN_BLOB_PIPE_NAME"]) if pipe_name else None
    try:
        while True:
            req = await transport.read()
            if req is None:
                break
            with open(os.environ["CANCELLATION_RPC_LOG"], "a") as log:
                log.write(json.dumps({**req, "_pid": os.getpid()}) + "\\n")
            rid, method, params = req["id"], req["method"], req.get("params", {})
            if method == "health":
                result = {"status": "ok", "pid": os.getpid(), "postgres": {"enabled": False}}
            elif method == "cancel":
                await transport.write({"id": params["request_id"], "ok": False, "error": {"type": "Cancelled", "message": "abandoned"}})
                result = {"accepted": True}
            elif method == "blob_put":
                if params.get("session_id") != "accept":
                    # Simulate queued work: no binary bytes consumed yet.
                    continue
                header, meta = await _read_blob_frame(blob, expected_session_id=params["session_id"], expected_blob_id=params["blob_id"])
                await blob.reader.readexactly(meta["data_len"])
                result = {"stored": True, "blob_id": params["blob_id"], "size_bytes": meta["data_len"]}
            elif params.get("question") == "fast":
                result = {"qid": params["qid"], "answer": "fresh", "session_id": params["session_id"], "metrics": {}}
            else:
                if method == "process_stream":
                    await transport.write({"id": rid, "event": "delta", "text": "partial"})
                continue
            await transport.write({"id": rid, "ok": True, "result": result})
    finally:
        await transport.close()
        if blob:
            await blob.close()

asyncio.run(main())
''', encoding="utf-8")
    env = os.environ.copy()
    for key in _CLEARED_ENV_VARS:
        env.pop(key, None)
    env.update(DUALBRAIN_REPO_ROOT=str(REPO_ROOT), DUALBRAIN_PYTHON=sys.executable,
               DUALBRAIN_ENGINE_PATH=str(script), DUALBRAIN_ENGINE_TRANSPORT=request.param,
               DUALBRAIN_ENGINE_AUTOSTART="0", DUALBRAIN_ENGINE_TIMEOUT_SECONDS="5",
               DUALBRAIN_ENGINE_HEALTH_TTL_MS="100", CANCELLATION_RPC_LOG=str(rpc_log))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            ["dotnet", "run", "--no-build", "--no-launch-profile", "--project", str(GATEWAY_PROJECT),
             "--urls", GATEWAY_LISTEN_URL], cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        try:
            yield _wait_for_gateway(log_path, proc), rpc_log, request.param
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _rpc_messages(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wait_rpc(path, predicate):
    deadline = time.monotonic() + 8.
    while time.monotonic() < deadline:
        matches = [m for m in _rpc_messages(path) if predicate(m)]
        if matches:
            return matches[-1]
        time.sleep(.01)
    raise AssertionError("Expected RPC frame did not arrive")


@pytest.mark.parametrize("stream", [False, True])
def test_http_disconnect_sends_attempt_scoped_cancel_and_next_request_succeeds(cancellation_gateway, stream):
    import http.client
    import socket

    base_url, rpc_log, transport = cancellation_gateway
    parsed = parse.urlparse(base_url)
    qid = f"disconnect-{uuid.uuid4().hex}"
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.)
    response = None
    try:
        connection.request("POST", "/v1/process/stream" if stream else "/v1/process",
                           body=json.dumps({"session_id": "same-session", "qid": qid, "question": "block"}),
                           headers={"content-type": "application/json"})
        if stream:
            response = connection.getresponse()
            while True:
                line = response.readline()
                assert line, "Stream ended before its first delta"
                if line.strip() == b"event: delta":
                    break
        original = _wait_rpc(rpc_log, lambda m: m.get("params", {}).get("qid") == qid)
        assert original["deadline_unix_ms"] > time.time()*1000
        connection.sock.shutdown(socket.SHUT_RDWR)
    finally:
        connection.close()
        if response is not None:
            response.close()
    cancel = _wait_rpc(rpc_log, lambda m: m["method"] == "cancel" and m["params"]["request_id"] == original["id"])
    assert cancel["id"] != original["id"] and original["id"] != qid

    payload = {"session_id": "same-session", "qid": qid, "question": "fast"}
    if stream:
        events = _read_sse(f"{base_url}/v1/process/stream", payload)
        assert events[-1][0] == "done" and events[-2][1]["answer"] == "fresh"
    else:
        status, body = _request_json("POST", f"{base_url}/v1/process", payload=payload)
        assert status == 200 and body["answer"] == "fresh"
    fresh = _wait_rpc(rpc_log, lambda m: m.get("params", {}).get("question") == "fast" and m["params"]["qid"] == qid)
    assert fresh["id"] != original["id"]
    assert not any(m["method"] == "cancel" and m["params"]["request_id"] == fresh["id"] for m in _rpc_messages(rpc_log))


@pytest.mark.parametrize("stream", [False, True])
def test_gateway_deadline_sends_cancel_and_reports_timeout(cancellation_gateway, stream):
    base_url, rpc_log, transport = cancellation_gateway
    qid = f"deadline-{uuid.uuid4().hex}"
    payload = {"qid": qid, "question": "block", "session_id": "timeout-session"}
    started = time.monotonic()
    if stream:
        events = _read_sse(f"{base_url}/v1/process/stream", payload, timeout=12.)
        assert events[-1][0] == "error" and "deadline" in events[-1][1]["message"].lower()
    else:
        status, body = _request_json("POST", f"{base_url}/v1/process", payload=payload, timeout=12.)
        assert status == 504 and "deadline" in body["error"].lower()
    assert time.monotonic()-started < 10.
    original = _wait_rpc(rpc_log, lambda m: m.get("params", {}).get("qid") == qid)
    _wait_rpc(rpc_log, lambda m: m["method"] == "cancel" and m["params"]["request_id"] == original["id"])


def test_restart_does_not_send_old_attempt_cancel_into_replacement_engine(cancellation_gateway):
    import http.client
    base_url, rpc_log, transport = cancellation_gateway
    parsed = parse.urlparse(base_url)
    qid = f"restart-{uuid.uuid4().hex}"
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5.)
    try:
        connection.request("POST", "/v1/process", body=json.dumps({"qid": qid, "question": "block"}),
                           headers={"content-type": "application/json"})
        original = _wait_rpc(rpc_log, lambda m: m.get("params", {}).get("qid") == qid)
        status, _ = _request_json("POST", f"{base_url}/v1/engine/restart", payload={})
        assert status == 200
    finally:
        connection.close()
    # Each replacement is warmed by a fresh call, with the same application qid.
    status, body = _request_json("POST", f"{base_url}/v1/process", payload={"qid": qid, "question": "fast"})
    assert status == 200 and body["answer"] == "fresh"
    assert not any(m["method"] == "cancel" and m["params"]["request_id"] == original["id"] for m in _rpc_messages(rpc_log))


@pytest.mark.parametrize("size", [16, 4*1024*1024])
def test_aborted_binary_upload_does_not_hold_the_rpc_dispatcher(cancellation_gateway, size):
    import http.client
    import socket
    base_url, rpc_log, transport = cancellation_gateway
    # Binary uploads are intentionally available only on the pipe transport.
    if transport != "pipes":
        pytest.skip("binary uploads require pipes")
    parsed = parse.urlparse(base_url)
    boundary = f"blob-{uuid.uuid4().hex}"
    content = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.bin\"\r\n"
               "Content-Type: application/octet-stream\r\n\r\n").encode() + b"x"*size + f"\r\n--{boundary}--\r\n".encode()
    seen = {m["id"] for m in _rpc_messages(rpc_log)}
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=8.)
    try:
        connection.request("POST", "/v1/blobs", body=content,
                           headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        original = _wait_rpc(rpc_log, lambda m: m["method"] == "blob_put" and m["id"] not in seen)
        connection.sock.shutdown(socket.SHUT_RDWR)
    finally:
        connection.close()
    # Both a partial frame and a fully buffered but unconsumed queued frame
    # require invalidation. Neither may be reused by the next upload.
    fresh_qid = f"after-blob-{uuid.uuid4().hex}"
    def fresh_call():
        return _request_json("POST", f"{base_url}/v1/process", payload={"question": "fast", "qid": fresh_qid}, timeout=10.)
    deadline = time.monotonic()+10.
    while True:
        try:
            status, body = fresh_call()
            if status == 200:
                break
        except (error.URLError, OSError):
            pass
        assert time.monotonic() < deadline, "Interrupted blob left the engine stuck"
        time.sleep(.05)
    assert body["answer"] == "fresh"
    fresh = _wait_rpc(rpc_log, lambda m: m.get("params", {}).get("qid") == fresh_qid)
    assert fresh["_pid"] != original["_pid"]
    # Verify the next binary frame too, not only the JSON control stream.
    content = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"session_id\"\r\n\r\naccept\r\n"
               f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"fresh.bin\"\r\n"
               "Content-Type: application/octet-stream\r\n\r\nfresh\r\n"
               f"--{boundary}--\r\n").encode()
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=8.)
    try:
        connection.request("POST", "/v1/blobs", body=content,
                           headers={"content-type": f"multipart/form-data; boundary={boundary}"})
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 200 and result["stored"] and result["size_bytes"] == 5
    finally:
        connection.close()
