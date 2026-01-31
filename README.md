# SpiralReality Dual Brain LLM

SpiralReality Dual Brain LLM is a research playground for **dual-agent orchestration**: a “left brain” (planner/drafter) and a “right brain” (specialist/deepener) collaborate via an in-memory *corpus callosum* and share a lightweight episodic memory. The goal is to make *why* the system chose a collaboration pattern observable via telemetry, traces, and structured “architecture path” summaries.

This repository intentionally optimizes for:
- Experimenting with orchestration policies (when to consult, when to braid, when to stay solo)
- Inspectable runtime behavior (telemetry, inner-dialogue steps, coherence signals)
- A stable **REST surface without Python web frameworks** (C# Minimal API gateway → Python engine over stdio)

## Architecture (Braided Co-Lead Flow)
```mermaid
flowchart TD
    U[User Input] --> P[Perception Layer<br/>Amygdala + Prefrontal Cortex + Schema Profiler]
    P --> D[Inner Dialogue Loop<br/>Left/Right Brains + Corpus Callosum]
    D --> I[Integration & Auditing<br/>Coherence Resonator + Auditor + Default Mode]
    I --> M[Memory & Consolidation<br/>Shared Memory + Hippocampus + Unconscious Field]
    M --> R[Final Response]
```

By default, `/v1/process` returns a **clean user-facing answer** (no internal debug blocks). The same signals are still available via `telemetry` + `dialogue_flow`, and you can opt-in to debug-style answers via `answer_mode`.

## Quick Start (Recommended): REST via C# Gateway
The recommended research loop is:
1) C# Minimal API exposes REST endpoints
2) C# launches and talks to the Python engine over stdio (JSONL)
3) Python owns orchestration, memory, and telemetry

### Prerequisites
- Python 3.10+
- .NET 8 SDK (ensure `dotnet` is on your `PATH`; on some macOS setups it may live at `/usr/local/share/dotnet/dotnet`)

### Python install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional (Postgres-backed state)
pip install -r requirements-pg.txt

# Optional (tests)
pip install -r requirements-dev.txt
```

### Run the gateway
```bash
export DUALBRAIN_REPO_ROOT="$PWD"
dotnet run --project csharp/SrDualBrain.Gateway --urls http://127.0.0.1:8080
```

Optional gateway timeout override (useful for long generations / auto-continue):
```bash
export DUALBRAIN_ENGINE_TIMEOUT_SECONDS=120
```

### Browser UI (chat + metrics)
Open:
- `http://127.0.0.1:8080/`

The UI lets you optionally pin `provider/model` for the session and set `max_output_tokens`.

#### Metrics pop-out
Click **Pop out metrics** to open a separate metrics monitor window (`/metrics.html`). This keeps the main window chat-focused (no metrics mixed into the chat transcript) while the pop-out window updates live as you send messages. Click **Dock** (or the same topbar button) to bring metrics back.

### Call it
```bash
curl -s http://127.0.0.1:8080/v1/process \
  -H 'content-type: application/json' \
  -d '{
    "session_id": "demo",
    "question": "Explain why a dual-agent workflow can improve quality.",
    "leading_brain": "auto",
    "return_telemetry": true,
    "llm": { "provider": "openai", "model": "gpt-4o-mini" }
  }'
```

Notes:
- API keys are read from environment variables (do not send keys in requests).
- `session_id` scopes in-memory state (shared memory + hippocampal episodes) inside the Python engine process.

## REST API

### `GET /v1/health`
Returns gateway + engine health.

### `GET /v1/trace/{qid}`
Fetches stored `telemetry` and/or `dialogue_flow` for a prior turn (even if `/v1/process` returned only the clean answer + metrics).

Query params:
- `session_id` (default: `"default"`)
- `include_telemetry` (default: `true`)
- `include_dialogue_flow` (default: `true`)

### `POST /v1/reset`
Resets a session inside the Python engine.

Request:
```json
{ "session_id": "demo" }
```

### `POST /v1/process`
Runs one orchestration turn.

Request fields:
- `question` (string, required)
- `session_id` (string, default: `"default"`)
- `leading_brain` (`"auto"|"left"|"right"`, default: `"auto"`)
- `answer_mode` (`"plain"|"debug"|"annotated"|"meta"`, default: `"plain"`)
- `return_telemetry` (bool, default: `false`)
- `return_dialogue_flow` (bool, default: `true`)
- `qid` (string, optional): supply your own ID for dataset runs / trace correlation
- `llm` (object, optional): select provider/model for the *session* (keys still come from env)
  - `provider` (string, required if `llm` provided): `openai|google|anthropic|mistral|xai|huggingface`
  - `model` (string, required if `llm` provided)
  - `left_model` / `right_model` (string, optional): per-hemisphere model override
  - Optional request-time knobs (no secrets): `api_base`, `organization`, `max_output_tokens`, `timeout_seconds`, `auto_continue`, `max_continuations`

Response fields:
- `qid` (string)
- `answer` (string)
- `session_id` (string)
- `metrics` (object): lightweight summary (coherence/policy/latency) for UI dashboards
- `dialogue_flow` (object, optional): inner steps + architecture path captured for this turn
- `telemetry` (array, optional): structured per-module events emitted during the turn

### `POST /v1/process/stream` (SSE)
Streams the answer via Server-Sent Events (SSE).

When using an OpenAI-style provider (`openai`, `mistral`, `xai`), `delta` events are emitted from the provider's true token stream (not "chunk the final answer"). The engine may emit a `reset` event if it decides to replace an initial draft with an integrated final answer.

SSE events:
- `start`: `{ qid, session_id }`
- `delta`: `{ text }`
- `reset`: `{}` (client should clear the current assistant text)
- `final`: `{ qid, answer, session_id, metrics }`
- `done`: `{}`
- `error`: `{ message }`

Fetch `/v1/trace/{qid}` afterward if you want `telemetry` or `dialogue_flow`.

Example:
```bash
curl -N http://127.0.0.1:8080/v1/process/stream \
  -H 'content-type: application/json' \
  -d '{
    "session_id":"demo",
    "question":"Write a 2 minute explanation of what coherence means here.",
    "leading_brain":"auto",
    "llm": { "provider":"openai", "model":"gpt-4o-mini" }
  }'
```

### `POST /v1/episodes/search`
Searches hippocampal episodic memory for a session.

Request:
```json
{ "session_id": "demo", "query": "coherence audit", "topk": 5 }
```

Response:
- `backend`: `"pgvector" | "postgres" | "memory" | "none"`
- `results`: array of `{ similarity, qid, question, answer, ts, tags, ... }`

### `POST /v1/telemetry/query`
Queries persisted telemetry events (Postgres only).

Request:
```json
{ "session_id": "demo", "limit": 250, "event": "policy_decision" }
```

### `POST /v1/schema/list`
Lists consolidated schema memories (Postgres only).

Request:
```json
{ "session_id": "demo", "limit": 16 }
```

### Concurrency model
The gateway currently runs **one Python engine process** and the engine processes requests sequentially (one line in → one line out). This is intentional for determinism during research runs. Scaling options are in the roadmap.

## LLM Configuration

### Environment variables (recommended)
The Python engine can call external providers directly via plain HTTP (no SDK deps).

Example (OpenAI):
```bash
export LLM_PROVIDER=openai
export LLM_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

Optional overrides:
- `LEFT_BRAIN_MODEL`, `RIGHT_BRAIN_MODEL`
- `LLM_API_BASE` or `<PROVIDER>_API_BASE`
- `LLM_MAX_OUTPUT_TOKENS` / `LEFT_BRAIN_MAX_TOKENS` / `RIGHT_BRAIN_MAX_TOKENS` (default: 1024)
- `LLM_TIMEOUT` / `LEFT_BRAIN_TIMEOUT` / `RIGHT_BRAIN_TIMEOUT`
- `LLM_AUTO_CONTINUE` / `LEFT_BRAIN_AUTO_CONTINUE` / `RIGHT_BRAIN_AUTO_CONTINUE` (default: on)
- `LLM_MAX_CONTINUATIONS` / `LEFT_BRAIN_MAX_CONTINUATIONS` / `RIGHT_BRAIN_MAX_CONTINUATIONS` (default: 2)
- `OPENAI_ORGANIZATION`

Auto-continue will issue follow-up calls when a provider reports that output stopped due to token limits (e.g., OpenAI-style `finish_reason=length`, Anthropic `stop_reason=max_tokens`).

### Provider keys
Use provider-specific keys (e.g., `ANTHROPIC_API_KEY`) or the shared `LLM_API_KEY`.

Supported providers: `openai`, `google`, `anthropic`, `mistral`, `xai`, `huggingface`.

## Postgres-backed Stateful Memory (Optional)
To persist state across engine restarts, set:
- `DUALBRAIN_PG_DSN` (e.g., `postgresql://user:pass@localhost:5432/dual_brain`)

If your DSN uses a Docker service name like `host` (e.g., `...@host:5432/...`), it will not resolve from your macOS host shell; use `localhost` (or `127.0.0.1`) with the published port instead.

The engine auto-creates tables on startup:
- `SharedMemory` traces
- Hippocampal episodes (+ embedding version)
- Telemetry events
- Schema memory (sleep/consolidation output) + consolidation cursor

### pgvector (optional)
If `pgvector` is installed/enabled in your database, the store will add a `vector_pg` column and use cosine search to accelerate `POST /v1/episodes/search`. If not available, vectors are still stored as JSON and search falls back to a Python similarity scan over recent candidates.

### Telemetry persistence toggle
Set `DUALBRAIN_PG_PERSIST_TELEMETRY=0` to disable telemetry writes (default: enabled when Postgres is enabled).

### Schema memory (sleep/consolidation)
Run the deterministic consolidation job:
```bash
python3 sr-dual-brain-llm/scripts/sleep_consolidate.py --session-id demo
```

On session creation, the engine loads recent schema memories (default: 16) and may inject them into the context as `[Schema memory] ...`. Override with:
- `DUALBRAIN_SCHEMA_MEMORY_LIMIT`

Resetting a session via `POST /v1/reset` will also delete the persisted rows for that session.

## Running without HTTP (Engine JSONL)
If you want to debug the engine protocol directly:
```bash
printf '{"id":"1","method":"health","params":{}}\n' | python3 sr-dual-brain-llm/scripts/engine_stdio.py
```

The engine also supports a streaming method (`process_stream`) which emits intermediate event lines before the final `ok` response:
```bash
python3 sr-dual-brain-llm/scripts/engine_stdio.py <<'EOF'
{"id":"1","method":"process_stream","params":{"session_id":"demo","question":"hello"}}
EOF
```

## Running the Interactive Orchestrator (Python-only)
```bash
python3 sr-dual-brain-llm/scripts/run_server.py
```

## Development
- Python tests: `python3 -m pytest -q`
- C# build: `dotnet build csharp/SrDualBrain.Gateway -c Release`

## Repository Layout
```
.
├── csharp/
│   └── SrDualBrain.Gateway/        # C# Minimal API REST gateway (HTTP → Python stdio)
├── docs/                           # Design notes / playbooks
├── sr-dual-brain-llm/
│   ├── core/                       # Orchestration + memory + telemetry modules
│   ├── dashboard/                  # Experimental UI (not required for REST path)
│   └── scripts/
│       ├── engine_stdio.py         # JSONL engine entrypoint (used by the C# gateway)
│       └── run_server.py           # Interactive CLI loop
└── tests/
```

## Roadmap (Next Research Upgrades)
- **Stateful memory backends**:
  - Redis for fast shared state + TTL / decay
  - Postgres for durable episodic traces + queryable analytics
- **Eval harness**: deterministic scenario packs + metrics for coherence, safety, memory retention
- **Scaling**: multi-engine pool behind the gateway (process pool / queue) + per-session routing

## License
See `LICENSE_NOTICE.txt` for third-party notices and licensing information.
