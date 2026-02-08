# SpiralReality Dual Brain LLM

SpiralReality Dual Brain LLM is a research playground for **dual-agent orchestration**: a “left brain” (drafting/speech) and a “right brain” (deepening/lead) collaborate via an in-memory *corpus callosum* while sharing short-term and long-term memory.

The goal is not “AGI”. The goal is **observability**: make it easy to see *why* the system chose a collaboration pattern via metrics, traces, and an explicit “architecture path” view — without mixing those internals into the chat output.

This repository intentionally optimizes for:
- Experimenting with orchestration policies (when to consult, when to braid, when to stay solo)
- Inspectable runtime behavior (telemetry, inner-dialogue steps, coherence signals, architecture path)
- A stable **REST surface without Python web frameworks** (C# Minimal API gateway → Python engine over stdio/pipes; `pipes` default on macOS/Linux)

## What you can do (today)
- Run a browser UI with **clean chat** + **separate metrics window**.
- Watch policy + module activation change as prompts shift from “small talk” to “hard reasoning”.
- Upload images via `POST /v1/blobs` and reference them on `POST /v1/process` (multimodal plumbing).
- Inspect any turn via `GET /v1/trace/{qid}` (metrics + dialogue flow + telemetry).
- Optionally persist memory/telemetry to Postgres.

## Try it (3 minutes)

### 0) (Optional) Set LLM provider/model + API key
The engine can run without external LLMs (deterministic heuristic fallbacks), but you’ll want real models for quality.

Example (OpenAI-style):
```bash
export LLM_PROVIDER=openai
export LLM_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

Tip: if you plan to use images at all, **start with a vision-capable model** even for text-only turns to keep the “thinking substrate” consistent across the session.

### 1) Install (Python)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional (Postgres-backed state)
pip install -r requirements-pg.txt

# Optional (tests)
pip install -r requirements-dev.txt
```

### 2) Run the gateway (C# REST → Python engine)
Prereqs:
- Python 3.10+
- .NET 8 SDK (`dotnet` on `PATH`; on some macOS setups it may live at `/usr/local/share/dotnet/dotnet`)

Run:
```bash
export DUALBRAIN_REPO_ROOT="$PWD"
dotnet run --project csharp/SrDualBrain.Gateway --urls http://127.0.0.1:8080
```

IPC transport:
- macOS/Linux: defaults to `pipes` (Unix domain sockets under the hood) for fast blob/image uploads.
- Windows: defaults to `stdio` (`pipes` transport is not supported by the current Python engine).

Optional: force a transport explicitly:
```bash
export DUALBRAIN_ENGINE_TRANSPORT=pipes  # macOS/Linux only
export DUALBRAIN_ENGINE_TRANSPORT=stdio  # portable fallback
```

Notes (macOS):
- .NET named pipes use Unix domain sockets under the hood and have a strict path length limit.
- If you hit socket path length issues, set `TMPDIR=/tmp/` before running `dotnet run`.

Optional gateway timeout override (useful for long generations / auto-continue):
```bash
export DUALBRAIN_ENGINE_TIMEOUT_SECONDS=120
```

Optional engine-side corpus-callosum timeout (bounds right-brain waits):
```bash
# Default: 20000 (20s)
export DUALBRAIN_CALLOSUM_TIMEOUT_MS=20000
```

### 3) Open the UI (chat + metrics)
Open:
- `http://127.0.0.1:8080/`

Recommended “research loop”:
- Keep the **main window** chat-only.
- Click **Pop out metrics** to open `/metrics.html` in a separate window and monitor:
  - coherence/tension/routing/policy/latency
  - metacognition (drift / repetition / cleanup)
  - active modules
  - architecture path (stage → modules)
  - executive memo (out-of-band)

### 4) One curl request (sanity check)
```bash
curl -s http://127.0.0.1:8080/v1/process \
  -H 'content-type: application/json' \
  -d '{
    "session_id":"demo",
    "question":"Explain (briefly) what the architecture path is.",
    "leading_brain":"auto",
    "executive_mode":"assist",
    "return_telemetry": true,
    "return_dialogue_flow": true,
    "llm": { "provider":"openai", "model":"gpt-4o-mini" }
  }'
```

Notes:
- API keys are read from environment variables (do not send keys in requests).
- `session_id` scopes memory state inside the Python engine process.

### 5) System2 benchmark (fixed set + trend tracking)
To measure reasoning refinement quality over time (`initial_issues -> final_issues`), run:

```bash
python3 sr-dual-brain-llm/scripts/benchmark_system2.py \
  --system2-mode on \
  --questions sr-dual-brain-llm/examples/system2_benchmark_questions_en.json
```

Outputs:
- Full report: `sr-dual-brain-llm/samples/system2_benchmark_last.json`
- History rows (for trend over repeated runs): `sr-dual-brain-llm/samples/system2_benchmark_history.jsonl`

Useful flags:
- `--limit 5` (quick smoke run)
- `--shuffle --seed 42` (order randomization)
- `--history-limit 50` (trend window size)

## Architecture (Braided Co-Lead Flow)
```mermaid
flowchart TD
    U[User Input] --> P[Perception Layer<br/>Amygdala + Prefrontal Cortex + Schema Profiler]
    P --> D[Inner Dialogue Loop<br/>Left/Right Brains + Corpus Callosum]
    D --> I[Integration & Auditing<br/>Coherence Resonator + Auditor + Default Mode]
    I --> M[Memory & Consolidation<br/>Working Memory + Shared Memory + Hippocampus]
    M --> R[Final Response]
```

By default, `/v1/process` returns a **clean user-facing answer** (no internal debug blocks).
Everything “internal” lives out-of-band:
- `metrics` (small, stable UI payload)
- `dialogue_flow` (inner steps + architecture path)
- `telemetry` (structured per-module events)
- `/v1/trace/{qid}` (fetch traces later without polluting the chat output)

## The Python engine: roles + interaction

The core orchestration happens in `sr-dual-brain-llm/core/dual_brain.py` (`DualBrainController.process`).

Treat “brains” as **roles** with strict I/O contracts:

| Role | Responsibility | Output | Where |
|---|---|---|---|
| **Left brain (speech/draft)** | Produce the primary user-facing answer draft | Draft answer text | `sr-dual-brain-llm/core/models.py` |
| **Right brain (deepener/lead)** | Add missing substance; can create a “lead prelude” | Additive notes (not a second full answer) | `sr-dual-brain-llm/core/models.py` |
| **Corpus callosum** | In-process message bus for right-brain requests/responses | `ASK_DETAIL` / `ASK_LEAD` messages | `sr-dual-brain-llm/core/callosum.py` |
| **Right worker** | Consumes callosum requests and calls the right model | `notes_sum` / `lead_notes` | `sr-dual-brain-llm/scripts/engine_stdio.py` |
| **Executive reasoner** | Private meta-control + optional “mix-in” | `memo` (private) + `mix_in` (user-facing micro-addition) + directives | `sr-dual-brain-llm/core/executive_reasoner.py` |
| **Policy + timing** | Decide whether to consult, set temperature/slot/timeouts | `action`, `temperature`, `slot_ms` | `sr-dual-brain-llm/core/policy*.py`, `sr-dual-brain-llm/core/hypothalamus.py` |
| **Auditor** | Sanity checks (keep answers clean) | ok/fail | `sr-dual-brain-llm/core/auditor.py` |
| **Coherence + motifs** | Post-hoc scoring + “why it routed” signals | `coherence_signal` + tags | `sr-dual-brain-llm/core/coherence_resonator.py` |

### Executive “tri-layer” (speech + advice + reasoning)
This is the “3-layer” structure:

1) **Speech brain** (left): produces the final answer
2) **Advice brain** (right): produces additive content that can be blended into the answer
3) **Reasoning brain** (executive): produces:
   - `memo`: private constraints + priorities (out-of-band only)
   - `mix_in`: tiny user-facing addition (blended only when `executive_mode=assist`)

Design intent:
- Keep the **chat transcript clean** (no “writing coach notes”, no internal orchestration talk).
- Still **observe the system** via the metrics panel and `/v1/trace/{qid}`.

## Memory model (short-term vs long-term)

This repo intentionally separates “sticky memory” from “recent context”:

- **Working memory (PrefrontalCortex)**: short-term, small, clipped summaries of the last few turns. Included only when the question structure suggests it’s needed (language-agnostic heuristics: punctuation, length, digits).  
  Code: `sr-dual-brain-llm/core/prefrontal_cortex.py`
- **Shared memory**: lightweight recall of prior Q/A traces.  
  Code: `sr-dual-brain-llm/core/shared_memory.py`
- **Hippocampus (episodic memory)**: embedding-backed episodic recall (powerful, but can be noisy if overused).  
  Code: `sr-dual-brain-llm/core/temporal_hippocampal_indexing.py`
- **Schema memory (optional, Postgres)**: consolidation output from a deterministic “sleep” job. Intended to be slower-changing and less chatty than raw episodic recall.

Reset behavior:
- `POST /v1/reset` resets in-process state for a `session_id` (and deletes persisted rows when Postgres is enabled).

## Observability (metrics vs traces)

You’ll see three levels of “what happened”:

1) **`metrics`** (small): coherence/policy/latency + active modules and stage→modules  
   - `metrics.modules.active`: union of modules seen in the architecture path
   - `metrics.modules.stages`: ordered stage breakdown (used by the UI “Architecture path” panel)
2) **`dialogue_flow`** (structured): inner steps + architecture path summary (what modules ran in what order)
3) **`telemetry`** (raw): per-module event payloads for dashboards/evals

If you stream via SSE (`/v1/process/stream`), fetch traces afterward:
- `GET /v1/trace/{qid}?session_id=demo&include_telemetry=true&include_dialogue_flow=true`

## UI tour (chat + metrics without pollution)

Open `http://127.0.0.1:8080/`.

### Chat
- Chat transcript stays **answer-only**.
- Internal data (executive memo, telemetry, dialogue flow) stays in the metrics pane / pop-out.
- `Restart engine`: restarts the Python engine process (drops all in-process sessions). Useful after code changes or when recovering from an engine error.

### Controls
- `Session`: scopes memory state inside the engine process.
- `Leading brain`: `auto|left|right` (force-leading is useful for experiments).
- `Provider/Model`: optional per-session override (keys still come from env).
- `Max output tokens`: request-time override (not a secret).
- `Executive mode`:
  - `off`: no executive
  - `observe`: executive memo only (answer unchanged)
  - `assist`: memo + **mix-in blended into answer**
  - `polish`: may apply directives as a second-pass rewrite (may reset streams)
- `System2 mode (reasoning)`:
  - `auto`: enables the **right-brain critic** only when both left/right are backed by external LLMs and the turn looks like a reasoning task (structural complexity + loaded context)
  - `on`: force critic mode (left drafts → right critiques → left revises → verify pass)
  - `off`: disable critic mode
  - Metrics panel (`system2`) now shows round progress and issue decay (initial → final) when available.
- `Executive observer (experiment)`:
  - `off`: disabled
  - `metrics`: after the turn, the Executive receives a compact metrics/context report and emits an out-of-band memo (never blended into chat)

### Metrics pane
- Key numbers: coherence / tension / routing / action / temperature / latency
- Active modules: quick chip list
- **Architecture path**: stage-by-stage module timeline (the “what ran” view)
- Executive memo: out-of-band `memo` + optional `mix_in` + directives metadata

### Metrics pop-out
Click **Pop out metrics** to open `/metrics.html` in a separate window. This is the recommended “research loop”:
- Main window: chat only
- Pop-out: metrics + executive memo + architecture path + traces

## REST API

### `GET /v1/health`
Returns gateway + engine health.

### `POST /v1/blobs` (multipart upload)
Uploads a binary blob (e.g., an image) into the Python engine.

This endpoint is intended for multimodal experiments and uses a separate named-pipe "blob" channel when `DUALBRAIN_ENGINE_TRANSPORT=pipes` is enabled.

Request (multipart/form-data):
- `session_id` (optional, default `"default"`)
- `file` (required)

Notes:
- The engine stores blobs under `DUALBRAIN_BLOB_DIR` (default: your OS temp dir under `srdb_blobs/`).
- The browser UI uses this endpoint to upload images and then references them via `attachments` on `/v1/process`.
- The engine keeps a small in-memory base64 cache for re-used images (`DUALBRAIN_VISION_CACHE_ITEMS`, default: `4`).

### `GET /v1/trace/{qid}`
Fetches stored `telemetry` and/or `dialogue_flow` for a prior turn (even if `/v1/process` returned only the clean answer + metrics).

Query params:
- `session_id` (default: `"default"`)
- `include_telemetry` (default: `true`)
- `include_dialogue_flow` (default: `true`)
- `include_executive` (default: `true`): include executive memo/mix-in/directives and any observer memo (if available)

### `POST /v1/reset`
Resets a session inside the Python engine.

Request:
```json
{ "session_id": "demo" }
```

### `POST /v1/engine/restart`
Restarts the Python engine process. This drops all in-process sessions (equivalent to restarting the gateway for the engine).

### `POST /v1/process`
Runs one orchestration turn.

Request fields:
- `question` (string, required)
- `session_id` (string, default: `"default"`)
- `leading_brain` (`"auto"|"left"|"right"`, default: `"auto"`)
- `answer_mode` (`"plain"|"debug"|"annotated"|"meta"`, default: `"plain"`)
- `executive_mode` (`"off"|"observe"|"assist"|"polish"`, default: `"off"`)
  - `off`: no executive reasoner run
  - `observe`: run executive reasoner and store memo (out-of-band only; does not change the answer)
  - `assist`: run executive reasoner and blend a small **user-facing mix-in** into the final answer (memo stays out-of-band)
  - `polish`: executive directives may trigger a second-pass integration (may reset stream)
- `executive_observer_mode` (`"off"|"director"|"metrics"|"both"`, default: `"off"`)
  - `director`: pre-turn steering (memory gating + optional clarifying question)
  - `metrics`: post-turn feedback memo (out-of-band)
  - `both`: director + post-turn feedback
- `system2_mode` (`"auto"|"on"|"off"`, default: `"auto"`)
  - `auto`: enable critic mode only when both left/right are backed by external LLMs and the turn looks like a reasoning task (structural complexity + loaded context)
  - `on`: force critic mode (left drafts → right critiques → left revises → verify pass)
  - `off`: disable critic mode
- `return_telemetry` (bool, default: `false`)
- `return_dialogue_flow` (bool, default: `true`)
- `return_executive` (bool, default: `false`)
- `qid` (string, optional): supply your own ID for dataset runs / trace correlation
- `llm` (object, optional): select provider/model for the *session* (keys still come from env)
  - `provider` (string, required if `llm` provided): `openai|google|anthropic|mistral|xai|huggingface`
  - `model` (string, required if `llm` provided)
  - `left_model` / `right_model` / `executive_model` (string, optional): per-role model override
  - Optional request-time knobs (no secrets): `api_base`, `organization`, `max_output_tokens`, `timeout_seconds`, `auto_continue`, `max_continuations`
- `attachments` (array, optional): references previously uploaded blobs (e.g., images) to include as vision inputs
  - Each item supports: `{ blob_id, content_type, file_name, size_bytes }`
  - Vision is currently implemented for **OpenAI-style providers** (`openai|mistral|xai`) via `data:<mime>;base64,...` URLs generated inside the Python engine.
  - Recommendation: always use a **vision-capable model** even for text-only turns to keep behavior stable across “text-only vs. image” conversations.

Response fields:
- `qid` (string)
- `answer` (string)
- `session_id` (string)
- `metrics` (object): lightweight summary (coherence/policy/latency/modules) for UI dashboards
- `dialogue_flow` (object, optional): inner steps + architecture path captured for this turn
- `telemetry` (array, optional): structured per-module events emitted during the turn
- `executive` / `executive_observer` (object, optional): out-of-band memos (only when `return_executive=true`)

### `POST /v1/process/stream` (SSE)
Streams the answer via Server-Sent Events (SSE).

To avoid “draft → reset → regenerate” artifacts in the browser, streaming mode emits the **final integrated answer only**. `delta` events are chunked slices of the final answer (currently ~512 characters per chunk). No `reset` event is emitted.

SSE events:
- `start`: `{ qid, session_id }`
- `delta`: `{ text }`
- `final`: `{ qid, answer, session_id, metrics, executive?, executive_observer? }`
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

## LLM configuration

### Environment variables (recommended)
The Python engine can call external providers directly via plain HTTP (no SDK deps).

Example (OpenAI):
```bash
export LLM_PROVIDER=openai
export LLM_MODEL_ID=gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

Optional overrides:
- `LEFT_BRAIN_MODEL`, `RIGHT_BRAIN_MODEL`, `EXECUTIVE_MODEL`
- `LLM_API_BASE` or `<PROVIDER>_API_BASE`
- `LLM_MAX_OUTPUT_TOKENS` / `LEFT_BRAIN_MAX_TOKENS` / `RIGHT_BRAIN_MAX_TOKENS` / `EXECUTIVE_MAX_TOKENS` (default: 1024)
- `LLM_TIMEOUT` / `LEFT_BRAIN_TIMEOUT` / `RIGHT_BRAIN_TIMEOUT` / `EXECUTIVE_TIMEOUT`
- `LLM_AUTO_CONTINUE` / `LEFT_BRAIN_AUTO_CONTINUE` / `RIGHT_BRAIN_AUTO_CONTINUE` / `EXECUTIVE_AUTO_CONTINUE` (default: on)
- `LLM_MAX_CONTINUATIONS` / `LEFT_BRAIN_MAX_CONTINUATIONS` / `RIGHT_BRAIN_MAX_CONTINUATIONS` / `EXECUTIVE_MAX_CONTINUATIONS` (default: 2)
- `OPENAI_ORGANIZATION`

Auto-continue will issue follow-up calls when a provider reports that output stopped due to token limits (e.g., OpenAI-style `finish_reason=length`, Anthropic `stop_reason=max_tokens`).

### Provider keys
Use provider-specific keys (e.g., `ANTHROPIC_API_KEY`) or the shared `LLM_API_KEY`.

Supported providers: `openai`, `google`, `anthropic`, `mistral`, `xai`, `huggingface`.

## Multimodal (images) without base64 in requests

Why this exists:
- Sending `data:<mime>;base64,...` directly from the browser is slow and bloats logs.
- Switching models when an image appears can change behavior mid-conversation.

Recommended flow:
1) Pick a **vision-capable model** from the start (even for text-only).
2) Upload images via `POST /v1/blobs` (multipart).
3) Reference them via `attachments` on `POST /v1/process`.

Under the hood:
- The engine stores blobs on disk under `DUALBRAIN_BLOB_DIR`.
- The engine converts blobs to data URLs *inside the Python process* (for providers that require it) and keeps a small in-memory cache (`DUALBRAIN_VISION_CACHE_ITEMS`) to avoid repeated base64 work.
- On macOS/Linux the gateway defaults to `DUALBRAIN_ENGINE_TRANSPORT=pipes`, so blob uploads use a dedicated named-pipe channel for higher throughput.

## Postgres-backed stateful memory (optional)
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

## Running without HTTP (engine JSONL)
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

## Running the interactive orchestrator (Python-only)
```bash
python3 sr-dual-brain-llm/scripts/run_server.py
```

## Development
- Python tests: `python3 -m pytest -q`
- C# build: `dotnet build csharp/SrDualBrain.Gateway -c Release`

## Repository layout
```
.
├── csharp/
│   └── SrDualBrain.Gateway/        # C# Minimal API REST gateway (HTTP → Python stdio/pipes)
├── docs/                           # Design notes / playbooks
├── sr-dual-brain-llm/
│   ├── core/                       # Orchestration + memory + telemetry modules
│   └── scripts/
│       ├── engine_stdio.py         # JSONL engine entrypoint (used by the C# gateway)
│       └── run_server.py           # Interactive CLI loop
└── tests/
```

## Roadmap (next research upgrades)
- **Stateful memory backends**:
  - Redis for fast shared state + TTL / decay
  - Postgres for durable episodic traces + queryable analytics
- **Eval harness**: deterministic scenario packs + metrics for coherence, safety, memory retention
- **Scaling**: multi-engine pool behind the gateway (process pool / queue) + per-session routing

## License
See `LICENSE.txt` for third-party notices and licensing information.
