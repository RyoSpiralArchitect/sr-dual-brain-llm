# Clarification calibration and natural persistent memory — 2026-11-06

The fixed live trajectory completes **12/12 turns**, including three actual
model-generated primary writes, explicit correction, PostgreSQL restart and two
byte-exact quotations of the actual stored sources. The six fresh calibration
packets match **24/24 authored expectations**; all nine read answers receive
pass/pass from the V4 judge. **The original frozen replay fails at a concurrent
preview boundary.** A separately labeled post-output offline diagnostic reproduces
all 50 ordered requests and the canonical outcomes by restoring recorded preview
scheduling constraints. That diagnostic does not turn the original replay into a
pass.

Implementation, fixtures, criteria, order and budget were frozen before outputs
at `96cbdde91a47514c9475cb1b989a7e0f5d2b8e3b`, on merged #96
(`f4015037c5bb753e2768ac20cf1ab4703733804f`). Production core bytes remain
unchanged. See the [prospective protocol](natural-memory-protocol.md),
[safe metrics](natural-memory-2026-11-06.metrics.json),
[calibration fixture](../../sr-dual-brain-llm/examples/clarification_calibration_v4.json)
and [trajectory](../../sr-dual-brain-llm/examples/natural_memory_trajectory.json).
Earlier studies and V3 scores were not rescored.

## What was measured

This is one small acceptance trajectory using normal `RightBrainPolicy`, actual
`gpt-6-luna` generation and real isolated PostgreSQL. No primary fixture was
injected, draft substituted or consultation action forced. The opt-in all-AUTO,
balanced profile keeps executive/Director work OFF; it does not change product
defaults. Transport is the production nonstream JSON path, not a streaming
latency experiment.

| Turn | Observed result | Runtime HTTP attempts | Handler ms |
| --- | --- | ---: | ---: |
| Initial period write | Complete primary committed | 2 | 3,475 |
| Backup write | Second complete primary committed | 4 | 4,477 |
| Ambiguous whole-record request | Ask which record; native selective skip | 1 | 1,276 |
| Listed period selected | Exact original source quotation | 4 | 3,771 |
| Period correction | New primary committed; old period superseded | 5 | 12,552 |
| Earlier selection reused | Request reselection; no substitution | 3 | 2,655 |
| Reference reused after restart | Request reselection; native selective skip | 1 | 1,014 |
| Relisted current period | Exact corrected source quotation | 4 | 5,009 |
| Explicit current period | Attribute the record and preserve verification caveat | 5 | 7,374 |
| Bare retention in empty scope | Ask which measure is intended | 2 | 2,708 |
| Explicit Japanese retention rate | Preserve unknown support; request evidence | 2 | 2,075 |
| Explicit period in empty scope | Request a current source for that parameter | 2 | 3,326 |

Every operation reports a committed receipt and consistent System2 execution
accounting. Candidate listing makes no provider call. Only the backup and
corrected period remain active primaries after correction, including after
restart; read answers do not become primary evidence. Selected references remain
outside provider payloads.

The short Japanese rate question does **not** run asynchronous retrieval
preparation. Its bounded condition is nevertheless actually delivered in the
generation inputs: `current_unapproved / unknown / preserve_unknown`. This is
not evidence that the value is absent. The harness records internal eligibility
separately and admits only actually exposed complete memory blocks and conditions
to semantic assessment.

## Calibration and interpretation

Assessment V4 separately grades attribution and information sufficiency using
observed context. The six packets yield agreement on **13/13 expected pass,
8/8 expected fail and 3/3 expected uncertain** criteria, with zero invalid packets
or disagreements. Expected labels, variant names and private comparison
references are not sent to the judge. All nine actual read judgments return
pass/pass, with literal supporting quotes validated privately. Three writes are
outside this read rubric and checked through completion and persistence instead.

These are implementing-assistant-authored calibration expectations and a single
`gpt-6-sol` judge, not independent human adjudication, general judge validation or
a before/after quality gain. V3 remains separate: source validity passes for the
two quotations and current-period lookup, and fails for all six source-free
clarifications (`no_exposed_record`). Its semantic fields remain unassessed.
Appropriate clarification does not establish source validity. Broader calibration
in #89 remains open.

## Cost and the remaining preview weakness

There are **50/64 actual HTTP attempts**: six calibration, **35/49 runtime** and
nine read judgments. Runtime roles are 12 draft, 7 critic, 5 support, 5 right
preview, 4 detail and 2 integration. The draft label includes nested source-write
generation. All 48 returned envelopes end in `stop`; two other attempts are
cancelled previews without response/usage receipts. No paid repeat, resampling,
rescue, continuation or post-output grader/fixture change was used.

Known subtotals are **56,622 input / 4,244 output tokens**. Full token totals
remain **unknown**, because cancellation does not establish zero provider cost.
The two cancelled right previews belong to the relisted quotation and Japanese
rate question. The latter ends with policy action 0 after speculative work has
already started. Five previews were admitted in total; three completed.

Measured episode time is **50,895 ms**, including owned setup/load, handlers,
candidate listing and restart, excluding calibration/read grading, user think
time and teardown. Handler times are instrumented observations from one run,
not a latency or savings comparison. The two native selective skips demonstrate
that the gate can apply without forced consultation pressure, but do not measure
general skip frequency or the benefit of suppressing earlier previews.

## Original replay failure and supplemental diagnosis

The frozen CLI consumes the first ten requests correctly, then diverges at
zero-based occurrence **10**: `backup_initial` expects a speculative `right`
request but receives `detail`. Its offline provider response returns immediately;
the preview task has not started before the left draft finishes and cancels it.
The run retains its partial journal, report and `ValueError` failure. No new
provider calls were made. Original full-trajectory replay equality is therefore
**not established**.

The [supplemental diagnostic](natural-memory-schedule-diagnostic.py) was authored
**after** observing that failure and is published as an investigation artifact,
not a prospectively frozen validation. It requires frozen offline transport and
leaves production code, inputs, envelopes and grades unchanged. At the transport
boundary it lets a recorded preview start before returning the draft; a recorded
completed preview must finish, while a recorded cancelled preview remains pending
until the actual owning controller cancels it. It does not supply policy actions,
rewards, success values or replacement answers. Existing feedback replay restores
only measured latency and verifies every non-timing feedback input.

That diagnostic consumes **50/50 exact ordered requests**, records **0 provider
calls / 0 unexpected blocked HTTP attempts**, and reproduces the canonical hash:

`17e1d3112502d1972d608a8ae8af111d3c638d40a00fd06f6d7dd6304defedc4`

Its source SHA-256 is
`f239c8cfd2e51a9e8ddab5ed99f7a76ef5bd28e5cedc64577d7edb473882666f`.
It establishes consistency under recorded scheduling constraints for this trace,
not general scheduler correctness or success of the unmodified replay CLI.
Three earlier diagnostic setup failures imported an old module from `/tmp` and
stopped before benchmark execution; all are retained separately. The published
helper prioritizes this checkout's scripts to prevent that import shadowing.

With the original private journal and isolated test PostgreSQL, run from the
repository root, choosing a new output directory:

```bash
PYTHONPATH=sr-dual-brain-llm:sr-dual-brain-llm/scripts \
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python docs/benchmarks/natural-memory-schedule-diagnostic.py \
  --journal sr-dual-brain-llm/samples/module-gating-natural-memory-2026-11-06-live/journal.json \
  --judge-model gpt-6-sol --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/module-gating-natural-memory-2026-11-06-diagnostic-new
```

The helper is a separately hashed post-output artifact outside the frozen
`core/` and `scripts/` digest, not part of the measured implementation. Raw
prompts, responses, judge quotes, references and outboxes remain private under
ignored `samples/`; safe metrics retain their artifact hashes, including the
failed original replay and diagnostic attempts.

## Verification and next bounded scope

At the frozen implementation, local verification passes **1,907 Python tests**
with real PostgreSQL and one intentional local-HTTP deselection, with zero
unexpected blocked provider requests. The final focused set passes **45**;
gateway/local-HTTP passes **18** with two platform skips; Node checks pass **13**.
Fake controls include failed writes with skipped dependencies, invalid judge
responses, unknown/unexposed conditions, malformed completion envelopes and
latency-only feedback replay. The Fake Provider's immediate responses did not
expose the real preview race; its passing replay is not claimed as coverage of
that race.

The next bounded improvement is **speculative preview admission before HTTP**,
with its transport scheduling recorded explicitly. Preserve manual System2,
source writes, source approval and task ownership; first add offline delayed
preview controls for completion, pre-admission cancellation and in-flight
cancellation. Then prospectively measure whether unnecessary requests fall while
normal answers and durable-memory contracts hold. No runtime optimization or new
default has been inferred from this single measured trace.
