# Completion-aware replanning and bounded critic reviews

The [frontier study](benchmarks/reasoning-frontier-2026-11-09.md) separated
completion recovery, semantic error and review availability. This prototype
carries those distinct states into existing stages. Offline tests demonstrate
contracts. The [first fresh hard-task measurement](benchmarks/replanning-stress-2026-11-09.md)
found no initial-to-final score gain or completion recovery; useful critic
availability and intermediate work handoff remain limitations.
The previous study's outputs and scores remain unchanged.

## Optional reconstruction

`Orchestrator` / `TaskState` own a prefrontal work plan. The separate TRN module
`completion_replanning` owns admission and stays **default OFF**:

```bash
export DUALBRAIN_MODULE_TASK_GOAL_ORCHESTRATION_MODE=auto
export DUALBRAIN_MODULE_COMPLETION_REPLANNING_MODE=auto
```

AUTO requires two to four explicitly anchored goals, an exact current candidate
with recorded partial/empty output from a length limit, and remaining ordinary
left-output allocation. ON retains these hard conditions. Unknown generation,
empty-stop responses, transport failure, completed candidates, unsupported goals
and exhausted allocation do not qualify. Source writes and literal replay remain
protected. No consultation, System2 or integration is forced by this module.

Before an existing integration, the controller proposes a deterministic work
plan from goal dependencies, candidate fingerprint, generation-record index and
remaining allocation. The orchestrator retains at most three turn-local plans;
duplicate preparation of the same snapshot is idempotent. Task cleanup removes
the plans without resetting round-trip limits.

The integration prompt assigns `derive_then_check` to every original goal in
dependency order, carries unresolved constraints forward and checks the combined
answer against all conditions and output requirements. Missing dependencies do
not establish independence. A matching incomplete snapshot is required; later
changed/completed candidates cannot inherit the plan. The model emits one answer,
without exposing its work plan or inventing missing facts or tool execution.

There is no planner-model call, per-goal execution loop, domain-specific answer
branch, new token allocation or fact authority. Ordinary generation keeps its
single-request/no-continuation contract and complete-draft retention. OFF retains
the existing integration recovery path. `metrics.task_goals.replanning` and
`completion_replanning` telemetry contain only bounded strategy/status/count/
allocation metadata and `verified_complete=false`, never goal or answer text.

## Critic roles and time

The critic receives structured-review instructions without the ordinary answer
role's conflicting “answer directly / no self-critique” rules. Question, candidate
and context are data, including when the original task requests another output
format. The review stays concise and covers every active goal. Existing strict
completion, shape, quotation and external-validity checks remain mandatory.

Each critic invocation shares one monotonic local window, capped by its existing
configured/client timeout, across initial requests, existing transport retries,
backoff, the optional single reformat and explicitly configured continuation.
Every JSON HTTP request rechecks the remainder. Normal non-critic requests keep
their existing timeouts. No timeout or output ceiling is automatically enlarged.

Admission, retry and late-result checks clip the remainder by the RPC turn and
actual callosum-owned lifetime. HTTP itself uses only the local/client remainder:
the RPC/callosum owner cancels work at its own unchanged deadline, without a
second HTTP timer racing that same boundary. A shorter local/client window
still bounds direct review HTTP. `ReviewBudgetExhausted` distinguishes proven
local review-window expiry (including whether HTTP was attempted) from generic
transport/auth/quota failure; it never makes an external review valid.
`remaining_consultation_ms()` reads the pending request in operation-local context,
ignores packet deadline edits, returns zero after abandonment, and resets after
the operation. Even a custom provider swallowing cancellation cannot use its
late response to start repair or supply a valid external review. No additional
`wait_for` timer is introduced inside model work; the owner retains cancel-once
and cleanup authority.

Expired reviews stay unavailable (`deadline_exhausted`), with external validity
false and existing failure/micro provenance retained. An already admitted native
direct fallback has its own invocation window under its RPC owner; no new
fallback/retry is introduced. Attempt accounting and unknown usage remain
independent of output quality and local token reservations.

## Verification and next measurement

Scripted-provider RPC tests cover empty/partial candidates across event ownership,
SQL NULL aggregates and evidence expiry, actual prompt handoff, dependencies,
finite allocation, OFF recovery, complete-draft retention, manual System2 OFF,
source exclusion and cancellation cleanup. These measure mechanisms, not model
intelligence. Deterministic-clock/HTTP tests cover shared repair time, late clean
JSON, retry inhibition, continuation, forged deadlines, swallowed cancellation
and the next request's fresh lifetime. Review-budget cases also run on Python 3.10.

Before claiming gains, freeze fresh cross-domain questions, repeats/order, arms,
actual-attempt/output ceilings and exact replay criteria. Report semantic repair,
completion recovery, regression, review availability and actual/unknown usage
separately. These development examples and old paid outputs are not confirmation
data. The first frozen measurement and its original cap/replay failure are
reported separately from the corrected fresh eight-task experiment linked above.
