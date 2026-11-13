# Completion-aware replanning and bounded critic reviews

The [frontier study](benchmarks/reasoning-frontier-2026-11-09.md) separated
completion recovery, semantic error and review availability. This prototype
carries those distinct states into existing stages. Offline tests demonstrate
contracts. The [first fresh hard-task measurement](benchmarks/replanning-stress-2026-11-09.md)
found no initial-to-final score gain or completion recovery; useful critic
availability and intermediate work handoff were observed limitations. The
optional mechanisms below are subsequent development work, not evaluated gains.
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

`DUALBRAIN_COMPLETION_WORK_PRODUCTS=1` separately enables work products and is
**default OFF**, including when replanning is ON or AUTO. This keeps the frozen
replanning experiments' request sequence reproducible. When both switches are
enabled, the completion-replanning gate admits a plan, both the stage ceiling
and remaining allocation are at least 1024 tokens, and at least 30 seconds remain in an RPC
turn (or no turn deadline exists), the left model first makes **one** bounded
work-product request for the first explicitly anchored goal. Its ceiling is at
most a quarter of both the stage ceiling and the remaining ordinary answer
allocation; the following integration receives the remainder. The product must
have a successful terminal completion and strict `goal_id`, `work`, `unresolved`
JSON shape. Only then does its text enter the turn-local integration prompt,
labeled as unverified internal work. A partial, malformed, duplicated-key or
stale product is never handed off. The final answer is generated separately;
an internal product never becomes a user-facing answer or verified source.

This adds at most one model request to an admitted recovery within the existing
output-ceiling ledger. It is not a per-goal execution loop or a domain-specific
answer branch. The complete-draft retention and no-continuation contracts still
apply. OFF retains the existing integration recovery path. `metrics.task_goals`
now reports `replanning` and `work_products` as bounded status, goal ID, cap and
handoff receipts with `verified_complete=false`; no goal, product or answer text
is exposed there. Request count, latency and billed usage can increase when the
gate is enabled; reserved output ceilings are not billed-token receipts.

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

## Optional completion-aware critic routing

`DUALBRAIN_CRITIC_COMPLETION_AWARE=1` enables a separate default-OFF routing
prototype. It checks the controller's turn-local generation receipt, not a
worker packet or the model's claim. A recorded partial/empty candidate is not
sent for semantic review. The existing integration first tries to complete the
answer; a successful terminally complete result can receive a critic request
in the verification slot. Existing failure fallback may add a direct attempt.
If integration stays incomplete, the review
remains unavailable. A complete initial candidate follows the usual review and
revision path. Unobserved or unverified candidates retain legacy routing.

For a complete candidate, the callosum owner may receive up to the configured
local critic window plus 500 ms, bounded by `DUALBRAIN_TIMEOUT_MAX_MS` and the
actual RPC turn deadline. With an RPC deadline, five seconds are reserved for
finalization; fewer than three available review seconds means no request. This
explicit mode can increase the old slot-derived callosum timeout, while the
critic local/client window, output cap, retries, turn owner and cancellation
rules stay bounded. It cannot guarantee a valid review. If the review sees only
a trimmed candidate, System2 cannot mark that full answer resolved.

`critic_status=incomplete_candidate` or `turn_budget_unavailable` records a
deferred review without an external request. `system2.attempted_rounds` counts
actual critic requests; a deferred routing slot is not a completed review.
Timeout fallbacks and internal products remain unverified. A later valid critic
report is still a model observation, not proof that the answer is correct.

## Verification and next measurement

Scripted-provider RPC tests cover empty/partial candidates across event ownership,
SQL NULL aggregates and evidence expiry, actual prompt handoff, dependencies,
finite allocation, OFF recovery, complete-draft retention, manual System2 OFF,
source exclusion, deferred review and cancellation cleanup. These measure mechanisms, not model
intelligence. Deterministic-clock/HTTP tests cover shared repair time, late clean
JSON, retry inhibition, continuation, forged deadlines, swallowed cancellation
and the next request's fresh lifetime. Review-budget cases also run on Python 3.10.

Before claiming gains, freeze fresh cross-domain questions, repeats/order, arms,
actual-attempt/output ceilings and exact replay criteria. Report semantic repair,
completion recovery, regression, review availability and actual/unknown usage
separately. These development examples and old paid outputs are not confirmation
data. The first frozen measurement and its original cap/replay failure are
reported separately from the corrected fresh eight-task experiment linked above.
