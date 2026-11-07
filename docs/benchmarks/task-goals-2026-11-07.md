# Explicit task-goal orchestration: first comparison — 2026-11-07

The bounded prototype carries explicit user obligations into existing model
inputs, but this first comparison **does not establish a reliable answer-quality
gain**. The two runtime-qualified pairs tie on all three criteria. One degraded
pair has an AUTO-only soundness pass; another fails in both conditions. The
original zero-network replay **fails** on a critic cancellation/fallback ordering
race. No paid repeat, rescue, resampling, budget change or post-output rubric
tuning was performed. The capability remains default OFF.

Implementation, fixture, references and [prospective protocol](task-goals-protocol.md)
were frozen at `ac61b3b5307908eb5945e76a5381bb2fb6e0d619`, on merged #99
(`01e0180d6d3e2fb65b3d82193b0b74e32b4125c4`). Subsequent changes add this
report, filtered metrics, discoverability links and a tracked private-output
ignore rule; measured core, scripts, fixture and protocol remain unchanged.
See [filtered metrics](task-goals-2026-11-07.metrics.json),
[capability contract](../task-goal-orchestration.md) and
[fixed fixture](../../sr-dual-brain-llm/examples/task_goal_comparison.json).

## What the prototype establishes

One complete headed list supplies two to four exact user-text anchors, with
explicit backward numbered dependencies. It does not infer unstructured goals.
Left generation receives solution obligations; right lead/detail receives
task-specific alternative/counterexample obligations; the existing critic may
report goal findings; integration re-evaluates requested goals and omissions.
Copied worker plans are re-derived against the question. All phases retain
existing native routing, manual modes, time/output/round limits and ownership.
There is no planner-model request or per-goal loop. Goals authorize no new source
approval, retrieval, persistence or external operation.

All four eligible AUTO cases receive assignments in their actual submitted
draft messages. The retry detail also receives assignments; schedule/queue
critics and unknown-rate reviews receive them, as do the queue/unknown-rate
integrations. The small arithmetic control has no plan or assignment in either
arm and returns exactly `102`. All ten turns clear the plan map.
Right lead wiring is covered by mechanism controls, not a live lead in this run.
The live right-detail contribution is empty on the retry case; input handoff
alone does not establish a useful alternative or successful goal fulfillment.

The schedule AUTO critic reports all goals addressed, but its first quote drops
Markdown trailing spaces from the reviewed draft. The exact substring check
rejects the whole goal packet as invalid; the normal external critic remains
separately valid. No goal repair is added. The queue critic reports all goals
unresolved. Final output differs from that snapshot, so its state is
`unreviewed_answer`, retaining the earlier unresolved reports. The final
unknown-rate review reports all goals addressed against the actual answer;
`verified_complete` still remains false. Retry uses manual System2 OFF and stays
`not_reviewed`. These states establish conservative bookkeeping, not truth.

## Answer observations and runtime failures

Each arm uses a fresh session with matched policy/RNG initialization, the actual
user question and unmodified RightBrainPolicy. Arm and blind-label orders are
fixed. Generation uses `gpt-6-luna`, a shared 2,048-token ceiling, the existing
450/900-token detail limits, critic cap 2,048 and no continuation. Director and
executive work are OFF. Other module settings use the balanced AUTO profile.

After all ten turns, one `gpt-6-sol` judgment per pair receives only the question,
authored reference and actual A/B answers. All five judgments satisfy the strict
JSON/literal-quote contract. References were authored prospectively by the
implementing assistant; this new single-model rubric has no independent human
calibration. Passes are model observations, not verified correctness.

The columns below show OFF / AUTO. A pair is runtime-qualified only when both
arms return normally without an observed critical cancellation, transport or
completion degradation. An engine `ok` alone does not qualify it.

| Case | Coverage | Soundness | Useful contribution | Runtime-qualified | Observed limitation |
| --- | --- | --- | --- | --- | --- |
| Schedule / deadline counterexample | pass / pass | pass / pass | pass / pass | Yes | AUTO goal-review quote is invalid; no inferred completion. |
| Retry / idempotency | pass / pass | fail / pass | pass / pass | No | Both ordinary details are empty `length` completions. |
| Bounded queue / deadlock | fail / fail | fail / fail | fail / fail | No | Both drafts and integrations are empty `length` completions. |
| Unknown queueing probability | pass / pass | pass / pass | pass / pass | No | Both drafts are empty `length`; later critical critics are cancelled before fallback. |
| Direct arithmetic | pass / pass | pass / pass | pass / pass | Yes | No eligible goals; exact direct answer in both arms. |

There are no uncertain or ungraded live judgments. Across all five pairs,
coverage and contribution have four pass/pass ties and one fail/fail tie;
soundness has three pass/pass ties, one fail/fail tie and one AUTO-only pass.
Across the two qualified pairs, every criterion has two pass/pass ties and no
directional difference. All degraded rows remain in the full table.

Unblinded implementing-assistant inspection finds that the retry judge's OFF
failure quote names an undefined `submit_with_transport` test helper. AUTO also
supplies a schematic test that assumes a fake harness. The judgment may apply
different strictness to those incomplete examples. This inspection neither
regrades the answers nor repairs the original score; it further weakens the
single directional signal. The sample cannot support a general capability or
efficiency claim, a causal attribution to right reasoning, or default promotion.

Six of ten rows are degraded. There are **eight empty `length` completions**:
two detail, four draft and two integration. In every such provider receipt,
`completion_tokens_details.reasoning_tokens` equals the entire completion-token
budget (450 for detail, 2,048 for draft/integration), leaving no visible text.
The existing detail contract withholds unusable detail. Ordinary draft and
integration paths can still return an engine-level `ok` with fallback material;
that does not establish a completed user answer. This availability/budget
boundary is a stronger next core target than further goal-label sophistication.

## Allocation and failed replay

The sole paid run makes **33/64 actual HTTP attempts**: 28 runtime attempts,
14 per arm, followed by five judges. Roles are 10 draft, 10 critic, 4 detail,
4 integration and 5 judge. All response-bearing envelopes end in `stop` or the
eight retained `length` completions. Two critical critic requests are cancelled
without response/usage, one per unknown-rate arm. Transport failures are zero;
cancellation is counted separately and is not free work.

Known usage is **30,960 input / 27,639 output tokens**. Full totals remain
**unknown** for the two cancelled requests. Runtime OFF known subtotals are
11,095 / 12,846 and AUTO 14,338 / 12,857; judges account for the rest. Extra
instruction tokens and independently generated answers prevent treating these
subtotals as an isolated cost effect. Handler durations are descriptive only.

The original replay runs with no provider transport under the offline HTTP
guard. Its first six non-timing rows match exactly. It stops establishing exact
reproduction on `unknown_rate:auto`: occurrence 17 (zero-based) is an owned
critic cancellation, followed by a live direct fallback at occurrence 18.
The partial replay ultimately contains 30 attempted journal rows and only the
first 36 of 66 matching transport events; later admission failures contaminate
subsequent rows and every replay judge. The first request-payload difference
appears later at occurrence 20 and is a consequence, not the initial fault.
The CLI exits 1 with `ValueError`; no equal canonical outcome is established.

A separate post-output, zero-provider diagnostic instruments the observer
without changing measured source, inputs or caps. It observes fallback
`start(18)` while the journal still expects `cancelled(17)`, at transport event
position 35. The recorded cancellation terminal arrives afterward. The journal
rejects the race and performs zero provider calls; it does not silently reorder
admissions or force a cancellation. A preliminary diagnostic could not start
because an older `/tmp` module shadowed an import and failed metadata matching;
that failure is also retained. Neither diagnostic replaces the original failed
replay or repairs its scores. Exact handling of this cancel/fallback schedule
remains open; the earlier preview study's different successful tape is unchanged.

Live canonical digest:

`f002ea639b8d010b43b84dc31cec65a5fcc04e95eeb6110561de9c24dde4be4a`

Live transport-event digest:

`4451e6bb79c1e9002686ff7f6a97f36a13aacf2917db6ef92a1936fd155c2916`

Full runtime messages, answers, judge quotes and raw journals stay private under
`samples/task-goals-2026-11-07-{live,replay,replay-diagnostic,replay-diagnostic2}`.
Filtered metrics publish bounded states, verdicts, receipts and artifact hashes.
The synthetic questions and authored references are in the tracked frozen
fixture. This is narrower than the protocol's private-artifact wording; runtime
artifacts remain private, and references never enter answering-model inputs.
The new directory prefix initially needed a local Git exclusion during the run;
no private file was staged. The result commit adds its tracked ignore rule.

## Verification and next capability work

The exact measured revision passes **2,029 local Python tests** with isolated
PostgreSQL, one intentional local-HTTP deselection and zero unexpected blocked
provider requests. Gateway/local HTTP passes 18 with two platform skips; Node
passes 13. Focused capability/critic checks pass 148 and the final paired/
lifecycle focus passes 54. All eight source-freeze CI checks pass, including
Python 3.10 anchor/lifecycle tests, PostgreSQL and gateway checks. This automated
verification does not make the failed live replay pass.

Mechanism controls cover real model/worker/controller prompt handoff, valid
unresolved findings entering existing integration, direct fallback, manual
System2 OFF, invalid/missing reports without extra repair, tampered plans,
source-write and literal-replay protection, context isolation and cancellation
cleanup without resetting round-trip limits. Scripted grades in those controls
are not model quality evidence.

#100's bounded prototype is implemented and measured, with a negative/limited
quality result and the replay gap retained. Next, make ordinary reasoning
completion-aware and allocate bounded reasoning/output work so difficult tasks
can emit a useful answer. Preserve unavailable/partial states, request ownership,
manual modes and total work caps. Any next paid comparison needs a fresh frozen
protocol; these observed cases become development evidence, not fresh test data.
Broad judge calibration remains in #89; critic-specific availability remains in
#87. The cancellation/fallback replay race needs its own narrow followup.
