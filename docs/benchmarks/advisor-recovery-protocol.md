# Bounded advisor recovery — prospective protocol (2026-11-03, Asia/Tokyo)

Freeze production source, runner, parameters and the complete independent fixture
before any paid calls. Preserve the #72 study and its original grades/known rubric
defect. Do not rescore it or tune a policy on these new outputs after measurement.

## Policy and comparison

Hypothalamus owns per-role/model timing and an opaque recovery lease. AUTO v3 may
explore known-unaffordable work after 15 seconds since the latest observation,
with **at most one recovery start per retained profile in a rolling 60 seconds**.
The same profile cannot hold two active leases. Queries do not consume allowances;
claim at actual work start, after checking the actual profile and remaining RPC
budget. OFF, missing caller modes, trivial turns and pre-start cancellation do
not spend a probe. ON remains ordinary execution, not a recovery probe.

A probe needs a known window of at least 1,600 ms for Director, 2,500 ms for metrics
or 2,000 ms for Executive. Native timers and the outer deadline remain unchanged.
Unknown windows do not establish recovery eligibility; existing normal unknown-
window/cold behavior remains compatible. Fast usable completion plus the existing
50 ms reserve must fit its claimed window before older observations are replaced.
Newer concurrent observations survive. Empty/error/late output cannot clear slow
history. Sample expiry does not erase an unexpired rolling probe limit. Eviction
keeps the bounded 32-profile resource store; a discarded profile becomes unknown,
with ordinary cold behavior rather than a preserved infinite budget history.

Compare `ttl_only_auto` (new ledger with recovery disabled internally, equivalent
to v2's recent-max/TTL admission) against `recovery_auto`. All twelve registered
module modes are AUTO in both profiles; no new user environment switch is added.
The same unknown-estimate fallback, manual precedence, evidence and ownership
contracts apply. Defaults stay ON within requested advisor modes.

## Live sequence and aligned rubric

Keep one initially empty production EngineSession per arm across five turns:
`initial`, `opportunity`, `followup`, `relapse`, `capped`. Names are planned
opportunities, not assertions about provider speed. Never seed/clone/reset history.
After both initial turns drain, idle both sessions for **16 real seconds** before
opportunity. Otherwise proceed immediately, alternating arm order by phase.
Natural processing time may expire samples or rolling allowances; retain and
report actual facts rather than forcing intended states.

Use ordinary primary generation per arm/turn, explicit left lead, System2
auto/balanced, evaluative dial, Executive observe and observer both. Each turn
uses a real TurnDispatcher with a 120-second cap. Match initial state, questions,
caller modes and per-turn policy seeds; let all ordinary source/learner state
evolve independently. No external source records or PostgreSQL are seeded;
conversation record context is disabled, but ordinary support/state updates and
every HTTP role still count. No free warmup or frozen/shared draft.

Question/task obligations were checked before generation:

| Question | Only required task obligations |
| --- | --- |
| LRU eviction and one recency-only limitation | Exactly two sentences; eviction choice and one limitation |
| Jitter, retry limit, supported idempotency key | Exactly three bullets; one requested benefit each |
| Running holder after lease expiry; stale writes | Exactly two sentences; cause and one requested rejection mechanism |
| Concurrent-query limit analogies | Exactly two sentences; one apt analogy per sentence |
| Retry budget and a stopping signal | Exactly two sentences; amplification control and one requested stopping signal |

Grounding judges asserted factual claims; omission can fail task independently.
Clarification judges unnecessary or missing clarification. No extra next action,
label or technical qualification is required unless the question asks for it;
false asserted guarantees may still fail grounding. Blind grade all available
finals and primary drafts using shuffled labels and literal quote validation.
Retain malformed grades, all planned cells and any criterion regressions.

Production roles use `gpt-6-luna`, judge `gpt-6-sol`, 2,048 output tokens, no
continuations and 60-second provider timeout. Hard cap: **96 paid HTTP attempts**
for the complete live study, including all ten turns, probes, support/repair and
five judge packets. Never resample/replace a failed cell. Transport/auth/quota
failure blocks new paid requests; preserve partial artifacts and unknown receipts.

## Controlled timing and replay

Run two separate network-free transport regimes with synthetic receipts, real
advisor waits, real TurnDispatcher ownership and the same retained sessions.
For causal isolation only, set System2 off, skip consultation and use fixed
`Controlled answer.` output. Executive and primary take 10/30 ms; support is 2 ms.
Advance only the resource history clock by 16 seconds at opportunity; label that
idle simulated, not a live timing measurement. No history is seeded.

In `recovery`, Director/metrics take 1.8/2.7 seconds on initial, become 5 ms on
opportunity/followup, then slow again on relapse/capped. Predict one recovery
probe per role, two useful early completions, continued ordinary fast admission,
then one ordinary timeout per role on relapse and inhibition on capped. Count
these relapse failures, not just successful probes. In `slow`, those roles remain
slow throughout: predict only two additional wasted attempts versus TTL-only,
with no second recovery attempt before its rolling limit. No simulated token
receipt is paid-cost evidence and structural usability is not correctness.

Offline tests additionally cover early cancellation, empty/errors/late output,
profile changes, newer concurrent evidence, expiry with a still-active limit,
duplicate/competing claims, OFF/ON, unknown windows and tight RPC floors. Preserve
cancel-once cleanup and actual deadlines; do not raise all timeouts.

Report quality, whole-turn attempts by role, probe starts/outcomes, ordinary
completions, cancellation/cleanup, source counts and pending tasks together.
Extra probes are a cost: fewer requests alone is not the acceptance metric.
Unknown billed usage remains unknown; do not claim exact monetary savings.

Live history advances normally. Record actual latency/status reads and lease
claims/finishes. In **replay only**, restore those observations/capabilities so
instantaneous cached responses do not masquerade as new live learning. Match
role/method occurrence counts, exact journal occurrences, statuses, drafts/finals,
gate decisions, grades and drained tasks with zero network fallback. Wall-clock,
timeout/cancellation labels and async completion order can differ; report them.
Controlled traces validate timers and are not replayed as real-time evidence.

Publish only bounded aggregate projections. Keep raw artifacts in ignored
`samples/module-gating-advisor-recovery-*/`. One correlated sequence per arm and
synthetic recovery cannot justify broad AUTO default promotion. A valid outcome
may show bounded extra waste with no live recovered utility; retain that result.
