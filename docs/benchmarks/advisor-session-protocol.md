# Advisor session sequence — prospective protocol (2026-11-03, Asia/Tokyo)

Freeze and commit the complete fixture, runner and production source before paid
calls. Preserve the earlier reseeded-history pilot and its frozen revision.
Do not tune policy thresholds/defaults on this sequence. The experiment evaluates
the existing 60-second, recent-max resource history, including its limitations.

The pre-freeze controlled RPC test exposed that Python 3.11+ `wait_for` can use
parent-task cancellation for a local timeout, consuming the turn's cancel-once
guard. Repair advisor waits to use a non-cancelling timed wait plus explicit
owned-child cancellation. Before live calls, verify successive local timeouts
and a real RPC deadline after a local timeout. No policy thresholds are changed.

## Live sequence

Create one production EngineSession per arm, initially empty. Keep each of those
two sessions for all six turns: `cold`, `warm`, `warm_again`, `expired`,
`recovered_warm`, `bounded_rpc`. Never seed, clear, clone or replace advisor history
between turns. All normal controller/learning/memory state evolves in its own arm;
only initial state, questions, caller modes and per-turn policy RNG seeds match.
Subsequent learned state and answers may differ and must be reported.

Compare `untimed_advisors` (three advisor modes ON) with `timed_advisors` (three
AUTO v2); other nine gates are AUTO in both arms. Explicit left lead, System2
auto/balanced, evaluative dial, Executive observe and observer both are fixed.
Conversation record context is disabled and no external records are seeded;
session-local ordinary source/state updates continue. No PostgreSQL restart or
persistence intervention. Generate each primary draft normally per cell and count
its HTTP attempts: drafts are not frozen/shared across arms. Alternate arm order
at each step. No separate warmup or training turn is free/excluded from accounting.

Run every cell through the real TurnDispatcher. The first five turns have its
ordinary 120-second cap; `bounded_rpc` has a prospectively fixed 20-second cap.
Before either `expired` cell, leave both sessions idle for at least 61 real
monotonic seconds after both preceding cells drain. Other steps start as soon as
the preceding paired step finishes. Scheduling/provider latency can make an
intended warm observation expire early; label actual recorded facts rather than
forcing the intended phase or retrying it. All failures/unavailable cells remain.

Use `gpt-6-luna` for production, `gpt-6-sol` for blinded judgment, 2,048 output
tokens, no continuation and 60-second provider timeout. Hard maximum: **96 HTTP
attempts total**, including twelve complete turns and six judge packets. Stop
additional paid requests on transport/auth/quota failure. No resampling,
selecting favorable cases or replacing failed generation. Preserve partial work.

After processing, blindly grade every returned final answer and available primary
draft using shuffled labels and independent task/grounding/clarification criteria.
Require literal quotes for pass/fail. Report twelve planned finals, their coverage,
six planned pairs, jointly graded pairs, criterion regressions, draft outcomes and
malformed grades. Natural draft/provider variation and evolving state preclude
attributing every request/quality difference to the gate.

## Controlled causal timing coverage

Use a separate, network-free transport with synthetic receipts, real production
EngineSessions, real advisor waits and TurnDispatcher timers. Use the same six
phase order and preserve history naturally. For causal isolation only, choose
System2 off and a skip-consult policy; these are explicit test interventions, not
the live profile. The primary output is the fixed string `Controlled answer.`.

Director and metrics take 1.8/2.7 seconds in the first two phases, exceeding their
unchanged 1.6/2.5-second local waits. They become fast (5 ms) in `warm_again`,
before the slow history expires, to measure suppression of newly useful work.
Advance only the history clock by 61 seconds at `expired`, explicitly labeling
this simulated idle; record() still advances the same ledger without seeding.
The last three phases retain fast Director/metrics output, exposing cold recovery
and positive fast AUTO completion in the same session.

Executive takes 1.35 seconds and ordinary primary generation 1.45 seconds,
demonstrating parallel completion. `recovered_warm` has a 3-second real RPC cap;
the final controlled phase shortens primary generation to 20 ms and uses a
400 ms real RPC cap. Expect AUTO to inhibit known-slow Executive work, preserve
an answer, and drain all tasks; ON may time out without committing. Keep all
responses/errors and assert mandatory cancellation/steering behavior. These
are controlled timing facts and deterministic answer checks, not live quality,
real token receipts or paid-cost evidence.

## Measurement and replay

Record per-turn gate facts, estimate kinds/sample counts, admission/start/outcome/
structural usability, cleanup, task leakage, RPC failure/commit state and session
turn number. Report HTTP roles including primary generation, consultation,
integration/repair and every advisor. Keep unknown cancellation usage unknown;
synthetic controlled receipts are separately labeled. Do not claim an exact
monetary reduction with unknown live receipts. Wall-clock comparisons are
descriptive, not a significance claim.

Live history is never seeded. A small wrapper records its actual estimate reads.
For exact-request replay only, substitute those recorded role/model timing reads
because cached completions cannot recreate real response timing. Record per-cell
wall-time snapshots for source timestamps and repeat policy seeds; replay skips
the real idle and never falls back to network. Validate all journal occurrences,
answers/drafts, statuses, gate decisions, history reads, grades and pending tasks.
Real deadline/cancellation ordering and elapsed times may differ; retain and
report any mismatch. Do not claim replay validates natural timing adaptation.

Raw artifacts stay under ignored `samples/module-gating-advisor-sessions-*/`;
publish a bounded, content-free projection and concise measured analysis. Report
unexpected expiry, slow-bound suppression duration, positive fast exposure and
Executive deadline coverage even when they contradict the intended hypothesis.
