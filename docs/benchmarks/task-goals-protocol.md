# Explicit task-goal orchestration comparison — 2026-11-07

Freeze implementation, fixture, authored references, criteria, ordering, caps
and replay contract before any paid output. Earlier studies/scores remain
immutable. This is the first bounded capability prototype for #100, not another
extension of the admission/measurement framework or a default promotion.

## Mechanism and fixed comparison

Use `task_goal_comparison.json` in its fixed five-case order: non-preemptive job
scheduling, timeout-after-commit retry/idempotency, bounded-queue deadlock,
insufficient queueing measurements and a single-number arithmetic control.
The first four have three explicit user goals; the control has no eligible plan.
References were authored before outputs by the implementing assistant, not
independently adjudicated humans. None is supplied to generation or planning.

For each case, run the prospectively declared OFF/AUTO arm order once. Each arm
has a fresh in-memory EngineSession with the **same session ID and policy/RNG
initialization**, clock and actual user question. Change only
`task_goal_orchestration=off|auto`; retain the other existing all-AUTO balanced
profile, Director/executive OFF and the case's declared System2 mode. The retry
and direct-answer cases use explicit OFF; the other three use AUTO. No primary
memory fixture, substituted/shared draft, forced consultation policy or planned
answer is injected. No persistent conversation state crosses arms.

Use real `gpt-6-luna` generation with production nonstream JSON, the same
2,048-token generation ceiling and 2,048-token critic cap in both arms, existing
450/900-token ordinary detail budgets, timeout/repair/fallback rules and no
continuation. User goals share existing phases; there is no planner-model call
or per-goal loop. Native routing and the explicit System2 modes retain authority.
Requests/rounds may differ because actual drafts and reported omissions differ;
count every resulting attempt rather than assuming equal realized work.

The prototype accepts complete explicit request lists of two to four goals.
Primary writes, literal reproduction and unsupported/simple shapes remain
direct. Each goal has an exact user-text anchor and only explicit backward
numbered dependencies. Workers revalidate copied plans. Right work requests
useful task-specific alternatives/counterexamples; critic observations require
known goal IDs and literal positive evidence from the actual reviewed snapshot.
All observations are provisional: no critic confidence, structured status or
goal label proves semantic completion or approves source facts. A revised or
partially reviewed answer retains unreviewed state. Missing/invalid goal fields
make no additional goal-repair request or review round. Valid unresolved reports
for an exact snapshot may use the existing integration phase.

Record actual goal-assignment blocks in submitted HTTP message payloads,
separately from internal eligibility or stage-selection metadata. A cancelled
request proves client admission, not that a provider consumed it. Capture final
answer, policy class/action, actual critic accounting, detail/preview outcomes,
goal-state cleanup, schema/completion failures, degradation, usage and elapsed
handler durations. Never replace an unavailable answer with a fixture, omit a
failed arm, or reclassify a local fallback as healthy external completion.

## Independent answer observations

After all ten runs, make at most one `gpt-6-sol` judgment per pair in its fixed
blind A/B order. Only the actual question, authored comparison reference and
labeled answers enter the grading packet. Do not supply arm names, plan fields,
gate decisions, internal goals/reviews or historical scores. Both conditions
already receive the same explicit user requirements. The reference can guide
evaluation but cannot leak into answering-model knowledge.

Grade three independent criteria: **goal coverage**, **reasoning soundness** and
**relevant contribution**. Coverage includes every explicit output/constraint;
soundness includes correctness, appropriate uncertainty and no invented facts or
test execution; contribution includes a useful alternative/counterexample when
requested and appropriate directness for the arithmetic control. Use
pass/fail/uncertain, with a validated literal answer quote of at most 500
characters for every pass/fail.
One strict terminal JSON completion, a 2,048-token cap, no grading repair or
resampling. Preserve malformed, failed, unavailable or uncertain judgments.

This new rubric is **uncalibrated single-model observation**, not independent
human validation or proof of general capability improvement. Report every
per-arm criterion and case, provider-degraded rows, disagreements and ungraded
pairs. Do not turn uncertain into a numeric half-pass. Compare only clearly
opposed pass/fail pairs as directional observations, label runtime qualification,
and retain the complete raw outcome table. The sample is too small for a stable
quality, efficiency or causal contribution claim. No measured self-report or
trace-field success can substitute for answer assessment.

## Budget, failures and replay

Hard cap **64 actual HTTP attempts total**, reserving five paired judge slots
before runtime: **59 runtime attempts maximum**. Count all generation, preview,
detail, critic, repair, fallback, cancellation and failures. Existing real
transport/auth/quota failure keeps the sticky admission latch closed across
runtime and grading. No paid repeat, rescue, continuation or extra diagnostic.
Unknown usage remains unknown with known subtotals; cancellation is not free.
Preserve unavailable answers and judge failures explicitly.

Use the existing ordered `ScheduledJournal` with strict starts and terminal
response/error/actual-owner-cancellation events. Replay has no provider transport
and runs under the offline HTTP guard. Each arm resets its session/policy state;
no turn-to-turn controller feedback is carried across arms. Canonical outcomes
exclude wall-clock timing/reward, retaining actual answers, goal snapshots,
completion/failure/degradation, routing, scores and usage. Require every exact
ordered request/event and equal canonical outcomes. The bounded local replay
wait is 30 seconds, allowing the unchanged native consultation timers to deliver
their own cancellation, and must preserve subsequent owner cancellation. Preserve any real divergence
without tuning or a paid repeat.

Keep full questions/answers, raw messages, references, judge quotes and private
journals under ignored samples. Publish filtered metrics, both source/replay
provenance and artifact hashes. Instrumented durations are descriptive, not a
latency-improvement claim. Fake controls demonstrate mechanism, failure
preservation and replay; their scripted grades are not real answer quality.

```bash
python sr-dual-brain-llm/scripts/benchmark_task_goals.py --model gpt-6-luna \
  --judge-model gpt-6-sol --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/task-goals-2026-11-07-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_task_goals.py \
  --journal sr-dual-brain-llm/samples/task-goals-2026-11-07-live/journal.json \
  --judge-model gpt-6-sol --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/task-goals-2026-11-07-replay
```
