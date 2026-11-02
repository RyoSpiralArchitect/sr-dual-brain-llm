# First TRN measurement — prospective protocol

Freeze and commit the fixture, runner and production code before provider calls.
This is a small frozen-draft mechanism pilot in fresh production EngineSessions,
with **8 trials / 5 problem groups**, not a natural-error-frequency or PostgreSQL
restart study. Six drafts are generated once with the real speech model; two
authored controls are separately labeled (one complete record replay, one wrong
arithmetic answer). The three Vireo trials share a source/problem, and the two
arithmetic trials share a problem; they are not independent replications.

Compare `legacy_default` (registered defaults/legacy switches absent) against
`all_auto` (all twelve canonical mode variables set to `auto`). Caller modes,
inputs, source snapshots and initial draft bytes are matched. Greetings and the
creative question request `executive_mode=observe` and observer `both`; the other
trials request both caller modes `off`. No arm can change these requested modes.
Use explicit left lead, System2 `auto` with balanced priority, evaluative dial,
normal round budgets, default local thresholds and temperature-drop amount zero.
Each fresh session has the same deterministic NumPy/Torch policy seed per trial.
Provider decoding and optional-task timing are not synchronized across arms.

Generate all six natural drafts before running comparisons. Alternate arm order
across the fixed shuffled trial order. A no-revision baseline keeps each frozen
draft. Grade every returned answer after generation/processing finish, using
randomized blind labels for the baseline and both arms. The rubric independently
assesses requested task/format, factual/numeric grounding and clarification.
Each pass/fail requires an exact nonempty quote; malformed grades are retained
as grader failures, with no retry or candidate exclusion. Report all planned
trials, available/graded/jointly graded pairs, per-criterion outcomes and origins.

Use `gpt-6-luna` for production roles and `gpt-6-sol` for the blind judge, with
2,048 output tokens, no auto-continuation and 60-second provider timeout. Hard
cap: **96 provider attempts total**, including draft generation, source support,
advice/director, cancelled requests, critic/repair/integration and judging.
The runner must not resample failed/completed cells or select favorable trials.

Journal exact request bodies and endpoint fingerprints without auth headers.
Retain incomplete/empty completions and expected optional-task cancellation.
Cancellation without a response leaves billed tokens unknown; it is not a
zero-token call. Production fallbacks can yield a graded answer but are separately
reported as degraded provider health. Transport/auth/quota errors stop additional
live requests. Returned invalid completions permit only later unattempted cells,
never a retry of the failed cell. Failed natural generation makes both corresponding
processing cells unavailable. Preserve partial artifacts on interruption.

Report processing request counts and token usage by role separately from shared
draft-generation and judge overhead. These are **post-draft mechanism** costs,
including fresh-session support preparation; they are not end-to-end savings.
Do not silently sum unknown usage as zero or publish an exact monetary saving
when any billed usage is unknown. Report handler latency as descriptive only:
one pair per trial, deadline cancellations and sequential provider load make it
unsuitable for a latency/quality significance claim.

Expose reached/admitted gates and bounded reasons, plus actual module events and
call roles. Cold sessions may not generate DMN reflections or stalled critiques;
zero exposure is an evaluation coverage gap, not proof of efficacy for that gate.
No broad AUTO default promotion is justified by this pilot. Investigate unique
AUTO quality failures before merge, preserve existing compatibility defaults,
and derive the next work item from paired quality, task waste and coverage gaps.

Raw journals/reports stay private under ignored `samples/`; publish safe aggregate
metrics and concise evidence. Attempt full exact-request replay with zero live
transport after measurement. Report coverage/status/answer/grade differences;
if task timing prevents exact replay, retain the mismatch rather than claiming
reproducibility or making fallback live requests.

## Post-review steering confirmation — 2026-11-02 (Asia/Tokyo)

The first pilot exposed coupling between an inhibited Director task and its free
trivial-chat steering; Codex review requested a fix before merge. Preserve the
original eight-trial measurement and its frozen revision. Repair the control flow
so a caller-requested Director keeps deterministic trivial steering without a
provider task, while explicit System2 ON still takes precedence over consult skip.
Do not retune gate thresholds or regenerate the first pilot's answers.

Before further live calls, commit that fix and a separate **two-trial natural-draft
confirmation** (`module_gating_steering_confirmation.json`). Its English `Hello!`
and Japanese `こんにちは！` inputs were not used in the first pilot. They test this
specific repair rather than generalization across workloads. Match explicit left
lead, System2 auto/balanced, Executive observe, observer both and frozen initial
drafts between the same two profiles. Use the same models, grading/quote rules,
timing behavior and receipt accounting, with a **24-attempt maximum** including
generation and judging. The prediction is fewer post-draft attempts without
replacement right-detail/left-integration work and without changed answer quality.
No resampling, arm filtering or output edits are allowed. Report this confirmation
separately from the first pilot, including any contradictory result, and perform
network-free exact replay against its own frozen source revision.
