# Bounded reasoning effort and consultation cleanup — 2026-11-08

The fresh five-pair pilot does **not** support promoting low reasoning effort.
Provider-default alone passes coverage and soundness on one qualified pair;
the other four pairs tie on every criterion. Low avoids one empty detail but
makes a concrete numeric-substitution error that the ordinary critic misses.
The sole paid run uses **28/64 HTTP attempts**. Its strict zero-network replay
matches **28 requests / 56 transport events** and canonical outcomes. Provider
default remains unchanged; low stays opt-in.

Code, fresh fixture/references and the [prospective protocol](reasoning-effort-protocol.md)
were frozen at `8feec824a55b24a8d9b9309bb9c58c62c9a197a9`, on merged #104
`37d86db4a8200fc45ad1c4b8f97253b32ef2f82b`. Subsequent changes add only this
report, filtered observations and discoverability links. No paid repeat, rescue,
resampling, cap/profile change or rubric tuning occurs. See the
[filtered metrics](reasoning-effort-2026-11-08.metrics.json),
[fixture](../../sr-dual-brain-llm/examples/reasoning_effort_comparison.json) and
[cleanup contract](../consultation-draining.md).

## Fixing the cancellation/fallback handoff

A consultation timeout returned before its cancelled operation finished async
cleanup, allowing a direct fallback to start first. The serial controller now
drains the channel before leaving that consultation wrapper. A later owner
cancellation waits for the owned cleanup, then propagates. Standalone Callosum
API timeout/capacity semantics remain intact. No new model request, local
`wait_for` timer, cancellation injection or event normalization is added.

Immediate draining also exposed a Python 3.12 idle-wait starvation: gathering
already-done operations could finish synchronously while the callback that
removes them was still queued. Reaping those done operations makes the wait
terminate and keeps exception/capacity cleanup idempotent. Native provider/RPC
limits remain authoritative; timeout reporting may include owned cleanup time.

The synthetic regression fails before the change with one operation still
running at fallback, then passes through the real controller/channel/worker and
JSON-round-tripped scheduled journal. Owner cancellation during/after draining,
RPC deadline and subsequent request reuse, repeated cancellation and the
subprocess-bounded queued-callback case also pass.

The original #101 failed replay and all old grades remain unchanged. Separate
historical-source validation applies **only the current drain fix** to measured
source `ac61b3b5307908eb5945e76a5381bb2fb6e0d619`. Only a derived metadata code
hash changes; raw requests, responses, event order, fixtures and criteria do not.
All **33 requests / 66 events** and non-timing canonical outcomes match, with zero
provider calls and zero blocked HTTP attempts. Both cancelled calls retain their
unknown usage. The overlay code digest is
`02ddf6043820b0d1304538078808998d0c72ee0275ba509c4d6a6cd8336e4269`;
the private exact patch, runner, artifacts and hashes are retained.

Current-source validation against that old tape also matches all **28 runtime
requests / 56 runtime events**, including both owned critic cancellations. Its
first different request is the old queue judge packet at occurrence 30
(zero-based): #102 intentionally changed both empty-generation failure answers.
The strict journal rejects that different packet; it does not reuse an old grade
for new answer text. That current-source run remains incomplete, with no equal
whole-study canonical result. This is distinct from both the original failed
replay and the historical-source overlay success. The fresh study below tests
current answers with its own frozen tape.

## Fresh profile comparison

Each arm has a fresh session with matched per-case policy/RNG initialization,
fixed arm and blind-label order, native RightBrainPolicy and the actual question.
Goals are AUTO in both arms. Other settings use balanced AUTO, with Director and
executive OFF; manual System2 modes remain fixed per case. No primary seed,
external persistence, forced action or substituted draft is supplied.

Generation uses `gpt-6-luna`, the same 2,048-token generation/critic ceilings,
existing 450/900-token detail ceilings, an 8,192-token left-generation allocation
and no continuation. Only ordinary draft/integration/detail receive the opt-in
low parameter in that arm. Critics retain provider default in both arms. Actual
submitted profile counts are published separately from requested settings.

Five paired `gpt-6-sol` judgments receive only question, prospectively authored
reference and actual blind A/B answers. All five pass strict terminal JSON and
literal-quote validation; none is uncertain, unavailable or ungraded. References
are authored by the implementing assistant and the rubric is an uncalibrated
single-model observation, not independent human validation. The GPT-6 payload
omits unsupported temperature rather than establishing temperature-zero sampling.

The table shows **provider-default / low** verdicts. Runtime qualification
requires both arms to avoid critical transport, empty/incomplete generation and
allocation degradation; handler `ok` or a complete left answer alone is not enough.

| Fresh case | Goal coverage | Reasoning soundness | Relevant contribution | Qualified pair |
| --- | --- | --- | --- | --- |
| Lower-bound binary search | pass / pass | pass / pass | pass / pass | Yes |
| Conditional test dependence | pass / pass | pass / pass | pass / pass | Yes |
| Join multiplicity / equal order totals | pass / pass | pass / pass | pass / pass | **No** — default detail empty |
| Inverse measurements | **pass / fail** | **pass / fail** | pass / pass | Yes |
| Direct arithmetic | pass / pass | pass / pass | pass / pass | Yes |

Across all five pairs, coverage/soundness have four pass/pass ties and one
default-only pass; contribution has five pass/pass ties. Across the four qualified
pairs, coverage/soundness have three pass/pass ties and the same default-only
pass; contribution has four ties. There is no low-only pass. The sample is too
small for a general profile ranking or isolated causal claim about right reasoning.

Unblinded implementing-assistant inspection corroborates the numeric failure:
the low answer derives the correct inverse, but evaluates the example as
`(x,y)=(1,2)` rather than `(3/2,2)`. Exact substitution gives `2*1+2=4`, contrary
to the visible `u=5`; its `v=6` check alone would not catch that mismatch. The
ordinary external critic returns `ok`. Its optional goal packet is invalid and
correctly withheld, so this never becomes verified goal fulfillment. The grade
is preserved; this arithmetic inspection does not regrade any other cell or
claim independent human calibration. [#106](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/issues/106)
proposes bounded checks of concrete numeric examples against explicit constraints,
using existing micro-critic/Cerebellum responsibilities and fresh later evaluation.

On search, default uses one integration and two critic attempts, while low uses
one critic with no integration. Both final answers pass. On SQL, default's
450-token detail receipt spends all 450 completion tokens on reasoning and emits
no text; the existing detail contract withholds it. Low detail ends in `stop`
with 262 completion tokens, including 96 reasoning tokens, and is integrated.
Both final SQL answers pass. These are realized path observations, not a general
quality, cost or latency benefit.

All ten selected answers have terminal-completion provenance; semantic correctness
remains not assessed by that field. All plan maps clear. Search has two valid
provisional goal reviews; conditional/inverse cases have four invalid goal
packets; SQL's two manual-OFF turns remain not reviewed. Direct control has no
goal plan/assignment and returns exactly `42` in both arms. Structured state and
critic confidence do not establish semantic correctness.

## Allocation, replay and verification

There are **23 runtime HTTP attempts** (default12, low11) and five judges: 10
draft, 7 critic, 2 integration, 4 detail and 5 judge. All 28 return envelopes;
27 end in `stop` and the single default SQL detail ends in empty `length`.
Transport failures, cancellations and unknown-usage calls are zero. Full observed
usage is **33,296 input / 11,735 output tokens**.

| Runtime profile | HTTP attempts | Input tokens | Output tokens | Handler ms |
| --- | ---: | ---: | ---: | ---: |
| Provider default | 12 | 15,066 | 6,123 | 61,692 |
| Low | 11 | 12,021 | 4,071 | 42,451 |

Judges account for the remaining 6,209 input / 1,541 output tokens. Different
realized routing/answers and one unreplicated sample preclude a stable efficiency
claim. Durations are descriptive. The left allocation is a ceiling reservation,
not all-provider billing; existing actual HTTP/usage accounting remains separate.

The first fresh replay makes **zero provider calls**, triggers zero unexpected
blocked HTTP attempts and matches every exact request/event and canonical row,
grade, usage and comparison. Its canonical digest is
`4d77a573047975ced373d47f0ad2031e59f05e41724ec3fcdf32b99ce2f213b9`;
transport-event digest is
`7b4b98843a075d9e9c31ed6041495bbe9bcf6829ded10007f4c02a13c18ac485`.
This fresh live tape has no cancellation; cancellation coverage comes from the
separate historical/current runtime validations and synthetic controls, not from
claiming that this no-cancellation trace exercised every asyncio schedule.

At frozen source, **2,086 local Python tests pass**, with isolated PostgreSQL,
one intentional local-HTTP deselection and zero blocked provider attempts.
Lifecycle/journal focus passes49, owner focus12, fresh-profile/replay focus32;
Node13 and gateway/local HTTP18 pass, with two gateway platform skips. Minimum
Python 3.10 CI includes the lightweight drain ownership/callback checks. Final
CI/review evidence is attached to PR #105 at its exact head.

Raw runtime/judge messages and original/partial journals stay in ignored
`samples/task-goals-*`; synthetic questions/references are tracked in the fixture.
Filtered metrics retain all rows, verdicts, failed-validation provenance and
artifact hashes. #103's handoff is fixed and the fresh measurement is recorded;
provider default remains unchanged. The concrete next capability target is
numeric consistency before trusting an ordinary critic `ok`, not automatic low
promotion based on faster output or successful trace fields. Broad calibration
remains #89 and critic-specific availability remains #87.
