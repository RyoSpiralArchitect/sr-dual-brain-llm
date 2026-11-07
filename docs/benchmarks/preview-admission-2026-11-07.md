# Speculative preview admission and scheduled replay — 2026-11-07

The fixed natural trajectory completes **12/12 turns**, with three actual
model-generated primary writes, correction/restart and two byte-exact quotations
of the actual current sources. One pending-record-choice preview is suppressed
before task/HTTP admission. The first frozen CLI replay consumes **32/32 exact
ordered requests and 64/64 transport events**, with zero provider calls, zero
unexpected blocked HTTP attempts and identical canonical outcomes. That first
predeclared replay needs no supplemental diagnostic or paid repeat. A subsequent
synthetic timeout regression and offline-only robustness correction are reported
separately below.

Implementation, fresh fixture and [prospective protocol](preview-admission-protocol.md)
were frozen at `6123b0d9c945291899423f4bb0cb9a65f3570f53`, on merged #97
(`092421ab0032b16034465a64f59a7cd2fd8369c2`). See the
[safe metrics](preview-admission-2026-11-07.metrics.json) and
[fixture](../../sr-dual-brain-llm/examples/preview_admission_trajectory.json).
#97's original replay failure, supplemental diagnostic and scores remain intact;
this freshly instantiated acceptance structure is not an independent capability
benchmark or a repaired score for the earlier trace.

## What the runtime change does

The default-OFF TRN module `speculative_preview_suppression` acts before drafting.
AUTO inhibits optional speculative work only for a locally typed pending whole-
record selection/reselection. It reads no draft or eventual policy action and
makes no model or evidence-check request. Source writes, explicit right lead,
manual System2 ON/OFF, precision priority and Director consultation overrides
retain their authority. Other unknown/missing-evidence conditions retain preview
admission. Later normal routing and consultation remain separate.

The owned-preview helper cancels and drains its child before the stage returns,
including outside RPC and during abandonment. `metrics.preview` distinguishes
not requested, suppressed, cancelled before coroutine start, cancelled after
start, completed, empty and failed work. Starting a model coroutine does not
establish HTTP admission, external completion or billed usage.

## Observed natural trajectory

Normal RightBrainPolicy runs under the opt-in all-AUTO balanced profile, with
Director/executive work OFF. No action is forced and no draft or primary seed is
substituted. This is the production nonstream JSON path with a 2,048-token
generation ceiling and the existing support/stage budgets.

| Turn | Observed preview | Runtime HTTP attempts | Handler ms |
| --- | --- | ---: | ---: |
| Initial period write | Not requested | 2 | 3,324 |
| Backup write | Completed; primary-write guard preserves admission | 4 | 4,846 |
| Ambiguous whole-record request | Not requested | 1 | 1,144 |
| Listed original period quoted | Completed | 4 | 3,479 |
| Period correction | Not requested | 5 | 12,392 |
| Stale selection reused | **Suppressed before task creation** | 2 | 1,903 |
| Reference reused after restart | Not requested | 1 | 1,027 |
| Relisted corrected period quoted | Cancelled after start | 4 | 2,947 |
| Explicit current period | Not requested | 3 | 5,342 |
| Bare retention in another scope | Not requested | 2 | 3,693 |
| Explicit Japanese retention rate | Completed; unknown support does not justify inhibition | 2 | 1,698 |
| Explicit period in empty scope | Not requested | 2 | 3,867 |

The stale-selection turn has a requested auto-rotated right preview. Its pre-draft
gate returns `record_choice_pending`, creates no preview task and makes no
preview HTTP attempt. Later policy action remains 1 and ordinary detail still
runs; the gate did not pretend the final action was already known. The Japanese
rate turn ends with action 0 but its pre-draft unknown-support condition retains
the preview. This illustrates the deliberately narrow authority of this change.

All twelve operations retain matching committed receipts and consistent System2
execution accounting. Correction leaves only the backup and corrected period as
active primaries, including after restart. Both quotation checks match the actual
source bytes; read answers do not become primary evidence. Listing makes no
provider request, and references stay outside provider packets.

V3 source validity passes for the two quotations and current-period lookup and
retains fail for six source-free clarifications. All semantic fields and the
study's twelve grade entries remain **not assessed**. Unblinded inspection by
the implementing assistant is not an independent human grade, calibrated model
score or evidence of a general answer-quality improvement. Broader calibration
remains in #89.

## Actual allocation and replay

There are **32/48 actual HTTP attempts**: 12 draft, 6 critic, 5 support, 4 right
preview, 4 detail and 1 integration. Nested source-write generation uses the
draft label. All 31 returned envelopes end in `stop`; one started preview is
cancelled without response/usage. Three previews complete, one is cancelled,
one is inhibited and seven turns request no preview. Cancellation before first
coroutine execution is covered by Fake controls, not observed in this live run.

Known subtotals are **22,439 input / 2,429 output tokens**. Full totals remain
**unknown**, because the cancelled relisted-quotation preview can have provider
cost without a receipt. No continuation, paid rescue, resampling, semantic judge
or extra paid diagnostic occurs. Episode time is **46,790 ms**, including owned
setup/load, handlers, listing and restart, excluding teardown and user think time.

These are descriptive observations from one unpaired trace. One inhibited
admission is not a measured counterfactual saving, and the totals must not be
compared with #97 as a quality, request or latency gain. The paired offline
synthetic-delay control observes one-versus-zero Fake preview HTTP attempts with
the same left draft; it establishes admission mechanics only.

The prospectively specified scheduled journal records each admission and its
response/error/actual-cancellation terminal in a hashed event tape. Replay waits
for that order with strict request identity and coverage. It does not inject a
recorded cancellation: the real owning controller must cancel the pending work.
Latency-only feedback restoration verifies every non-timing input. No action,
reward, success flag or replacement answer is supplied as an oracle.

The first ordinary replay CLI passes with canonical hash:

`7b20c3b54b8710de88f0682c32ebc05ee52a1fa4fd829c4c52a851f98c510ab0`

Transport-event digest:

`2a47c1af1fde2aba4e1dd74b8b7166f0d14fd52686b1bff8c387834dbdd7e5f9`

All raw envelopes, requests, contextual messages, outboxes and full journals stay
private under ignored `samples/module-gating-preview-admission-2026-11-07-{live,replay}`.
Safe metrics publish filtered observations and artifact hashes. The result
establishes replay for this recorded transport ordering, not wall-time replay,
all possible asyncio schedules or success of #97's original immediate replayer.

## Subsequent offline timeout correction

After the successful measured run and first replay, implementing-assistant review
reproduced a separate fault: the replay wait's local `asyncio.wait_for` timeout
could spend an RPC `_TurnTask`'s cancel-once guard, preventing later cancellation.
This does not explain or change the measured successful trajectory. The replay
wait now observes and drains a separate local waiter with `asyncio.wait`, keeping
the actual caller cancellable. Two real-dispatcher regressions cover later cancel
and deadline, and a 24-case delayed/replay/ownership focus passes with zero
unexpected provider requests. This is a post-study replay fault-path fix; the
production core, live transport path, fixture, criteria and original artifacts
remain unchanged. No additional paid run is made.

The fix is pinned at `3d1afe899ae3c00877914ad83d7653afe547613c`.
A separately declared **post-study offline validation** also consumes all 32
requests / 64 events and matches the original canonical hash with zero provider
calls / zero blocked attempts. Its derived input changes only
`metadata.code_sha256` to the new code digest; every raw occurrence, envelope,
transport event and feedback input is unchanged. Original journals are never
overwritten, and their artifact hashes are checked again. This explicit metadata
bridge validates the subsequent fix, not the original prospective protocol or a
new paid measurement. Original replay uses the measured revision; current source
continues to require its own exact code digest. Safe metrics retain both
provenances and all artifact hashes separately.

## Verification and return to the intelligence work

The measured frozen implementation passes **1,964 Python tests** with isolated real
PostgreSQL, one intentional local-HTTP deselection and zero unexpected blocked
provider requests. The delayed/persistence focus passes 60; gateway/local HTTP
passes 18 with two platform skips; Node passes 13. Python 3.10 CI includes the
lightweight ownership controls. Delayed full-store Fake controls replay successful
and cancelled previews, failed writes/skipped dependencies and every exact
request after JSON round-trip. Schedule tampering, missing owner cancellation,
explicit protected modes and unavailable local facts fail without inventing
permission to inhibit work.

This bounds #98. The next architectural target returns to **task goals and
orchestration**. Source inspection shows that `Orchestrator` currently tracks
round-trip counts and a leading role; Director controls steer existing stages
without an explicit set of subgoals and unresolved obligations. A useful next
prototype should carry user-grounded goals through left reasoning, right
alternatives, critic checks and integration, with bounded work per goal and
explicit unresolved items. That is a subsequent capability change, not a result
or new behavior claimed by this operational study.
