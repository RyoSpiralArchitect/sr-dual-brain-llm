# Prospective clarification calibration and natural memory trajectory — 2026-11-06

Freeze implementation, fixtures, expectations, criteria, ordering, repeat rules
and budgets before paid outputs. Earlier #85–#96 records and scores stay immutable.
This study adds assessment V4 beside V3, not a replacement score for old studies.
Production core bytes remain those of merged #96.

## Fixed calibration

Execute the six fresh packets in `clarification_calibration_v4.json` once, with
two authored contrasting answers per packet. Sort variant names, alternate their
blind A/B order by packet index, and never submit variant names or expectations.
The 24 authored criterion expectations contain 13 pass, 8 fail and 3 uncertain
labels. Report all matches, negative/uncertain agreement, malformed judgments and
disagreements. These are implementing-assistant expectations, not independent
human adjudication or proof of general grader validity.

Cases contrast explicit data-retention period, explicit retention rate, bare
retention without current supporting context, an available relevant record, a
complete requested quotation, and unknown context. Comparison references and
expired/unexposed values remain outside the semantic packet. V4 separately
assesses attribution and information sufficiency, with strict literal quotes for
pass/fail. A missing or malformed context stays unknown. Do not infer that a
stored value is absent merely because its support is unknown. One judge request
per packet, no repair/resampling or selecting a favorable order.

Use `gpt-6-sol` for all V4 judgments. Schema-invalid, empty, nonterminal or failed
judge responses remain failures with unassessed/unknown criteria. Do not tune the
rubric after these results. Preserve raw scores even when calibration disagrees;
their interpretation must retain that qualification. V3 deterministic equality
and source validity, and its unassessed semantic fields, remain separate.

## Natural persistent trajectory

Use real `gpt-6-luna` generation/support, a 2,048-token generation ceiling,
the existing 4,096-token support cap, and unchanged production stage budgets,
timeout/repair rules and completion guards. The V4 judge cap is 2,048 tokens. Keep the
existing all-AUTO module profile with balanced AUTO priority and Director/
executive work OFF. The stdio API defaults to left leading/System2 AUTO.
Use normal `RightBrainPolicy`: no forced consult/skip, no frozen/substituted draft,
no warm-up or authored primary seed injected directly into storage. Existing
session RNG seeds and logical timestamps are prospectively fixed for replay.

The 12 fixed turns in `natural_memory_trajectory.json` execute in order:

1. Model-generated period statement, committed as an unverified primary record.
2. Model-generated backup statement, committed as a second primary.
3. Whole-record request with both current records; ask which record is intended.
4. List candidates, select period, quote its actual complete stored answer.
5. Model-generated correction, atomically superseding the first period record.
6. Fresh operation with the earlier selection; reselect without substitution.
7. Mint a current reference, restart the same PostgreSQL session, submit old ref.
8. Relist after restart and quote the actual corrected current source.
9. Explicit current data-retention-period lookup.
10. Bare retention for a different caller-declared scope without current evidence.
11. Explicit Japanese retention-rate lookup with current records for other facts.
12. Explicit data-retention-period lookup in the empty declared scope.

Quotes compare exact UTF-8 bytes against the named actual current source, never
an authored expected output. In turn 11, both `request_current_source` and
`preserve_unknown` are acceptable trusted-state actions: existing short-question
suppression can leave current-source support unprepared. Unknown is not denial.
Only successful writes may supply later dependencies. Failed writes and skipped
dependent turns remain explicit; never insert substitute fixtures, choose a
different record or retry a favorable answer. Read answers never become primary.
Candidate listing must make zero provider calls.

Capture current retrieval eligibility and finite missing-evidence state before
generation, including turns without asynchronous preparation. Separately match
the complete admitted memory block and condition JSON against actual provider
stage inputs. Only confirmed supplied blocks enter the semantic context; an
unobserved candidate stays unknown. Do not backfill an internal condition that
was never delivered to the model. Retain full actual generation messages as
untrusted context data, including working/draft/reviewer text without promoting
it to current approved evidence. Keep all references out of provider packets.

Grade the nine successful read answers individually with their own actual
context after the trajectory. Source writes use strict completion/commit checks
and are outside the semantic read rubric. Preserve failed/unexecuted turns and
unavailable judgments. Confirmation can be appropriate while V3 source validity
remains fail for no exposed record. Neither score overwrites the other.

Use isolated `DUALBRAIN_TEST_PG_DSN`, a fresh owned namespace and private outbox
under ignored output. Preserve atomic commits, explicit correction and restart
semantics; drop only the study's owned tables. This is a tiny natural-routing
acceptance trajectory, not a before/after quality or savings comparison.

## Cost, timing and exact replay

Hard cap **64 actual provider HTTP attempts total**: six calibration packets,
up to **49 runtime attempts**, and at most nine read judgments. Reserve the 15
judge slots before runtime admission. Count support, generation, detail, critic,
repair, integration, failures and existing fallbacks. No continuation, paid
rescue or extra diagnostics. Transport/auth/quota failure keeps the sticky paid
admission latch closed. Unknown usage retains known subtotals rather than zero.

Report actual per-handler, choice-listing, restart and episode durations without
judge time or user think time, and all provider costs separately. Latency feeds
the normal controller's feedback: observe live calls unchanged, then replay only
the measured `latency_ms` inputs to Hypothalamus/BasalGanglia feedback. Verify all
non-timing feedback inputs and exact order/coverage; never replay an action,
reward or success as an oracle. Hash this control-input tape in the raw journal.

Replay every exact ordered raw envelope under the offline HTTP guard, with no
transport object. Require equal canonical non-timing outcomes, including failed
and unknown receipts, feedback digest, calibration disagreements and semantic
grades. Replay is not a latency benchmark. A real timing race or divergence is
retained without post-output tuning or paid rescue. Raw prompts, responses,
reference signatures, judge quotes and outbox stay private; publish filtered
outcomes, actual allocation and artifact hashes.

```bash
python sr-dual-brain-llm/scripts/benchmark_natural_memory.py --model gpt-6-luna \
  --judge-model gpt-6-sol --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/module-gating-natural-memory-2026-11-06-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_natural_memory.py \
  --journal sr-dual-brain-llm/samples/module-gating-natural-memory-2026-11-06-live/journal.json \
  --judge-model gpt-6-sol --max-calls 64 \
  --output-dir sr-dual-brain-llm/samples/module-gating-natural-memory-2026-11-06-replay
```
