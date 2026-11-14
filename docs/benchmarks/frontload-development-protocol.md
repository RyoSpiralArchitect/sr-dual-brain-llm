# Frontloaded generation: prospective development comparison

## Why this experiment

The completed 2026-11-13 study found that every Boolean draft and integration
used its full 4096 (or 3072) output cap as hidden reasoning tokens and returned
zero visible text. The one usable 1024-cap work product did not recover an
answer. A previous fresh global low-effort comparison produced a numeric
error, so this study keeps provider-default reasoning effort. The hypothesis
is that allocating the same 8192 ordinary-left ceiling to the first native
draft, instead of two 4096 attempts, can produce a terminal answer on fresh
deep tasks. The critic-aware arm tests whether withholding review of an
incomplete long draft avoids spending on a candidate that cannot be graded.
This is a development screen, not a default-change qualification.

## Freeze and conditions

Commit this protocol, source, fixture, exact graders and manifest before any
provider output. Five newly seeded, never-sent closed-world cases cover two
Boolean entailment systems at level 4 and one each of SQL bag semantics at
level 7, event replay at level 6 and scoped source selection at level 6.
Expected answers are evaluator-only and never enter requests. The fixture is
public; raw provider payloads and outputs remain in ignored private samples.
Lock the five case objects, four arms, five arm orders, scoring, all caps and
stop rules. No output-based task replacement, resampling or tuning in this
campaign.

1. `native_4096`: native manual System2 ON, completion replanning AUTO,
   work product OFF, completion-aware critic OFF. Draft/integration cap4096.
   This is the primary baseline.
2. `frontload_8192`: identical native policy, with the first draft cap8192
   and ordinary-left total8192. Integration can receive no remaining ordinary
   allowance; do not borrow from critic or silently enlarge the total.
3. `frontload_8192_aware`: the same frontloaded allocation with
   completion-aware critic ON.
4. `self_review`: existing four-stage reference using caps4096/2048/4096/2048.
   It is a contextual benchmark, not a one-variable comparison with native.

All native arms use the same goals AUTO, balanced AUTO gates, explicit left
lead, Director/executive OFF and independent fresh sessions with matching IDs,
RNG seeds and logical clocks. Requested model `gpt-6-luna`, provider-default
effort, no ordinary continuation. Require `SSL_CERT_FILE=certifi.where()` and
at least 100 certificates in Python's default SSL context before any live
admission; no provider health request. Native client timeout120s across arms;
self-review's existing stage timeout60s. Timeouts, actual compute, prompts and
request counts are not matched. Assert actual serialized request caps before
admission. Each trial offers the same 12288 all-provider output-ceiling
allowance; native ordinary-left total is8192. At most 80 admissions for 20
cells. Never refund reservations for failure, cancellation, malformed output or
pre-HTTP denial. Stop future paid work on unclassified transport/auth/quota/
provider errors or configuration mismatch. No health call, paid judge, retry,
rescue, or outcome-driven follow-up in this frozen campaign.

## Scoring, decision and replay

Use the unchanged exhaustive Boolean, evaluator-authored SQLite/Python,
event-state and scoped-source oracles. Preserve strict whole JSON, duplicate
key, schema and SQL bag rules. Retain every incomplete or failed cell. Grade
terminally complete returned answers even if critic review fails; report
workflow validity separately. Report exact counts, mean item accuracy, initial
and final completion, semantic repair/regression, actual serialized caps,
reasoning-token and visible-output receipts, critic availability, reservations,
usage missingness, and all case pairs. If fewer than 20 cells finish, do not
rank the full comparison.

Advance at most one frontloaded candidate to a separately frozen untouched
holdout only if, on all five development cases, it gains at least one exact
answer over `native_4096`, has no lower per-case item score on the three
non-logic cases, and has no lower overall mean item score. If both pass, choose
the one with fewer admissions, then fewer unknown-usage receipts, with the
`frontload_8192` arm winning any remaining tie. If neither passes, report a
negative development result and investigate another mechanism on new tasks;
do not tune these five outputs or call them confirmation evidence. Even a pass
does not justify promotion without untouched holdout evidence.

Replay exactly once at the unchanged frozen source with provider HTTP blocked,
recorded durations and ordered admissions/terminals. Owner cancellation must
arise from the real owner. Require request/event coverage, canonical rows,
grades, usage, reservations and per-arm diagnostics to match. Retain replay
failure without changing timers or tape. Publish filtered metrics and private
artifact hashes. Independent initial provider samples and five authored cases
do not identify an isolated causal effect or statistical superiority.
