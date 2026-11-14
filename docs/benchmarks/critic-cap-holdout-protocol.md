# Critic-cap recovery: untouched eight-case holdout

The [development screen](critic-cap-development-2026-11-14.md) completed
20/20 cells, passed its precommitted candidate gate, and replayed exactly.
This protocol freezes a **separate, previously unused** eight-case holdout.
It tests the selected `low_critic4096` against the primary `default_off`
baseline and the `low_aware` cap control. The development result guided
candidate selection; these holdout questions were generated only after that
result and have no overlap with any preceding fixture. The wrapper verifies
byte hashes, canonical equality, zero replay network attempts and the exact
development gate before live admission.

Freeze this protocol, source, fixture, unchanged exact graders and manifest
before provider output. Use two deterministic fresh seeds per domain:
Boolean entailment at level 4, SQL bag semantics at level 7, event replay
at level 6, and scoped source selection at level 6. The authored fixture is
public, with references evaluator-only. Preserve the eight cyclic arm
orders, matched session IDs/RNG/logical clocks and independent provider
samples. No regeneration, replacement, scoring change or prompt tuning in
this holdout.

All arms use explicit left lead, manual System2 ON, task goals AUTO,
completion replanning AUTO, balanced AUTO gates, Director/executive OFF,
work products OFF and no ordinary continuation. Draft/integration caps are
4096, ordinary-left allowance8192 and all-provider per-trial output
ceiling12288. The requested model is `gpt-6-luna`; native client timeout is
120s. Only draft/integration requests use explicit low effort; critic
requests retain provider-default effort and the same 24s review window.

1. `default_off`: provider-default ordinary effort, completion-aware critic
   OFF, critic cap2048. Primary baseline.
2. `low_aware`: low ordinary effort, completion-aware critic ON, critic
   cap2048. Control for the cap change.
3. `low_critic4096`: low ordinary effort, completion-aware critic ON, critic
   cap4096. Selected candidate.

Require `SSL_CERT_FILE=certifi.where()` and at least 100 certificates in the
default Python SSL context before live admission. Assert serialized effort
and cap at every admission. At most 80 conservative admissions across 24
cells; reserve full caps without refund for failures or cancellations.
Stop future paid work on genuine or unclassified provider, transport, auth
or quota failure. No paid judge, health call, retry, rescue, model switch or
outcome-driven revision. Missing usage is unknown. If fewer than 24 cells
finish, do not rank all arms.

Use the unchanged exhaustive Boolean, SQLite/Python, event-state and
source-scope exact graders. Preserve duplicate-key, whole-JSON, schema and
SQL bag rules. Report all cells, unavailable outcomes, initial/final
completion, exact and mean item accuracy, paired deltas, workflow validity,
external review validity, cap/effort receipts, admissions, reservations,
cancellations and missing usage. A complete answer retains its semantic
grade even if review failed.

The **precommitted held-out signal gate** requires all 24 cells, at least
two more exact cases for `low_critic4096` than `default_off`, strictly
higher mean item accuracy, no lower item score on any SQL/events/evidence
case, no fewer workflow-valid cases or external reviews than `low_aware`,
and no configuration fault. If any condition fails, record `winner: none`.
Even a passed gate is a small, independent-sample signal, not statistical
superiority or proof that the critic cap caused the exact gain. Report the
`low_aware` comparison separately to isolate what the cap may contribute.
Do not change the product default from this one holdout.

Run one exact provider-network-forbidden replay on the frozen source. Preserve
recorded durations/admission order and actual owner cancellation. Require
full request/event coverage, canonical rows, grades, usage and reservations.
Preserve failed replay evidence without tape fitting. Publish filtered
metrics, denominators and hashes; keep raw provider payloads/outputs private.
