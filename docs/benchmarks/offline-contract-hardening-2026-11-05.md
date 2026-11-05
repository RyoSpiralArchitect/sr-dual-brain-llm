# Offline contract hardening — 2026-11-05

The five followups from the offline audit are implemented on top of PR #90:
critic acceptance, System2 execution accounting, explicit missing-evidence
conditions, outbox target binding, and shared offline HTTP enforcement. The
implementation and fixtures were frozen at
`618ddcd9ad96e9c3e140ae9e7b2b1dfc4f78297f` before the recorded development runs.
Its Git tree is `687f9c9d910f947680a4691e3411f9a9ae4d1285`; the combined
core/scripts SHA-256 is
`1e2b7350b2643052c0b2e1ea3e664d8136855abc84ea19e8658713df1f792455`.

[Safe metrics](offline-contract-hardening-2026-11-05.metrics.json) retain the
individual outcomes, metadata, usage receipts, original artifact hashes and
exact replay checks. All provider responses in these runs are authored Fake
responses. No real model request was made.

## 1. Critic completion and review availability

Both the initial critic and the existing single text-repair call now require a
nonempty, successful terminal completion and a valid review shape. The review
must contain an allowed verdict and arrays of nonempty issue/fix strings;
duplicate fields, scalar issues, missing fields and contradictory verdicts are
rejected before normalization can erase their defects. Existing tolerated JSON
formatting and the single unstructured-text repair remain available.

Strict completion paths also check provider envelopes. This closes cases where
a Hugging Face object was stringified into an apparently valid Python review
dictionary, or an Anthropic/Google tool block could masquerade as text. Ordinary
permissive completions keep their previous behavior. No retry policy or timeout
was widened.

The result exposes initial/repair validation, effective review provenance,
`critic_review_valid` and `critic_external_review_valid`. Local micro review and
external review remain distinguishable. A completed answer can remain available
when its review fails. Conversely, returned local fallback text does not prove
that its own provider generation completed. Review validity describes the review
contract; it does not verify the truth of a remembered claim.

The [prospective critic protocol](critic-failure-development-protocol.md) fixes
18 controls across initial review, repair, verification, integration and caller
cancellation. The current complete Cedar source, question, clock, per-case seed,
module settings and 128-attempt ceiling were defined before execution.

| Recorded result | Observation |
| --- | --- |
| Initial critic: empty, length, filtered, missing finish, missing fields, scalar issue, transport failure | Completed approved-record draft retained; external review invalid |
| Repair: empty, filtered, invalid schema | Draft retained; failed repair status retained |
| Verification: empty, invalid schema | Integrated answer retained; latest external review invalid, earlier review still visible |
| Integration transport failure | Local fallback returned; final text has no confirmed provider completion |
| Cancellation during initial review or integration | No abandoned answer/trace/source publication; consultation drained; next turn succeeds in the same session |
| Clean, successful repair and missing-usage controls | Valid review retained independently of receipt completeness |

The run used **71/128 scripted attempts**, including recovery turns, with no
unexpected fixture outcomes. Five receipts have unknown usage: two transport
failures, two cancellations and one deliberately missing usage record. The known
subtotal is 660 input and 330 output tokens; complete totals remain `null`.
These token counts are synthetic receipts, not measured provider costs.

Exact replay consumed all 71 occurrences, performed **zero new Fake transport
calls**, reproduced the raw journal, and matched the canonical report. The
canonical critic comparison excludes only the count of newly invoked Fake
transport calls.

## 2. Routing, actual execution and cancellation

The controller publishes its initial decision and a final routing decision after
ACC, Director and selective-consultation adjustments. ACC admission is therefore
visible in final metrics. A Director skip clears AUTO admission and its round
target; explicit System2 ON retains its authority. Suppression keeps the existing
inhibition of an extra post-answer correction.

A separate execution event records each critic request, its source, round,
outcome and review validity. `system2_used` in feedback reflects actual critic
invocation, including a failed attempt. Ordinary detail consultation is not a
critic attempt. The latest attempt determines current review completion, while
earlier outcomes remain inspectable. Custom adapters without validity fields
retain an unknown review state.

Cancellation regressions cover rounds 1, 2 and 3. Exported `rounds` is derived
from distinct attempted rounds whenever execution metadata is present, so a
late cancellation cannot leave a misleading zero. Callosum plus direct fallback
in one round counts as one round. Historical events without execution metadata
retain their prior projection. Tests also assert child draining, no post-cancel
fallback, and no persistence or hippocampal/Basal feedback for abandoned turns.

## 3. Explicit missing evidence and caller identity

Issue #89 now has an enum-only `missing_evidence_v1` packet carried through
memory, prefrontal context, drafting, review and existing integration calls.
Identity-only caller declarations can resolve the current turn without becoming
factual sources or approval cache entries.

| Available facts in a bounded request | Carried action |
| --- | --- |
| Resolved identity, explicit data-retention period or retention rate, no current source | Request a current source for that parameter |
| Resolved identity, all current sources explicitly denied for the exact request | Request a current source for that parameter |
| Resolved identity, bare retention | Clarify which measure is intended |
| Unknown or conflicting identity | Clarify the reference |
| Literal request with several current unapproved records | Clarify the intended complete record |
| Current records with unknown approval | Preserve the unknown state instead of asserting that saved content is absent |

Recognition uses bounded English/Japanese question forms and the existing
literal grammar. Parameter identity comes from explicit wording. Historical
values and units do not supply a missing parameter. The packet contains no
project/source identifiers, historical text, units or values; currentness,
completeness, scope and exact source approval remain separate checks.

Independent review found two interactions and both now have regressions:

- Negative support can be aggregated only when **every resident current primary
  record** has an exact negative decision. Ineligible records and a seventh
  record outside the six-candidate cap keep the aggregate unknown. The check
  reads existing decisions without adding a support call.
- Live aliases are rechecked against the caller declaration after awaits. A new
  alias that would redirect a Cedar-declared pronoun to Birch makes the reference
  ambiguous. Candidate selection and retrieval withhold that conflicting scope,
  including an older positive approval and a disabled support-gate path. Expiry
  or supersession of the conflicting alias restores the surviving declaration.

The prototype establishes deterministic branch and context propagation. Fresh
natural answer/clarification quality and model-judge calibration remain a later
measured scope for #89; previous ambiguous-retention grades are unchanged.

## 4. Outbox database target binding

Explicit nonempty DSN `options`, including `search_path`, now participate in the
outbox namespace. Absent/empty options preserve the old hash, and password
rotation leaves identity unchanged. Binding uses the exact parsed option string;
it does not guess equivalent routing from environment or server defaults.

An older namespace cannot identify which schema its entries originally targeted.
When explicit options are present, pending **and acknowledged** legacy rows for
the same session block ambiguous recovery before a backend call. Rows are
retained. Reconciliation installs this guard even on an outbox made through the
original two-argument constructor; subsequent access remains guarded. Namespace
conflicts also fail before backend access. Guarded SQLite writes use the same
transaction for the legacy check and mutation.

The real PostgreSQL fixture uses two disposable schemas, the same session and
operation IDs, and one outbox file. It checks target isolation, recovery in the
original target, password-independent identity and both legacy receipt states.
See [recovery contract](../memory-commit-recovery.md) for upgrade behavior.

## 5. Offline HTTP enforcement and reproducible evidence

Both development runners use the same opt-in HTTP guard. It blocks production
JSON and SSE requests at `aiohttp.ClientSession._request`, plus standard-library
HTTP fallback entries. A blocked attempt remains fatal on guard exit even if
application fallback catches its immediate exception. Task draining occurs inside
the guarded lifetime. The guard is process-wide and used serially within each
experiment process; intentional local Gateway/HTTP integration tests retain
their own service setup.

Frozen critic receipts validate their envelope, error category and response
checksum before execution. A request mismatch or unsupported recording exception
cannot yield a complete journal after a caller catches it. Atomic partial
checkpoints retain prior cases and the failing receipt for inspection.

The existing eleven-case record-replay development comparison was rerun on this
same code freeze: **66 support preparations, 22 runtime sessions, 22 V3 packets,
111/192 scripted attempts**, and no unexpected fixture outcomes. Exact replay
made zero new Fake transport calls, consumed the same journal and matched its
canonical report. Both suites and both replays recorded **zero real provider
calls and zero unexpected blocked HTTP attempts**.

The development environment was Python 3.12.14, NumPy 2.3.5, without Torch.
Code and fixture hashes bind each replay; the critic journal additionally records
dependency versions. The existing record-replay canonical comparison retains its
documented removal of diagnostic timings. The V3 contract, previous fixtures,
#85/#88 reports, legacy model grades and historical availability rule are unchanged
relative to baseline `a19b861407bd94e0b33fbd81269476e4069d5b00`.

## Validation

The full local Python run was wrapped in `forbid_provider_network`. Intentional
local HTTP/Gateway tests ran separately because their local service traffic
would correctly be rejected by that process-wide guard.

| Validation on the frozen implementation | Result |
| --- | --- |
| Local full Python suite under HTTP guard, excluding Gateway and one local HTTP-server test | 1,668 passed; 17 environment skips; 1 deliberately deselected; zero unexpected blocked requests |
| Separately run local HTTP-server test | 1 passed |
| Local Gateway checks | 2 passed; 17 skipped because .NET is unavailable locally |
| Node trace/history UI | 13 passed |
| CI isolated PostgreSQL 16 job | 16 passed, including schema/target isolation |
| CI Gateway job with .NET | 17 passed; 2 expected skips |
| CI minimum Python 3.10 consultation lifetime | 43 passed; 3 deliberately deselected |

The 17 skips in the local full Python run are the 16 database cases covered by the
separate CI service and one optional Torch case. Configured integration outcomes
above come from the [frozen-code CI run](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/actions/runs/41974198483).
Independent reviews covered critic/provider envelopes, routing/cancellation,
missing-evidence identity/support, outbox target handling and journal/HTTP
boundaries. Each concrete finding was repaired with a regression. Final diff
checks pass.

## Reproduce the recorded controls

Choose new output directories; the commands refuse to overwrite an existing run.

```sh
python sr-dual-brain-llm/scripts/benchmark_critic_failures.py \
  --output-dir sr-dual-brain-llm/samples/module-gating-critic-hardening-2026-11-05
python sr-dual-brain-llm/scripts/benchmark_critic_failures.py \
  --journal sr-dual-brain-llm/samples/module-gating-critic-hardening-2026-11-05/journal.json \
  --output-dir sr-dual-brain-llm/samples/module-gating-critic-hardening-2026-11-05-replay
python sr-dual-brain-llm/scripts/benchmark_record_replay_development.py \
  --output-dir sr-dual-brain-llm/samples/module-gating-replay-hardening-2026-11-05
python sr-dual-brain-llm/scripts/benchmark_record_replay_development.py \
  --journal sr-dual-brain-llm/samples/module-gating-replay-hardening-2026-11-05/journal.json \
  --output-dir sr-dual-brain-llm/samples/module-gating-replay-hardening-2026-11-05-replay
```

Raw journals remain ignored development artifacts. The linked safe metrics retain
the complete public observations without publishing raw request/response payloads.
