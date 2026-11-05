# Whole-record selection after the live operational pilot

## Baseline and question

PR #92 records a live failure at baseline
`53498ef57e6ed41ec5f885154391b6e66e0ae16b`: an unspecific bounded whole-record
request has two current primary records in one scope. Ordinary support approves
both, the local unique replay binding is absent, and generation receives the two
records without the required record-selection clarification.

This followup asks whether the recognized operation can survive an unavailable
binding through retrieval, clarification and generation. The #92 protocol,
fixture, report, metrics and historical response journal remain unchanged.

## Prospective behavioral contract

Recognition uses the existing bounded literal grammar. A request to replay a
complete record and a binding to one specific record are separate facts. The
binding is recomputed from all current primary records in the resolved scope,
before relevance ranking, candidate limits or support approval. A binding never
constitutes approval; exact complete-source replay approval remains necessary
for delivery.

| Fixed control | Required behavior |
| --- | --- |
| Two current records, both ordinary decisions positive | Preserve the replay operation; clarify which record; expose neither body as selected evidence |
| Two current records, only one positive and the other negative/unknown | Preserve the same ambiguity; approval does not identify the user's intended target |
| A second current record omitted by eligibility/ranking/candidate limits | Preserve ambiguity across the full current primary set |
| Two records become one through expiry, supersession or derived-record status | Recompute the binding; require its exact replay approval before exposing the survivor |
| One previously approved record becomes two | Withhold the prior binding and source text; carry record clarification |
| The last current record/alias disappears after context admission | Retain this turn's recognized operation, re-resolve identity and remove stale evidence at the next existing generation boundary |
| Ordinary semantic or compound questions over the same records | Retain their existing approved multi-source behavior |
| Missing checker for a bounded replay | Preserve approval uncertainty; do not substitute ordinary unchecked source delivery |
| Non-conversation legacy retrieval | Retain its existing path |

An ambiguous, recognized replay carries `clarify_record` even when ordinary
approved evidence exists. `current_unselected` describes this condition without
claiming that the sources are missing or have all been denied. No historical
values or source bodies are placed in the clarification packet.

The controller must carry the condition into the initial draft and each existing
consultation/revision path. A previously assembled context, draft or reviewer
note cannot itself establish a new target binding. Module modes, authority and
model-call budgets retain their existing policy.

The last-source lifecycle control was added after independent review exposed
stale admitted context when a named request lost its final alias. It is reported
separately from the fixed 23-case before/after matrix. Only current-turn request
recognition is retained; scope, source identity and approvals are recomputed.
The same review also exposed a copied right-worker packet becoming stale during
queue/admission waits. Separate consumer tests pause at that boundary, mutate the
source set, and resume the existing call. The consumer must use the same current
binding/evidence check as the controller. These checks govern inputs to the next
existing generation; no new call is scheduled to rewrite an already completed
draft when the turn has no remaining generation stage.

## Comparison and verification

An independently authored, self-contained black-box regression file is run
against both the unchanged #92 production code and the patched production code.
The same test bytes, synthetic sources, approval patterns and expected behavior
are used in both arms. JUnit observations retain every passing/failing case;
baseline failures are not removed to improve the comparison.

Separate Fake Provider engine tests inspect the actual generation inputs and
the resulting clarification, including existing consultation/integration paths.
Lifecycle tests cover source changes after approval. Controls include bounded
English/Japanese forms, caller-scoped/pronominal references, normal semantic
questions and ordinary approved-source cache entries.

All new checks use local Fake responses under the shared offline HTTP guard.
The real-model call budget for this followup is **zero**. No model judge, live
rerun, new retry policy or classifier is introduced. The original 33-call
operational observation remains attached to its original code and inputs.

Full memory, source-approval, controller and offline experiment regressions run
after integration. Existing configured CI also covers isolated PostgreSQL,
Gateway, the minimum Python consultation lifecycle and trace/history UI.
