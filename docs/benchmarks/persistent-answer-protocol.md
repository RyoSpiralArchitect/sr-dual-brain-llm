# Persistent final-answer protocol — 2026-10-28 (Asia/Tokyo)

Freeze fixture, runtime configuration, rubric and runner before paid calls. Use
one development episode before/after repairs, then two untouched evaluation
episodes once. Repairs may use development answers only. Preserve incomplete
runs and invalid grading; do not regenerate favorable evaluation answers.

Use a real isolated PostgreSQL DB and EngineSession/_handle_process, with complete
source generation, outbox/receipts, explicit supersession, 1,030 authored derived
QAs and fresh spawned Python workers. Each arm runs four probes, exits, then
recreates the engine in another process for its final conditional probe. Provider
configuration crosses only the private multiprocessing pipe, never command-line
arguments or artifacts. Journal/budget state transfers between sequential workers. Three real left-only source turns per episode produce
one frozen source snapshot, cloned identically to both arms. Baseline development
sources are reused as data for repaired development, not regenerated. Source
metadata/aliases/correction IDs are caller supplied; this is not automatic fact
extraction. Author-provided reference values enter grading only.

Arms: left-centered (left lead, System2 off, consult policy skip plus a routing clamp to prevent salience overrides) and collaborative
(left lead, System2 on with the production round policy and real right worker).
Both instantiate normal EngineSession brain components. Optional executive and
director models are off, and responses are nonstreaming for complete transport
journals. This is a bounded collaboration configuration, not every optional LLM
or the default auto-mode. No benchmark-specific answer instruction is appended to
production prompts. The support checker uses the configured generation model.

Five probes cover corrected conditional facts, unknown fees for an explicitly
named project, a two-project comparison, the subsequent ambiguous reference and
an explicit conditional query after a second process starts. Grade task,
grounding and clarification independently with arm-blind shuffled labels using a
separate gpt-6-sol judge (generation/support gpt-6-luna). Required JSON schema and
literal answer quotes are validated atomically; an invalid packet remains
ungraded with its usage retained. No partial-score salvage or automatic retry.
This is authored, correlated, same-provider-family model grading, not human
assessment or a generalization estimate. Unknown facts and ambiguous references
have distinct criteria.

Hard provider allocations: 64 baseline-development + 64 repaired-development +
112 evaluation = 240 calls total. Each command enforces its allocation. Source/left/right/judge requests use at most
2,048 completion tokens; the production support backend requests 4,096. No
continuations, automatic retries or embeddings.
All provider calls, failures, requested/served models and token usage are recorded
by role. Missing usage stays unknown. Abort generation failures before later turns
consume a heuristic fallback; unavailable/partial runs are not successful answers.

Raw answers, snapshots and HTTP journals are private ignored artifacts. Publish
aggregate/per-probe grades, source/fixture/policy/request hashes and exact code
revisions. Probe QA is timestamped deterministically in the benchmark process to
support reproducible transport requests. Wall-clock, UUID and random sources are
controlled only inside that process; latency still uses real monotonic time.
Report whether complete request replay actually succeeds; do not claim exact
replay merely because saved answers can be regraded. Latency is descriptive with
fixed arm order and a small number of episodes, not a controlled service speedup.

Development and evaluation commands will be recorded in the result report with
exact output directories and source revisions. Read-only provider configuration
is reused; no key or DSN value enters public artifacts. New disposable table
prefixes are dropped in cleanup; the application DB is never used.

Technical development amendment (before evaluation): the first repaired-development
attempt stopped before its first answer after one successful support call and one
provider POST failure. Its partial journal/failure artifacts remain intact; the
original implementation recorded the error type but not its HTTP status. A
read-only model-list check returned HTTP 200 with both configured models present.
A separate explicit development rerun may use at most 62 calls, preserving the
original 64-call allocation. Status-only error/cancellation telemetry is added;
no prompts, rubric, fixtures or output limits change. This is not an automatic
retry or favorable answer resampling. Missing failed-call token usage stays unknown.

Further technical record before evaluation: two additional repaired-development
attempts also failed before their first answer; status telemetry identified HTTP
429. Three isolated support diagnostics include one success and two HTTP 429
responses, with provider error type `insufficient_quota` under both normal and
frozen Python clocks. Paid calls are stopped pending usable quota. The three
failed development attempts used two calls each, and diagnostics used three:
9 of the repaired-development 64-call allocation are consumed, leaving at most
55. No completed repaired answer has been sampled or discarded. Retain all
failures and unknown usage; do not infer zero tokens for rejected requests.

Cap disclosure: the initial protocol incorrectly described 2,048 as a universal
cap. Inspection of recorded payloads shows the unchanged production support
backend always requested 4,096 (both initial and repaired attempts); other roles
requested 2,048. No payload cap was changed to obtain a better result. New metadata
explicitly reports the support cap; historical exact request journals retain their
original fields. This deviation and all actual request caps must appear in the
result report rather than retrofitting historical artifacts.

Review hardening: reject nonempty DUALBRAIN_MEMORY_EMBEDDING_MODEL and
DUALBRAIN_MEMORY_ANSWERABILITY_MODEL before creating any session, in both live and
frozen modes. The benchmark creates its own journaled support checker and permits
no environment-selected embedding/support client. Parent/worker API checks prevent
untracked provider calls and retrieval-configuration drift. These selectors are
disabled in the observed environment; the guard is not a prompt/rubric change.
