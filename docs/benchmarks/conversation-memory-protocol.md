# Controlled conversation-memory protocol — 2026-10-21

Measure the path from model-generated source turns through stored QA to final
answers, rather than crediting retrieval alone. Four authored Japanese episodes
have three source turns and eight probes each. Two development episodes use
deadlines/parcel weights; two evaluation episodes use retention/concurrency.
Scope names and values differ; dialogue templates repeat. This is a controlled
synthetic pilot, not blind natural conversation or independent human labeling.

Each episode records two projects, a complete long policy with a regional
exception at the end, and a correction replacing an identified earlier turn.
Probes cover the corrected fact, an explicit follow-up, exact repeats, the long
record's exception, an unknown fee, two-project comparison and an ambiguous
follow-up. Caller metadata supplies only scope IDs/aliases, an explicit remember
flag and exact source IDs to supersede. It contains no expected answer values.
This is caller-assisted memory ingestion; it does not test automatic project or
correction extraction from unrestricted natural language.

Compare memory off, the current support checker, and conversation-aware checking.
Use the actual DualBrainController and LeftBrainModel answer path, fixed left
lead, System2/executive/remote right consultation disabled. No working-memory,
hippocampal or unconscious factual hints are supplied, so all arms have the same
short-context boundary. Source writes have no retrieved context. Identical source
generation requests share frozen real model completions across arms; source QA
must match, rather than letting random source rewrites confound the comparison.
Final probe answers are separately generated and stored through the controller.

First run development with off/current checking. Diagnose source capture, scope
resolution, replacement, complete-source limits and repeated verification calls.
Repair those mechanisms, then generate the conversation arm on development and
grade the three arms together. Freeze implementation before evaluation; run the
two untouched evaluation episodes once with all three arms. Do not tune on those
responses. Preserve failed and incomplete stages; do not silently score a local
model fallback as a successful live generation.

Use gpt-6-luna for generation, support checking and an arm-blinded rubric grader.
The grader sees shuffled answer labels and the reference/criteria, never arm
names or retrieval diagnostics. Require complete pass/fail/uncertain decisions
and literal evidence quotes. Model grading is exploratory and uses the same
model family, not an independent human assessment. Report per-criterion outcomes,
complete-case task success, unsupported-fact failures and grader failures, with
all denominators. Distinguish correct abstention from missing supported facts.

Budget at most 256 live chat requests across this experiment, no continuations or
automatic retries, 4,096 completion tokens per request. Record actual input/output
usage, latency by role, checker calls/cache hits, source hashes, prompts, exact
request/response journals, revision and fixture hashes. No embeddings are needed
for this small scope-registry experiment. Paid commands are explicit; replay must
use complete exact matching request sequences with zero network calls. Private
answers/journals stay ignored; publish metrics and artifact hashes.

Keep default behavior unchanged. Any new conversation binding is opt-in and uses
declared scopes, conservative follow-up phrases and fail-closed ambiguity. Long
QA must remain complete within a bounded larger budget, not lose conditions by
prefix truncation. Remembered replies must not multiply into independent primary
evidence. Supersession requires explicit caller IDs within the selected scope,
not an assumption that any newer answer is true. Validate normal/stream RPC,
metadata persistence and both worker evidence paths. No live PostgreSQL or
adversarial robustness claim follows from simulated storage tests.

Development amendment before evaluation: the first rubric grader incorrectly
treated missing recall as unsupported factual assertion, and rejected one valid
unknown-fee admission. Retain that initial grading journal as a diagnostic. Clarify
that omissions can fail task while passing grounding, and equivalent uncertainty
phrases pass unknown-fact tasks. Regrade the same saved development answers once
with `--grade-only`; do not regenerate favorable answers. Freeze this rubric before
any evaluation responses. Include both grading costs in the 256-call budget.
