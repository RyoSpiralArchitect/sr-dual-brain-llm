# Prospective bounded clarification/detail confirmation — 2026-11-06

Freeze production code, runner, inputs, criteria and this protocol before paid
calls. Prior #95 results remain immutable. Use two fresh authored scopes in
`clarification_detail_confirmation.json`, real `openai/gpt-6-luna`, the existing
all-AUTO profile with Director/executive work OFF, balanced AUTO priority and the
unchanged production token/time budgets. No paid judge or semantic rescore.

Elicit one natural model draft per case through the production left generator,
using the fresh memory condition and PFC replay context. Retain that exact draft
across all arms. Record whether it matches an allowed canonical clarification;
an unfavorable/noncanonical draft stays in the study without another sample.
Every arm uses a fresh real EngineSession and the same explicit caller scope,
records and source-change setup. Candidate selection uses zero model calls.

This is a **controlled routing comparison**: all evaluated arms use fixed
consult pressure (policy action 1) to ensure positive exposure. It is not a
natural policy-frequency estimate. The only between-arm admission change is
selective consultation OFF versus AUTO. The English case also includes explicit
System2 ON and OFF with the selective module AUTO. Those controls must retain
authority; OFF may still use ordinary detail.

The English case asks which of two current complete records to quote. The
Japanese case obtains a selection reference and changes that record before
generation. Expect source-free `clarify_record` / `reselect_record`, retaining
all old/new/unselected value canaries. AUTO may skip only when the actual shared
draft is an exact allowed source-free reply for the fresh condition and all
existing protected controls and finite confidence/risk requirements pass.
Canonical reply matching is a bounded structural contract, not semantic grading.

Report actual source exposure, final text, gate proposal/application, critic
execution, detail delivery outcomes and provider HTTP receipts separately.
Known returned unusable ordinary detail must retain the draft without a second
detail request. Delivery failure can retain the existing single direct fallback;
its result must also be checked. Do not claim zero transport failures means all
stage outputs were usable. Detail is not counted as critic execution.

Hard cap: **24 provider HTTP attempts total**, including both shared drafts,
all support, detail, critic/repair, integration, retries and existing fallbacks.
No warm-up, resampling, continuation, paid rescue or additional diagnostic budget.
Retain unknown usage with known subtotals. Real transport/auth/quota failure
closes paid admission via the existing sticky raw journal.

The live study uses in-memory primary fixtures, not application persistence.
Real PostgreSQL remains offline integration coverage in the full suite.
Injected empty/nonterminal/provider failures are separately labeled offline
controls; a natural live occurrence is reported only if it actually happens.

Replay every exact ordered raw HTTP envelope under the offline provider guard
with no transport object. Require identical canonical non-timing outcomes and
complete occurrence coverage, including failed/empty completions. Keep raw
requests, responses and shared draft text private; publish filtered metrics and
artifact hashes. Do not tune code, inputs or grading after seeing live outputs.

```bash
python sr-dual-brain-llm/scripts/benchmark_clarification_detail.py --model gpt-6-luna \
  --max-calls 24 --output-dir sr-dual-brain-llm/samples/module-gating-clarification-detail-2026-11-06-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_clarification_detail.py \
  --journal sr-dual-brain-llm/samples/module-gating-clarification-detail-2026-11-06-live/journal.json \
  --max-calls 24 --output-dir sr-dual-brain-llm/samples/module-gating-clarification-detail-2026-11-06-replay
```
