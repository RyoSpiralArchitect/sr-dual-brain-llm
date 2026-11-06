# Prospective read-only record-choices confirmation — 2026-11-06

Freeze implementation, API contract, inputs, criteria and runner before paid
calls. Prior #92–#94 studies and scores remain immutable. This study measures
the new caller selection round trip, not a general quality/latency improvement.

Use production nonstream JSON `record_choices` and `process` dispatch, real
`openai/gpt-6-luna`, 2,048 output tokens, no continuation, existing timeouts and
retry/repair rules, all-AUTO with Director/executive advice/metrics OFF. Explicit
ON and OFF controls retain authority. No paid judge is used; V3 semantic grades
stay unknown. Manual response observations are unblinded assistant inspection.

The six fixed turns in `record_choices_confirmation.json` are executed once:

1. English: list two current records, choose backup, quote with System2 ON.
2. Japanese: list two, choose retention, quote with OFF.
3. List/choose backup, prospectively prepare its full-source support once, then
   change its content before the request. Expect stale selection and reselect.
   Keep any true/false/unknown setup receipt; count the call, no rescue.
4. Real isolated PostgreSQL: seed two authored primary fixtures, list and quote
   the selected backup with a durable operation receipt.
5. Restart the same persistent session, try the previous reference in a fresh
   operation. Expect invalid selection and reselect, without the old source text.
6. List again in that restarted session, choose current backup and quote it with
   a new committed operation. Primary source identities must remain unchanged.

Candidate enumeration must make **zero model calls**. It returns labels and
signed references, never answer bodies or approvals. Select by the fixed label
substring in the fixture, not by a favorable output. All four intended quotes
use the prospectively named actual current record as the UTF-8 equality oracle;
observed V3 source-validity guards remain independent. The two invalidation
responses have no equality oracle. Record the actual `record_selection` result,
missing condition, exposure, HTTP usage and final System2 execution independently.
Inspect every answer, retaining source re-entry wording and other limitations.

Hard cap: **32 provider HTTP attempts total**, including the one pre-change
support preparation, all support, drafting, detail, critic, repair, retries and
integration. No warm-up, resampling, paid rescue or extra diagnostic allocation.
Unknown usage remains unknown with known subtotals. Genuine transport/auth/quota
failure closes paid admission with the existing sticky raw journal.

Use `DUALBRAIN_TEST_PG_DSN` only, a fresh owned table prefix and private outbox
under the ignored output directory. Seed fixtures are not model write-quality
observations. Cleanup drops only that owned namespace. No application data or
outbox is used. Keep raw journals and requests private. References/signatures
vary by memory incarnation and must not enter provider prompts or public metrics;
canonical comparisons use observed identity/binding outcomes, not token bytes.

Replay every exact ordered HTTP occurrence with no transport object under the
offline HTTP guard. Require identical canonical non-timing outcomes, including
all failed/unknown receipts. A divergence is preserved without paid rescue or
post-output tuning. Every raw returned envelope reaches production validation.

```bash
python sr-dual-brain-llm/scripts/benchmark_record_choices.py --model gpt-6-luna \
  --max-calls 32 --output-dir sr-dual-brain-llm/samples/module-gating-record-choices-2026-11-06-live
python sr-dual-brain-llm/scripts/run_offline_checks.py -- \
  python sr-dual-brain-llm/scripts/benchmark_record_choices.py \
  --journal sr-dual-brain-llm/samples/module-gating-record-choices-2026-11-06-live/journal.json \
  --max-calls 32 --output-dir sr-dual-brain-llm/samples/module-gating-record-choices-2026-11-06-replay
```
