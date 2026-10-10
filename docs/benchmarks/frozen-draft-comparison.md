# Frozen-draft review comparison

This experiment isolates review and revision from initial answer generation. It
does **not** estimate the end-to-end benefit of `system2_mode=on`, compare a
single brain to a dual brain, or evaluate the full controller's routing and
fallbacks. Every arm receives the same frozen question, context and draft:

| Arm | Review | Revision | Maximum calls |
| --- | --- | --- | --- |
| `no_revision` | None | None; unchanged draft | 0 |
| `self_review` | Left model | Left model | 2 |
| `right_critic` | Right model | Left model | 2 |

Both active arms use identical review instructions and revision instructions,
requested temperature 0.2, token caps and per-call timeouts. Continuation and retry
calls are disabled. Each call uses a new client and complete explicit input,
with no shared conversation or controller memory. Matching caps and call
budgets does not imply matching actual tokens, latency, cost, or model capability.
Provider-specific request options are recorded: the existing client omits
temperature for OpenAI GPT-5/6 models and uses `max_completion_tokens`, which also
covers reasoning tokens. Other models use `temperature` and `max_tokens`.
Consequently, identical configured caps need not allow identical visible answer
lengths, and the requested temperature is not guaranteed to be effective.
If both roles use the same model, this measures independent review rather than
model diversity. Different providers/routes can introduce additional differences.

The fixed `system2-hard-repairs-v1` rubric covers concurrency, incident triage,
log-sharing security and distributed payments: four existing questions and five
criteria each. Reviewers record `pass`, `fail`, or `uncertain` with evidence from
each answer. Missing required content fails. Incompatible recommendations fail
even when the answer also contains a correct phrase. Critical criteria identify
dangerous regressions without changing the weighting. The reviewer must be
independent of answer generation; do not use the critic's own verdict as a score.

## Prepare frozen drafts

Create a JSON corpus with `schema_version: 1` and one `drafts` entry per rubric
case. Each entry needs the case `id`, complete `draft`, `origin` (`model`, `human`
or `synthetic`), and optional text `context`. Preserve the original draft without
selecting a different starting answer for either review arm. Record model-draft
collection settings and inclusion decisions separately before seeing revisions.

The checked-in `system2_frozen_drafts.synthetic.json` contains deliberately bad,
artificial answers. It is a harness fixture, **not model performance evidence**.
All reports start as `unscored`. Dry runs make no LLM calls and cannot be prepared
for quality scoring:

```bash
python sr-dual-brain-llm/scripts/benchmark_frozen_drafts.py run \
  --drafts sr-dual-brain-llm/examples/system2_frozen_drafts.synthetic.json \
  --dry-run --output-dir target/benchmarks/frozen-dry-run
```

## Run the live comparison

Configure the existing `LEFT_BRAIN_*` and `RIGHT_BRAIN_*` (or shared `LLM_*`)
provider/model/key environment variables. The initial harness supports the
OpenAI-style `openai`, `mistral` and `xai` providers. Missing configuration fails
explicitly; there is no heuristic substitute. Do not put credentials in a corpus
or command line. Live mode sends corpus content to the configured providers.

```bash
python sr-dual-brain-llm/scripts/benchmark_frozen_drafts.py run \
  --drafts target/benchmarks/frozen-model-drafts.json \
  --live --repetitions 6 --max-tokens 1024 --timeout 40 \
  --output-dir target/benchmarks/frozen-live
```

With four cases and six repetitions, a successful run makes **96 calls** (48
reviews and 48 revisions). Six repetitions cover every arm ordering once per
case; each arm occupies each position twice. The seed controls case order;
actual arm order is recorded. Fewer repetitions are allowed for smoke checks
but do not guarantee full counterbalancing. Repetitions share the frozen draft
and are **not additional independent questions**.

Reports preserve input/corpus/rubric hashes, source revision and dirty flag,
origin, sanitized effective role configuration, each stage's text, wall time,
attempted calls, finish reason and provider-reported input/output tokens. Missing
usage is `null`, not zero. Budget failures are visible: invalid review JSON,
contradictory verdicts, empty output, provider errors and truncated completions
retain their rows; they cannot become successful revisions. After a failed
review, the revision call is skipped. Raw provider exception messages are omitted
because they may contain credentials or private input.

Run outputs need a new directory, and JSON files use mode `0600` and exclusive
creation. Store generated answers and reveal keys under ignored `target/` paths.

## Blind review and scoring

```bash
python sr-dual-brain-llm/scripts/benchmark_frozen_drafts.py prepare \
  --report target/benchmarks/frozen-live/report.json \
  --output-dir target/benchmarks/frozen-review
```

Give the reviewer `blind.json` and `ratings.json`. Keep `reveal_key.json` separate.
Answer labels and packet order are shuffled; arm names, models and critic text
are excluded. Identical answers or recognizable wording can still reveal an
arm, so blinding is not guaranteed. Failed answers remain `null` and their
criteria are `unavailable`; do not drop them or replace them with the draft.

The reviewer fills their identity and every pending criterion's verdict/evidence
in `ratings.json`. Use `uncertain` for genuinely ambiguous evidence and adjudicate
disagreements before claiming a quality difference. The scorer validates the
packet digest, full packet/criterion coverage, evidence, and failure availability:

```bash
python sr-dual-brain-llm/scripts/benchmark_frozen_drafts.py score \
  --blind target/benchmarks/frozen-review/blind.json \
  --key target/benchmarks/frozen-review/reveal_key.json \
  --ratings target/benchmarks/frozen-review/ratings.json \
  --output target/benchmarks/frozen-review/scores.json
```

Repairs are baseline `fail` → final `pass`; regressions are baseline `pass` →
final `fail`. The report separately counts critical regressions, all criterion
units, pass/fail/uncertain/unavailable counts and failed answers. Rate denominators
include only comparisons with determinate baseline and final labels:

- Repair rate: repaired defects / eligible baseline failures.
- Regression rate: introduced defects / eligible baseline passes.

Unavailable or uncertain outcomes are excluded from these rate denominators,
but remain in counts. Always report those counts beside rates; exclusions can
bias comparisons. Empty denominators produce `null`. Counts are at
case/repetition/criterion level; the report also states unique-case and packet
counts. These correlated units must not be treated as independent observations.
Defects outside the frozen criteria require separate adjudication.

## Latency compatibility

New System2 benchmark cases use `latency_schema_version: 2`,
`latency_scope: process_wall`, and measure the entire awaited `process()` call,
including failures. `controller_latency_ms` records controller telemetry;
`consult_latency_ms` retains the old consultation/integration interval. The
controller's `controller_turn` interval includes initial draft, persistence,
observer and feedback; final telemetry emission/return adds a small outer-call
overhead. Phase timings are diagnostic and may overlap; their sum is not the
definition of wall time. Adaptive feedback retains its existing consult interval;
changing its learned behavior is a separate experiment.

Historical reports lacking the schema/scope fields remain unchanged. Their
`latency_ms` can describe a partial consultation interval and must not be pooled
with version 2 full-turn measurements. New summaries retain the scope/version;
mixed case scopes suppress latency averages, and comparison/history delta tables
mark differing scopes/versions `incomparable` instead of assigning a speed gain.
Two legacy reports without metadata retain the old comparison behavior; this
does not certify their unknown measurement scopes. See the
[evidence analysis](system2-evidence-analysis-2026-10-10.md) for the archive audit.
