# Director steering confirmation — 2026-11-02 (Asia/Tokyo)

The [first TRN pilot](module-gating-2026-11-01.md) exposed replacement work on
trivial turns. Codex review requested preserving the Director's free steering
before merge. The repair separates caller-selected deterministic control from
optional provider task admission, and prevents `consult=skip` from suppressing
an explicitly active System2 `on` request. No gate predicate/threshold was retuned.

This is a separate, prospectively committed two-input confirmation at revision
`6a1e97fa81c8bd4742cd42bf365acade21069813`. The first eight-trial result and its
frozen revision are unchanged. See the [confirmation protocol](module-gating-protocol.md#post-review-steering-confirmation--2026-11-02-asiatokyo).

`Hello!` and `こんにちは！` were not inputs in the first pilot. Each received one
natural `gpt-6-luna` draft, frozen byte-for-byte for both legacy-default and all-AUTO
profiles. Both requested explicit left lead, System2 auto/balanced, Executive
observe and observer both. Blind `gpt-6-sol` grading used the same task, grounding,
clarification and exact-quote validation rules. This is a targeted regression
confirmation on greetings, not a general workload evaluation.

| Profile | Available / graded | All criteria passed | Post-draft API attempts | Right detail / left integration |
| --- | ---: | ---: | ---: | ---: |
| Frozen drafts | 2/2 | 2/2 | 0 | 0 / 0 |
| Legacy defaults | 2/2 | 2/2 | 4 | 0 / 0 |
| All AUTO | 2/2 | 2/2 | **0** | **0 / 0** |

Both profiles retained each original draft exactly. Legacy attempted Executive
advice and post-turn metrics for each greeting; AUTO attempted neither and did
not replace them with consultation/integration. In both profiles the requested
free Director controls remained available: consultation skip, short output and
long-term-memory suppression. Automated tests separately cover observer modes
that do not request Director, missing providers, manual gate modes, explicit
System2 on/off, forced-right lead and nontrivial input.

The full confirmation used **8 provider attempts** under its 24-attempt cap:
2 shared draft-generation, 4 legacy processing and 2 judge attempts. There was
no resampling or malformed grade packet. Three legacy attempts were cancelled
at existing wait limits (two Executive advice, one metrics); their usage is
unknown. Known receipts contain 2,080 input and 588 output tokens across the
whole confirmation, including generation/judging. The zero AUTO processing
attempts are directly observed, but complete billed savings cannot be priced.
Primary draft generation is shared/excluded from the processing comparison;
this is not a claim of zero end-to-end model use.

A subsequent review fixed a measurement-runner race: an allowed cancellation or
invalid completion could clear another concurrent request's genuine transport
failure. The runner now holds that blocking failure independently and tests both
exempt outcomes after a concurrent failure. Neither live journal contains a
genuine transport failure (the first has one recorded output-limit completion),
so this repair does not change either study's data or replay record. The
source-pinned commands below reproduce the measured runner; current-head tests
cover the strengthened stop behavior.

Full transport replay consumed all **8 recorded request events**, including the
three cancellations, with **0 new provider calls**. Status, final answers, draft
identity, gate payloads and grades all matched. The
[safe metrics projection](module-gating-steering-2026-11-02.metrics.json) preserves
these checks and per-cell health without raw prompts, responses or auth headers.

```bash
git worktree add /tmp/trn-steering-confirmation 6a1e97fa81c8bd4742cd42bf365acade21069813
cd /tmp/trn-steering-confirmation
python sr-dual-brain-llm/scripts/benchmark_module_gating.py \
  --fixture sr-dual-brain-llm/examples/module_gating_steering_confirmation.json \
  --model gpt-6-luna --judge-model gpt-6-sol --max-calls 24 \
  --output-dir sr-dual-brain-llm/samples/module-gating-steering-NEW-live
python sr-dual-brain-llm/scripts/benchmark_module_gating.py \
  --fixture sr-dual-brain-llm/examples/module_gating_steering_confirmation.json \
  --journal /absolute/path/to/confirmation/journal.json \
  --judge-model gpt-6-sol --max-calls 24 \
  --output-dir sr-dual-brain-llm/samples/module-gating-steering-NEW-replay
```

The original control-flow issue is addressed in PR #68 / #69. The next target is
[deadline-aware optional advisor admission](https://github.com/RyoSpiralArchitect/sr-dual-brain-llm/issues/70):
observe useful completion versus cancellation, then supply bounded remaining-time
and known latency facts to TRN. Do not simply increase every timeout. The first
pilot's seven gates without positive AUTO exposure still need controlled live
coverage; neither study justifies broadly enabling AUTO defaults.
