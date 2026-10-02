# GPT-6 Luna System2 paired retest — 2026-10-02

Two live, counterbalanced blocks ran the same 15-question English/Japanese
suite on the clean source commit `60d45dcc1b562eeeae4da4b698c04f71a1e09e88`.
Block AB ran `off,on`; block BA ran `on,off`. Both hemispheres and the
executive were configured for OpenAI `gpt-6-luna`; executive mode was `off`.
The effective question-list SHA-256 in both reports is
`4cc0dd11caacf3a0b4b10ab61b312da917fb8003c9936e379672650272aaa378`.

## Conditions and validity

- All 15 IDs ran in fixture order in each mode and block: 60 attempted turns.
  No question was removed after seeing an answer.
- `LLM_MAX_OUTPUT_TOKENS=4096`, `LLM_TIMEOUT=90`,
  `LLM_AUTO_CONTINUE=0`, `DUALBRAIN_CRITIC_MAX_OUTPUT_TOKENS=4096`,
  `DUALBRAIN_CRITIC_TIMEOUT_SECONDS=60`,
  `DUALBRAIN_SYSTEM2_TIMEOUT_MULTIPLIER=3`, and
  `DUALBRAIN_TIMEOUT_MAX_MS=180000`.
- Critic preflight required 2/2 valid external responses in each mode, with
  no retries. All four mode preflights passed. All 30 `on` cases have
  `critic_validity=valid`; the 39 recorded critic round statuses are `ok`.
  The 30 `off` cases have no critic measurement.
- Complete answers and dialogue flows were saved only in local, ignored
  reports with owner-only permissions. The report contains loaded model
  settings, mode order, source revision, fixture hashes, and per-case statuses.

| Block | Mode | Completed | Invalid critic | Internal issues | Mean turn latency |
| --- | --- | ---: | ---: | ---: | ---: |
| AB, first | off | 15/15 | — | — | 9,035 ms |
| AB, second | on | 15/15 | 0 | 6 → 2 | 6,634 ms |
| BA, first | on | 15/15 | 0 | 7 → 5 | 8,709 ms |
| BA, second | off | 15/15 | — | — | 7,457 ms |

The internal issue counts are produced by the evaluated critic. They are
diagnostics, not an answer-quality score. The latency direction reverses
between blocks; two sequential blocks do not establish a speed difference.

## Independent fixed-reference scoring

`score_system2_reference.py` compared explicit final answers on the seven
base-suite items with fixed numeric or category references. It did not ask
Luna to judge Luna. Across the 14 potential question/block pairs, 12 were
machine-scoreable for both modes: all 12 were **correct/correct ties**.
There were 0 `on` wins, 0 `off` wins, and 0 incorrect/incorrect ties.
Two pairs were indeterminate because the frozen recognizer did not match
the answer wording (`table_inference_001` in AB and `bayes_001` in BA).
The other eight questions per block are open-ended and unscored by this
fixed-reference method.

This result supports no answer-quality advantage for either mode in the
machine-scoreable subset. It does not score explanations, completeness,
unsupported claims, or the open-ended items. A 30-packet blinded answer set
was prepared locally for the two-rater protocol in
[`system2-paired-evaluation-protocol.md`](system2-paired-evaluation-protocol.md);
human ratings and adjudication have not been collected. A separate
[single-model blinded review by Sol-6.1](gpt-6-luna-system2-sol-judge-2026-10-02.md)
found 27 ties, 1 `on` preference, and 2 `off` preferences across all 30 pairs.

### Additional Mistral and human review in progress

`mistral-large-latest` completed all 30 blind packets in a fresh API request
using the same rubric, without the reveal key or other ratings. The request
used [Mistral JSON mode](https://docs.mistral.ai/studio/conversations/structured-output/json_mode),
temperature 0, random seed 7, and an 8,192-token output limit. The response
reported the same model alias; an underlying model revision was not exposed.
Packet IDs, score ranges, and completion status were validated before locking
the ratings. Their SHA-256 is
`379828b528818d6c087f4fa4814de1c184255d5af48f821c542460e9ad48c881`.

Those scores are withheld until the human ratings are locked. The human
reviewer has already seen the earlier Sol aggregate; the form records prior
exposure explicitly. Human scoring and adjudication remain pending, so this
follow-up is not yet a completed two-rater result. The assistant coordinating
the study is not an additional blinded rater because it has seen mode mappings.

## Reproduce and receipts

Set `OPENAI_API_KEY` outside the repository. Run the same benchmark command
twice, once with `--modes off,on` and once with `--modes on,off`, using the
environment values above plus `LLM_PROVIDER=openai LLM_MODEL_ID=gpt-6-luna`.
For both invocations use:

```bash
python3 sr-dual-brain-llm/scripts/benchmark_system2_ab.py \
  --critic-health-attempts 2 --critic-health-min-successes 2 \
  --critic-health-retries 0 --critic-health-timeout 60 \
  --require-critic-health --diagnostics all \
  --include-full-answers --include-traces --history '' \
  --modes off,on --output target/benchmarks/luna-system2-ab.json
```

Switch only the final `--modes` to `on,off` and the output filename to
`luna-system2-ba.json` for the second block. Then run
`score_system2_reference.py --ab ... --ba ... --output ...` against the two
reports. The score output has case IDs and labels but no answer text.

Local receipt hashes (SHA-256) for this run:

| Local file under `target/benchmarks/` | SHA-256 |
| --- | --- |
| `luna-system2-ab-2026-10-02.json` | `c555699b8ac33fb8000c6a423127baf50e9a165779b5afcf2b315b53bca9c03c` |
| `luna-system2-ba-2026-10-02.json` | `42d0712e24b567d042367976181bb6c2f4fd805461cffb8f4ce647c05552fce1` |
| `luna-system2-reference-2026-10-02.json` | `e1e4c3a60c55a5157ece2bbd1b53b548eb2a2b5685d57e16aa173dbdfddb7d03` |

Run IDs: AB `system2_ab_20261002_113100`; BA
`system2_ab_20261002_113710`. The full reports and blinded packets remain
outside Git because they contain complete generated answers and traces.
