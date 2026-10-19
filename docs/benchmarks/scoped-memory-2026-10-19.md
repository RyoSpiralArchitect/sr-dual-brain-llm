# Scoped multilingual memory pilot — 2026-10-19

Separating declared project identity from question content improves the recall /
abstention operating point on this synthetic fixture. At their predeclared default
thresholds, the previous hybrid retrieves the relevant first source on 14/25
positive evaluation queries and the scoped hybrid on 22/25; irrelevant selected
slots fall from 7 to 0. Japanese first-source hits rise from 0/8 to 5/8.

This requires trusted scope/alias annotations. The gains are not solely an
embedding-model improvement. Unannotated records and the raw threshold remain
unchanged; model-produced QA is not automatically annotated. See the
[usage guide](scoped-memory-recall.md), [fixed protocol](scoped-memory-protocol.md),
and [machine-readable metrics](scoped-memory-2026-10-19.metrics.json).

## Comparison

The fixture contains 17 records and 72 queries. Four calibration scopes supply
34 queries (24 positive, 10 no-answer). Four other evaluation scopes supply 38
(25 positive, 13 no-answer). Every scope has two competing topics. Two Tidal rate
limit records conflict; both must be retrieved for full-source recall.

| Selector | First source relevant | All sources within 3 | Correct no-answer abstention | Irrelevant slots / selected | Macro set-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous lexical | 16/25 | 17/25 | 6/13 | 50/70 | 0.410 |
| Scoped lexical | 16/25 | 16/25 | 13/13 | 2/21 | 0.746 |
| Previous hybrid, default 0.70 | 14/25 | 14/25 | 10/13 | 7/23 | 0.623 |
| Scoped hybrid, content default 0.50 | 22/25 | 22/25 | 13/13 | 0/25 | 0.921 |
| Previous hybrid, calibrated 0.30 | 25/25 | 25/25 | 5/13 | 25/53 | 0.727 |
| Scoped hybrid, calibrated 0.25 | 25/25 | 25/25 | 13/13 | 0/28 | 1.000 |

Calibration used only calibration source labels: a fixed 0.20–0.80 grid in 0.05
steps, macro source-set F1, and the stricter threshold on ties. The calibrated
values are not installed as runtime defaults. After inspecting evaluation, no
alias, relevance label, ranking rule or threshold was tuned.

The default comparison changes both the representation and its threshold. A
post-run diagnostic at the already predeclared 0.50 threshold, using the same raw
vectors without scopes, gives 18/25 first hits, 6/13 correct abstentions and 12/32
irrelevant slots (F1 0.623). Merely lowering the old threshold does not reproduce
the new operating point here. This additional diagnostic was not used for tuning
and still does not isolate alias information from the content representation.

Scope/alias handling supplies much of the precision gain: scoped lexical search
already reduces irrelevant slots from 50 to 2. It does not solve translation:
Japanese first hits remain 1/8 locally. With learned content comparison they are
5/8 at the default and 8/8 at the calibration-selected threshold. The width/case
variant is retrieved by both scoped hybrid settings, whereas scoped lexical
search misses it because the question's topic is Japanese.

The default hybrid's three failures remain: `beacon-keys-japanese`,
`tidal-cache-japanese`, and `kestrel-rollback-japanese`. All abstain. Scoped lexical
search additionally misses most Japanese queries and the Kestrel rollback
paraphrase; it includes an irrelevant same-scope key/backup record on each of the
two Beacon echo queries. Its full-source recall falls from the old lexical 17/25
to 16/25: removing project-name assistance is a real tradeoff without semantics.
Every evaluation source selection is retained in the metrics file.

## Cost, regression checks, and limits

The run used `text-embedding-3-small`, served under the same name, dimension 1,536.
Preparing 106 distinct raw/content question views made **two requests, 1,553
reported input tokens**, taking about 2.88 seconds. This includes both baseline
and scoped representations, not a production per-turn measurement. Answers,
drafts, scope IDs, and verification labels were not embedded.

Complete frozen-vector replay used zero provider calls and reproduced all
summaries and calibration candidates exactly. The untouched October 16 fixture
was also replayed with its original vectors: all source selections, aggregate
metrics, runtime-default summaries and field-ablation recovery decisions match
the earlier report. Thus these checks establish preservation of the unannotated
path, not an improvement to its previous Japanese misses without annotations.

Local checks passed: 399 Python unit tests, five gateway contracts and 13 Node
tests. Tests cover scoped controller data flow, both detail/critic routes in both
actual worker/model prompt builders, ambiguous names, Unicode boundaries, original
UTF-8 budgets, expiry/supersession, conflict preservation and PostgreSQL metadata
round-trip/legacy defaults. Model calls are fake in these new path tests; no new
live chat or final-answer grading was performed. PostgreSQL uses a simulated
cursor, not a running database.

The perfect calibrated score is a small synthetic result. Queries and labels
were authored together, variants are correlated, and topic templates repeat
across scenes. **27/38 evaluation content cues exactly occur in calibration,
including 18/25 positive cues.** Project/source separation does not make this an
unseen-topic or blind external evaluation. Negatives sample only a few unrelated
topics; ambiguous/unknown-name abstention is also an explicit policy in the labels.
More diverse same-scope near misses and independently authored multilingual
questions are needed before reducing production thresholds. Missing or mistaken
caller annotations remain a limitation. Retrieval scores do not establish final
answer correctness, current applicability, or action quality.

## Provenance

Protocol and fixture were committed as `d03afaf` before the run; implementation,
default thresholds and selectors were frozen at clean commit
`2726cc2d529f4ee63f9094ec5e686543911a3484`. Live embedding, new frozen replay, old
fixture regression and the post-run fixed-threshold diagnostic used that revision.
Fixture and private-artifact hashes are in the metrics. Raw vectors/full reports
stay under ignored `target/benchmarks/`; hashes identify them but do not by
themselves provide an independent audit of execution. No historical answer,
fixture label or human rating was revised.

Review found that shared question text could make a latent seed's scope ambiguous
even though its stored source was uniquely identified. Commit
`241955a` resolves that scope by exact unique seed/source-QA identity, with tests
for both project directions and duplicate binding rejection. Zero-call new/old
replays on that clean revision preserve the reported summaries, calibration,
older source selections/defaults and field recovery decisions. These follow-up
revisions and hashes are recorded separately in the metrics; they do not change
the provenance of the original paid run or retune the selector.

A second review correction, `426887c`, makes name boundaries Unicode-aware:
`Café` cannot select `Caféine`, and letter/number/mark/format continuations cannot
turn a different name into a known scope. Japanese adjoining context remains
supported. Twelve boundary cases pass. New and old zero-call replays on this
clean revision again preserve every selection, summary, calibration choice and
older field recovery decision; their hashes are retained separately.
