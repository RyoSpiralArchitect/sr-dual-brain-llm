# Same-topic memory precision protocol — 2026-10-19

The preceding scope/content pilot left three Japanese misses at the default
0.50 similarity threshold. Its calibration-selected 0.25 threshold retrieved
every evaluation source, but nearby unsupported facets were scarcely tested.
That historical fixture is development/regression evidence, not fresh evaluation.
This study measures the cost of that lower operating point before changing any
runtime default.

`memory_precision_benchmark.json` contains 19 synthetic records and 90 queries.
Three calibration scopes cover webhook delivery, CSV ingestion and report
artifacts; three evaluation scopes cover container lifecycle, queue consumption
and search indexing. Each scope contains three competing question facets. Each
facet has an English echo, English paraphrase, Japanese positive, English
unanswered adjacent-facet question, and Japanese unanswered question. A conflicting
queue batch-size record must be retrieved alongside the original, not resolved
by recency. Both splits contain 27 positive and 18 no-answer queries. All questions
name a known scope: abstention cannot be credited to an unknown-name shortcut.

Scopes, source IDs and declared topic families are disjoint across splits. Query
content cues must not exactly repeat across splits. This is still authored,
correlated synthetic data, not a blind or independently labeled test. Explicit
`no_answer_reason` annotations explain why a nearby fact does not answer the
requested facet. General domain knowledge must not fill the gap. Verification
labels describe fixture assumptions, not verified operational facts.

Retain all six previous lexical/hybrid comparators unchanged, including raw 0.70,
content 0.50 and F1 calibration on 0.20–0.80 in 0.05 steps. Add the previously
selected scoped 0.25 setting as a fixed development comparator. Separately search
scoped 0.20–1.00 in 0.05 steps, maximizing calibration macro source-set F1 subject
to **zero calibration no-answer queries selecting any source**, preferring the
stricter threshold on ties. This is a query-level false-retrieval constraint,
not a claim of statistical safety or zero irrelevant slots on positive queries.

If no threshold satisfies the constraint, return `feasible: false` and no chosen
threshold; never silently recommend the least bad threshold. Report the best
effort point separately, minimizing false-retrieval queries, then maximizing F1,
then preferring stricter thresholds. It is diagnostic and cannot be published as
a calibrated configuration. Runtime defaults, ranking, aliases and labels stay
fixed after evaluation inspection. Infeasibility is a valid result.

Report every source selection, language/kind breakdowns, positive recall,
no-answer false retrievals, irrelevant slots, macro source-set F1 and failures.
For the fixed 0.50 and 0.25 scoped selectors, inspect compatible candidate cosine
scores, score gaps, and whether a below-threshold source was retained via the
strong lexical override. A gap is between distinct source questions, so a
duplicate conflicting question is not its own competitor. These diagnostics are
post-selection explanations, not additional ranking rules or confidence scores.

Commit this fixture and protocol before embedding; freeze implementation before
the paid run. Embed questions only, record model/usage/revision and hashes, and
save complete private vectors for zero-provider-call replay. Run the unchanged
older fixtures as regressions. Do not grade final answers, claim a live database
test, or revise historical results. Any later runtime change needs a separate
experiment; this study adds evaluation and calibration guardrails.
