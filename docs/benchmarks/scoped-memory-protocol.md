# Scoped multilingual retrieval protocol — 2026-10-19

This follow-up addresses two known development failures from the October 16
pilot: Japanese paraphrases and an unsupported question sharing a project name.
The previous fixture and results remain unchanged and are regression evidence,
not a fresh held-out test for the new design.

The new fixture declares trusted scope IDs and aliases separately from answer
verification. Aliases link names, not facts. No scope, translation, or verification
is inferred by a chat model. Project names and generic project wrappers are removed
from the content cue only after a declared alias matches. A declared scope must
match before its facts can be retrieved; ambiguous aliases must not pick a scope.
Unannotated historical memories retain their existing retrieval behavior.

The planned selectors are the previous lexical/hybrid rules, scoped lexical
selection, and scoped hybrid selection. The hybrid compares question content
after scope resolution. The fixed raw-question default stays 0.70; the initial
content-cue default is 0.50. These are similarity thresholds, not probabilities.
Calibration searches 0.20 through 0.80 in increments of 0.05, maximizing macro
source-set F1 (empty/empty = 1), preferring the stricter threshold on ties.
Raw and scoped selectors are calibrated separately using calibration labels only.
Both defaults and calibrated results must be reported. No threshold, ranking
weight, alias, or relevance label is changed after the evaluation run is inspected.

Four calibration scopes and four different evaluation scopes each have two
competing topics, English echoes/paraphrases and Japanese alias queries. Negatives
include unsupported English/Japanese topics within a known scope, unknown scopes,
ambiguous aliases, and unrelated requests. An evaluation pair has conflicting
active answers; both are relevant. Variants are correlated and authored alongside
the fixture: this is a synthetic scene-disjoint experiment, not a blind external
test. Explicit scope annotations are additional information relative to the old
pilot; gains must not be attributed solely to the embedding model.

Report every selected source ID, full-source recall within three, no-answer
abstention, irrelevant selected slots, macro set-F1, language/kind breakdowns,
embedding input/call usage, source revision, fixture/cache hashes, and failures.
Freeze code and fixture in git before the paid run. Store a complete private
vector checkpoint for zero-call replay. Never embed answers or use answer labels
to create cues. A fresh run is not needed when replay answers the question.

Scope boundaries also apply to recalled QA evidence, even when an unconscious
seed passed its own hash-based trigger. Test persistence, actual worker prompts,
ambiguous names, Unicode width/case normalization, entity-only matches, provider
outages, and preservation of unannotated behavior. Field scope checks do not
replace answer-quality grading. Retrieval gains are not answer-quality gains;
no historical answer rating is revised by this study.
