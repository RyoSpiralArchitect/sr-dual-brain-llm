# Question/evidence support protocol — 2026-10-19

The precision study showed that high cosine and large candidate gaps do not
establish support for the requested facet. Test an optional, bounded model check
between candidate recall and historical QA delivery. Each retained source must
support at least one fact explicitly requested by the current question. Compound
questions may need multiple individually partial sources; conflicts remain
independent historical records, not a license to select a convenient answer.
An explicit statement of absence can answer a question negatively. Silence cannot.

The checker receives the current question and candidate Q&A excerpts as data,
never drafts, relevance labels or no-answer rationales. It returns a boolean and
an exact quote from each source's own answer. Missing, malformed, duplicate or
unknown decisions, nonliteral booleans and ungrounded quotes cannot approve
evidence. Provider failure must withhold evidence, not silently restore an
unchecked lexical result. Source changes, query changes, model identity and
policy changes invalidate approvals. Scope/lifecycle and existing controller
memory gates remain necessary. Approval does not change verification status or
prove final-answer correctness. The feature is off unless explicitly configured.

Use `gpt-6-luna`, at most one chat request per query, six candidates per request,
4,096 maximum completion tokens, 20-second request timeout and no continuations
or automatic retries. Up to 150 live chat calls across development, repairs and
evaluation are authorized by this bounded experiment. Reuse old embedding vectors
where applicable. New fixture embeddings use `text-embedding-3-small` and are
checkpointed for zero-call replay. Save exact request/policy identities, complete
response checkpoints, usage and source revision. Incomplete replay is an error.

The new fixture has 11 sources and 60 queries. Two development scopes (24 queries)
cover image processing and invoice settlement. Three different evaluation scopes
(36 queries) cover coordinates, parcel tracking and meeting recordings. Relevant
source IDs, scopes and declared topic families are disjoint across splits. The
fixture includes English/Japanese paraphrases, adjacent unsupported facets,
compound questions, explicit absence and conflicting recording formats. Labels
are authored and correlated, not blind independent human judgments.

First measure the existing 0.25 content shortlist and the same shortlist followed
by support checking on development queries. Diagnose remaining misses separately
as candidate absence versus checker rejection. If candidate absence is material,
test widening to at most six available compatible scoped records before checking;
never bypass scope, ambiguity or expiry. Preserve ranking among accepted sources,
but do not prune checked independent facts solely by their score relative to the
top hit. For unscoped data keep the existing candidate policy. Runtime embedding
threshold defaults remain unchanged.

Development findings may drive implementation/prompt repairs, with every measured
revision retained. Freeze the final implementation before inspecting evaluation
responses. Then evaluate the untouched 36-query split once; report residual
failures honestly. The preceding 90-query fixture is development/regression data,
not a fresh held-out proof; its calibration split may be measured as an additional
45-query regression after the design is frozen. Do not tune on new evaluation.

Report source selections before/after the checker, first-source and full-source
recall, query-level false retrieval, irrelevant slots, source-set F1, by-kind
breakdowns, candidate coverage and checked recall, error/failure counts, call and
token usage and latency. Reproduce decisions from complete frozen responses with
zero provider calls. Run controller/actual worker evidence tests and unchanged
historical selectors as offline regressions. No final-answer grading or live
database claim is made by this retrieval experiment.
