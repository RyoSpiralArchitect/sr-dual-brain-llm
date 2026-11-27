# Luna reviewer-only results — 2026-11-27

In a 72-call development screen, OpenAI gpt-6-luna with provider-default
effort identified all 12 planted errors and accepted all 12 correct drafts.
Low and medium effort each missed one exact issue witness and raised a false
issue on one correct draft. All 72 reviews were valid and returned within
the 24-second owner window.

| Effort | Detected / wrong | Exact misses | False ok / wrong | False issue / correct | Unavailable / all |
| --- | ---: | ---: | ---: | ---: | ---: |
| Provider default | 12/12 | 0 | 0/12 | 0/12 | 0/24 |
| Low | 11/12 | 1 | 0/12 | 1/12 | 0/24 |
| Medium | 11/12 | 1 | 0/12 | 1/12 | 0/24 |

The frozen [protocol](critic-reviewer-only-protocol.md) used the native
reviewer prompt and parser with a field-witness instruction and no repair
or fallback. The 12 tasks cover Boolean logic, SQLite, event state, and
evidence authority. Each supplied draft had three answer fields. The bad draft
changed exactly one field, and only one draft per call was visible to the
critic. The three effort arms used identical messages for each draft with a
4,096 completion-token cap, a 24-second deadline, one admission per cell,
and no retries. The three calls for each draft ran concurrently. Each arm
reported the requested model alias; no pinned provider revision or effective
default effort was observed.

## Case review

- **Low, event order, wrong draft.** The issue named the Completed field,
  flagged 2 and gave 1 as the replacement. Its witness syntax did not match
  the frozen format. The exact score stays 11/12. In the separate semantic
  review, low found the planted error in this case too.
- **Medium, Boolean XOR, wrong draft.** The issue targeted the correct Y
  field instead of the planted error in Z. The fixes mentioned that Z is
  false, alongside an incorrect claim that Y is false. The exact score
  records a miss. The review contained the true correction mixed with a
  false diagnosis.
- **Low and medium, Boolean XOR, correct draft.** Both incorrectly read the
  equivalence between A and the exclusive-or of B and C as a global
  constraint that B and C cannot both be true. The assignment A=false,
  B=true, C=true, D=false satisfies the constraints and makes Y true.
  There are also satisfying assignments where Y is false.

All other cells matched the frozen outcome without extra issue strings on
detected wrong drafts. The controlled drafts and field-witness instruction
are part of the experimental conditions. Each cell was sampled once.

## Time and usage

| Effort | Median elapsed | Input tokens | Output tokens | Reasoning tokens included in output |
| --- | ---: | ---: | ---: | ---: |
| Provider default | 1.873 s | 15,358 | 3,381 | 2,415 |
| Low | 1.693 s | 15,358 | 2,916 | 1,938 |
| Medium | 1.819 s | 15,358 | 3,378 | 2,300 |

The longest cell was 3.296 s. All 72 usage receipts were present, totaling
46,074 input and 9,675 output tokens. These times describe one run with
concurrent calls in each batch. Deadline behavior and speed under other load
conditions remain open.

## Verification

The runner, fixture and scoring rule were frozen at
76188fd9228e9c28b780b54cb698e22f050e8456 before the first admission. The fixture
SHA-256 is 05244dd01c21d4c48e5846e422fb7fac96d46e36b876ffbbc7d4c33c35889631.
The frozen score file was recomputed byte for byte from the recorded critic
outputs and terminal decisions with network access blocked. This checks
scoring for the isolated reviewer. The full-turn timing replay failure from
the [previous holdout](critic-cap-holdout-2026-11-14.md) remains recorded there.

- [Per-cell metrics](critic-reviewer-only-2026-11-27.metrics.json), SHA-256
  524f7739135b770d9ef05299475d31f7de439f5bf02708f36f9f23367b3d17be
- Private journal SHA-256
  d8765c2c25d73bf586ea3d88dbf28ba21201a2c58754b6ced77b6246339bce0f

This screen supports retaining the existing reviewer setting for these
conditions. An end-to-end experiment would have to show that accurate
critic feedback actually repairs wrong answers without degrading correct
answers. The present study observed reviews only.
