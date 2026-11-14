# Critic-cap recovery: eight-case holdout (2026-11-14)

**Winner: none.** The fresh holdout was frozen at
`06a8f4fbbac1ed389db608fa7192424b1fe23ef7` after the
[development gate](critic-cap-development-2026-11-14.md). Its
[protocol](critic-cap-holdout-protocol.md) and
[manifest](critic-cap-holdout-manifest.json) fixed eight unused cases,
three arms, exact graders, caps, order and the winner rule before the first
provider response. The requested model was `gpt-6-luna` through the
OpenAI-compatible provider. All **24/24 cells completed**.

| Arm | Exact | Complete | Mean item accuracy | Workflow valid | Valid external reviews | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default_off` baseline | 6/8 | 6/8 | 0.750 | 2/8 | 4/8 | 19 |
| `low_aware` (critic 2048) | 3/8 | 8/8 | 0.785 | 2/8 | 2/8 | 21 |
| `low_critic4096` | 6/8 | 8/8 | 0.844 | 2/8 | 2/8 | 19 |

Both Boolean tasks were inexact for all arms. The default arm returned no
complete answer on either; the two low-effort arms returned complete but
inexact answers. On the six SQL, event and evidence cases, baseline and
`low_critic4096` were exact on every case. `low_aware` lost both SQL cases
and one event case. The higher mean item score for `low_critic4096` comes
from partial correctness on one Boolean case, not an extra exact answer.

The precommitted held-out signal gate required at least **two more exact
cases** than baseline. The candidate tied baseline at 6/8, so the gate
failed even though it had 8/8 complete answers and a higher item mean.
The larger critic cap also produced **no increase** in valid external
reviews or workflow-valid cases over `low_aware` (both 2/8). On the SQL
case where `low_critic4096` beat `low_aware`, its initial answer was already
exact; no critic semantic repair explains that difference. Independent
provider samples prevent a causal claim about the cap. There is no basis to
change product defaults or claim baseline superiority.

The journal records 59 admissions, 118 transport events and four denied
attempts at the per-trial output ceiling. Eight admitted requests were
owner-cancelled; one local `ReviewBudgetExhausted` had no usage receipt.
The known subset used 132,129 input and 69,524 output tokens. Nine calls
lack usage, so complete token totals and cost are **unknown**.

One network-forbidden replay consumed the full recorded sequence. It
reproduced the 24 semantic grades, 59 request admissions, 118 transport
events, usage, cap receipts and denied attempts. **Exact replay failed**
on `events_b:low_aware`: at a review deadline boundary, live recorded a
failed owner timeout while replay recorded a returned external review with
`deadline_exhausted`. The row diagnostics and canonical digest differ;
this is a timing-sensitive harness limitation, not evidence of a different
semantic score. No second replay or tape adjustment was made. The live
canonical digest is
`a8eb87bc3801c510433cd3edc38d4964bfcf6bbbfcd0308d323f25e3e918f7e3`;
the failed replay digest is
`0b1f198335bac1f53c50be883d9e314fb5c9a915cb0ccb9cf1b5cc47b2213774`.

The next mechanism study should separate the 24-second critic deadline
from critic reasoning effort while holding the token cap fixed. It must
score **whether a critic detects known wrong candidates**, not just whether
it emits JSON. Any new live study needs a fresh fixture, frozen source and
another bounded admission plan; these results cannot be retuned or retried.

Private local receipts (provider payloads and outputs are not published):

| Artifact | SHA-256 |
| --- | --- |
| live `report.json` | `de8d5bbfab7a36c76f6dd2ead5ca0852523a0e5b767ce89eb6876b65cd29d06b` |
| live `journal.json` | `1ff22ce00c91b723debf18c815d6dbf7a2195844e58298856bb39d86ad551020` |
| failed replay `report.partial.json` | `1e4ce5867d2a373d85b6fe0aefda3561b5c5f37f99f58c3eb5e6685f6d98c541` |
| failed replay `failure.json` | `3c1222d95add4c543c9507d17c3fbe7fbc30fdae4a6f8b0ff36887b1626cd0cf` |
