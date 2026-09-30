# System2 paired answer evaluation protocol

This is a design for a future quality study. The 2026-09-27 Luna run is one
fixed-order exploratory pass; its critic issue counts and latency are not an
independent quality result. No provider run is part of this protocol's
preparation.

## Freeze before running

1. Commit the code and run both blocks from the same clean commit. Freeze the
   question file, selected IDs, mode settings, provider/model, token and timeout
   limits, critic health gate, and scoring rubric. Keep both hemispheres'
   effective settings identical between blocks.
2. Use the same question order in each mode and block. Run one block with
   `--modes off,on`, then one with `--modes on,off`. Do not select the order
   after inspecting answers. The order reversal reduces a single fixed-order
   confound but does not eliminate time or provider drift.
3. Request `--include-full-answers` for both reports. The flag is opt-in because
   complete answers may contain sensitive text. Save reports locally outside
   the repository and do not commit them by default. Use `--history ''` if no
   local history row is wanted.
4. Retain all attempted cases, errors, health outcomes, and missing answers in
   the run reports. Do not repair a failed denominator by dropping cases after
   looking at outcomes. A scored comparison needs complete off/on answers for
   every frozen case in both blocks; otherwise report the failure separately.

The comparative report automatically includes the clean source revision,
question file SHA-256 values, a hash of the effective filtered question list,
actual loaded model settings, mode order, shuffle seed, and answer-retention
choice. Secrets such as API keys and custom endpoint URLs are excluded.

## Prepare scoring material offline

After the two reports exist, run:

```bash
python3 sr-dual-brain-llm/scripts/prepare_system2_scoring.py \
  --ab /local/path/off-on.json \
  --ba /local/path/on-off.json \
  --output-dir /local/path/scoring-packets \
  --seed 7
```

The tool makes `blind_packets.json` and `reveal_key.json` in a new local
directory with owner-only file permissions. It rejects changed code, question
hashes, model settings, incomplete answers, errors, duplicate IDs, and mode
overrides. Keep the reveal key away from scorers until ratings are locked.
The reveal key also retains each case's recorded System2 activation state, which
may be unknown. The tool only pairs answers; it does not score them or run a
provider.

## Score and report

- For each blinded answer, rate **correctness** (0 incorrect, 1 partly correct,
  2 correct), **completeness** (0 misses core request, 1 partial, 2 complete),
  and **unsupported claims** (0 none, 1 minor, 2 material). Record ties and
  uncertainty explicitly. Use the same rubric for English and Japanese items.
- Have at least two raters score independently, then adjudicate disagreements
  without seeing mode labels. Save the original ratings as well as adjudication.
- Lock ratings before opening the reveal key. Compare off/on within each case
  and block; report wins, losses, ties, missing/errored cases, and score
  differences with the full denominator. Show block-specific results and
  activation state before any aggregate. Treat critic issue counts as a
  separate diagnostic channel.
- Report latency by block and mode as observed cost. Do not infer a speed
  advantage from a single sequence or from answers of unequal completeness.

For a small fixed suite, case-level results and disagreement counts are more
informative than a lone summary score. A later study can predefine additional
repetitions and uncertainty intervals before collecting new provider outputs.
