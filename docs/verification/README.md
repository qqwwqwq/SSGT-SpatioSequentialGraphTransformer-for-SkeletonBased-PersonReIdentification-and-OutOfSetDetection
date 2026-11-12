# Measured validation records

This directory separates recorded CPU regression evidence from the earlier
training and GPU measurements. Published reference values remain in
`config/paper_targets.json`.

The current [Python 3.10](test-suite-temporal-encoding-python310.json) and
[Python 3.7](test-suite-temporal-encoding-python37.json) records report the
**81-test suite after the temporal-eigenbasis determinism fix**. Both passed
locally on CPU on 2026-11-13 (Asia/Tokyo), in 25.55 and 15.98 seconds respectively,
with zero failures or errors. The checks include repeated temporal-eigenbasis
construction, strict explicit-seed replay, and synthetic CLI training/evaluation.
The records include exact source/configuration hashes and do not claim a full
dataset training rerun or a remote CI result.

The earlier [Python 3.10](test-suite-random-seed-python310.json) and
[Python 3.7](test-suite-random-seed-python37.json) reports retain the **80-test
suite before that fix**, which passed locally in 24.38 and 14.18 seconds.
Their original source hashes and scope remain unchanged; they are historical
regression evidence rather than results for the corrected implementation.

The training, GPU evaluation, and retrieval measurements listed below were
collected on 2026-11-12 with the previous fixed-seed version. Source defaults now
generate run seeds, and new models use a deterministic temporal eigensolver.
The full training benchmark has not been repeated after either change. Saved
checkpoints load their recorded `time_pe` values; the reported metrics remain
evidence for those weights. These earlier records retain their original seed
settings, scores, and checksums and must not be relabeled as benchmarks of the
updated training implementation.

- `wut_exp_01.json` through `wut_exp_05.json` and the corresponding `kgbd` files
  retain the ten measured OOS trials, hashes, settings, and runtime versions.
- `wut_oos_summary.json` and `kgbd_oos_summary.json` aggregate five trials each.
  Trial filenames are made explicit here; each listed SHA-256 hashes the exact
  included trial record. Metrics are unchanged from the local aggregate.
- `*_historical_checkpoint.json` record four closed-set reevaluations. Only
  machine-specific path fields were removed. `original_record_sha256` identifies
  the full local record; measured scores and artifact fingerprints are unchanged.

- `*_fresh_training.json` and `*_training_comparison.json` report new training
  followed by independent selected-checkpoint evaluation. These do not reuse
  archived model weights. A comparison's `evaluation_sha256` identifies the
  original local evaluation record and matches the evaluation copy's
  `original_record_sha256`; it does not hash the sanitized distributed copy.
- `wut_oos_fresh_trial.json` and `wut_oos_fresh_verdict.json` report one new
  held-out-identity training run and its actual public-CLI OOS measurements.
  They do not claim a five-trial mean.
- `retrieval_benchmark.json` records the CPU aggregation speed comparison on real
  KGBD embeddings. `kgbd_retrieval_parity.json` verifies full-data metric equality.

Historical checkpoint metadata are included as evidence, not accepted as a
substitute for fresh evaluation. No private conversation, raw dataset,
checkpoint binary, or local experiment log is distributed here.

Read [VALIDATION.md](../../VALIDATION.md) for the measured agreement and unresolved
differences, and [PAPER_REPRODUCTION.md](../PAPER_REPRODUCTION.md) for commands.
