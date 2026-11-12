# Historical SSGT OOS trial manifests

`wut/exp01.json` through `exp05.json` and `kgbd/exp01.json` through `exp05.json` are explicit, portable manifests for the five archived OOS trials per dataset. Their internal trial IDs remain `exp_01` through `exp_05`.

Each manifest includes original identity labels, known/unknown membership, per-sequence split assignments, relative raw filenames, file sizes and SHA-256 fingerprints, expected prepared-window counts, and the matching checkpoint hash. Historical evidence contains filenames and digests, never a user-machine absolute path. Raw data and checkpoint binaries are not embedded.

These fixed identity and sequence assignments define the historical data protocol.
They remain unchanged when training and evaluation generate new run seeds.
Manifests can prepare data for fresh training; they do not fix a new model's seed.

Supply the full original identity root to `protocols.py --raw-root`. The source filenames are resolved below that root. Outputs belong in a separate directory passed as `--prepared-root`; identity directories are never moved.

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/full/WUT --prepared-root reproduction_runs/wut/exp01/data
```

See [Paper reproduction](../../docs/PAPER_REPRODUCTION.md) for evaluation, five-trial aggregation, and the distinction between archived checkpoint settings and the published training rule. See [Data protocol](../../docs/DATA.md) for coordinate formats and historical sequence-count/coverage details.
