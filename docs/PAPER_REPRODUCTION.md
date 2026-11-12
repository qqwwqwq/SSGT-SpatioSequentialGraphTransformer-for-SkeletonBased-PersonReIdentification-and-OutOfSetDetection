# Reproducing the final SSGT method

This release implements the final Spatio-Sequential Graph Transformer with dynamic loss balancing and k-nearest-neighbor out-of-set detection described in [He et al., IEEE Robotics and Automation Letters, 2026](https://doi.org/10.1109/LRA.2026.3734851). The paper's main SSGT row corresponds to the `SSGT + DLoss` configuration in its component analysis. Readers can prepare the raw datasets, train the final method, and evaluate their own checkpoints without archived weights.

`config/paper_targets.json` records the publication identifier, the supplied PDF fingerprint, and reference values for Tables II–IV and Figure 5. Those values are reference targets. Newly measured results are written by the evaluation tools; their comparison with the references is reported separately in [VALIDATION.md](../VALIDATION.md).

## Optional environment for the measured GPU runs

The default installation target is the modern environment in `config/environment.yml`. The separate [environment-reproduction.yml](../config/environment-reproduction.yml) records the package versions and numerical-library builds used for the measured GPU runs on Linux x86_64:

| Component | Measured version |
| --- | --- |
| Python | 3.7.13 |
| PyTorch runtime / CUDA / cuDNN | 1.13.1+cu117 / 11.7 / 8.5.0 |
| NumPy / MKL package | 1.19.2 / 2020.2 |
| SciPy / its wheel's OpenBLAS | 1.4.1 / 0.3.7 |
| scikit-learn | 0.24.2 |
| Matplotlib / threadpoolctl | 3.5.3 / 3.1.0 |

```bash
conda env create -f config/environment-reproduction.yml
conda activate ssgt-reproduction
python -c "import torch, numpy, scipy, sklearn, threadpoolctl; print(torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, sklearn.__version__, threadpoolctl.__version__)"
python -m unittest discover -s tests -v
```

The recipe preserves the measured conda MKL build of NumPy and the pip wheels of SciPy and PyTorch. The installed PyTorch distribution metadata say `1.13.1`, while its runtime reports `1.13.1+cu117`; the accompanying NVIDIA runtime packages are pinned explicitly. PyTorch also documents the separately suffixed CUDA 11.7 wheel in its [official historical installation instructions](https://pytorch.org/get-started/previous-versions/#v1131). A compatible NVIDIA host driver is required and is outside this environment file. CPU-library thread counts are fixed to four, as in the measured training commands.

This is a recipe reconstructed from installed metadata, **not a freshly created environment that has been tested**. The existing environment ran the reported checks. Package availability, host libraries, GPU hardware, and numerical kernels can still differ, so this recipe does not guarantee identical training results. Use it as an alternative to the modern requirements; installing `requirements.txt` afterward requests newer NumPy, SciPy, and scikit-learn versions and changes this environment.

## Published settings and training presets

| Component | Paper setting | Release option |
| --- | --- | --- |
| Skeleton joints | WUT 19; KGBD 20; KS20 25 | Selected by `dataset` |
| Sequence length | 6, 12, or 18 frames | `length` |
| Embedding dimension | 128 | `H` |
| Transformer layers / heads | 2 / 8 in both spatial and temporal branches | `L_transformer`, `n_heads` |
| Positional encoding | Laplacian; 10 spatial and 4 temporal eigenvectors | `spa_pos_type`, `enc_k_spa`, `enc_k_time` |
| Sequence / skeleton loss balance | α = 0.5, β = 0.5 | `seq_lambda`, `prompt_lambda` |
| Initial GPC weight | λ = 0.5 | `GPC_lambda` |
| Dynamic weighting | initial Up = 10, step τ = 0.05, threshold ι = 10 | `gpc_up_period`, `gpc_tau`, `gpc_iota` |
| Contrastive temperatures | τ1 = 0.07, τ2 = 14 | `t_1`, `t_2` |
| Optimizer / learning rate / batch size | Adam / 0.00035 / 256 | `lr`, `batch_size` |
| Sequence embedding | Average pooling across joints and time | `frame_attn_pool=0` |
| Main OOS rule | K = 1; P = 95; L2-normalized embeddings | `unk_knn_k=1`, `unk_sigma=0.95` |

The Algorithm 2 implementation updates λ from the **signed** ratio of the recent GPC and STPR loss slopes. It adds τ below 1/ι, subtracts τ above ι, and otherwise leaves λ unchanged. The epoch threshold doubles after each update, giving updates at epochs 10, 20, 40, 80, and so forth. λ is limited to its stated domain [0, 1]. The paper does not specify the finite-difference formula for the normalized slope; `loss_balancing.py` retains the research implementation's central normalized difference and documents the near-zero-denominator handling.

The paper does not specify an exact numerical random seed, dropout rates, mask counts, batch-normalization options, or training epoch limit. The release records its implementation defaults explicitly. The current defaults generate a new run seed and retain the stopping controls as operational settings, not values asserted to appear in Table I. The seed-42 measurements in [VALIDATION.md](../VALIDATION.md) came from the previous fixed-seed version; the full benchmark has not been repeated after changing the default seed policy.

Run from the release directory, after preparing the data described in [DATA.md](DATA.md):

```bash
python main.py --config config/paper/wut_5c_f6.json --task train
python main.py --config config/paper/kgbd_f6.json --task train
python main.py --config config/paper/ks20_f6.json --task train

python main.py --config config/paper/wut_5c_f6.json --task evaluate \
  --ckpt /path/to/the/matching/best.ckpt
```

Additional presets under `config/paper/` cover the paper's sequence lengths, WUT camera configurations, and frame-rate variants. A camera preset expects the corresponding fused skeleton files; it does not perform camera calibration or reconstruct a missing camera combination. Changing the input sequence selection, camera fusion, or frame rate changes the protocol.

For KS20, the raw MAT converter uses the recovered per-identity settings in `config/ks20_split.json`. The complete collection has 100 sequences per split across all five views. The converter preserves trim-10 processing and shifted half-stride training augmentation; at N=6 it reproduces 1,886/212/218 train/gallery/probe windows before resampling. Run `python main.py --task preprocess --dataset KS20 --input_dir /path/to/KS20 --length 6`, then train with `config/paper/ks20_f6.json`. See [KS20 data preparation](DATA.md#ks20-inputs-and-split) for identity mapping, zero-window recordings, and provenance.

## Random seeds and repeatable runs

Because the source code now uses randomly generated seeds by default, results may differ slightly from those reported in the paper.

This is not a bound on the difference. Initialization, masks, balanced sampling,
KGBD sequence splitting, numerical kernels, and data preparation can affect the
scores. Each stochastic entry point selects an unsigned 32-bit seed from
operating-system entropy, prints it, and records it in its output metadata.

| Stage | Default | Explicit replay option |
| --- | --- | --- |
| Training / closed-set evaluation | `--seed_val -1 --fix_seed 0` | `--seed_val N`; optionally `--fix_seed 1` for deterministic cuDNN |
| Unified WUT/KGBD/KS20 preprocessing | `--seed -1` | `--seed N`; only KGBD uses a random sequence split |
| OOS evaluation | `--seed -1` | `--seed N` |
| Archived OOS verification | `--seed -1` | `--seed N` |

Use the recorded integer in place of `N`. `--fix_seed 1` controls cuDNN
settings and does not choose a numerical seed. Keep each stage's seed,
configuration, data hashes, and software version when repeating an experiment.
Training checkpoints store the resolved `args.seed_val`, `requested_seed_val`,
`seed_selection`, and `paper_protocol.seed_control`. Preprocessing writes
`preprocessing_metadata.json` with `requested_seed`, `resolved_seed`, and
`seed_selection`. OOS result records include a `randomness` object with the
resolved seed, requested seed, and selection mode.
The loader uses an independent sampler initialized from the selected run seed,
so repeated loads with the same explicit seed retain consistent sampling. It
no longer resets the application's global NumPy state to a fixed value. A seed
alone does not guarantee identical scores across versions and devices.

WUT/KS20 split JSON and the ten historical OOS manifests are protocol definitions,
not random-seed settings. Their identity and sequence assignments stay fixed.
The original validation records are retained unchanged, including their seed-42
settings; they are evidence for the previous fixed-seed version. No full
random-default training benchmark is claimed here.

## Train and evaluate your own OOS model

Use one explicit identity split for both training and OOS evaluation. The released manifests can prepare that split for a new model without access to an archived checkpoint:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/full/WUT --prepared-root reproduction_runs/wut/exp01/data

python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir reproduction_runs/wut/exp01/data \
  --model-dir reproduction_runs/wut/exp01/weights

python main.py --task oos --data-dir reproduction_runs/wut/exp01/data \
  --ckpt reproduction_runs/wut/exp01/weights/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt \
  --data_source unknown --proto_source gallery --unk_knn_k 1 --unk_sigma 0.95
```

Use `--task evaluate` with that same checkpoint and data root for closed-set gallery/probe metrics. For KGBD, prepare a KGBD manifest, train with `config/paper/kgbd_f6.json`, and use `--task oos-kgbd`. Repeat independently for each identity split; do not reuse a model trained on held-out people. Fresh-run scores are measured results and may differ from the paper or archived trials.

The ordinary `main.py` workflows accept the checkpoint you trained. The stricter `reproduce.py evaluate` commands below are optional tools for verifying a particular archived checkpoint; their historical checkpoint-hash checks do not apply to the self-training recipe.

## Archived weights and the published training rule

The archived checkpoint metadata show training settings that differ from Table I and Algorithm 2. These differences are preserved as evidence instead of being hidden by relabeling the weights:

| Item | Published rule used for new training | Archived research implementation |
| --- | --- | --- |
| Slope-ratio threshold ι | 10 | Common archived value 1.2 |
| λ increment | Fixed ±0.05 | Smooth `tanh`-scaled increment |
| Update schedule | Doubling epoch threshold | Repeated fixed update interval |
| λ domain | [0, 1] | Additional bounds [0.25, 0.75] |
| Convergence adjustments | No separate adjustment in Algorithm 2 | Additional branch-convergence and learning-rate actions |
| Main OOS neighbors | K = 1 | Old standalone defaults 200/500; the recorded trial commands explicitly used K = 1 |
| OOS prototype count | One prototype per known training label: 23 for WUT or 148 for KGBD | Full dataset capacity of 25/164, leaving unused zero prototypes in the contrastive denominator |

Checkpoint evaluation restores the saved model architecture and parameters. It does not retrain the checkpoint or replace its learned weights with a model trained under revised settings. Reproducing an archived trial measures the released inference and data reconstruction against that trial; it is not evidence that a fresh run under the published loss schedule will reproduce the same weights or score exactly. Training and evaluation metadata distinguish the current loss schedule from the checkpoint's recorded settings.

New OOS training intentionally corrects the archived prototype-count behavior: the number of one-hot training-label columns determines the prototype count, and every known identity must contribute samples to the complete batches used for prototypes. This avoids empty-class averages and extra zero prototypes, consistent with the observed-class prototype in Equation 3. The loader's final tuple field still reports legacy dataset capacity for compatibility. Closed-set runs with all 25/164/20 identities keep the same prototype count, and evaluation of saved weights is unchanged.

## Inherited implementation conventions

The formulas do not uniquely specify every part of the research implementation.
The release retains the following conventions rather than silently rewriting the
encoder or loss to fit a target score:

- Training sequence embeddings and prototypes average the retained frames from
  the sampled temporal mask (two masked frames by default). Inference averages
  all frames. Equation 2 describes the unmasked average.
- Non-KGBD training windows undergo the inherited class-balanced resampling.
  Prototype construction consumes full batches from that sampled set; it does
  not necessarily include every original window exactly once.
- The implementation weights spatial reconstruction by `prompt_lambda` and
  temporal reconstruction by its complement. Equation 5 places beta on the
  temporal term. The final setting 0.5 gives identical weighting.
- The encoder stacks attention/fusion blocks and then applies a shared residual,
  feed-forward, and normalization stage. The paper does not unambiguously assign
  a separate such stage to every attention layer.
- Input projection, attention clamping, dropout, mask placement, and normalization
  follow the inherited implementation. Their defaults are exposed in the CLI and
  saved in checkpoints; they are not all specified by the paper's equations.
- Checkpoints are selected by probe Rank-1, with mAP/Rank-5/Rank-10 tie breakers.
  The inherited workflow has no separate validation identity split. Reported
  selected-checkpoint scores should be interpreted with that selection protocol.

New checkpoint files include the spatial Laplacian eigenbasis used during
training. Evaluation prefers this saved basis and records its source and hash.
Older files without the field recompute it from the graph; such files do not
fully pin the numerical positional encoding across dependency versions.

## Optional verification of historical OOS trials

The ten portable manifests under `config/trials/` describe the five WUT and five KGBD trials used by the archived OOS experiments. They contain no machine-specific absolute paths and require no access to the original experiment logs for preparation.

Each trial has its own held-out identities and its own checkpoint. WUT holds out two of 25 people; KGBD holds out sixteen of 164. Held-out identities are excluded from both training and gallery. The manifest retains every original identity label even though prepared unknown samples share label 26 for WUT or 165 for KGBD.

A public manifest pins exact source filenames, byte counts, and SHA-256 values. A file with equivalent-looking coordinates but different bytes is not silently accepted. Data preparation and fresh training need only the raw data. To verify an archived experiment with `reproduce.py evaluate`, obtain its checkpoint separately; that checkpoint must match `checkpoint_sha256` in the trial's manifest.

Prepare a WUT trial without moving any source directories:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/full/WUT \
  --prepared-root reproduction_runs/wut/exp01/data
```

For KGBD:

```bash
python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root '/path/to/full/kinect gait raw dataset' \
  --prepared-root reproduction_runs/kgbd/exp01/data
```

The output is a dedicated data root with `<dataset>/6/` and `provenance.json`. Preparation verifies input fingerprints and recorded window counts before writing arrays. Provenance includes the canonical manifest hash, hashes for all prepared files, original known/unknown labels, per-sequence window counts, and missing-known-identity coverage.

The WUT manifests preserve all 293 original sequences rather than truncating each identity to ten. They also preserve the recorded absence of valid Julia probe windows. See [the data protocol](DATA.md#wut-inputs-and-split) for these concrete differences between the aggregate archive and the paper's simplified split description.

## Evaluate one trial

```bash
python reproduce.py evaluate \
  --manifest config/trials/wut/exp01.json \
  --data-root reproduction_runs/wut/exp01/data \
  --checkpoint /path/to/wut_trial_01/best.ckpt \
  --output reproduction_runs/wut/exp01/result.json \
  --gpu 0 --batch-size 256 --k-values 1,5,20,50,200 --sigma 0.95
```

Use the corresponding KGBD manifest, data root, and checkpoint for KGBD. `--gpu cpu` selects CPU inference. `--distance-batch-size` and `--distance-threads` control bounded-memory distance computation.

The archived measurements used the previous fixed-seed implementation with seed 42. To request that numerical seed explicitly, add `--seed 42`; this is a historical-settings reference, not a guarantee of identical scores across execution settings or versions. The result records the selected sampling seed. For new runs, the default generates a seed automatically.

Keep `--batch-size 256` for the historical protocol. This option also enters the inherited WUT class-balanced training sampler; changing it changes the calibration library, not just the inference throughput. The evaluator encodes the selected training and unknown windows once, then reuses their embeddings for all requested K values.

The inherited score convention calibrates on the training library and excludes the first self-distance when selecting each training sample's K-th neighbor. Duplicate windows created by balanced sampling remain in the library. The unknown queries use their K-th training-library neighbor. Both feature sets are L2-normalized. The in-set scores supplied to AUROC/FPR95 are these training-library calibration scores. The 95th percentile of the calibration scores defines the acceptance threshold, and the saved result includes AUROC, FPR95, effective neighbor ranks, sample counts, runtime versions, and provenance hashes. These calibration conventions must remain unchanged when comparing with the archived metrics.

## Aggregate five trials

Repeat preparation and evaluation with `exp02.json` through `exp05.json`, using the checkpoint bound to each manifest. Aggregate exactly five distinct trial result files for one dataset:

```bash
python reproduce.py aggregate \
  reproduction_runs/wut/exp01/result.json \
  reproduction_runs/wut/exp02/result.json \
  reproduction_runs/wut/exp03/result.json \
  reproduction_runs/wut/exp04/result.json \
  reproduction_runs/wut/exp05/result.json \
  --paper-targets config/paper_targets.json \
  --output reproduction_runs/wut/summary.json
```

Run a separate aggregation for KGBD. The tool checks distinct trial IDs and checkpoint hashes, consistent sequence length, quantile, batch size, and K values. It reports means and sample standard deviations (`ddof=1`) in percent and differences from the recorded paper targets in percentage points. It never substitutes target values for measured results.

## Manifest recovery and Python API

Users with original experiment evidence can recover another explicit manifest:

```bash
python protocols.py --dataset KGBD --trial exp_01 \
  --group-root /path/to/recorded_experiment_group \
  --raw-root '/path/to/full/kinect gait raw dataset' \
  --manifest custom_trials/kgbd_exp01.json
```

Recovery reads the known sequence selections from `logs/exp_01_preprocess.log` and the unknown identity list from `summary.json`. If present, it also records checkpoint and metric-reference hashes and window counts from training/evaluation logs. The recorded logs contain positions in sorted sequence lists, not historic file hashes: recovered SHA-256 values pin the files available during recovery. Matching historical scores is a separate validation step.

```python
from pathlib import Path
from protocols import load_manifest, prepare_trial, verify_provenance

manifest = load_manifest("config/trials/wut/exp01.json")
provenance = prepare_trial(
    manifest,
    raw_root=Path("/path/to/full/WUT"),
    prepared_root=Path("reproduction_runs/wut/exp01/data"),
)
verify_provenance(manifest, provenance["data_root"])
```

`recover_historical_manifest(dataset, group_root, trial_id, raw_root)` returns a manifest dictionary. `validate_manifest(manifest)` checks identity and sequence uniqueness, known/unknown separation, relative paths, fingerprints, and declared counts. `manifest_digest(manifest)` hashes canonical JSON using sorted keys, compact separators, ASCII escaping, and SHA-256. Renaming a manifest file or changing indentation leaves that digest unchanged; changing its actual content invalidates its prepared-data binding.

For a new experimental split, create a version-one manifest with `dataset`, `length`, `data_subdir`, and an `identities` list. Each identity needs `name`, `original_label`, `role`, and explicit `sequences`; each sequence needs `name`, `split`, relative `source`, `format`, `size_bytes`, and `sha256`. Known identities require train, gallery, and probe assignments. Unknown identities may appear only in the unknown split. Include `checkpoint_sha256` only for the model actually trained with those known identities. A newly designed split is a new experiment and should not retain historical expected counts or be reported as one of the published five trials.
