# SSGT: Spatio-Sequential Graph Transformer for Skeleton-Based Person Re-Identification and Out-of-Set Detection

**Authors:** Xin He, Vibekananda Dutta, Teresa Zielinska, Takafumi Matsumaru, and Robert Sitnik.

![Python](https://img.shields.io/badge/Default_Python-3.10_%7C_3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Tests](https://github.com/qqwwqwq/SSGT/actions/workflows/tests.yml/badge.svg)](https://github.com/qqwwqwq/SSGT/actions/workflows/tests.yml)

## Paper

*IEEE Robotics and Automation Letters*, 11(11), 12727–12734, 2026.
[Paper](https://doi.org/10.1109/LRA.2026.3734851) · [Reproduction guide](docs/PAPER_REPRODUCTION.md) · [Data protocol](docs/DATA.md) · [Measured validation](VALIDATION.md)

## Table of contents

- [Introduction](#introduction)
- [Key features](#key-features)
- [Dataset downloads](#dataset-downloads)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Final model settings](#final-model-settings)
- [Note on reproducibility](#note-on-reproducibility)
- [Measured training checks](#measured-training-checks)
- [Prepare and train](#prepare-and-train)
- [Closed-set identification](#closed-set-identification)
- [Out-of-set inference](#out-of-set-inference)
- [Train an OOS model from scratch](#train-an-oos-model-from-scratch)
- [Configuration, testing, and packaging](#configuration-testing-and-packaging)
- [Limitations](#limitations)
- [API and code conventions](#api-and-code-conventions)
- [Citation and license](#citation-and-license)

## Introduction

This repository contains the final SSGT method: parallel spatial and temporal
attention with Laplacian positional encodings, prototype contrastive learning,
masked reconstruction, dynamic loss balancing, and kNN out-of-set detection.
The published final method corresponds to **SSGT + DLoss** in Table III.
No comparison-method implementations or baseline experiment scripts are bundled.
All source documentation, comments, and command-line help are in English.

The organization and documentation conventions follow the
[DualSOM reference](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-): a directly
accessible `main.py`, separate algorithm modules, JSON configuration, and
`Datas/` and `weight/` directories. SSGT also keeps generated outputs in `results/`.
The README follows the reference's paper, quick-start, data, configuration,
results, reproducibility, and API sections; [the style guide](docs/CODE_STYLE.md)
records the concrete correspondence.

## Key features

- Spatial and temporal skeleton attention with Laplacian positional encodings.
- Final SSGT training with prototype contrastive learning, masked reconstruction,
  and the published dynamic loss rule.
- Raw-data preparation for WUT, KGBD, and KS20, with explicit split definitions.
- Closed-set identification and kNN OOS inference from your own checkpoints.
- JSON presets, recorded run seeds, data provenance, and source-only packaging.

## Dataset downloads

Dataset packages are separate from the source archives. Download the raw
skeleton data using the links below, then preprocess them locally.

| Package | Intended use | Download |
| --- | --- | --- |
| `SSGT-WUT-raw.zip` | WUT skeleton inputs for preprocessing and explicit OOS splits | [Google Drive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) |
| `SSGT-KGBD-raw.zip` | KGBD skeleton inputs for preprocessing and explicit OOS splits | [Google Drive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) |

[Local archive checksums](config/dataset_SHA256SUMS) ·
[Package inventory](config/dataset_downloads.json)

Only WUT and KGBD raw datasets are included in the dataset delivery. Run
preprocessing locally before training; no prepared-data package is distributed.
The raw packages support the sequence lengths, frame-rate variants, and explicit
OOS splits described below. Training generates a new seed by default.

KS20 is not redistributed because its original dataset notice requires permission
for distribution. The KS20 converter, protocol, and configuration remain available
for users who obtain the dataset independently.

See [download and extraction instructions](docs/DOWNLOADS.md) for checksums,
paths on a new computer, and dataset access conditions.

## Quick start

Run from this directory with Python 3.10 or 3.11 and your WUT skeleton data
available locally. Choose a PyTorch installation suitable for your hardware;
see [Installation](#installation) for the supported environment choices.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --task preprocess --dataset WUT --length 6 --input_dir /path/to/WUT
python main.py --config params.json --task train
```

Raw data are separate from the source package; train checkpoints locally. Replace `/path/to/WUT`
with your dataset root. Training time depends on the dataset and hardware.

## Repository layout

```text
SSGT/
├── main.py                # Training, closed-set evaluation, preprocessing, OOS inference
├── SSGT.py                # Spatial and temporal graph encoder
├── training.py            # Training, reconstruction/contrastive objectives, ReID metrics
├── loss_balancing.py      # Published Algorithm 2
├── retrieval.py           # Exact mAP/CMC aggregation with shared CPU sorting
├── evaluation.py          # OOS inference; WUT defaults
├── evaluation_kgbd.py     # KGBD OOS inference
├── visualization.py       # Diagnostic projections and plots
├── preprocessing.py       # Raw-data conversion entry point
├── ks20_preprocessing.py  # KS20 MATLAB parsing and recovered sequence split
├── data.py                # Skeleton loading and graph construction
├── protocols.py           # Explicit identity/sequence splits and data integrity
├── randomness.py          # Generated run seeds and optional replay controls
├── reproduce.py           # Verified OOS trials and five-trial aggregation
├── params.json            # Final WUT 5C, N=6 configuration
├── config/
│   ├── paper/             # Final SSGT settings for Tables II–IV
│   ├── paper_targets.json # Published reference values, separate from measurements
│   └── trials/            # Recovered, explicit OOS trial manifests
├── Datas/                 # Prepared datasets generated locally
├── weight/                # Trained checkpoints generated locally
├── results/               # Generated measurements and figures
├── docs/                  # Data, reproduction, source mapping, and code conventions
├── tests/                 # Synthetic regression and integration tests
└── tools/build_release.py # Source archives from an explicit file manifest
```

## Installation

For a new environment, use Python 3.10 or 3.11 and install a PyTorch build
compatible with your hardware before the remaining requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --help
python -m unittest discover -s tests -v
```

`config/environment.yml` provides the same installation target. The actual GPU
checkpoint reproduction used the existing archival environment: Python 3.7.13,
PyTorch 1.13.1+cu117, NumPy 1.19.2, SciPy 1.4.1, scikit-learn 0.24.2, and an
NVIDIA RTX A6000. Modern-environment CPU checks and archival GPU measurements are
reported separately in [VALIDATION.md](VALIDATION.md); dependency changes can alter
numerical results. An optional [reproduction environment](config/environment-reproduction.yml)
records the measured package builds; see the [environment guide](docs/PAPER_REPRODUCTION.md#optional-environment-for-the-measured-gpu-runs)
for its scope and installation status.

## Final model settings

| Parameter | Published value |
| --- | --- |
| Sequence length | 6, 12, or 18 frames |
| Joint count | KGBD 20; KS20 25; WUT 19 |
| Embedding / layers / attention heads | 128 / 2 / 8 |
| Spatial / temporal Laplacian dimensions | 10 / 4 |
| Contrastive / reconstruction branch weights (α / β) | 0.5 / 0.5 |
| Initial dynamic weight / period / step / ratio threshold | 0.5 / 10 / 0.05 / 10 |
| Contrastive temperatures | 0.07 / 14 |
| Optimizer / learning rate / batch size | Adam / 0.00035 / 256 |
| Main OOS setting | K=1; known-library threshold percentile=95 |

Algorithm 2 uses fixed weight increments and doubles the update threshold
(10, 20, 40, 80, ... completed epochs). The finite-difference estimator, handling
of zero slopes, and clipping the weight to `[0, 1]` are documented implementation
conventions. Exact seeds, dropout, masks, batch-normalization options, and stopping
limits are not specified in the paper; the configuration makes those choices
explicit instead of presenting them as published facts.

## Note on reproducibility

Because the source code now uses randomly generated seeds by default, results may differ slightly from those reported in the paper.

The size of any difference is not guaranteed: data splits, random sampling,
initialization, numerical libraries, and hardware can also affect results.
Training defaults to `seed_val=-1` and `fix_seed=0`: a new unsigned 32-bit seed is
chosen from operating-system entropy, printed, and saved with the run.
`--seed_val N` selects a seed explicitly; `--fix_seed 1` additionally requests
deterministic cuDNN settings. It does not choose a fixed seed on its own.

Preprocessing, OOS evaluation, and `reproduce.py evaluate` use `--seed -1` for
the same automatic selection and `--seed N` for an explicit seed. Keep the
recorded seed for each stage when repeating a run. Fixed WUT/KS20 split JSON and
historical trial manifests define the data protocol and remain unchanged.
Explicit seeds do not guarantee identical results across devices or versions.

## Measured training checks

The following measurements were obtained on 2026-11-12 with the **previous
fixed-seed version**, using seed 42 on the three main N=6 settings. They are
retained as historical validation evidence. The full training benchmarks have
**not been rerun with the new random-seed defaults or the subsequent
temporal-eigenbasis determinism fix**. Each selected checkpoint was evaluated
again after loading; measurements remain separate from paper reference values.
Saved checkpoints retain their recorded temporal encoding (`time_pe`) when loaded;
the fix changes its construction for new models, not the stored checkpoint values.

| Dataset | Paper R1 / R5 / mAP (%) | Local R1 / R5 / mAP (%) |
| --- | --- | --- |
| WUT 5C | 80.2 / 90.2 / 59.0 | **79.93 / 89.96 / 59.58** |
| KS20 | 75.2 / 92.7 / 55.0 | **73.39 / 89.91 / 54.13** |
| KGBD | 61.7 / 75.7 / 25.5 | **68.16 / 80.18 / 30.72** |

WUT and KS20 are close to the reference scores. KGBD is higher by
6.46 / 4.48 / 5.22 percentage points; its fixed 201-epoch validation run does
not establish an exact reproduction of the paper's numerical results.

A separate fresh WUT OOS trial reached **98.49% AUROC / 6.12% FPR95**
with K=1 and P=95. It is one training trial, not a five-trial mean.

The current **81-test regression suite passes locally in both CPU environments**,
including repeated temporal-eigenbasis construction and strict explicit-seed
replay. See the [Python 3.10](docs/verification/test-suite-temporal-encoding-python310.json)
and [Python 3.7](docs/verification/test-suite-temporal-encoding-python37.json)
records. These checks are separate from the earlier full GPU benchmarks.
The [validation report](VALIDATION.md) links
the test evidence and provides exact training limits, data and checkpoint hashes,
actual metrics, and unresolved differences. The reported GPU results are
individual fixed-seed checks, not a complete repetition of every paper table
or all five OOS trainings. Checkpoint binaries are not bundled.

## Prepare and train

Prepared data follow `Datas/<dataset>/<length or variant>/`. Source archives
contain no raw data or trained weights. Download the WUT or KGBD raw package,
then run the converter before training.

For a new closed-set WUT run from raw skeletons:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/WUT
python main.py --config params.json --task train
```

For KGBD:

```bash
python main.py --task preprocess --dataset KGBD --length 6 \
  --input_dir '/path/to/kinect gait raw dataset'
python main.py --config config/paper/kgbd_f6.json --task train
```

For KS20 obtained independently, the recovered split configuration assigns one
distinct gait-cycle recording to each split per identity, across all five views:

```bash
python main.py --task preprocess --dataset KS20 --length 6 \
  --input_dir /path/to/KS20
python main.py --config config/paper/ks20_f6.json --task train
```

Additional final-method presets cover WUT 5C/3C/2C, sequence lengths 6/12/18,
and the published frame-rate settings. Match the raw fusion file and prepared
subdirectory to the chosen preset. See [the reproduction guide](docs/PAPER_REPRODUCTION.md).

Existing WUT preprocessing uses gallery indices `[8, 0]`, probe index `[1]`, and
all remaining sequences for training. Thus the paper's 7:2:1 description applies
to an identity with ten sequences; extra recordings remain in training. Exact
sequence selections, valid-window coverage, and identity membership matter more
than an approximate ratio. For archived experiments, use the supplied manifests
rather than creating a new random split.

The paper does not specify a numerical seed or patience limit. New runs use a
generated seed and retain patience 500 as an implementation setting.
`--max_epochs` retains the source's inclusive,
zero-based meaning: `1` permits epochs 0 and 1; `0` disables this additional cap.

## Closed-set identification

```bash
python main.py --task evaluate --ckpt /path/to/best.ckpt \
  --data-dir /path/to/prepared_data --metrics_output results/reid.json
```

Evaluation loads architecture metadata and model parameters before inference,
requires matching state keys, and records the checkpoint hash, dataset hashes,
actual metrics, and saved training settings. Saved checkpoint scores are metadata;
they are not substituted for newly calculated scores. Historical checkpoint
compatibility and protocol discrepancies are detailed in [VALIDATION.md](VALIDATION.md).

## Out-of-set inference

```bash
python main.py --task oos --ckpt /path/to/wut/best.ckpt \
  --data-dir /path/to/this_trial_data --unk_knn_k 1
python main.py --task oos-kgbd --ckpt /path/to/kgbd/best.ckpt \
  --data-dir /path/to/this_trial_data --unk_knn_k 1
```

The unknown identities must be excluded from both training and gallery data.
The paper uses two held-out WUT identities or sixteen held-out KGBD identities,
with five independent identity selections. A closed-set model trained on all
identities must not be combined with an old unknown split.

For an external sequence:

```bash
python main.py --task oos --ckpt /path/to/wut/best.ckpt \
  --data_source npy --input /path/to/skeleton_sequence.npy
```

Prepared training/gallery data are still needed to build feature libraries.
The external NPY array must use the training skeleton convention; see the
[data guide](docs/DATA.md) for KGBD joint order and cleaned-frame indexing.

## Train an OOS model from scratch

Use a separate data and model root for each held-out identity selection. The
supplied manifests can prepare the five paper-style identity selections without
moving raw directories. You can train new weights on those splits; no archived
checkpoint is needed.

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/WUT --prepared-root reproduction_runs/wut01/data

python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir reproduction_runs/wut01/data \
  --model-dir reproduction_runs/wut01/weights

python main.py --task oos --dataset WUT \
  --data-dir reproduction_runs/wut01/data \
  --ckpt reproduction_runs/wut01/weights/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt \
  --output-dir reproduction_runs/wut01/results --unk_knn_k 1 --proj2d 0
```

Repeat with `exp02.json` through `exp05.json`, keeping separate output roots.
For KGBD use its manifest, `config/paper/kgbd_f6.json`, and `--task oos-kgbd`.
The sequence split and known/unknown identities remain fixed, while each new
checkpoint is produced by the released final training method.

The inherited OOS metric uses leave-one-out distances within the known training
library as ID scores. These are calibration-library scores, not held-out known
probe scores. Identity retrieval uses the enrolled gallery. Two-dimensional
plots are diagnostics; OOS decisions use the full embedding space.

`reproduce.py` is an optional audit tool for the archived checkpoints, with
strict checkpoint hashes and a bounded-memory K sweep. The
[reproduction guide](docs/PAPER_REPRODUCTION.md) distinguishes that audit from
training new models. Published targets and locally measured validation results
are recorded separately; no pretrained weights are included in this repository.

## Configuration, testing, and packaging

CLI arguments override JSON options. JSON-relative paths are resolved beside the
JSON file; CLI-relative paths are resolved from the working directory. Default
locations are relative to the source tree. `--data-dir`, `--model-dir`, and
`--output-dir` override them. Individual modules also support `SSGT_DATA_DIR`,
`SSGT_MODEL_DIR`, and `SSGT_OUTPUT_DIR`.

```bash
python main.py --config params.json --dry-run
python main.py --task train --help
python -m unittest discover -s tests -v
python tools/build_release.py --output-dir ../dist
```

Source archives contain only the explicit `RELEASE_FILES.json` entries. Datasets,
checkpoints, private experiment logs, local environments, videos, and Cursor
conversation history are excluded. Load only trusted checkpoint and NumPy
object-array files.

## Limitations

The release supports the stated WUT/KGBD/KS20 skeleton formats and protocols.
Dataset packages are distributed separately under their own access conditions;
checkpoint binaries are not included. Probe Rank-1
selects checkpoints in the inherited workflow; there is no separate validation
identity split. The earlier measurements cover selected N=6 runs and archived
OOS evaluations. They do not cover every configuration or repeated runs under
the new random-seed defaults.
See [VALIDATION.md](VALIDATION.md) for unresolved differences and measurement scope.

## API and code conventions

`main.py` is the public task entry point. The [data guide](docs/DATA.md) documents
prepared arrays and the loader interface; the [reproduction guide](docs/PAPER_REPRODUCTION.md#manifest-recovery-and-python-api)
documents manifest preparation and verification. [SOURCE_MAP.md](docs/SOURCE_MAP.md)
explains how the research implementation was separated into modules.

English module docstrings, section banners, and `Args`/`Returns`/`Raises`
documentation follow the inspected DualSOM examples. See
[CODE_STYLE.md](docs/CODE_STYLE.md) for the reference mapping and a comment example.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff). The model and data utilities
build on [TranSG](https://github.com/Kali-Hac/TranSG); the original MIT copyright
notice is retained in [LICENSE](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
