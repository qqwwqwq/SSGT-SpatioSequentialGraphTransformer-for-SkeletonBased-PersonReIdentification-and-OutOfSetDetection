# SSGT: Spatio-Sequential Graph Transformer for Skeleton-Based Person Re-Identification and Out-of-Set Detection

**Authors:** Xin He, Vibekananda Dutta, Teresa Zielinska, Takafumi Matsumaru, and Robert Sitnik.

![Python](https://img.shields.io/badge/Default_Python-3.10_%7C_3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Tests](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/actions/workflows/tests.yml/badge.svg)](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/actions/workflows/tests.yml)

## Paper

*IEEE Robotics and Automation Letters*, 11(11), 12727–12734, 2026.
[Paper](https://doi.org/10.1109/LRA.2026.3734851) · [Parameter reference](docs/HYPERPARAMETERS.md) · [Data guide](docs/DATA.md)

## Introduction video

An overview of SSGT with English narration and subtitles (3 min 21 s).

https://github.com/user-attachments/assets/bb093a94-6c90-4a86-b90e-16c2bb76c52a

Enable sound in the player for the narration. You can also
[download the original MP4](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/raw/refs/heads/main/docs/media/video-RAL2026.mp4).

## Table of contents

- [Introduction](#introduction)
- [Quick start](#quick-start)
- [System requirements](#system-requirements)
- [Data preparation](#data-preparation)
- [Configuration (`params.json`)](#configuration)
- [Prepare and train](#prepare-and-train)
- [Closed-set identification](#closed-set-identification)
- [Out-of-set detection](#out-of-set-detection)
- [Note on reproducibility](#note-on-reproducibility)
- [Repository layout](#repository-layout)
- [API reference](#api-reference)
- [Testing and packaging](#testing-and-packaging)
- [Citation and license](#citation-and-license)

## Introduction

SSGT learns skeleton representations through spatial and temporal graph attention,
prototype contrastive learning, and masked reconstruction. Dynamic loss balancing
combines the training objectives, and k-nearest-neighbor distances detect unknown
identities. This repository implements the final **SSGT + DLoss** method for WUT,
KGBD, and KS20.

The workflow is: **prepare skeleton data → train SSGT → identify known people or
reject unknown people**. Train your own model; pretrained weights are not provided.

## Quick start

Download and extract WUT using the [data preparation guide](#data-preparation).
Replace `/path/to/WUT` below with its extracted directory.

```bash
git clone https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection.git SSGT
cd SSGT
python -m venv .venv
source .venv/bin/activate
python -m pip install -r config/requirements.txt
python main.py --task preprocess --dataset WUT --length 6 --input_dir /path/to/WUT
python main.py --config params.json --task train
```

Run commands from the repository root. For CPU training, add `--device cpu` to the
training command. Use a PyTorch build compatible with your CUDA installation for
GPU training.

## System requirements

- Python 3.10 or 3.11.
- PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, and threadpoolctl; dependency
  ranges are listed in [config/requirements.txt](config/requirements.txt).
- A CPU or a CUDA-capable GPU. GPU acceleration is optional.

For a Conda environment:

```bash
conda env create -f config/environment.yml
conda activate ssgt
```

Check the installation with `python main.py --help`.

## Data preparation

Only raw WUT and KGBD data are distributed. Convert them locally before training.

| Dataset package | Download |
| --- | --- |
| `SSGT-WUT-raw.zip` | [Google Drive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) |
| `SSGT-KGBD-raw.zip` | [Google Drive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) |

[Extraction instructions](docs/DOWNLOADS.md) · [Checksums](config/dataset_SHA256SUMS) ·
[Package inventory](config/dataset_downloads.json)

KS20 is not redistributed because its dataset notice requires permission.
Obtain it independently before using the provided converter.

| Dataset | Raw input | Prepared input |
| --- | --- | --- |
| WUT | Per-person recording folders with fused skeleton NPY arrays | 19 joints |
| KGBD | `PersonXXX` folders with text skeleton sequences and optional NPY caches | 20 joints |
| KS20 | MATLAB recordings under five viewpoint directories | 25 joints |

Prepared coordinates and identity labels are written to
`Datas/<dataset>/<length or variant>/`. See [docs/DATA.md](docs/DATA.md) for file
names, split rules, joint order, and external NPY input formats.

## Configuration

`params.json` provides the default WUT configuration. Use `config/kgbd.json` for
KGBD or a preset in `config/paper/` for another camera setup, sequence length, or
frame rate. Configuration sections match the tasks accepted by `main.py`:

| Section | Purpose |
| --- | --- |
| `task` | Default task: `train`, `evaluate`, `oos`, `oos-kgbd`, or `preprocess` |
| `paths` | Prepared-data, model, and output directories |
| `train` | Model and optimization settings |
| `evaluate` | Closed-set evaluation options |
| `oos` / `oos-kgbd` | Unknown-identity detection options |
| `preprocess` | Raw-data conversion options |

A minimal configuration is shown below. Omitted options use their parser defaults.

```json
{
  "task": "train",
  "paths": {
    "data_dir": "Datas",
    "model_dir": "weight",
    "output_dir": "results"
  },
  "train": {
    "dataset": "WUT",
    "length": 6,
    "H": 128,
    "n_heads": 8,
    "L_transformer": 2,
    "batch_size": 256,
    "lr": 0.00035,
    "seed_val": -1,
    "fix_seed": 0
  }
}
```

### Main hyperparameters

Values below are the default training settings unless noted. JSON option names
also work as command-line flags, for example `--batch_size 128`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | String | `WUT` | `WUT`, `KGBD`, or `KS20` |
| `length` | Integer | `6` | Frames per window; match the prepared data |
| `H` | Integer | `128` | Embedding width; must be divisible by `n_heads` |
| `n_heads` | Integer | `8` | Attention heads in each spatial and temporal layer |
| `L_transformer` | Integer | `2` | Layers in each attention branch |
| `enc_k_spa` / `enc_k_time` | Integer | `10` / `4` | Spatial / temporal Laplacian dimensions |
| `attn_dropout` | Float | `0.4` | Attention dropout probability |
| `St_mask_num` / `Tr_mask_num` | Integer | `10` / `2` | Masked joints / frames for reconstruction |
| `seq_lambda` | Float | `0.5` | Sequence contrastive weight; skeleton weight is its complement |
| `prompt_lambda` | Float | `0.5` | Spatial reconstruction weight; trajectory weight is its complement |
| `GPC_lambda` | Float | `0.5` | Initial contrastive weight; reconstruction weight is its complement |
| `t_1` / `t_2` | Float | `0.07` / `14.0` | Sequence / skeleton contrastive temperatures |
| `gpc_up_period` | Integer | `10` | Initial dynamic-weight update epoch; doubles after each update |
| `gpc_iota` | Float | `10.0` | Signed loss-slope ratio threshold |
| `gpc_tau` | Float | `0.05` | Dynamic-weight update step |
| `lr` | Float | `0.00035` | Adam learning rate; KGBD uses this fixed value |
| `batch_size` | Integer | `256` | Windows per training batch |
| `patience` | Integer | `500` | Epochs without improvement in the selected checkpoint metrics |
| `max_epochs` | Integer | `0` | Additional inclusive epoch cap; `0` uses the stopping schedule |
| `seed_val` | Integer | `-1` | Generate a random seed; a nonnegative value selects an explicit seed |
| `fix_seed` | Integer | `0` | `1` requests deterministic cuDNN settings |
| `unk_knn_k` | Integer | `1` | Neighbor rank used by OOS detection |
| `unk_sigma` | Float | `0.95` | Known-library quantile for the OOS acceptance threshold |

The final training method requires temporal attention, Laplacian positional
encoding, both reconstruction losses, and dynamic loss balancing. The supplied
presets select these settings. The [full parameter reference](docs/HYPERPARAMETERS.md)
covers every command-line option, valid ranges, loss formulas, dataset-specific
behavior, preprocessing controls, and evaluation outputs.

CLI arguments override JSON values. Paths inside JSON resolve relative to that
file; CLI paths resolve from the working directory. Use `--data-dir`,
`--model-dir`, and `--output-dir` to override the default directories.

```bash
python main.py --config params.json --task train --batch_size 128 --max_epochs 200
python main.py --config params.json --dry-run
```

## Prepare and train

For WUT:

```bash
python main.py --task preprocess --dataset WUT --length 6 --input_dir /path/to/WUT
python main.py --config params.json --task train
```

For KGBD:

```bash
python main.py --task preprocess --dataset KGBD --length 6 \
  --input_dir '/path/to/kinect gait raw dataset'
python main.py --config config/kgbd.json --task train
```

For KS20 obtained independently:

```bash
python main.py --task preprocess --dataset KS20 --length 6 --input_dir /path/to/KS20
python main.py --config config/paper/ks20_f6.json --task train
```

Match the raw WUT fusion file and prepared subdirectory to the selected 5C, 3C,
or 2C preset. The [data guide](docs/DATA.md) lists these variants.

Training writes `best.ckpt` below `weight/<dataset>/<configuration>/`. Its full
path is printed in the training output. The epoch cap is zero-based and inclusive:
`--max_epochs 1` allows epochs 0 and 1. With the default `0`, early stopping applies
within the training loop's 50,000-iteration ceiling.

## Closed-set identification

Evaluate a checkpoint produced by your training run:

```bash
python main.py --task evaluate --ckpt /path/to/best.ckpt \
  --data-dir /path/to/prepared_data --metrics_output results/reid.json
```

The checkpoint restores the model configuration. The JSON output contains
Rank-1, Rank-5, Rank-10, and mAP calculated on the supplied gallery/probe data.

## Out-of-set detection

Unknown identities must be excluded from both training and gallery data. Use the
same identity split for preparation, training, and evaluation. Example split
configurations are provided under `config/trials/`.

### Train an OOS model

Prepare and train one WUT split in separate directories:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/WUT --prepared-root runs/wut01/data

python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir runs/wut01/data --model-dir runs/wut01/weights

python main.py --task oos --dataset WUT \
  --data-dir runs/wut01/data \
  --ckpt runs/wut01/weights/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt \
  --output-dir runs/wut01/results --unk_knn_k 1 --proj2d 0
```

Repeat with `exp02.json` through `exp05.json`, using separate data, model, and
result directories. For KGBD, use its manifest, `config/kgbd.json`, and
`--task oos-kgbd`. See the [split guide](config/trials/README.md).

### Evaluate unknown or external sequences

```bash
python main.py --task oos-kgbd --ckpt /path/to/kgbd/best.ckpt \
  --data-dir /path/to/trial_data --unk_knn_k 1

python main.py --task oos --ckpt /path/to/wut/best.ckpt \
  --data-dir /path/to/trial_data \
  --data_source npy --input /path/to/skeleton_sequence.npy
```

External sequences must use the model's joint order. Prepared training/gallery
data are still required to build the feature libraries. OOS calibration uses
leave-one-out distances in the known training library; identity retrieval uses
the gallery. OOS decisions use the full embedding space.

## Note on reproducibility

Each run uses a randomly generated seed by default, so training results may differ
slightly from those reported in the paper. Software, hardware, and data splits can
also affect results. The selected seed is saved with the run.

Use `--seed_val N` for an explicit training seed, or `--seed N` for preprocessing
and OOS evaluation. `--fix_seed 1` additionally requests deterministic cuDNN
settings; it does not select a seed. Run seeds do not change explicit split files.

## Repository layout

```text
SSGT/
├── main.py                # JSON configuration and task entry point
├── SSGT.py                # Spatial and temporal graph encoder
├── training.py            # Training and closed-set evaluation
├── loss_balancing.py      # Dynamic loss weights
├── evaluation.py          # OOS evaluation
├── evaluation_kgbd.py     # KGBD OOS entry point
├── preprocessing.py       # WUT/KGBD/KS20 data preparation
├── ks20_preprocessing.py  # KS20 MATLAB conversion
├── data.py                # Skeleton loading and graph construction
├── protocols.py           # OOS identity and sequence splits
├── retrieval.py           # ReID metrics
├── randomness.py          # Run seed selection
├── visualization.py       # Optional result plots
├── params.json            # Default WUT settings
├── config/                # Dependencies, dataset presets, and split files
├── Datas/                 # Prepared data
├── weight/                # Checkpoints created by training
├── results/               # Evaluation outputs
├── docs/                  # Parameter and data guides
├── tests/                 # Synthetic workflow tests
└── tools/build_release.py # Source archive builder
```

## API reference

| Entry point | Purpose |
| --- | --- |
| `main.main(argv=None)` | Apply JSON settings and dispatch a task |
| `training.main(argv=None)` | Train SSGT or evaluate a checkpoint |
| `preprocessing.main(argv=None)` | Convert raw skeleton recordings |
| `protocols.prepare_trial(manifest, raw_root, prepared_root)` | Prepare an OOS split without moving raw files |
| `evaluation.main(argv=None)` | Run OOS detection and identity retrieval |
| `SSGT.SSGT(...)` | Construct the spatial and temporal graph encoder |

Each task exposes its options through `python main.py --task TASK --help`.
Parameter definitions are in [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md);
data formats are in [docs/DATA.md](docs/DATA.md).

## Testing and packaging

```bash
python -m unittest discover -s tests -v
python tools/build_release.py --output-dir ../dist
```

The tests use synthetic inputs and require no dataset download or GPU. Source
archives contain the files listed in `RELEASE_FILES.json`; data and trained
checkpoints are generated separately.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the paper. This project builds on
[TranSG](https://github.com/Kali-Hac/TranSG) and retains its MIT copyright notice.
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
