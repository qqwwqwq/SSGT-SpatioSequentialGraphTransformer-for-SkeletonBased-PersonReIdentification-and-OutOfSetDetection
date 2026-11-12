# SSGT: Spatio-Sequential Graph Transformer for Skeleton-Based Person Re-Identification and Out-of-Set Detection

**Authors:** Xin He, Vibekananda Dutta, Teresa Zielinska, Takafumi Matsumaru, and Robert Sitnik.

![Python](https://img.shields.io/badge/Default_Python-3.10_%7C_3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
[![Tests](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/actions/workflows/tests.yml/badge.svg)](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/actions/workflows/tests.yml)

## Paper

*IEEE Robotics and Automation Letters*, 11(11), 12727–12734, 2026.
[Paper](https://doi.org/10.1109/LRA.2026.3734851) · [Reproduction guide](docs/PAPER_REPRODUCTION.md) · [Data guide](docs/DATA.md)

## Introduction video

An overview of SSGT with English narration and subtitles (3 min 21 s).

https://github.com/user-attachments/assets/bb093a94-6c90-4a86-b90e-16c2bb76c52a

Enable sound in the player for the narration. You can also
[download the original MP4](https://github.com/qqwwqwq/SSGT-SpatioSequentialGraphTransformer-for-SkeletonBased-PersonReIdentification-and-OutOfSetDetection/raw/refs/heads/main/docs/media/video-RAL2026.mp4).

## Table of contents

- [Introduction video](#introduction-video)
- [Introduction](#introduction)
- [Dataset downloads](#dataset-downloads)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Final model settings](#final-model-settings)
- [Note on reproducibility](#note-on-reproducibility)
- [Prepare and train](#prepare-and-train)
- [Closed-set identification](#closed-set-identification)
- [Out-of-set inference](#out-of-set-inference)
- [Train an OOS model from scratch](#train-an-oos-model-from-scratch)
- [Configuration, testing, and packaging](#configuration-testing-and-packaging)
- [Documentation](#documentation)
- [Citation and license](#citation-and-license)

## Introduction

SSGT combines spatial and temporal graph attention, prototype contrastive learning,
masked reconstruction, and dynamic loss balancing for skeleton-based person
re-identification and out-of-set (OOS) detection. This repository implements the
final **SSGT + DLoss** method and provides data preparation, training, and evaluation
for WUT, KGBD, and KS20.

## Dataset downloads

Download the raw skeleton data and preprocess them locally before training.

| Package | Download |
| --- | --- |
| `SSGT-WUT-raw.zip` | [Google Drive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) |
| `SSGT-KGBD-raw.zip` | [Google Drive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) |

[Download and extraction guide](docs/DOWNLOADS.md) ·
[Checksums](config/dataset_SHA256SUMS) ·
[Package inventory](config/dataset_downloads.json)

KS20 is not redistributed because its dataset notice requires permission.
Obtain it independently before using the provided converter.

## Quick start

Use Python 3.10 or 3.11 and run these commands from the repository root.
Replace `/path/to/WUT` with the extracted WUT dataset directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --task preprocess --dataset WUT --length 6 --input_dir /path/to/WUT
python main.py --config params.json --task train
```

Use a PyTorch build compatible with your CPU or CUDA environment.
An alternative Conda specification is provided in
[config/environment.yml](config/environment.yml).

## Repository layout

```text
SSGT/
├── main.py                # Training, closed-set evaluation, preprocessing, OOS inference
├── SSGT.py                # Spatial and temporal graph encoder
├── training.py            # Training, reconstruction/contrastive objectives, ReID metrics
├── loss_balancing.py      # Dynamic loss balancing
├── retrieval.py           # ReID metrics
├── evaluation.py          # OOS inference; WUT defaults
├── evaluation_kgbd.py     # KGBD OOS inference
├── visualization.py       # Diagnostic projections and plots
├── preprocessing.py       # Raw-data conversion entry point
├── ks20_preprocessing.py  # KS20 data conversion
├── data.py                # Skeleton loading and graph construction
├── protocols.py           # Identity and sequence splits
├── randomness.py          # Run seeds and replay controls
├── reproduce.py           # Historical OOS evaluation and trial aggregation
├── params.json            # Final WUT 5C, N=6 configuration
├── config/
│   ├── paper/             # Experiment presets
│   ├── paper_targets.json # Paper reference values
│   └── trials/            # OOS trial manifests
├── Datas/                 # Prepared datasets generated locally
├── weight/                # Trained checkpoints generated locally
├── results/               # Generated measurements and figures
├── docs/                  # Data, reproduction, source mapping, and code conventions
├── tests/                 # Synthetic regression and integration tests
└── tools/build_release.py # Source archives from an explicit file manifest
```

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

Presets are in `params.json` and `config/paper/`. See the
[reproduction guide](docs/PAPER_REPRODUCTION.md) for implementation details.

## Note on reproducibility

Each run uses a randomly generated seed by default, so training results may differ
slightly from those reported in the paper. Hardware, software versions, and data
splits can also affect results. The selected seed is recorded with the run.

Training defaults to `seed_val=-1` and `fix_seed=0`. Use `--seed_val N` for an
explicit training seed; `--fix_seed 1` additionally requests deterministic cuDNN
settings. Preprocessing and OOS evaluation use `--seed -1` by default and
`--seed N` for an explicit seed. Run seeds do not change fixed split definitions.

No pretrained weights are provided. Train SSGT from scratch and evaluate the
checkpoints generated by your runs.

## Prepare and train

Prepared data are stored in `Datas/<dataset>/<length or variant>/`.

For WUT:

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

For KS20 obtained independently:

```bash
python main.py --task preprocess --dataset KS20 --length 6 \
  --input_dir /path/to/KS20
python main.py --config config/paper/ks20_f6.json --task train
```

Presets cover WUT 5C/3C/2C, sequence lengths 6/12/18, and frame-rate variants.
Match the raw fusion file and prepared subdirectory to the chosen preset.
Dataset splits and input formats are described in the [data guide](docs/DATA.md).

Training uses patience 500 by default. `--max_epochs` is inclusive and zero-based:
`1` permits epochs 0 and 1; `0` disables this additional cap.

## Closed-set identification

```bash
python main.py --task evaluate --ckpt /path/to/best.ckpt \
  --data-dir /path/to/prepared_data --metrics_output results/reid.json
```

Evaluation restores the model configuration from the checkpoint and writes
retrieval metrics to the specified JSON file.

## Out-of-set inference

```bash
python main.py --task oos --ckpt /path/to/wut/best.ckpt \
  --data-dir /path/to/this_trial_data --unk_knn_k 1
python main.py --task oos-kgbd --ckpt /path/to/kgbd/best.ckpt \
  --data-dir /path/to/this_trial_data --unk_knn_k 1
```

Exclude unknown identities from both training and gallery data. The paper uses
two held-out WUT identities or sixteen held-out KGBD identities across five trials.
Use a checkpoint trained on the matching known-identity split.

For an external sequence:

```bash
python main.py --task oos --ckpt /path/to/wut/best.ckpt \
  --data_source npy --input /path/to/skeleton_sequence.npy
```

External inputs still require prepared training/gallery data for the feature
libraries and must follow the training skeleton format. See the
[data guide](docs/DATA.md) for array shapes and joint order.

## Train an OOS model from scratch

Use the supplied manifests to prepare each identity split, with separate data,
model, and result directories for each trial:

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

OOS metrics use leave-one-out distances in the known training library as ID
scores; these are calibration scores, not held-out probe scores. Identity
retrieval uses the gallery, and OOS decisions use the full embedding space.

## Configuration, testing, and packaging

CLI arguments override JSON options. Relative paths in JSON resolve beside the
configuration file; CLI paths resolve from the working directory. Override the
default directories with `--data-dir`, `--model-dir`, and `--output-dir`, or
`SSGT_DATA_DIR`, `SSGT_MODEL_DIR`, and `SSGT_OUTPUT_DIR`.

```bash
python main.py --config params.json --dry-run
python main.py --task train --help
python -m unittest discover -s tests -v
python tools/build_release.py --output-dir ../dist
```

Source archives contain the files listed in `RELEASE_FILES.json`.
Load only trusted checkpoint and NumPy object-array files.

## Documentation

- [Data formats and protocols](docs/DATA.md)
- [Experiment settings and Python API](docs/PAPER_REPRODUCTION.md)
- [Validation records](VALIDATION.md)
- [Module structure](docs/SOURCE_MAP.md) and [code conventions](docs/CODE_STYLE.md)

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the paper. This project builds on
[TranSG](https://github.com/Kali-Hac/TranSG) and retains its MIT copyright notice.
See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
