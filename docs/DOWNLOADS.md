# Dataset downloads

Download raw skeleton data, preprocess them locally, and train your own model.
Prepared arrays and pretrained weights are not distributed.

| Archive | Download | Extracted data root |
| --- | --- | --- |
| `SSGT-WUT-raw.zip` | [Google Drive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) | `raw/WUT/` |
| `SSGT-KGBD-raw.zip` | [Google Drive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) | `raw/KGBD/` |

See the [package inventory](../config/dataset_downloads.json) for sizes and contents.
KS20 is not redistributed because its dataset notice requires permission. Obtain
it independently to use the [KS20 converter](DATA.md#ks20-inputs-and-split).
The source-code license does not replace dataset access conditions.

## Verify and extract

Save downloads with the filenames above. From the download directory, check them
against the repository's [checksums](../config/dataset_SHA256SUMS):

```bash
cd /path/to/downloads
sha256sum --check --ignore-missing /path/to/SSGT/config/dataset_SHA256SUMS
```

Replace `/path/to/SSGT` with your checkout. `--ignore-missing` lets you download
only the dataset you need; each downloaded archive should report `OK`. These
checksums identify the packaged ZIP files.

Extract the required archives:

```bash
python -m zipfile -e /path/to/downloads/SSGT-WUT-raw.zip /path/to/ssgt_data
python -m zipfile -e /path/to/downloads/SSGT-KGBD-raw.zip /path/to/ssgt_data
```

Keep original filenames and directory names. WUT includes 5C, 3C, and 2C fusion
arrays plus camera CSVs; preprocessing uses the fusion arrays. KGBD includes TXT
sequences and fusion caches; keep both for the supplied OOS manifests.

## Convert raw data

Install dependencies as described in the [README](../README.md). Run conversion
from the repository root:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/prepared
python main.py --task preprocess --dataset KGBD --length 6 \
  --input_dir /path/to/ssgt_data/raw/KGBD --data-dir /path/to/prepared
```

Pass `/path/to/prepared` to training with `--data-dir`, rather than the generated
`WUT/6/` or `KGBD/6/` subfolder. WUT uses fixed sequence assignments. KGBD generates
a split seed and saves it in `preprocessing_metadata.json`; use `--seed N` when
you need an explicit seed.

For WUT camera variants, select the matching fusion file and output directory:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/prepared \
  --fusion_file fusion_master_3c.npy --data_subdir 6_3c
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/prepared \
  --fusion_file fusion_master_2c.npy --data_subdir 6_2c
```

The default `fusion_master.npy` is the 5C input. WUT 10-FPS and 5-FPS variants are
derived by downsampling its 15-FPS arrays; they have no separate raw downloads.
See [data formats and split rules](DATA.md) and
[matching training presets](HYPERPARAMETERS.md).

## Prepare an OOS split

Use separate data roots so held-out identities are excluded from training and
gallery data:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/ssgt_data/raw/WUT \
  --prepared-root /path/to/trials/wut01/data
python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root /path/to/ssgt_data/raw/KGBD \
  --prepared-root /path/to/trials/kgbd01/data
```

Follow the [README training and OOS commands](../README.md) using that trial's
data root. Repeat with the other manifests in separate directories, training a
checkpoint for each known-identity split.
