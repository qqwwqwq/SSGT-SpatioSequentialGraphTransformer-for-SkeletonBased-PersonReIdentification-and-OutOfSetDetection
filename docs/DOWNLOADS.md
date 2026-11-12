# Dataset downloads and setup on a new computer

Dataset packages are separate from `SSGT-source.zip` and `SSGT-source.tar.gz`.
The source archives contain code, configuration, documentation, and tests.
The dataset delivery contains only the WUT and KGBD raw skeleton archives.
Preprocess them locally, then train your own checkpoints. No prepared-data
package or model checkpoint is distributed.

Download WUT and KGBD from the links below. The
[machine-readable inventory](../config/dataset_downloads.json) records their
URLs, expected archive filenames, contents, sizes, and local archive hashes.

## Choose a package

| Archive | Download | Extracted data root |
| --- | --- | --- |
| `SSGT-WUT-raw.zip` | [Google Drive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) | `raw/WUT/` |
| `SSGT-KGBD-raw.zip` | [Google Drive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) | `raw/KGBD/` |

Choose the raw dataset you need, extract it, and run its converter to generate
training, gallery, and probe arrays. The raw packages also support the documented
sequence-length and frame-rate variants and explicit OOS identity splits.
Raw inputs and generated arrays can be stored outside the source tree.

## Included raw skeleton files

- WUT contains 293 sequences with all three fused variants per sequence:
  `fusion_master.npy` for 5C, `fusion_master_3c.npy` for 3C, and
  `fusion_master_2c.npy` for 2C, giving 879 fusion arrays. The package also retains
  1,631 original per-camera CSV files. Fusion arrays have shape `(frames, 19, 4)`
  and dtype `float64`; the converter exports XYZ from the first three channels.
  Its NaN-window filter still examines every channel. The CSVs contain 32 joints
  in 128 columns; the converter does not recreate fusion from those CSV files.
- KGBD contains 822 sequence TXT files and 822 existing `fusion_master.npy`
  caches. Keep both: the explicit historical OOS manifests bind the cache files
  for known sequences by filename and SHA-256. Removing or regenerating a cache
  can invalidate that data binding.

## Why KS20 is not included

The original KS20 dataset notice requires permission for redistribution, so KS20
is excluded from this dataset delivery. Users who obtain it independently can
still use the included converter, split configuration, and training preset; see
[KS20 data preparation](DATA.md#ks20-inputs-and-split). The source-code license
does not replace a dataset's access conditions.

## Verify downloaded archives

Save the downloads with the archive filenames listed above. The repository
includes [dataset_SHA256SUMS](../config/dataset_SHA256SUMS) for both ZIP files and
[dataset_downloads.json](../config/dataset_downloads.json) for their inventory.
The published hashes identify the locally built release archives; the hosted
copies have not been checksum-verified.

On Linux, run this from the directory containing your downloaded ZIP files:

```bash
cd /path/to/downloads
sha256sum --check --ignore-missing /path/to/SSGT/config/dataset_SHA256SUMS
```

Replace `/path/to/SSGT` with your source checkout. Alternatively, copy
`config/dataset_SHA256SUMS` into the download directory and check that local file.
`--ignore-missing` allows downloading only the dataset you need. Each downloaded
archive must report `OK` before extraction.

Each archive also contains `DATASET_README.md` and `MANIFEST.json` under its
own metadata directory:

```text
dataset_info/
  WUT-raw/
  KGBD-raw/
```

A given ZIP contains only its corresponding metadata directory. These records
identify its data files and fingerprints. Keep the metadata beside the extracted
data when moving the package to another computer.

## Convert raw data

Extract only the raw packages you intend to use:

```bash
python -m zipfile -e /path/to/downloads/SSGT-WUT-raw.zip /path/to/ssgt_data
python -m zipfile -e /path/to/downloads/SSGT-KGBD-raw.zip /path/to/ssgt_data
```

Install the source dependencies using the [quick-start instructions](../README.md#quick-start).
From the source directory, prepare a separate output root:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/new_prepared

python main.py --task preprocess --dataset KGBD --length 6 --skip_unknown \
  --input_dir /path/to/ssgt_data/raw/KGBD --data-dir /path/to/new_prepared
```

After preprocessing, train from the source directory:

```bash
python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir /path/to/new_prepared --model-dir /path/to/ssgt_weights
python main.py --config config/paper/kgbd_f6.json --task train \
  --data-dir /path/to/new_prepared --model-dir /path/to/ssgt_weights
```

Pass the prepared-data root, not its `WUT/6/` or `KGBD/6/` subdirectory, to
`--data-dir`. The loader adds those dataset and length components. WUT follows
its fixed JSON sequence assignments. KGBD uses a generated preprocessing seed
by default; keep its `preprocessing_metadata.json` when recording the experiment. To request
the numerical seed used for the earlier KGBD validation split, add `--seed 42`.
This is an optional historical-split reference, not the default for new runs.

For WUT 3C or 2C, select the matching fusion array and prepared variant:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/new_prepared \
  --fusion_file fusion_master_3c.npy --data_subdir 6_3c
python main.py --config config/paper/wut_3c_f6.json --task train \
  --data-dir /path/to/new_prepared

python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/new_prepared \
  --fusion_file fusion_master_2c.npy --data_subdir 6_2c
python main.py --config config/paper/wut_2c_f6.json --task train \
  --data-dir /path/to/new_prepared
```

There are no separate raw 10-FPS or 5-FPS WUT files; create those variants by
downsampling the 15-FPS 5C fusion during preprocessing. See [DATA.md](DATA.md) for frame-rate options,
NaN filtering, and the exact prepared-array layout.

## Prepare an OOS split

To train an OOS model, create a separate trial from raw data so held-out people
are excluded from training and gallery:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/ssgt_data/raw/WUT \
  --prepared-root /path/to/oos_trials/wut01/data

python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root /path/to/ssgt_data/raw/KGBD \
  --prepared-root /path/to/oos_trials/kgbd01/data
```

Then follow the [OOS self-training recipe](PAPER_REPRODUCTION.md#train-and-evaluate-your-own-oos-model)
with that trial's data root. Repeat with the other manifests in separate output
locations. A model trained on all closed-set identities must not be evaluated
as though those same people had been held out during training.
