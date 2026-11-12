# Data preparation

Prepare raw skeletons before training. Download WUT or KGBD using the
[download guide](DOWNLOADS.md). KS20 must be obtained independently because its
dataset notice requires permission for redistribution. No pretrained weights
or prepared-data package is provided.

## Locations and commands

Run from the repository root:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/raw/WUT --data-dir /path/to/prepared
python main.py --task preprocess --dataset KGBD --length 6 \
  --input_dir /path/to/raw/KGBD --data-dir /path/to/prepared
python main.py --task preprocess --dataset KS20 --length 6 \
  --input_dir /path/to/KS20 --data-dir /path/to/prepared
```

The output root defaults to `Datas/`; override it with `--data-dir` or
`SSGT_DATA_DIR`. Pass the root, not its `WUT/6/` subdirectory. Raw inputs are
read only. Use a fresh output directory when changing splits, because preprocessing
does not remove files from previous runs.

`--length` selects the window length; `--data_subdir` names a separate variant.
For WUT/KGBD, `--fusion_file` selects the fusion array and `--unknown_dir` supplies
optional held-out inputs. `--skip_unknown` disables unknown preparation.
See [hyperparameters](HYPERPARAMETERS.md) for matching training presets.

## WUT inputs and split

```text
WUT/
  Jacek/
    sequence_00/fusion_master.npy
    sequence_01/fusion_master.npy
    ...
  Aleks/
    ...
```

Keep original identity directory names: the converter maps them to IDs 1–25
and skips unrecognized names. Each sequence contains `fusion_master.npy` (5C),
`fusion_master_3c.npy` (3C), and `fusion_master_2c.npy` (2C). Select the matching
file and output variant for the chosen preset; see the
[example commands](DOWNLOADS.md#convert-raw-data).

Packaged fusion arrays have shape `(frames, 19, 4)` and dtype `float64`.
The converter exports the first three channels as XYZ, but checks **all channels**
for NaNs when filtering windows. Do not remove the fourth channel before conversion.
Arrays with at least three channels are accepted. The accompanying camera CSVs
contain 32 joints and 128 columns; the converter does not regenerate fusion arrays
from these CSVs.

[`wut_split.json`](../config/wut_split.json) selects zero-based positions in each
identity's alphabetically sorted sequence directories:

- Gallery: positions 8 and 0.
- Probe: position 1.
- Training: all remaining positions, including extra recordings.

Keep at least nine sequence directories and the selected fusion files.
`--split_file` accepts an alternative JSON with two distinct gallery positions
and one distinct probe position. Short or invalid recordings can leave an identity
without probe windows; the supplied Julia probe sequence, for example, has no
valid six-frame windows.

Unknown inputs may use `person/sequence/fusion_master.npy`, sequence folders
directly below the unknown root, or a fusion file directly below that root.
Their raw identity label is 26.

## KGBD inputs and split

```text
KGBD/
  Person001/
    sequence_01.txt
    sequence_01/fusion_master.npy
    sequence_02.txt
    ...
  Person002/
    ...
```

Each TXT frame contains 20 joint records, with a leading joint field followed by
semicolon-separated X, Y, and Z values. The converter reads an existing
`PersonXXX/<sequence-stem>/<fusion_file>` cache when present; otherwise it parses
the TXT file without writing a cache. Keep both TXT and fusion files from the
download: the supplied OOS manifests select specific source files.

Two sequences per identity become gallery data and one becomes probe data;
the rest become training data. Person015 uses one gallery and one probe sequence.
Provide at least four sequences per identity, or three for Person015.

Sequence selection uses a new random seed by default. The converter prints it
and saves it in `preprocessing_metadata.json` beside the prepared splits.
Use `--seed N` to repeat a selection with the same input files and environment.

Known labels are 1–164. Unknown inputs are TXT files directly below
`--unknown_dir` or one level below it in person directories; their raw label is 165.
The unknown path must be supplied explicitly.

## KS20 inputs and split

Supply `frontal`, `left_diagonal`, `left_lateral`, `right_diagonal`, and
`right_lateral`, containing all 300 `gc*_body*.mat` files. The converter reads the
nested `gc` MATLAB objects and keeps 25 Kinect v2 joints. Body numbers
`1–4, 7–9, 11–17, 19–24` map in order to labels `0–19`.

[`ks20_split.json`](../config/ks20_split.json) assigns one distinct gait-cycle
recording to training, gallery, and probe per identity across all five views,
for a 1:1:1 sequence ratio. `--split_file` can supply another assignment covering
all identities with three distinct recording choices each.

The converter removes ten frames from each end by default. For length T, training
uses shifts `0..T-1` and stride `T // 2`; gallery and probe use shift zero and stride T.
Short recordings may produce no window. KS20 preprocessing supports closed-set
data only, without unknown inputs or FPS resampling.

## Explicit OOS splits

Unknown identities must be absent from training and gallery data. The manifests
in `config/trials/` define five selections per dataset, holding out two WUT or
sixteen KGBD identities. Prepare each trial in its own directory:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/raw/WUT --prepared-root /path/to/trials/wut01/data
python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root /path/to/raw/KGBD --prepared-root /path/to/trials/kgbd01/data
```

The manifest selects identities and sequences without moving raw files.
Preparation checks source fingerprints and records generated files in
`provenance.json`; keep it with the prepared data. Repeat with `exp02.json` through
`exp05.json` using separate output roots, then follow the
[training and evaluation commands](../README.md). Train a checkpoint for each
known-identity split; a closed-set checkpoint trained on all people is unsuitable.

OOS metrics use leave-one-out distances within the known training library as ID
calibration scores, rather than held-out probe scores. Identity retrieval uses
the gallery; rejection uses distances in the full embedding space.

## Windowing and output format

WUT/KGBD known sequences and WUT unknown sequences use complete, non-overlapping
windows. Incomplete final windows and windows containing NaNs are dropped.
KGBD unknown TXT inputs do not receive that NaN filter: supply finite coordinates.
Unknown windows are deduplicated within their split after rounding coordinates
to four decimal places; known and unknown data are not deduplicated against each other.

Use `--src_fps` and `--dst_fps` for downsampling: frames are selected on a nearest
uniform time grid before windowing, without interpolation. WUT is 15 FPS and
KGBD is 30 FPS. WUT 10-FPS and 5-FPS variants come from the 15-FPS 5C fusion arrays.
Keep each variant in its own `--data_subdir`.

```text
Datas/WUT/6/
  train_npy_data/
    source_x_WUT_6.npy
    source_y_WUT_6.npy
    source_z_WUT_6.npy
    ids_WUT_6.npy
    frame_id_WUT_6.npy
  test_npy_data/
    gallery/
    probe/
  unknown_npy_data/          # Only when unknown inputs are prepared
```

Gallery, probe, and unknown folders each contain the same five files, with
`t_source_` instead of `source_` for coordinates. Replace `WUT` with the dataset
name. KS20 has no unknown folder. A variant directory such as `6_3c` still uses
length `6` in filenames.

| File | Contents |
| --- | --- |
| `source_{x,y,z}_*.npy`, `t_source_{x,y,z}_*.npy` | Coordinates, each `(N * T, J)` for N windows, T frames, J joints |
| `frame_id_*.npy` | One identity label per window, shape `(N,)` |
| `ids_*.npy` | NumPy object dictionary mapping raw identity IDs to window-index lists |

Load object arrays only from trusted sources.

## Loader and external queries

The loader reshapes coordinates into `(N, T, J, 3)` and centers each frame on
original joint zero. KGBD then applies the joint permutation
`[11,10,1,0,2,4,6,8,3,5,7,9,12,14,16,18,13,15,17,19]`.
One-hot label columns follow sorted training identity keys. Unknown labels become
all-zero rows; do not use `argmax` on these rows to infer an identity.

Use at least three known training identities, enough windows for a complete batch,
and a batch size no smaller than the known identity count. Every identity must
have windows in the complete training batches used to build prototypes.

External OOS NPY queries should have shape `(frames, J, 3)` and match the training
coordinate units and **model joint order**. The external-query adapter does not
apply the KGBD permutation. For KGBD, subtract original joint zero and apply the
permutation above before inference with `--task oos-kgbd --center_root 0`.
For WUT in its original joint order, the default root centering is appropriate.
Prepared training/gallery arrays are still needed for feature libraries.

External-query inference removes NaN-containing frames before windowing, so
exported frame spans refer to the cleaned sequence. Keep an original-frame map
when aligning predictions with a video.

Retrieval IDs retain original dataset identities. `--one_based_ids 1` is the
default; `0` selects zero-based exports. WUT also accepts identity names for
`--groundtruth`. CSV `oos_window` marks rejected windows while retaining their
best known candidate; `_win_id.npy` instead uses unknown ID 26 for WUT or 165
for KGBD under the default indexing. `_topk_id.npy` keeps known candidates.
