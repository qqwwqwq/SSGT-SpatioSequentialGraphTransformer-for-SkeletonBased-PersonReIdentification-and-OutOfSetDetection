# Data preparation and protocol

The source release includes raw-data converters for WUT, KGBD, and KS20, explicit WUT/KS20 split configurations, and ten recorded OOS trial manifests. Readers can prepare their own copy of the datasets, train SSGT, and evaluate the resulting checkpoints. Training does not require archived weights. Raw skeletons and checkpoint binaries are not bundled in the source archives. WUT and KGBD raw archives are packaged separately; download links and extraction instructions are in [DOWNLOADS.md](DOWNLOADS.md).

## Separate raw dataset packages

Only `SSGT-WUT-raw.zip` and `SSGT-KGBD-raw.zip` are supplied in the dataset
delivery. They extract to `raw/WUT/` and `raw/KGBD/` below your chosen directory.
These folders are inputs to the converters. Generate the prepared arrays locally
before training; no prepared-data archive is distributed.

Each package includes `dataset_info/<package>/MANIFEST.json` and
`DATASET_README.md`. See [DOWNLOADS.md](DOWNLOADS.md) for checksum verification,
extraction, and preprocessing commands. OOS trials can be prepared separately
from these raw packages with the explicit manifests in `config/trials/`.

KS20 is excluded from dataset delivery because its original notice requires
permission for redistribution. The KS20 converter and protocol below remain
available for users who obtain the data independently. Dataset access terms
are separate from the source-code license.

## Locations and commands

`paths.data_root()` defaults to `Datas/` beside the source files. Set `SSGT_DATA_DIR` or use `main.py --data-dir` to select another prepared-data root. Raw input paths are supplied explicitly and may be anywhere readable; preprocessing does not move input directories or write fusion caches into them.

Run from the release directory:

```bash
python main.py --task preprocess --dataset WUT \
  --input_dir /path/to/WUT --unknown_dir /path/to/Unknown --length 6

python main.py --task preprocess --dataset KGBD \
  --input_dir '/path/to/KGBD/kinect gait raw dataset' \
  --unknown_dir /path/to/KGBD/Unknown --length 6

python main.py --task preprocess --dataset KGBD \
  --input_dir '/path/to/KGBD/kinect gait raw dataset' \
  --skip_unknown --length 6 --src_fps 30 --dst_fps 10 --data_subdir 6_10fps

python main.py --task preprocess --dataset KS20 \
  --input_dir /path/to/KS20 --length 6
```

The same options work with `python preprocessing.py`. `--fr` is an alias for `--length`. For WUT/KGBD, `--fusion_file` selects the per-sequence array filename. `--out_tag 10fps` is the legacy equivalent of `--data_subdir 6_10fps` for a six-frame window. `--clip_num` retains the original trimming behavior: the default is zero for WUT/KGBD and ten frames from each end for KS20.

Random seeds do not replace split definitions. WUT and KS20 use their fixed JSON assignments; explicit OOS trial manifests also retain their recorded identities and sequence choices. Random choices in training and evaluation use the seed selected by their own entry point. See [the reproducibility note](../README.md#note-on-reproducibility) for replay options.

The public datasets are WUT, KGBD, and KS20. All three use the prepared coordinate format below. Use a separate prepared-data root or variant directory for each experimental protocol.

## WUT inputs and split

```text
/path/to/WUT/
  Aleks/
    sequence_00/fusion_master.npy
    sequence_01/fusion_master.npy
    ...
  Jacek/
    sequence_00/fusion_master.npy
    ...
/path/to/Unknown/
  person/
    sequence/fusion_master.npy
```

The separate WUT raw package contains 293 sequences with
`fusion_master.npy` (5C), `fusion_master_3c.npy` (3C), and
`fusion_master_2c.npy` (2C) for each sequence. Use `--fusion_file` to select one
variant and `--data_subdir 6_3c` or `6_2c` for the corresponding N=6 preset.
The included per-camera CSV files are retained source data; the converter uses
the fused NPY arrays. See [the camera-variant commands](DOWNLOADS.md#convert-raw-data).

The packaged fusion arrays have shape `(frames, 19, 4)` and dtype `float64`.
The converter exports channels 0, 1, and 2 as X, Y, and Z. It does not export the
fourth channel, but its inherited NaN-window filter examines **all channels**:
a NaN in that channel can still remove a window. Preserve the original array
when reproducing the recorded preparation. More generally, the converter
accepts `(frames, 19, C)` with at least three channels. The per-camera CSV files
instead contain 32 joints in 128 columns, with headers such as
`x0,y0,z0,p0,...,x31,y31,z31,p31`; they are not the fused 19-joint model input.

Unknown data may also be supplied as sequence directories directly under the
unknown root, or as one fusion file directly under that root. Raw identity
directory names must match the original WUT name-to-ID mapping in
`process_dataset_WUT`; unrecognized names are skipped. For a custom dataset,
adapt that mapping explicitly.

`config/wut_split.json` replaces the original `random3.npy` array `[8, 0, 1]` without changing its meaning:

- Gallery: zero-based positions **8 and 0** in each identity's alphabetically sorted sequence directories.
- Probe: zero-based position **1**.
- Training: every other sequence position.

These are sequence-list positions, not frame indices or literal folder names. The default assumes at least nine sequence directories per identity, with the selected fusion files present. Use `--split_file /path/to/split.json` for an explicit alternative with two distinct gallery indices and one distinct probe index. A new split changes the experimental protocol.

The archived WUT collection contains **293 sequences from 25 identities**: most identities have ten sequences, while Julia has 27, Moncy 19, Showni 19, and Vibek 18. Applying the recorded indices to the complete collection assigns 218 sequences to training, 50 to gallery, and 25 to probe. All additional sequences remain in training. The paper's 7:2:1 description therefore describes the typical ten-sequence identity; the aggregate archived ratio is 218:50:25, approximately 8.72:2:1. Do not truncate every identity to ten sequences when reproducing the recorded checkpoints.

The selected Julia probe sequence produces **zero valid six-frame windows** after the original filtering. Consequently, all five archived WUT OOS trials omit original ID 3 from their prepared probe arrays, while retaining it in training and gallery. This concrete coverage limitation is preserved and recorded as `missing_known_ids.probe: [3]` in preparation provenance. The reconstruction does not invent replacement probe windows or silently select a different sequence.

Known WUT identities use the original IDs 1–25. Unknown windows receive raw label 26. The unknown directory must be disjoint from the known data according to the intended experimental protocol; preprocessing does not choose or move held-out identities automatically.

## KGBD inputs and split

```text
/path/to/kinect gait raw dataset/
  Person001/
    sequence_01.txt
    sequence_02.txt
    sequence_03.txt
    sequence_04.txt
  Person002/
    ...
/path/to/KGBD/Unknown/
  person/
    sequence.txt
```

Known-person directories must contain `.txt` sequence filenames. Each valid line has a leading joint field followed by semicolon-separated X, Y, and Z values; each frame has 20 joint records. If `PersonXXX/<sequence-stem>/<fusion_file>` already exists, preprocessing reads that array. Otherwise it parses the text directly, without creating a cache in the input directory. The separate raw package preserves both the 822 TXT sequences and their 822 existing fusion caches, since historical OOS manifests reference and hash the cache files for known identities.

The original KGBD split selects two gallery sequences and one probe sequence independently per identity, leaving the others for training. Person015 uses one gallery and one probe sequence. Provide enough sequences to leave training data: normally at least four per identity, or three for Person015. The CLI defaults to `--seed -1`, which generates a new unsigned 32-bit seed from operating-system entropy, prints it, and records `requested_seed`, `resolved_seed`, and `seed_selection` in `<dataset>/<variant>/preprocessing_metadata.json` below the prepared-data root. Use `--seed N` to repeat the selection for the same sorted input filenames and software version. A generated seed can change KGBD sequence assignments. Retain the prepared arrays or the recorded seed when comparing runs.

The complete archived KGBD collection contains 822 text sequences across 164 identities. Its closed-set allocation is 331 training, 327 gallery, and 164 probe sequences, including the Person015 exception. For historical OOS checkpoints, use the released per-trial manifests: those manifests record the actual per-identity sequence choices from the logs, so reconstruction does not depend on guessing the old random seed.

KGBD unknown inputs are `.txt` files directly below the supplied unknown root or inside one level of person directories. The unknown path is never inferred from the current working directory. Unknown windows receive raw label 165, after the original 164 known identity IDs. `--skip_unknown` or `--skip_unknown=1` disables construction of the unknown split.

## KS20 inputs and split

The official KS20 collection contains five viewpoint folders: `frontal`, `left_diagonal`, `left_lateral`, `right_diagonal`, and `right_lateral`. Each contains 60 MATLAB files named `gc1_body1.mat`, `gc2_body1.mat`, and so on: three gait-cycle recordings for each of 20 identities. Supply all 300 files. The converter reports missing viewpoints or sequences instead of silently training on a partial collection.

`ks20_preprocessing.py` reads the native nested `gc` MATLAB objects and preserves all 25 Kinect v2 joints. It uses the recorded joint type as the index, fills an entirely zero joint row from the previous frame when one exists, and converts coordinates to float32 before writing the original float64 coordinate format. Body numbers `1–4, 7–9, 11–17, 19–24` map in order to labels `0–19`.

`config/ks20_split.json` records one distinct `gc1`/`gc2`/`gc3` setting for train, gallery, and probe for every identity. The same assignment applies across all five views, giving exactly **100 raw sequences per split** and the paper's 1:1:1 sequence ratio. These assignments were recovered by matching the existing prepared windows to the raw MAT files; their original random seed was not recorded. The fixed JSON, rather than `--seed`, controls KS20 splitting. An explicit `--split_file` must preserve all 20 labels and assign three distinct settings per identity.

The default converter removes ten frames from each end of every recording. For length N, training uses shifts `0..N-1` and stride `N // 2`; gallery and probe use shift zero and non-overlapping stride N. This inherited training augmentation deliberately repeats some windows. For N=6, the verified collection produces **1,886 training, 212 gallery, and 218 probe windows** before loader resampling. Short recordings can contribute no complete window: this occurs for two train, seven gallery, and six probe sequences in the recovered split. Every identity nevertheless has valid windows in every split.

The converter writes all 15 coordinate/label/index files under `Datas/KS20/<length or variant>/` and records source hashes, sequence assignments, trimming, window counts, and output hashes in `preprocessing_provenance.json`. Inputs remain unchanged. KS20 preprocessing supports the closed-set protocol; it does not create an unknown split or apply FPS resampling. The standalone command is also available:

```bash
python ks20_preprocessing.py --input_dir /path/to/KS20 --length 6
python main.py --config config/paper/ks20_f6.json --task train
```

## Explicit trial manifests

The published OOS trial manifests are `config/trials/wut/exp01.json` through `exp05.json` and `config/trials/kgbd/exp01.json` through `exp05.json`. Each records original identity membership, exact sequence filenames and split assignments, source byte counts and SHA-256 fingerprints, the matching checkpoint digest, and historical window-count evidence. Paths inside these manifests are relative to the raw identity root. Their internal trial names retain the original form `exp_01` through `exp_05`.

Prepare a trial directly from the full original identity collection:

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root /path/to/full/WUT \
  --prepared-root reproduction_runs/wut/exp01/data

python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root '/path/to/full/kinect gait raw dataset' \
  --prepared-root reproduction_runs/kgbd/exp01/data
```

Known and held-out people remain in their original raw directories. The manifest selects 23 known and two unknown WUT identities, or 148 known and sixteen unknown KGBD identities. Outputs are written only below the separate prepared root. That root contains `<dataset>/6/` and `provenance.json`, which binds the manifest to source fingerprints, every prepared file, original labels, per-sequence window counts, and any missing known IDs. A fingerprint or recorded-count mismatch stops preparation. Existing prepared data are reused only after provenance verification.

These prepared splits can be used for **fresh training with your own checkpoints**. `main.py --task train`, `evaluate`, `oos`, and `oos-kgbd` do not require an archived checkpoint digest. The `checkpoint_sha256` restriction applies to the optional `reproduce.py evaluate` workflow that verifies a specific archived experiment. See [the self-training recipe](PAPER_REPRODUCTION.md#train-and-evaluate-your-own-oos-model) before using that archival verifier.

Users with their own recorded experiment logs can recover a manifest explicitly:

```bash
python protocols.py --dataset WUT --trial exp_01 \
  --group-root /path/to/recorded_experiment_group \
  --raw-root /path/to/full/WUT --manifest custom_trials/wut_exp01.json
```

Recovery reads the group's `summary.json`, `logs/exp_01_preprocess.log`, and available training/evaluation evidence. Historical logs record sequence positions and counts; the newly recovered hashes pin the files present during recovery and do not establish their past contents. The released manifests already contain the recovered selections, so end users do not need the original logs. See [Paper reproduction](PAPER_REPRODUCTION.md) for checkpoint verification, evaluation, aggregation, and the manifest API.

## Windowing and output format

Known WUT/KGBD sequences and WUT unknown sequences are divided into complete, non-overlapping windows. Incomplete final windows and complete windows containing NaNs are dropped. KS20 uses the shifted training windows described above and requires finite parsed coordinates. The inherited KGBD unknown-text branch retains its original behavior: it groups records into frames and windows without applying the known-data NaN filter. Provide finite coordinates in KGBD unknown files. Unknown windows are deduplicated within the unknown split using coordinates rounded to four decimal places; there is no automatic deduplication across known and unknown splits.

Optional downsampling selects existing frames on the nearest uniform time grid before windowing; it does not interpolate poses. Native frame-rate defaults are WUT 15 FPS and KGBD 30 FPS. The WUT raw package has no separate 10-FPS or 5-FPS files; these variants are derived from the 5C fusion by downsampling during preprocessing. Use a separate `--data_subdir` for each frame-rate or split variant. Rerunning preprocessing does not delete stale files from earlier runs, so use a fresh output subdirectory when changing which splits are produced.

```text
Datas/
  WUT/
    6/
      train_npy_data/
        source_x_WUT_6.npy
        source_y_WUT_6.npy
        source_z_WUT_6.npy
        ids_WUT_6.npy
        frame_id_WUT_6.npy
      test_npy_data/
        gallery/
          t_source_x_WUT_6.npy
          t_source_y_WUT_6.npy
          t_source_z_WUT_6.npy
          ids_WUT_6.npy
          frame_id_WUT_6.npy
        probe/
          ...
      unknown_npy_data/
        t_source_x_WUT_6.npy
        t_source_y_WUT_6.npy
        t_source_z_WUT_6.npy
        ids_WUT_6.npy
        frame_id_WUT_6.npy
```

Replace `WUT` with `KGBD` or `KS20` for their files; KS20 has no `unknown_npy_data` directory. A variant such as `6_10fps` changes the directory name, while filenames retain the actual window length `6`.

| File | Contents |
| --- | --- |
| `source_{x,y,z}_*.npy` | Training coordinates, each shape `(N * T, J)` |
| `t_source_{x,y,z}_*.npy` | Gallery, probe, or unknown coordinates in the same shape |
| `frame_id_*.npy` | One integer identity label per **window**, shape `(N,)`, despite the historical filename |
| `ids_*.npy` | NumPy object containing a dictionary from raw identity ID to window-index list |

The object dictionary format is retained for compatibility. Load only prepared arrays from a trusted source.

## External NPY queries

`main.py --task oos --data_source npy --input sequence.npy` accepts an external
sequence with shape `(frames, J, 3)`; the inherited shape adapter also accepts
channel-first or flattened coordinate layouts. Coordinate units, joint order,
and normalization must agree with training. This adapter does not infer a
skeleton convention or apply the KGBD joint permutation.

For a KGBD external query, first subtract original joint zero and apply the
permutation listed below, then use `--task oos-kgbd --center_root 0`. For a WUT
query in the original joint order, the default root centering is appropriate.

The external-query path removes frames containing NaNs before windowing. Exported
frame predictions and window spans refer to this cleaned sequence, not the original
frame numbers; maintain your own original-frame mapping if aligning to a video.
This behavior is inherited and differs from dropping entire NaN-containing windows
in the raw-data converters.

## OOS identity labels

OOS retrieval labels are decoded through the prepared training identity keys,
so holding out identities does not shift names or original dataset IDs.
`--one_based_ids 1` (default) uses one-based numeric ground truth and exports;
`0` uses zero-based values. WUT also accepts its original identity names as
`--groundtruth`.

CSV rows retain the best known retrieval candidate and report rejection in the
`oos_window` column. The exported `_win_id.npy` applies that rejection directly:
its default one-based unknown ID is 26 for WUT, 165 for KGBD, or 21 for KS20.
`_topk_id.npy` retains the ranked known candidates. The result JSON records the
indexing convention, original known IDs, and unknown sentinel.

## Loader contract

`data.gen_train_data(...)` retains the original function signature. The SSGT path uses `D=False`, `graph=False`, and `L3=True`, and returns **26 entries**:

| Index | Value |
| --- | --- |
| 0, 1, 2 | Training joint, part, and body windows |
| 3, 4 | Legacy placeholders, zero |
| 5 | Training one-hot labels |
| 6, 7, 8 | Requested split joint, part, and body windows |
| 9, 10 | Legacy placeholders, zero |
| 11 | Requested split one-hot labels |
| 12 | Joint adjacency, shape `(1, J, J)` |
| 13 | Legacy placeholder, zero |
| 14 | Joint Laplacian encoding, shape `(J, enc_k)` |
| 15, 16, 17 | Part adjacency, zero placeholder, part Laplacian encoding |
| 18, 19, 20 | Body adjacency, zero placeholder, body Laplacian encoding |
| 21–24 | Legacy placeholders, zero |
| 25 | Configured number of classes: WUT 25, KGBD 164, KS20 20 |

Joint windows have shape `(N, T, J, 3)`, part windows `(N, T, 10, 3)`, and body windows `(N, T, 5, 3)`. Coordinates are centered on original joint zero. KGBD then applies the inherited joint permutation `[11,10,1,0,2,4,6,8,3,5,7,9,12,14,16,18,13,15,17,19]`; the centered root consequently appears at model joint index three.

One-hot columns follow the sorted **training identity keys**. Labels outside that known set become all-zero rows, including unknown labels 26/165. These rows indicate unknown samples and must not be interpreted using `argmax` as a known identity. The final tuple field retains the fixed dataset capacity even if a protocol uses fewer known identities. New training uses the number of one-hot columns for its actual prototype count: 23/148 for the WUT/KGBD OOS splits, rather than the legacy capacities 25/164. Every known identity must appear in the complete training batches used to build prototypes; missing coverage raises an error. This corrects the archived unused-zero-prototype behavior without changing full closed-set class counts or saved-checkpoint inference.

The legacy class-balanced training sampler is retained for non-KGBD datasets. The loader raises `ValueError` when the number of training windows is less than `batch_size`, when `batch_size` is less than the number of known classes, or when an identity has no windows. Use sufficient windows and an appropriate batch size. All datasets require at least **three known training identities**: smaller protocols raise `ValueError` because the inherited scikit-learn binary label encoding would not match the multiclass downstream code. Custom binary datasets need an explicit label-encoding adaptation. Alternative legacy `D=True` or `graph=True` branches raise `NotImplementedError`; the public SSGT path uses both options as `False`.
