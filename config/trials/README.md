# SSGT open-set training splits

The five manifests in each of `wut/` and `kgbd/` define complete identity and sequence assignments for open-set training and evaluation. WUT uses 23 known and 2 unknown identities per split; KGBD uses 148 known and 16 unknown identities. Each `exp01.json` through `exp05.json` is an independent split. Training and evaluation select their own run seeds; changing a seed does not change these explicit assignments.

Run the following commands from the project directory after extracting the raw data packs. The raw roots must contain all original identities, including those assigned to the unknown split.

```bash
python protocols.py --manifest config/trials/wut/exp01.json \
  --raw-root raw/WUT --prepared-root Datas/oos/wut/exp01

python protocols.py --manifest config/trials/kgbd/exp01.json \
  --raw-root raw/KGBD --prepared-root Datas/oos/kgbd/exp01
```

Pass the resulting root to training with `--data-dir` and use a separate model directory for each split:

```bash
python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir Datas/oos/wut/exp01 --model-dir weight/oos/wut/exp01

python main.py --task oos --dataset WUT --length 6 \
  --data-dir Datas/oos/wut/exp01 --model-dir weight/oos/wut/exp01 \
  --output-dir results/oos/wut/exp01
```

For KGBD, use `config/paper/kgbd_f6.json` for training, `--task oos-kgbd --dataset KGBD` for evaluation, and the corresponding KGBD data/model/output roots. Evaluation locates `best.ckpt` in that model root; `--ckpt` can select a specific checkpoint. See [Data preparation](../../docs/DATA.md) for the directory layout and coordinate formats.

## Manifest fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest format version, currently `1`. |
| `dataset` | `WUT` or `KGBD`; determines the skeleton layout and unknown label. |
| `trial_id` | Optional name identifying the split. |
| `length` | Frames per complete, non-overlapping window; the supplied manifests use `6`. |
| `data_subdir` | Directory below the dataset name, normally the window length. |
| `preprocessing` | Optional `src_fps`, `dst_fps`, and `clip_num`; zero disables resampling or clipping. Clipping removes complete windows from known sequences and WUT unknown sequences, and frames from KGBD unknown sequences. |
| `expected_identities` | Declared counts of known and unknown original identities. |
| `identities` | Ordered list of original identity names, positive labels, roles, and sequence definitions. |

Each identity has `name`, `original_label`, `role`, and `sequences`. A known identity has sequences assigned to `train`, `gallery`, and `probe`; an unknown identity has only `unknown` sequences. Each sequence specifies `name`, `split`, a relative `source` filename, `format` (`npy` or `kgbd_txt`), `size_bytes`, and `sha256`. List order defines output window order. WUT uses NPY skeleton arrays; KGBD supports both NPY arrays and semicolon-separated TXT coordinates. The KGBD raw pack includes the arrays referenced by these manifests.

Preparation checks every input file before writing arrays, keeps raw files in place, and rejects overlapping identities or sources. Unknown windows are deduplicated and receive the dataset's unknown label. The output `provenance.json` records actual window counts, source/output hashes, and identities without valid windows in each split. The training split must cover every known identity; gallery and probe must be nonempty. An individual gallery/probe sequence may yield no complete valid windows, which is reported without reassigning it.

Reusing an output directory verifies its manifest binding and output hashes. A directory containing different prepared data is not overwritten. Use a new output root when selecting another manifest or changing preparation settings.
