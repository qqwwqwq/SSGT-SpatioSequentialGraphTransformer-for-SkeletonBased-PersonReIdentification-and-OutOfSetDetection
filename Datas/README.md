# Prepared datasets

Generate arrays here from the [WUT and KGBD raw downloads](../docs/DOWNLOADS.md).
Prepared data are not bundled. Run from the repository root:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/raw/WUT
python main.py --config params.json --task train
```

The converter writes `Datas/WUT/6/`. KGBD and KS20 use the same dataset/variant
layout. To store arrays elsewhere, pass `--data-dir /path/to/prepared` to both
preprocessing and training, or set `SSGT_DATA_DIR`. Pass the parent root, not the
dataset or length folder.

KS20 must be obtained independently because its dataset notice requires
redistribution permission. See [data formats and split rules](../docs/DATA.md)
and [training presets](../docs/HYPERPARAMETERS.md).
