# Prepared datasets

The source archive includes only this guide in `Datas/`. Generate dataset arrays
locally from raw skeletons before training. No prepared-data package or model
checkpoint is distributed.

Download the [WUT raw archive](https://drive.google.com/file/d/1PBWrIvwNyMTTh-ZoyUujaY6j6vtHPlhH/view?usp=sharing) or
[KGBD raw archive](https://drive.google.com/file/d/1MUjeOndmCqsn2VPEVC_mmB1jQ1hszsDo/view?usp=sharing) separately.
See [download instructions](../docs/DOWNLOADS.md) for extraction and checksums.
For example, after extracting WUT:

```bash
python main.py --task preprocess --dataset WUT --length 6 \
  --input_dir /path/to/ssgt_data/raw/WUT --data-dir /path/to/prepared_data
python main.py --config config/paper/wut_5c_f6.json --task train \
  --data-dir /path/to/prepared_data
```

Run these commands from the source directory containing `main.py`. The converter
writes `WUT/6/` below the prepared-data root. The KGBD converter follows the same
layout. Pass the parent root to `--data-dir`, not the dataset or length folder.
Training generates a seed by default; KGBD preprocessing also generates a split
seed and records it beside the output arrays.

KS20 data are not redistributed because the original dataset notice requires
permission. Users who obtain KS20 independently can use the included converter
to create `KS20/6/` locally. See [the data protocol](../docs/DATA.md) for all input
formats, split definitions, and prepared files.
