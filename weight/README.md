# Model checkpoints

Training writes `<dataset>/<configuration>/best.ckpt` here. No pretrained weights
are provided; train a model using the [README instructions](../README.md).

Use `--model-dir /path/to/weights` to choose another checkpoint root and
`--ckpt /path/to/best.ckpt` for evaluation. Load only trusted checkpoints.
