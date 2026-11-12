# Model checkpoints

Training creates `<dataset>/<configuration>/best.ckpt` here.
Pretrained weights are not included in the source distribution.
Pass `--ckpt /path/to/best.ckpt` when evaluating an existing model.

The `model_state` and `args` checkpoint fields retain their original meanings.
`TranSG` remains an alias of `SSGT` for source compatibility.
