# SSGT Hyperparameter Reference

This reference describes the options accepted by [main.py](../main.py). Defaults below are the parser defaults; a JSON configuration or command-line option can replace them. Integer switches use `0` for disabled and `1` for enabled. Dataset layout and preparation instructions are in [DATA.md](DATA.md).

## 1. Configuration and task selection

```bash
python main.py --config params.json --task train
python main.py --config params.json --task train --batch_size 128 --device cpu
python main.py --config params.json --task train --dry-run
python main.py --task train --help
python main.py --task oos-kgbd --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--task` | `train` | One of `train`, `evaluate`, `preprocess`, `oos`, or `oos-kgbd`. An explicit value overrides the JSON `task`. |
| `--config` | None | UTF-8 JSON configuration. It is loaded only when supplied; running without it uses parser defaults. |
| `--dry-run` | Disabled | Print the selected module, resolved path overrides, and argument tokens without loading data or running the task. It does not perform the task parser's numerical validation. |
| `--help` | Disabled | Without a task/configuration, show entry-point help. With one, show that task's options. |

JSON uses a `paths` object and a separate object for each task. Task option names omit the leading `--` and preserve capitalization and underscores. Values must be scalars; `null` omits an option, and JSON booleans become `0` or `1`. Lists and nested option objects are not accepted.

```json
{
  "task": "train",
  "paths": {
    "data_dir": "Datas",
    "model_dir": "weight/run_a",
    "output_dir": "results/run_a"
  },
  "train": {
    "dataset": "WUT",
    "length": 6,
    "data_subdir": "6",
    "batch_size": 256,
    "H": 128,
    "seed_val": -1
  },
  "evaluate": {
    "ckpt": "weight/run_a/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt",
    "metrics_output": "results/run_a/closed_set.json"
  }
}
```

Only the selected task's object is read. `train` settings are not automatically merged into `evaluate` or `oos`. Task arguments from the command line appear after configured arguments, so explicit CLI values take precedence. For training/evaluation, `main.py` sets `--mode Train` or `--mode Eval` from `--task`; choose the task instead of overriding `--mode`.

### Paths

| Entry-point option | JSON key under `paths` | Environment variable | Default location |
| --- | --- | --- | --- |
| `--data-dir` / `--data_dir` | `data_dir` | `SSGT_DATA_DIR` | `Datas/` beside `main.py` |
| `--model-dir` / `--model_dir` | `model_dir` | `SSGT_MODEL_DIR` | `weight/` beside `main.py` |
| `--output-dir` / `--output_dir` | `output_dir` | `SSGT_OUTPUT_DIR` | `results/` beside `main.py` |

For these roots, precedence is CLI > JSON > environment > source-tree default. Relative JSON paths are resolved against the JSON file's directory. Relative command-line paths and relative environment overrides are resolved against the working directory. Recognized task paths, including `ckpt`, `input`, `input_dir`, `unknown_dir`, `split_file`, `metrics_output`, and the `save_*` destinations, follow the same JSON-relative rule. `data_subdir` is a dataset-directory selector, not a filesystem root.

## 2. Training data and dataset variants

| Training/evaluation option | Default | Meaning and constraints |
| --- | --- | --- |
| `--dataset` | `WUT` | `WUT`, `KGBD`, or `KS20`. Their input skeletons have 19, 20, and 25 joints, respectively. |
| `--length` | `6` | Frames per window, integer at least 2. Must match the prepared arrays and filename suffix. |
| `--data_subdir` | Empty | Directory under `<data-dir>/<dataset>/`; empty selects the decimal window length, such as `6`. |
| `--probe` | `probe` | Prepared query split under `test_npy_data/`. The supplied protocols use `probe`; the gallery is loaded from `test_npy_data/gallery/`. |
| `--batch_size` | `256` | Positive window count per training or feature-extraction batch. Training uses complete batches. WUT/KS20 balanced sampling additionally requires at least as many batch slots as known identities and at least one batch of raw training windows. |

The loader requires at least three known training identities. Prototypes are formed for the identities actually present in the training labels. Each identity must have features in the complete batches used for prototype construction. An error reports missing identities rather than substituting an empty prototype. WUT/KS20 use balanced sampling followed by a seeded permutation; KGBD uses a seeded permutation without that balancing step. Changing `batch_size` can therefore change the sampled training/calibration library, not only memory use.

The files in [config/](../config/) provide these input variants while retaining the same final encoder and loss defaults:

| Dataset/variant | `length` | `data_subdir` | Preparation setting |
| --- | --- | --- | --- |
| WUT 5-camera fusion | 6, 12, or 18 | `6`, `12`, or `18` | `fusion_master.npy` |
| WUT 3-camera fusion | 6 | `6_3c` | `fusion_master_3c.npy` |
| WUT 2-camera fusion | 6 | `6_2c` | `fusion_master_2c.npy` |
| WUT 5/10 FPS | 6 | `6_5fps` / `6_10fps` | 5-camera fusion, downsample from 15 FPS |
| KGBD | 6, 12, or 18 | `6`, `12`, or `18` | Native-rate skeletons |
| KGBD 10/20 FPS | 6 | `6_10fps` / `6_20fps` | Downsample from 30 FPS |
| KS20 | 6 | `6` | KS20 view/sequence split |

Selecting a training preset does not perform preprocessing or create an unknown split. In [params.json](../params.json), the default WUT subdirectory is explicitly `6`; the parser's empty default resolves to the same directory. If you change `length` in a configuration that explicitly sets `data_subdir`, update both values.

## 3. Encoder

| Option | Default | Meaning and usable values |
| --- | --- | --- |
| `--H` | `128` | Hidden feature width. Must be positive and divisible by `n_heads`; use at least 2 so the trajectory reconstruction hidden layer has nonzero width. |
| `--n_heads` | `8` | Positive number of heads in each spatial and temporal attention layer; per-head width is `H / n_heads`. |
| `--L_transformer` | `2` | Number of spatial/temporal attention-and-merge blocks. Use a positive integer. |
| `--enc_k_spa` | `10` | Number of nonconstant joint-graph Laplacian eigenvectors. Use an integer from 1 through `J - 1`, where `J` is the dataset's joint count. The loader does not pad an oversized request. |
| `--enc_k_time` | `4` | Number of nonconstant chain-graph temporal eigenvectors. The model clamps this to `[1, length - 1]`. |
| `--pre_norm` | `1` | Apply LayerNorm before each spatial/temporal attention block. |
| `--attn_dropout` | `0.4` | Attention-probability dropout. Use a probability in `[0, 1]`; disabled during evaluation. |
| `--attn_clamp` | `1` | Clamp attention logits to `[-5, 5]` before softmax. |
| `--qkv_merged` | `1` | Produce query/key/value projections with one combined linear layer. `0` uses separate projections. |
| `--rand_flip` | `0` | When `1`, randomly flip spatial eigenvector signs during gradient-update batches. Prototype construction and closed-set evaluation use the stored basis without these flips. KGBD forces this option to `0`. |

The output/FFN dropout probability is fixed at `0.5` by the training entry point and has no separate CLI flag. Spatial and temporal position features are projected to `H` dimensions. The temporal basis is computed with a dense symmetric eigensolver and saved in the model state; the spatial basis is also saved with checkpoints.

### Required final-model settings

Training enforces the following values. Keep them in a custom configuration:

| Option | Required value | Effect |
| --- | --- | --- |
| `--pos_enc` | `1` | Add spatial and temporal position information. |
| `--spa_pos_type` | `laplacian` | Use the skeleton graph's spatial Laplacian basis. |
| `--use_time_attn` | `1` | Enable temporal attention together with spatial attention. |
| `--frame_attn_pool` | `0` | Use mean pooling rather than a learned frame-pooling layer. |
| `--learnable_temp` | `0` | Keep contrastive temperatures fixed at `t_1` and `t_2`. |
| `--enable_dynamic_gpc` | `1` | Apply the dynamic loss-weight update. |
| `--train_eval_gallery` | `split` | Select checkpoints using the prepared gallery/query split. |

The parser also recognizes other values for some of these fields so the evaluation task can reconstruct a checkpoint's recorded architecture. They are not accepted as alternative final-model training settings.

## 4. Losses and dynamic balancing

The coefficient names map to the implemented objective as follows:

```text
alpha = seq_lambda
beta  = prompt_lambda
lambda starts at GPC_lambda and is updated during training

L_GPC  = alpha * L_sequence + (1 - alpha) * L_skeleton
L_STPR = beta  * L_spatial  + (1 - beta)  * L_trajectory
L      = lambda * L_GPC    + (1 - lambda) * L_STPR
```

`seq_lambda` follows the sequence-loss coefficient in Equation (4) of the paper. For reconstruction, Equation (5) places its beta on the **trajectory** term, whereas the CLI's `prompt_lambda` weights the **spatial** term. To translate a nondefault paper beta, use `prompt_lambda = 1 - beta_paper`. Both conventions give equal weighting at the default `0.5`.

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--seq_lambda` | `0.5` | Weight of sequence-level GPC. Use `[0, 1]` for a convex mixture with skeleton-level GPC. This range is not checked by argparse. |
| `--prompt_lambda` | `0.5` | Weight of spatial reconstruction. Use `[0, 1]` for a convex mixture with trajectory reconstruction. This range is not checked by argparse. |
| `--GPC_lambda` | `0.5` | Initial GPC weight; finite and in `[0, 1]`. Dynamic balancing can change it later. |
| `--t_1` | `0.07` | Finite positive temperature dividing cosine-similarity sequence-GPC logits. |
| `--t_2` | `14.0` | Finite positive temperature dividing projected skeleton-GPC dot products. Skeleton cross-entropy is summed over frames and averaged over the batch. |
| `--S_prompt` | `l1` | Spatial reconstruction: `l1` is mean absolute error, `l2` is half the mean squared error, and `MSE` is mean squared error. |
| `--T_prompt` | `l1` | Trajectory reconstruction, with the same three loss definitions. |
| `--gpc_up_period` | `10` | Initial absolute completed-epoch update threshold; integer at least 2. Subsequent thresholds double. |
| `--gpc_iota` | `10.0` | Finite slope-ratio threshold, strictly greater than 1. |
| `--gpc_tau` | `0.05` | Finite positive increment/decrement of the GPC weight. |

Both reconstruction losses are required; `S_prompt=0` and `T_prompt=0` are rejected for training. The default final configuration uses L1 for both.

With the default period, weight updates occur after 10, 20, 40, 80, ... completed epochs. The ratio is the signed average normalized slope of GPC divided by that of STPR. Each normalized difference is `(previous_loss - next_loss) / (previous_loss + next_loss)`, with a denominator floor; a two-value history uses the adjacent difference. If the ratio is below `1 / gpc_iota`, lambda increases by `gpc_tau`; above `gpc_iota`, it decreases by that amount; otherwise it stays unchanged. Lambda is clipped to `[0, 1]`. Two near-zero slopes cause no change; a near-zero STPR slope uses a signed epsilon denominator. The slope discretization and boundary handling make the numerical update explicit; no learning-rate changes are coupled to this rule.

### Reconstruction masks

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--Tr_mask_num` | `2` | Number of frames excluded from the trajectory-prompt feature average. Required range: `1 <= Tr_mask_num < length`. |
| `--St_prompt_type` | `3` | Joint-mask selection rule; one of `1`, `2`, or `3`. |
| `--St_mask_num` | `10` | Number of excluded joints for type `3`. Required range for that type: `1 <= St_mask_num < J`. |

Type `3` selects `St_mask_num` random joints to exclude. Type `1` excludes a graph-based set produced by greedily selecting a joint and marking its neighbors as visited. Type `2` excludes the same number of randomly selected joints as that graph-based set. `St_mask_num` does not determine the size for types `1` and `2`.

Masks act on encoded joint/frame features used by the reconstruction heads and sequence prototype pooling. They do not erase coordinates before the encoder. One joint mask and one frame mask are sampled per complete batch, shared across that batch, and reused during the same epoch's gradient update. Reconstruction targets contain all joints or the full trajectory.

## 5. Optimization, stopping, and checkpoints

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--lr` | `0.00035` | Adam learning rate; use a positive finite value for learning. The KGBD path sets it to `0.00035` even if another value is supplied. |
| `--patience` | `500` | Consecutive evaluated epochs without a better selected checkpoint before stopping. Use a positive integer. |
| `--max_epochs` | `0` | `0` leaves stopping to patience, within a 50,000-iteration ceiling. A positive value is the maximum **zero-based epoch index, inclusive**. Thus `1` permits epochs 0 and 1, and `1000` permits up to 1,001 epochs. |
| `--amp` | `0` | Enable automatic mixed precision only when `1` and running on CUDA. CPU runs remain full precision. |
| `--bn_freeze_train` | `0` | When `1`, use stored BatchNorm statistics during gradient updates. The default updates them in training mode. |
| `--bn_freeze_proto` | `0` | Force stored BatchNorm statistics for prototype extraction. This path already puts the model in evaluation mode, so both values normally use stored statistics. |
| `--bn_freeze_eval` | `0` | Force stored BatchNorm statistics for closed-set evaluation. Evaluation mode already uses stored statistics. |
| `--save_model` | `1` | Save improved `best.ckpt` files when exactly `1`. A value of `0` disables checkpoint writes, but training and its CSV log still run. |

Adam uses its standard beta/epsilon defaults with zero weight decay. There is no exposed scheduler or gradient-clipping option. Each epoch builds prototypes, evaluates the current model, possibly saves it, then performs gradient updates. Consequently, epoch 0 evaluates the initialized model before the first optimizer step. Patience can stop before an epoch's update. A final console result labeled `Last-epoch model` describes the last in-memory weights; evaluate `best.ckpt` explicitly to report the selected checkpoint.

Selection prioritizes Rank-1, with mAP, Rank-5, and Rank-10 as successive tie-breaks. The default WUT checkpoint path is:

```text
<model-dir>/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt
```

The directory encodes the dataset, probe name, length, attention depth/heads, initial loss weights, and optional data variant. It does not encode every option or the random seed. Use a separate `--model-dir` for independent runs to keep their checkpoints and CSV logs separate. `train` starts a new model; `--ckpt` is an evaluation input and does not resume training.

### Device and random seed

| Training/evaluation option | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | `auto` selects CUDA when available, otherwise CPU. `cpu` forces CPU; `cuda` fails if CUDA is unavailable. |
| `--gpu` | `0` | CUDA visibility string, for example `0` or `1`. This does not enable multi-GPU training; the model uses one CUDA device. For training/evaluation on CPU, use `--device cpu`. |
| `--seed_val` | `-1` | `-1` selects a fresh unsigned 32-bit seed from system entropy. An explicit integer in `[0, 2^32 - 1]`, including `0`, seeds Python, NumPy, PyTorch, and the data sampler. |
| `--fix_seed` | `0` | `1` enables deterministic cuDNN settings and disables cuDNN benchmarking. It does not select a seed: explicit `seed_val` is honored with either setting. |

The chosen training seed is printed and stored in checkpoint metadata. Use that number to request the same random choices in another run. OOS and preprocessing use `--seed` instead of `--seed_val`. Automatic seeds are chosen once per run and shared with its data-loading calls. Matching seeds do not promise identical results across different devices, numerical libraries, or software versions.

## 6. Closed-set evaluation

```bash
python main.py --task evaluate --device cpu \
  --ckpt weight/run_a/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt \
  --metrics_output results/run_a/closed_set.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--ckpt` | Empty | Explicit checkpoint file. If empty, use `best.ckpt` in the directory constructed from the current dataset/model/loss-weight/variant arguments. No recursive latest-file search is used by this task. |
| `--metrics_output` | Empty | Optional JSON result destination. Reports Rank-1/5/10 and mAP as fractions in `[0, 1]`, plus configuration and input metadata. |
| `--metric` | `euclidean` | Closed-set feature distance: Euclidean on unnormalized embeddings, or cosine distance with normalization. The checkpoint's saved value takes precedence during evaluation. |
| `--meanap_mode` | `tf` | Accepts `tf` or `pt`; both select the same AP computation. |
| `--probe_type` | Empty | Leave empty for the supplied protocols. A nonempty value changes checkpoint-directory naming; `nm.nm`, `cl.cl`, and `bg.bg` also remove the nearest gallery entry during metric computation. |

The checkpoint restores dataset, window length, data variant, probe settings, architecture, position encoding, temperatures, `bn_freeze_eval`, `metric`, and `meanap_mode`. Supplying those fields again on the CLI does not replace the saved settings. Runtime device/GPU, batch size, seed controls, path roots, checkpoint path, and JSON output destination remain under current CLI/configuration control. A missing checkpoint or incompatible state fails before evaluation; evaluation does not train a model.

Prepared training, gallery, and query arrays remain necessary because the loader reconstructs the dataset label mapping and graph as well as the evaluation inputs. Spatial and temporal encodings stored with the checkpoint are used when present.

## 7. Out-of-set detection and identity retrieval

Use `--task oos` for WUT and `--task oos-kgbd` for KGBD. Both expose the common options below. The main differences are the default dataset, identity-retrieval metric, and extra visualization controls available in `oos`.

```bash
python main.py --task oos --dataset WUT --gpu cpu \
  --ckpt weight/run_a/WUT/probePytorch_6_layers_2_heads_8_alpha_0.5_beta_0.5_lambda_0.5/best.ckpt \
  --data_source unknown --proto_source gallery \
  --unk_knn_k 1 --unk_sigma 0.95 --sigma_eval 0.95 \
  --proj2d 0 --save_npy_base results/run_a/oos
```

### Inputs, library, and identity prediction

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--dataset` | `WUT` / `KGBD` | `oos` / `oos-kgbd` default, respectively. Accepted names are `WUT`, `KGBD`, `KS20`. Must match the checkpoint dataset. |
| `--ckpt` | Empty | Checkpoint path. If omitted, search `<model-dir>/<dataset>/` recursively for `best.ckpt` or `*_best.ckpt`, then choose the most recently modified file. Use an explicit file when several runs exist. |
| `--data_subdir` | `None` | Inherit the checkpoint's prepared-data variant. An explicit value overrides it for this task. |
| `--probe` | `probe` | Known split loaded alongside training data. An unknown input is selected separately by `data_source`. |
| `--data_source` | `unknown` | `unknown` reads the prepared `unknown_npy_data/` split; `npy` segments an external skeleton sequence. Neither option creates held-out identities automatically. |
| `--input`, `--npy_path`, `--npy-path` | Empty | Aliases for the existing external NPY file, required with `data_source=npy`. |
| `--proto_source` | `gallery` | `gallery` or `train` feature library for identity retrieval. This is a library of window embeddings; each identity is ranked by its minimum library distance. |
| `--batch_size` | `256` | Positive extraction batch size; also passed to the data sampler that constructs the training library. |
| `--topk` | `5` | Positive number of distinct retrieved identities, capped at the number present in the library. |
| `--metric_override` | `euclidean` / `cosine` | Identity-retrieval distance for `oos` / `oos-kgbd`, respectively. Both accept either value. |
| `--l2norm_features` | `1` | Normalize identity-retrieval embeddings. Cosine retrieval always normalizes, even when this flag is `0`. |
| `--gpu` | `0` | CUDA visibility string; `cpu` or `-1` forces CPU. These tasks fall back to CPU when CUDA is unavailable and do not expose `--device`. |
| `--seed` | `-1` | Fresh recorded seed, or explicit unsigned 32-bit seed for sampling and projections. |
| `--debug` | `0` | Additional retrieval diagnostics. |
| `--groundtruth` | Empty | Optional single identity for window-prediction accuracy. Numeric IDs follow `one_based_ids`; WUT also accepts a subject name. This option does not relabel OOS metric inputs. |
| `--one_based_ids` | `1` | Export dataset identity indices as one-based (`1`) or zero-based (`0`), preserving gaps in a held-out identity split. KS20 uses its contiguous 20-class labels rather than raw body-file numbers. |

Unknown identities must be excluded from training and the gallery during data preparation. The training, known query, and selected gallery files are required even when the input is an external NPY file. The model width, sequence length, joint count, and architecture come from the checkpoint.

`batch_size` bounds feature-extraction batches, not the OOS distance matrices. The current scoring helpers construct full training-to-training and input-to-training matrices, so their memory use also depends on the total number of library and input windows.

### External NPY segmentation

| Option | Default | Meaning |
| --- | --- | --- |
| `--sliding` | `0` | `0` uses non-overlapping windows of the checkpoint's length; `1` uses sliding windows. |
| `--stride` | `6` | Positive step in frames when sliding is enabled; ignored for non-overlapping segmentation. |
| `--center_root` | `1` | Subtract the selected root joint from every frame. Applies to external NPY input, not prepared arrays. |
| `--root_idx` | `0` | Root index, required in `[0, J - 1]` when centering is enabled. |

Preferred input shape is `(frames, joints, channels)` with at least XYZ channels. `(frames, channels, joints)`, flattened per-frame XYZ/XYZP, and compatible one-dimensional arrays are also recognized. Only XYZ enters the model. Supply joints in the model's expected order: the external-input path does not apply the prepared KGBD loader's joint permutation. Frames containing NaNs are removed before segmentation; reported frame spans refer to that cleaned sequence. An incomplete final window is omitted. These options do not alter existing prepared windows.

For KGBD, the prepared loader centers on raw joint 0 and then reorders joints with zero-based indices `[11, 10, 1, 0, 2, 4, 6, 8, 3, 5, 7, 9, 12, 14, 16, 18, 13, 15, 17, 19]`. Supply the same order for external NPY input. If those coordinates are still uncentered, use `--root_idx 3`; if already centered as in the prepared loader, use `--center_root 0`.

### OOS score and threshold

OOS scoring always uses the **training feature library**, independently of `proto_source`. Both library and query embeddings are L2-normalized; their score is the distance to the requested k-th neighbor, using `sqrt(max(2 - 2 * dot_product, 0))`. `metric_override` and `l2norm_features` affect identity retrieval, not this OOS score.

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--unk_detect` | `1` | Emit rejection decisions when enabled. Disabling it does not disable feature extraction or score calibration. |
| `--unk_method` | `knn` | The available OOS method. |
| `--unk_knn_k` | `1` | Positive neighbor rank. Calibration caps it at `N - 1` for `N` training samples; queries cap it at `N`. At least two calibration samples are required. |
| `--unk_sigma` | `0.95` | Quantile in `[0, 1]` of known calibration distances used for rejection. A window is unknown when its distance is strictly above the threshold. |
| `--unk_scope` | `window` | `window` rejects each window separately; `sequence` prints one decision from the median distance across the entire input. Sequence mode does not write per-window rejection masks. |
| `--eval_oos` | `1` | Compute AUROC and unknown acceptance at the selected known-data quantile. Use `0` for unlabeled/mixed external input when those OOS metrics would not describe the input. |
| `--sigma_eval` | `0.95` | Quantile in `[0, 1]` used for metric reporting, independently of `unk_sigma`. Keep `0.95` when interpreting the output key `FPR95` literally. |

Calibration uses each training sample's neighbor distance after skipping the first distance rank to account for its self match. Reported AUROC treats these calibration samples as known and the supplied input windows as unknown. `FPR95` is the fraction of those unknown windows accepted at the `sigma_eval` threshold; `TPR_ID_at_tau` reports the measured calibration acceptance. Known/unknown ties can make that acceptance differ slightly from the requested quantile. These are window-level metrics even when `unk_scope=sequence`.

### Outputs and diagnostic figures

| Option | Default | Meaning |
| --- | --- | --- |
| `--save_csv` | Empty | Optional prediction CSV destination. |
| `--save_npy_base` | `<output-dir>/oos` | Prefix for prediction/distance arrays, optional window/frame rejection masks, and `_unk_verdict.json`. Give each run a distinct prefix. An empty string disables these NPY/JSON exports. |
| `--save_plot` | Empty | Explicit distance-scatter image destination; otherwise derive it from the output prefix. The histogram uses a separate filename: replace `_oos_scatter` with `_oos_score_hist` in the filename stem, or append `_oos_score_hist` before the extension when absent. |
| `--proj2d` | `1` | Enable diagnostic 2D projections when OOS metrics are enabled. Set `0` to skip this projection stage. |
| `--proj2d_method` | `pca` | `pca`, `tsne`, `fisher`, or `svm`. |
| `--svm_C` | `1.0` | Positive regularization parameter for a linear-SVM projection. |
| `--proj2d_perplexity` | `30.0` | Requested t-SNE perplexity; the plotting path adjusts it to sample availability. Use enough samples for t-SNE, or choose PCA for a small input. |
| `--proj2d_max_points` | `3000` | Maximum samples selected for the subset projection; integer at least 4. |
| `--save_plot2d` | Empty | Explicit projection image destination; otherwise derive it from the output prefix. |
| `--proj2d_standardize` | `1` | Standardize projected features using known-data statistics. |
| `--proj2d_whiten` | `1` | Whiten projected features using known-data covariance. |
| `--boundary_metric` | `mahalanobis` | `euclidean`, `mahalanobis`, or `cosine` distance for the displayed 2D boundary only. |
| `--boundary_id_core_q` | `1.0` | Core-known-point quantile for that boundary; must be in `(0, 1]`. |
| `--boundary_grid` | `500` | Boundary grid resolution per axis, integer at least 2. |

With `eval_oos=1`, the distance scatter and score histogram are produced even if `proj2d=0`. Projected boundaries are visual diagnostics; they do not replace the high-dimensional OOS threshold. The subset projection/boundary is skipped when there are no rejected windows or no per-window rejection mask.

The following additional controls belong to `--task oos`; `oos-kgbd` does not accept them:

| Option | Default | Meaning |
| --- | --- | --- |
| `--plot_fisher_all` | `1` | Plot all supplied unknown samples in a Fisher plane. |
| `--boundary_draw_hull` | `0` | Draw identity convex hulls on the diagnostic boundary. |
| `--plot_id_oos` | `1` | Plot known identity clusters with an unknown overlay. |
| `--id_oos_method` | `tsne` | `tsne` or `pca` for the identity overlay. |
| `--id_oos_per_id_max` | `120` | Per-identity window cap for the overlay; use a positive integer. |
| `--id_oos_oos_max` | `800` | Unknown-window cap for the overlay; use a positive integer. |
| `--id_oos_feat_mode` | `l2norm` | `l2norm` or `whiten` feature processing for that overlay. |

## 8. Preprocessing

```bash
python main.py --task preprocess --dataset WUT \
  --input_dir raw/WUT --length 6 --fusion_file fusion_master.npy --skip_unknown

python main.py --task preprocess --dataset WUT \
  --input_dir raw/WUT --length 6 --data_subdir 6_10fps \
  --src_fps 15 --dst_fps 10 --skip_unknown
```

| Option | Default | Meaning and constraints |
| --- | --- | --- |
| `--dataset` | Required | `WUT`, `KGBD`, or `KS20`. |
| `--input_dir` | Required | Explicit raw root: WUT subject directories, KGBD `PersonXXX` directories, or KS20 view directories. |
| `--unknown_dir` | None | Optional existing raw unknown directory for WUT/KGBD; never inferred from the working directory. |
| `--length` / `--fr` | `6` | Frames per output window. Use at least 2 for model-compatible data; KS20 enforces that minimum. |
| `--data_subdir` | Empty | Output below `<data-dir>/<dataset>/`. Empty uses `length` or the `out_tag` fallback. Absolute paths and `..` components are rejected. |
| `--out_tag` | Empty | If `data_subdir` is empty, produce `<length>_<out_tag>`; otherwise ignored. |
| `--fusion_file` | `fusion_master.npy` | WUT fusion filename. For KGBD, select an existing sequence fusion array when available; otherwise parse its raw text. Not used by KS20. |
| `--split_file` | None | WUT/KS20 split JSON; when omitted, use `config/wut_split.json` or `config/ks20_split.json`. KGBD does not use this option. |
| `--clip_num` | Dataset-dependent | Nonnegative trimming amount: default 0 for WUT/KGBD, 10 for KS20. Units differ as described below. |
| `--src_fps` | `0` | Nonnegative source rate. Zero selects the WUT 15-FPS or KGBD 30-FPS convention. |
| `--dst_fps` | `0` | Nonnegative target rate; zero disables resampling. Only a target below the source rate downsamples. Equal/higher targets leave frames unchanged. |
| `--seed` | `-1` | Fresh seed, or explicit unsigned 32-bit seed. Controls KGBD random sequence selection; WUT/KS20 explicit split assignments remain fixed. |
| `--skip_unknown` | `0` | `--skip_unknown` alone or `--skip_unknown 1` disables optional unknown preprocessing; `0` allows it when an unknown directory is supplied. |

WUT/KGBD known sequences are downsampled, divided into complete non-overlapping windows, and windows containing NaNs are removed. `clip_num` then removes that many retained windows from each end. WUT unknown inputs use the same rule. The KGBD external unknown-text path instead interprets `clip_num` as frames per end. Keep the default 0 unless this distinction is intended. KS20 trims `clip_num` raw frames from each end before its windowing procedure; its training windows use shifts `0..length-1` with stride `length//2`, while gallery/query windows use one non-overlapping partition.

FPS resampling selects existing frames on a nearest-neighbor time grid; it does not synthesize poses. KS20 preparation rejects unknown inputs and nonzero FPS settings. It uses the explicit view/sequence split, so a new runtime seed does not reshuffle that protocol.

Prepared output is written below the selected dataset variant. `preprocessing_metadata.json` records the requested and resolved seed and split policy. Use distinct `data_subdir` values for distinct fusion/FPS/window settings, then use the same selector when training. An optional unknown directory supplies unknown examples; it does not automatically remove matching identities from the known training/gallery input.
