# Source migration map

The release reorganizes the final SSGT pipeline. Original experiment files remain
outside the release tree and have not been overwritten. Their migration-time
SHA-256 fingerprints are recorded in `SOURCE_MANIFEST.json`.

| Original source | Release module | Retained responsibility |
| --- | --- | --- |
| `TranSG_pytorch2.py` | `SSGT.py` | Spatial/temporal encoder and positional encoding |
| `TranSG_pytorch2.py` | `training.py` | Reconstruction, prototype contrastive learning, training, ReID metrics |
| `utils/process_SG.py` | `data.py` | Skeleton loading, class sampling, joint ordering, graph construction |
| `Preprocess.py` | `preprocessing.py`, `ks20_preprocessing.py` | WUT/KGBD/KS20 conversion with explicit paths |
| `oostest.py` | `evaluation.py` | kNN OOS inference with WUT defaults |
| `oostestKGBD.py` | `evaluation_kgbd.py` | KGBD OOS inference and label convention |
| `id_oos_plot_utils.py` | `visualization.py` | Diagnostic projections and plots |
| `random3.npy` | `config/wut_split.json` | WUT fixed split, represented as JSON |
| `LICENSE` | `LICENSE` | Unchanged upstream MIT notice |

`main.py`, `paths.py`, `randomness.py`, `loss_balancing.py`, `retrieval.py`, `protocols.py`, `reproduce.py`,
configuration presets, documentation, regression tests, and the archive builder
are new. The public model class is `SSGT`; a `TranSG` alias retains compatibility
with existing imports. This alias does not include a second model implementation.

## Numerical preservation and intentional changes

The default encoder, class sampling, root centering, KGBD joint permutation,
reconstruction objectives, prototype losses, and metric formulas retain their
research-source conventions. Random choices now follow the selected run seed
instead of internal fixed seeds. Encoder equivalence was checked with matching
parameters. Prepared data for all ten recovered OOS trials reproduce the recorded
window counts; actual GPU reevaluation is reported in [VALIDATION.md](../VALIDATION.md).

New training uses the final architecture and the published Algorithm 2:
signed loss-slope ratio, fixed increments, threshold 10, and doubling update
thresholds. The archived smooth updates and convergence-triggered adjustments
are not a second selectable training method. This is an intentional numerical
change from the archived training code, explained in the
[reproduction guide](PAPER_REPRODUCTION.md#archived-weights-and-the-published-training-rule).

Retrieval aggregation reuses one sorting pass and an equivalent AP reduction.
Tests against the original implementation, including full KGBD evaluation,
verify bitwise equality of metrics; distance arithmetic is unchanged.

Additional execution and portability corrections include:

- Closed-set evaluation loads model weights and metadata before inference;
  the original branch evaluated newly initialized parameters.
- Checkpoint loading restores temporal attention, position encoding, data
  variants, and the saved spatial eigenbasis when available. Model parameter
  keys are checked strictly. Missing eigenbases in old weights are disclosed.
- OOS frame labels convert zero-based/one-based IDs once, including ID zero.
  Unknown sentinels are outside the complete known identity range.
- New training builds one prototype per actual known identity. Historical OOS
  training retained extra zero prototypes for held-out identities; those unused
  contrastive negatives are removed. Full closed-set training and archived-weight
  inference are unaffected. Missing prototype samples fail explicitly.
- OOS retrieval indices decode through the actual training identity keys, so
  held-out identities do not shift exported names or numeric IDs.
- Raw and prepared paths are explicit. KGBD conversion no longer creates caches
  inside the original raw-data directories.
- Stochastic entry points now generate and record a run seed by default;
  explicit seeds remain available. Internal fixed sampling and plotting seeds
  have been removed. Recovered experiments retain their explicit manifests;
  a new seed does not reconstruct an old random split. Earlier seed-42 metrics
  remain evidence for the previous version, not a rerun of the new defaults.
- Training defaults and public presets use the final method. Evaluation can
  restore architecture metadata from older trusted SSGT weights.
- Missing checkpoints, invalid data, and undersized batches fail explicitly.
- OOS defaults use K=1 and percentile 95. The reproduction runner verifies
  manifest, data, identity separation, and checkpoint hashes before inference.

The training epoch cap retains its inclusive, zero-based meaning:
`--max_epochs 1` permits epochs 0 and 1; `0` disables that additional cap.

## Release boundary

The public dataset scope is WUT, KGBD, and KS20, with raw-data converters for
all three and recovered WUT/KS20 sequence splits. TensorFlow TranSG, comparison methods, older training variants,
CNN/GCN ablations, SOM experiments, capture utilities, and scratch scripts are
excluded. Diagnostic plots and old checkpoint compatibility do not expose
additional baseline training workflows.

The archive allowlist also excludes raw/prepared data, checkpoint binaries,
private experiment logs, editor conversations, videos, and local environments.
Portable split manifests, measured validation summaries, and file hashes are
included so the verification remains inspectable without copying those assets.
