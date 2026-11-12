# Code and documentation conventions

The reference is [DualSOM](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-).
Its current README, repository tree, `main.py`, `Dualmap_api.py`, and
`preprocessing.py` were inspected directly on 2026-11-13. SSGT follows their
organization and documentation patterns while documenting its own model and
supported workflows.

## Reference correspondence

| Inspected DualSOM pattern | SSGT adaptation |
| --- | --- |
| Paper and authors, badges, linked contents, quick start | README paper header, environment/license badges, contents, and WUT quick start |
| Data, configuration, execution, results, reproducibility, and API sections | README task commands and links to detailed data, reproduction, validation, and API guides |
| Root `main.py`, `params.json`, separate model and preprocessing modules | Root task entry point, JSON presets, `SSGT.py`, `training.py`, and converters |
| `Datas/`, `config/`, and `weight/` directories | Same folder names; SSGT additionally gives generated outputs a `results/` directory |
| English module descriptions and usage examples | English purpose/usage docstrings at SSGT entry points |
| Three-line `# ===` section banners | Banners for major model, data, training, and evaluation stages |
| Public routine docs with `Args`, `Returns`, and `Raises` | Shape-aware argument, result, and failure documentation |

The inspected examples are [README.md](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-/blob/main/README.md),
[main.py](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-/blob/main/main.py),
[Dualmap_api.py](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-/blob/main/Dualmap_api.py),
and [preprocessing.py](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-/blob/main/preprocessing.py).
The organization is adapted, not an identical tree: SSGT needs separate training,
OOS evaluation, dataset protocols, and regression tests. No DualSOM-specific
model, GUI, dataset, or cached-model claim is implied by this style alignment.

## Source conventions

- Start modules with an English docstring describing their role and usage.
- Use section banners for major stages, not individual statements.
- Document public routines using `Args`, `Returns`, and `Raises` where applicable.
- Explain tensor shapes and protocol assumptions near numerical operations.
- Explain why a mask, normalization, or split exists rather than narrating Python.
- Keep user configuration in JSON or argparse; never embed a developer's paths.
- Generate and record run seeds by default; expose explicit replay seeds.
- Distinguish fixed protocol definitions from stochastic run settings.
- Label historical measurements with their original seed policy and version.
- Keep old parameter names when changing them would break stored checkpoints.
- Do not retain commented copies of old implementations or numbered debug prints.

```python
# =====================================================================
# Feature Library Construction
# =====================================================================

def normalize_features(features, epsilon=1e-12):
    """Normalize sequence representations before distance-based matching.

    Args:
        features (np.ndarray): Sequence features with shape (samples, channels).
        epsilon (float): Lower bound preventing division by zero.

    Returns:
        np.ndarray: Row-normalized features with the same shape.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, epsilon)
```

## README conventions

Keep the main task commands visible after the introduction and quick start.
Use tables for parameter and result comparisons, and link detailed protocol/API
explanations from the README. Separate paper targets from measured results and
state which source version and seed policy produced each measurement. A note
about stochastic results must not promise a fixed bound on their variation.
