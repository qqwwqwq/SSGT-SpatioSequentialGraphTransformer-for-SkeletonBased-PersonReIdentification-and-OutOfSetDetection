# Code style

The project follows the module and documentation style of
[DualSOM](https://github.com/qqwwqwq/DualSOM--SOFTWAREX-): a root `main.py`, separate
model and preprocessing modules, JSON configuration, and `Datas/` and `weight/`
directories.

- Write module docstrings, comments, and command-line help in English.
- Start each module with a short purpose statement and a usage example when useful.
- Separate major processing stages with `# ===` section banners.
- Document public functions with `Args`, `Returns`, and `Raises` as needed.
- State array shapes, coordinate conventions, and split assumptions explicitly.
- Explain non-obvious decisions without repeating what the code already says.
- Keep paths and model settings in CLI arguments or JSON configuration.
- Remove commented-out implementations and temporary debugging output.

```python
# =====================================================================
# Feature Normalization
# =====================================================================

def normalize_features(features, epsilon=1e-12):
    """Normalize sequence features before distance-based matching.

    Args:
        features (np.ndarray): Features with shape (samples, channels).
        epsilon (float): Lower bound preventing division by zero.

    Returns:
        np.ndarray: Row-normalized features with the same shape.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, epsilon)
```

Keep runnable commands in the [README](../README.md), dataset details in the
[data guide](DATA.md), and model settings in [hyperparameters](HYPERPARAMETERS.md).
