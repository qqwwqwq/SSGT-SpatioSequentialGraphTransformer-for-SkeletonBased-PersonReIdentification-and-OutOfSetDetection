"""Random run seeds and optional replay controls shared by SSGT entry points.

Each run draws a seed from operating-system entropy unless an explicit seed is
supplied. Record the returned seed so the same stochastic choices can be
requested later; identical results across devices or versions are not implied.
"""

import numbers
import random
import secrets


def resolve_seed(seed=None):
    """Return a valid NumPy-compatible seed, drawing one for None or -1.

    Args:
        seed (int or None): Explicit unsigned 32-bit seed, or automatic mode.

    Returns:
        int: The selected seed, including zero when explicitly requested.
    """
    if seed is None or (isinstance(seed, numbers.Integral) and seed == -1):
        return secrets.randbits(32)
    if not isinstance(seed, numbers.Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError('Seed must be -1, None, or an integer in [0, 2**32 - 1].')
    return int(seed)


def seed_everything(seed=None, deterministic=False):
    """Seed Python, NumPy, and PyTorch and return the selected run seed.

    Args:
        seed (int or None): Explicit seed, or None/-1 to generate a new seed.
        deterministic (bool): Enable deterministic cuDNN settings for replay.

    Returns:
        int: Seed to record in run metadata and pass to isolated data samplers.
    """
    import numpy as np
    import torch

    selected = resolve_seed(seed)
    random.seed(selected)
    np.random.seed(selected)
    torch.manual_seed(selected)
    torch.cuda.manual_seed_all(selected)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return selected
