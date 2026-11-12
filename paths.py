"""Portable locations for prepared data, checkpoints, and generated results.

Defaults are relative to this source tree. The main entry point can override
them with command-line options or JSON configuration, including from another
working directory. Importing this module does not create any directories.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _location(variable, directory):
    """Resolve an environment override or a directory in the source tree."""
    return Path(os.environ.get(variable, PROJECT_ROOT / directory)).expanduser().resolve()


def data_root():
    """Return the root containing prepared dataset directories."""
    return _location("SSGT_DATA_DIR", "Datas")


def model_root():
    """Return the root containing dataset-specific checkpoint directories."""
    return _location("SSGT_MODEL_DIR", "weight")


def output_root():
    """Return the root for generated evaluation results."""
    return _location("SSGT_OUTPUT_DIR", "results")
