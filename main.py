"""SSGT training, identification, and open-set evaluation pipeline.

This entry point combines task-specific JSON settings with command-line
overrides. Numerical routines live in separate model, training, evaluation,
and preprocessing modules.

Usage:
    python main.py --config params.json --task train
    python main.py --task oos-kgbd --help
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


# =====================================================================
# Configuration and Task Dispatch
# =====================================================================

TASK_MODULES = {
    "train": "training",
    "evaluate": "training",
    "oos": "evaluation",
    "oos-kgbd": "evaluation_kgbd",
    "preprocess": "preprocessing",
}
PATH_VARIABLES = {
    "data_dir": "SSGT_DATA_DIR",
    "model_dir": "SSGT_MODEL_DIR",
    "output_dir": "SSGT_OUTPUT_DIR",
}
TASK_PATH_KEYS = {
    "ckpt", "input", "npy_path", "npy-path", "input_dir", "unknown_dir", "split_file",
    "save_csv", "save_npy_base", "save_plot", "save_plot2d", "metrics_output",
}


def _resolve_path(value, base):
    """Resolve a configuration path relative to its configuration file."""
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(filename):
    """Read and validate a task configuration without importing ML libraries.

    Args:
        filename (str): Path to a UTF-8 JSON configuration file.

    Returns:
        dict: Validated top-level configuration. Task options are validated by
        their corresponding argument parsers when the task is dispatched.

    Raises:
        ValueError: If the document has unknown sections or invalid types.
    """
    with open(filename, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    unknown = set(config) - {"task", "paths", *TASK_MODULES}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if config.get("task", "train") not in TASK_MODULES:
        raise ValueError("Unknown task in configuration.")
    for key, value in config.items():
        if key != "task" and not isinstance(value, dict):
            raise ValueError(f"Configuration section '{key}' must be an object.")
    if set(config.get("paths", {})) - set(PATH_VARIABLES):
        raise ValueError("paths accepts only data_dir, model_dir, and output_dir.")
    for value in config.get("paths", {}).values():
        if not isinstance(value, str) or not value:
            raise ValueError("Configured paths must be nonempty strings.")
    return config


def task_arguments(settings, base):
    """Convert JSON scalars to task arguments, resolving input/output paths.

    Args:
        settings (dict): Task-specific options without leading dashes.
        base (Path): Parent directory of the JSON file.

    Returns:
        list[str]: Argument tokens suitable for the task's argument parser.
    """
    result = []
    for key, value in settings.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError("Configuration option names must not start with '-'.")
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            raise ValueError(f"Option '{key}' must be a scalar value.")
        if key in TASK_PATH_KEYS and value:
            if not isinstance(value, str):
                raise ValueError(f"Path option '{key}' must be a string.")
            value = _resolve_path(value, base)
        if isinstance(value, bool):
            value = int(value)
        result.append(f"--{key}={value}")
    return result


def main(argv=None):
    """Dispatch one task, with explicit CLI options taking precedence."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False, allow_abbrev=False,
    )
    parser.add_argument("--task", choices=TASK_MODULES, help="Pipeline task; default: train")
    parser.add_argument("--config", help="Optional JSON configuration file")
    for name in PATH_VARIABLES:
        parser.add_argument("--" + name.replace("_", "-"), "--" + name, dest=name)
    parser.add_argument("-h", "--help", action="store_true", help="Show general or task help")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved options without running")
    args, remaining = parser.parse_known_args(argv)

    if args.help and args.task is None and args.config is None:
        parser.print_help()
        print("\nTask options: python main.py --task TASK --help")
        return

    try:
        config = load_config(args.config) if args.config else {}
        config_base = Path(args.config).resolve().parent if args.config else Path.cwd()
        task = args.task or config.get("task", "train")
        environment = {}
        for key, variable in PATH_VARIABLES.items():
            cli_value = getattr(args, key)
            configured = config.get("paths", {}).get(key)
            if cli_value is not None:
                environment[variable] = _resolve_path(cli_value, Path.cwd())
            elif configured is not None:
                environment[variable] = _resolve_path(configured, config_base)
        options = task_arguments(config.get(task, {}), config_base) + remaining
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    if task in ("train", "evaluate"):
        options += ["--mode", "Train" if task == "train" else "Eval"]
    if args.help:
        options.append("--help")
    if args.dry_run:
        print(json.dumps({"task": task, "module": TASK_MODULES[task],
                          "paths": environment, "arguments": options}, indent=2))
        return

    # Set device visibility before importing PyTorch through a task module.
    gpu_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    gpu_parser.add_argument("--gpu")
    gpu, _ = gpu_parser.parse_known_args(options)
    if gpu.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu.gpu.lower() == "cpu" else gpu.gpu
    os.environ.update(environment)
    module = importlib.import_module(TASK_MODULES[task])
    module.main(options)


if __name__ == "__main__":
    main(sys.argv[1:])
