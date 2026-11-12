"""Check configuration precedence, path portability, and release boundaries."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main
import paths


class EntrypointTests(unittest.TestCase):
    def test_configuration_paths_follow_config_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = folder / "run.json"
            config.write_text(json.dumps({
                "task": "train", "paths": {"data_dir": "prepared"},
                "train": {"dataset": "KGBD", "max_epochs": 10},
            }), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main.main(["--config", str(config), "--max_epochs", "3", "--dry-run"])
            resolved = json.loads(stream.getvalue())
            self.assertEqual(resolved["paths"]["SSGT_DATA_DIR"], str(folder / "prepared"))
            # The task parser sees the CLI setting last, so it overrides JSON.
            tokens = resolved["arguments"]
            self.assertLess(tokens.index("--max_epochs=10"), tokens.index("--max_epochs"))
            self.assertEqual(tokens[-2:], ["--mode", "Train"])

    def test_explicit_cli_path_overrides_configuration(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main.main(["--config", str(ROOT / "params.json"),
                       "--data-dir", "my-data", "--dry-run"])
        resolved = json.loads(stream.getvalue())
        self.assertEqual(resolved["paths"]["SSGT_DATA_DIR"], str(Path("my-data").resolve()))

    def test_invalid_configuration_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "invalid.json"
            for value in ([], {"training": {}}, {"task": "typo"}, {"paths": []}):
                filename.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    main.load_config(filename)

    def test_scalar_and_path_conversion(self):
        actual = main.task_arguments({"fix_seed": True, "ckpt": "weights/best.ckpt",
                                      "input": None}, ROOT)
        self.assertEqual(actual, ["--fix_seed=1", "--ckpt=" + str(ROOT / "weights/best.ckpt")])
        with self.assertRaises(ValueError):
            main.task_arguments({"dataset": ["WUT"]}, ROOT)

    def test_default_paths_do_not_depend_on_working_directory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.data_root(), ROOT / "Datas")
            self.assertEqual(paths.model_root(), ROOT / "weight")
            self.assertEqual(paths.output_root(), ROOT / "results")

    def test_all_training_presets_select_automatic_run_seeds(self):
        from training import get_args

        presets = [ROOT / 'params.json', ROOT / 'config/kgbd.json']
        presets += sorted((ROOT / 'config/paper').glob('*.json'))
        for filename in presets:
            with self.subTest(config=filename.name):
                config = main.load_config(filename)
                options = main.task_arguments(config['train'], filename.parent)
                parsed = get_args(options)
                self.assertEqual(parsed.seed_val, -1)
                self.assertEqual(parsed.fix_seed, 0)
                # An explicitly supplied seed must still take precedence.
                replay = get_args(options + ['--seed_val', '193', '--fix_seed', '1'])
                self.assertEqual(replay.seed_val, 193)
                self.assertEqual(replay.fix_seed, 1)

    def test_release_manifest_contains_only_source_files(self):
        spec = importlib.util.spec_from_file_location("build_release", ROOT / "tools/build_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = module.release_files()
        self.assertIn("LICENSE", names)
        self.assertIn("config/wut_split.json", names)
        excluded = {".npy", ".npz", ".ckpt", ".pth", ".mp4", ".mkv", ".log", ".pyc"}
        for name in names:
            self.assertNotIn(Path(name).suffix, excluded)
            self.assertNotIn("__pycache__", Path(name).parts)


if __name__ == "__main__":
    unittest.main()
