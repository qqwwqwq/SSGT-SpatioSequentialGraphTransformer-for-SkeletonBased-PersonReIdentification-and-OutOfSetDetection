"""Regression checks for preprocessing and loading synthetic skeleton data."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import data
import preprocessing


class DataPipelineTests(unittest.TestCase):
    """Exercise complete raw-to-prepared-to-loader paths without real datasets."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssgt-test-data-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "prepared"
        self.environment = patch.dict(os.environ, {"SSGT_DATA_DIR": str(self.output)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.rng = np.random.RandomState(7)

    def run_preprocessing(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            preprocessing.main(list(arguments))
        return output.getvalue()

    def load_split(self, dataset, split, subdir="", sampling_seed=None):
        return data.gen_train_data(
            dataset=dataset, split=split, time_step=6,
            nb_nodes=19 if dataset == "WUT" else 20,
            nhood=1, global_att=False, batch_size=6, data_subdir=subdir,
            sampling_seed=sampling_seed,
        )

    def snapshot(self, root):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()
        }

    def write_wut(self):
        root = self.root / "wut_raw"
        for identity in ("Jacek", "Aleks", "Julia"):
            for sequence in range(9):
                directory = root / identity / f"seq{sequence:02}"
                directory.mkdir(parents=True)
                coordinates = self.rng.randn(24, 19, 3).astype(np.float32)
                if sequence == 3:
                    coordinates[0, 0, 0] = np.nan
                np.save(directory / "fusion_master.npy", coordinates)
        unknown_root = self.root / "wut_unknown"
        directory = unknown_root / "person" / "seq00"
        directory.mkdir(parents=True)
        coordinates = self.rng.randn(12, 19, 3).astype(np.float32)
        np.save(directory / "fusion_master.npy", np.concatenate([coordinates, coordinates]))
        return root, unknown_root

    def write_kgbd(self):
        root = self.root / "kgbd_raw"
        for identity in (1, 2, 15):
            for sequence in range(4):
                path = root / f"Person{identity:03}" / f"seq{sequence:02}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                coordinates = self.rng.randn(24, 20, 3)
                path.write_text("\n".join(
                    f"{joint};{x};{y};{z}"
                    for frame in coordinates
                    for joint, (x, y, z) in enumerate(frame)
                ))
        unknown_root = self.root / "kgbd_unknown"
        unknown_root.mkdir()
        (unknown_root / "unknown.txt").write_bytes(
            (root / "Person001" / "seq00.txt").read_bytes()
        )
        return root, unknown_root

    def assert_default_tuple(self, result, joints, classes):
        self.assertEqual(len(result), 26)
        self.assertEqual(result[0].shape[1:], (6, joints, 3))
        self.assertEqual(result[6].shape[1:], (6, joints, 3))
        self.assertEqual(result[12].shape, (1, joints, joints))
        self.assertEqual(result[14].shape, (joints, 10))
        self.assertEqual(result[25], classes)
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertTrue(np.isfinite(result[6]).all())

    def test_wut_split_unknown_and_variant_loading(self):
        raw, unknown = self.write_wut()
        snapshot = self.snapshot(raw)
        self.run_preprocessing(
            "--dataset", "WUT", "--input_dir", str(raw),
            "--unknown_dir", str(unknown), "--length", "6",
            "--data_subdir", "6_synthetic",
        )
        self.assertEqual(snapshot, self.snapshot(raw))
        prepared = self.output / "WUT" / "6_synthetic"
        gallery = np.load(prepared / "test_npy_data/gallery/t_source_x_WUT_6.npy")
        # Identities and sequences are traversed alphabetically. Split JSON
        # indices [8, 0] select seq00 and seq08 as gallery sequences.
        expected = np.concatenate([
            np.load(raw / identity / f"seq{seq:02}" / "fusion_master.npy")[:, :, 0]
            for identity in ("Aleks", "Jacek", "Julia") for seq in (0, 8)
        ])
        np.testing.assert_array_equal(gallery, expected)
        probe = np.load(prepared / "test_npy_data/probe/t_source_x_WUT_6.npy")
        expected_probe = np.concatenate([
            np.load(raw / identity / "seq01/fusion_master.npy")[:, :, 0]
            for identity in ("Aleks", "Jacek", "Julia")
        ])
        np.testing.assert_array_equal(probe, expected_probe)
        # Each identity has six training sequences; one NaN window is dropped.
        labels = np.load(prepared / "train_npy_data/frame_id_WUT_6.npy")
        self.assertEqual(labels.size, 3 * (6 * 4 - 1))
        unknown_labels = np.load(prepared / "unknown_npy_data/frame_id_WUT_6.npy")
        np.testing.assert_array_equal(unknown_labels, [26, 26])
        for split in ("gallery", "probe", "unknown"):
            result = self.load_split("WUT", split, "6_synthetic")
            self.assert_default_tuple(result, 19, 25)
            np.testing.assert_array_equal(result[0][:, :, 0], 0)
            np.testing.assert_array_equal(result[6][:, :, 0], 0)
            if split == "unknown":
                self.assertEqual(result[11].shape, (2, 3))
                np.testing.assert_array_equal(result[11], 0)
            else:
                np.testing.assert_array_equal(result[11].sum(axis=1), 1)

    def test_kgbd_text_conversion_seed_unknown_and_joint_order(self):
        raw, unknown = self.write_kgbd()
        snapshot = self.snapshot(raw)
        self.run_preprocessing(
            "--dataset", "KGBD", "--input_dir", str(raw),
            "--unknown_dir", str(unknown), "--length", "6", "--seed", "42",
        )
        self.assertEqual(snapshot, self.snapshot(raw))
        self.assertFalse(any(raw.rglob("*.npy")))
        first = self.snapshot(self.output / "KGBD" / "6")
        self.run_preprocessing(
            "--dataset", "KGBD", "--input_dir", str(raw),
            "--unknown_dir", str(unknown), "--length", "6", "--seed", "42",
        )
        self.assertEqual(first, self.snapshot(self.output / "KGBD" / "6"))
        unknown_labels = np.load(
            self.output / "KGBD/6/unknown_npy_data/frame_id_KGBD_6.npy"
        )
        np.testing.assert_array_equal(unknown_labels, [165] * 4)
        for split in ("gallery", "probe", "unknown"):
            result = self.load_split("KGBD", split)
            self.assert_default_tuple(result, 20, 164)
            # Centering occurs before the inherited permutation. Original
            # joint zero appears at model joint index three for KGBD.
            np.testing.assert_array_equal(result[0][:, :, 3], 0)
            if split == "unknown":
                np.testing.assert_array_equal(result[11], 0)
        self.run_preprocessing(
            "--dataset", "KGBD", "--input_dir", str(raw),
            "--unknown_dir", str(unknown), "--length", "6",
            "--skip_unknown=1", "--data_subdir", "6_no_unknown",
        )
        self.assertFalse((self.output / "KGBD/6_no_unknown/unknown_npy_data").exists())

    def test_fps_selection_and_nan_windows(self):
        np.testing.assert_array_equal(preprocessing._fps_keep_indices(10, 30, 10), [0, 3, 6, 9])
        np.testing.assert_array_equal(preprocessing._fps_keep_indices(9, 15, 10), [0, 2, 3, 5, 6, 8])
        self.assertIsNone(preprocessing._fps_keep_indices(10, 15, 30))
        sequence = self.rng.randn(14, 19, 3).astype(np.float32)
        sequence[0, 0, 0] = np.nan
        x, y, z = preprocessing._lines_to_xyz_blocks(sequence, fr=6)
        np.testing.assert_array_equal(np.stack([x, y, z], axis=-1), sequence[6:12])

    def test_automatic_preprocessing_seed_is_logged_saved_and_replayable(self):
        raw, _ = self.write_kgbd()
        arguments = ['--dataset', 'KGBD', '--input_dir', str(raw), '--length', '6']
        with patch('preprocessing.resolve_seed', return_value=1234) as resolve:
            output = self.run_preprocessing(*arguments)
        resolve.assert_called_once_with(-1)
        self.assertIn('preprocessing seed 1234', output)
        prepared = self.output / 'KGBD' / '6'
        metadata = json.loads((prepared / 'preprocessing_metadata.json').read_text())
        self.assertEqual(metadata['requested_seed'], -1)
        self.assertEqual(metadata['resolved_seed'], 1234)
        self.assertEqual(metadata['seed_selection'], 'automatic')
        first = {name: value for name, value in self.snapshot(prepared).items() if name.endswith('.npy')}
        self.run_preprocessing(*(arguments + ['--seed', str(metadata['resolved_seed'])]))
        second = {name: value for name, value in self.snapshot(prepared).items() if name.endswith('.npy')}
        self.assertEqual(first, second)

    def test_loader_sampling_seed_alignment_and_global_rng_isolation(self):
        raw, _ = self.write_wut()
        self.run_preprocessing('--dataset', 'WUT', '--input_dir', str(raw), '--length', '6', '--seed', '0')
        state = np.random.get_state()
        probe = self.load_split('WUT', 'probe', sampling_seed=0)
        gallery = self.load_split('WUT', 'gallery', sampling_seed=0)
        different = self.load_split('WUT', 'probe', sampling_seed=1)
        np.testing.assert_array_equal(probe[0], gallery[0])
        np.testing.assert_array_equal(probe[5], gallery[5])
        self.assertFalse(np.array_equal(probe[0], different[0]))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        with patch('data.resolve_seed', side_effect=[11, 12]) as resolve:
            first = self.load_split('WUT', 'probe')
            second = self.load_split('WUT', 'probe')
        self.assertEqual(resolve.call_count, 2)
        self.assertFalse(np.array_equal(first[0], second[0]))
        metadata = json.loads((self.output / 'WUT/6/preprocessing_metadata.json').read_text())
        self.assertFalse(metadata['seed_affects_sequence_split'])

    def test_split_configuration_validation(self):
        np.testing.assert_array_equal(preprocessing._load_wut_split(), [8, 0, 1])
        custom = self.root / "split.json"
        custom.write_text(json.dumps({"gallery_indices": [0, 2], "probe_indices": [1]}))
        np.testing.assert_array_equal(preprocessing._load_wut_split(custom), [0, 2, 1])
        custom.write_text(json.dumps({"gallery_indices": [0, 1], "probe_indices": [1]}))
        with self.assertRaisesRegex(ValueError, "overlap"):
            preprocessing._load_wut_split(custom)

    def test_unsupported_dataset_fails_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, "supports WUT"):
            self.load_split("CASIA_B", "probe")

    def test_unsupported_graph_flags_fail_before_reading_files(self):
        for option in ("D", "graph"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(NotImplementedError, "D=False and graph=False"):
                    data.gen_train_data(
                        "KGBD", "probe", 6, 20, 1, False, 6, **{option: True}
                    )

    def test_invalid_sampler_settings_are_explicit(self):
        windows = np.zeros((6, 6, 19, 3), dtype=np.float32)
        labels = np.array([1, 1, 2, 2, 3, 3])
        ids = {1: [0, 1], 2: [2, 3], 3: [4, 5]}
        with self.assertRaisesRegex(ValueError, "batch_size >= known classes"):
            data.class_samp_gen(windows, labels, ids, batch_size=2)
        with self.assertRaisesRegex(ValueError, "at least one batch"):
            data.class_samp_gen(windows, labels, ids, batch_size=8)
        with self.assertRaisesRegex(ValueError, "at least one window"):
            data.class_samp_gen(windows, labels, {1: [], 2: [2, 3], 3: [4, 5]}, 6)

    def test_binary_identity_protocol_requires_explicit_adaptation(self):
        raw, unknown = self.write_kgbd()
        self.run_preprocessing(
            "--dataset", "KGBD", "--input_dir", str(raw), "--length", "6"
        )
        ids_path = self.output / "KGBD/6/train_npy_data/ids_KGBD_6.npy"
        ids = np.load(ids_path, allow_pickle=True).item()
        ids.pop(15)
        np.save(ids_path, ids)
        with self.assertRaisesRegex(ValueError, "at least three known"):
            self.load_split("KGBD", "probe")


if __name__ == "__main__":
    unittest.main()
