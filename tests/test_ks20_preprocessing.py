"""KS20 MAT parsing, recovered splits, and end-to-end coordinate regression."""

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
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data
import ks20_preprocessing as ks20
import preprocessing


class KS20PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssgt-ks20-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "prepared"
        environment = patch.dict(os.environ, {"SSGT_DATA_DIR": str(self.output)})
        environment.start()
        self.addCleanup(environment.stop)

    def write_numeric_mat(self, path, frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = np.empty(len(frames), dtype=object)
        for index, frame in enumerate(frames):
            cells[index] = frame
        savemat(path, {"gc": cells})

    def test_complete_conversion_preserves_exact_windows_and_raw_files(self):
        assignments = ks20.load_split()
        source_arrays = {}
        for view_index, view in enumerate(ks20.VIEWS):
            for setting_index, setting in enumerate(("gc1", "gc2", "gc3")):
                for label, body in enumerate(ks20.BODY_IDS):
                    frames = 24 if (view, setting, body) == ("left_diagonal", "gc1", 1) else 30
                    coordinates = (np.arange(frames * 25 * 3).reshape(frames, 25, 3) / 1000
                                   + 1 + view_index * 1000 + setting_index * 100 + label).astype(np.float32)
                    relative = f"{view}/{setting}_body{body}.mat"
                    source_arrays[relative] = coordinates
                    self.write_numeric_mat(self.raw / relative, coordinates)
        before = {relative: hashlib.sha256((self.raw / relative).read_bytes()).hexdigest()
                  for relative in source_arrays}
        with contextlib.redirect_stdout(io.StringIO()):
            preprocessing.main(["--dataset", "KS20", "--input_dir", str(self.raw), "--length", "6"])
        after = {relative: hashlib.sha256((self.raw / relative).read_bytes()).hexdigest()
                 for relative in source_arrays}
        self.assertEqual(before, after)
        output = self.output / "KS20/6"
        self.assertEqual(len(list(output.rglob("*.npy"))), 15)
        provenance = json.loads((output / "preprocessing_provenance.json").read_text())
        self.assertEqual(len(provenance["source_files"]), 300)
        self.assertEqual(provenance["splits"]["gallery"]["windows_by_sequence"]["left_diagonal/gc1_body1.mat"], 0)
        # With ten retained frames and N=6, the inherited training offsets are
        # [0,3], [1,4], [2], [3], [4], [] for shifts zero through five.
        train_offsets = ((10, 13), (11, 14), (12,), (13,), (14,), ())
        for split in ("train", "gallery", "probe"):
            expected_windows, expected_labels = [], []
            for offsets in (train_offsets if split == "train" else ((10,),)):
                for view in ks20.VIEWS:
                    for relative in sorted(key for key in source_arrays if key.startswith(view + "/")):
                        filename = Path(relative).name
                        setting, body_text = filename.split("_body")
                        label = ks20.BODY_IDS.index(int(body_text[:-4]))
                        if assignments[label][split] != setting:
                            continue
                        sequence = source_arrays[relative]
                        for start in offsets:
                            if start + 6 <= len(sequence) - 10:
                                expected_windows.append(sequence[start:start + 6])
                                expected_labels.append(label)
            expected = np.asarray(expected_windows, dtype=np.float64).reshape(-1, 25, 3)
            directory = output / ("train_npy_data" if split == "train" else "test_npy_data/" + split)
            prefix = "" if split == "train" else "t_"
            for axis, name in enumerate("xyz"):
                actual = np.load(directory / f"{prefix}source_{name}_KS20_6.npy")
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_array_equal(actual, expected[:, :, axis])
            labels = np.load(directory / "frame_id_KS20_6.npy")
            np.testing.assert_array_equal(labels, expected_labels)
            identities = np.load(directory / "ids_KS20_6.npy", allow_pickle=True).item()
            for label in range(20):
                np.testing.assert_array_equal(identities[label], np.flatnonzero(labels == label))
            self.assertEqual(provenance["splits"][split]["sequence_count"], 100)
            self.assertEqual(provenance["splits"][split]["missing_identity_windows"], [])
        loaded = data.gen_train_data("KS20", "probe", 6, 25, 1, False, 256)
        self.assertEqual(len(loaded), 26)
        self.assertEqual(loaded[0].shape, (960, 6, 25, 3))
        self.assertEqual(loaded[6].shape, (100, 6, 25, 3))
        np.testing.assert_array_equal(loaded[0][:, :, 0], 0)

    def test_nested_mat_joint_indices_and_previous_frame_fill(self):
        cells = np.empty(2, dtype=object)
        for frame, positions in enumerate((((1, 2, 3), (4, 5, 6)), ((0, 0, 0), (7, 8, 9)))):
            joints = np.empty(2, dtype=object)
            for index, (joint, xyz) in enumerate(zip((24, 0), positions)):
                joints[index] = {"type": joint, "position": np.asarray([str(value) for value in xyz], dtype=object)}
            cells[frame] = {"body": {"joints": joints}}
        path = self.root / "nested.mat"
        savemat(path, {"gc": cells})
        parsed = ks20.extract_frames_from_mat(path)
        self.assertEqual(parsed.shape, (2, 25, 3))
        self.assertEqual(parsed.dtype, np.float32)
        np.testing.assert_array_equal(parsed[:, 24], [[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(parsed[:, 0], [[4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(parsed[:, 1:24], 0)
        without_fill = ks20.extract_frames_from_mat(path, fill_missing_with_prev=False)
        np.testing.assert_array_equal(without_fill[1, 24], 0)

    def test_split_rejects_overlap_and_identity_remapping(self):
        document = json.loads(ks20.DEFAULT_SPLIT.read_text())
        custom = self.root / "split.json"
        document["identities"][0]["probe"] = document["identities"][0]["gallery"]
        custom.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "distinct"):
            ks20.load_split(custom)
        document = json.loads(ks20.DEFAULT_SPLIT.read_text())
        document["identities"][0]["body_id"] = 5
        custom.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "sparse-to-contiguous"):
            ks20.load_split(custom)

    def test_incomplete_or_nested_inputs_fail_before_writing(self):
        self.raw.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "viewpoint"):
            ks20.process_dataset_KS20(self.raw)
        self.assertFalse(self.output.exists())
        with patch.dict(os.environ, {"SSGT_DATA_DIR": str(self.raw / "outputs")}):
            with self.assertRaisesRegex(ValueError, "non-nested"):
                ks20.process_dataset_KS20(self.raw)
        with self.assertRaisesRegex(ValueError, "length >= 2"):
            ks20.process_dataset_KS20(self.raw, fr=1)


if __name__ == "__main__":
    unittest.main()
