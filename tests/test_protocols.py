"""Synthetic tests for immutable, explicitly partitioned reproduction datasets."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocols


class ProtocolTests(unittest.TestCase):
    """Verify source integrity, identity holdout, and prepared-data provenance."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssgt-protocol-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "prepared"
        self.raw.mkdir()
        self.rng = np.random.RandomState(17)

    def build_manifest(self, dataset):
        joints = protocols.JOINTS[dataset]
        identities = []
        names = ["Jacek", "Aleks", "Julia", "Vibek"] if dataset == "WUT" else [f"Person{i:03}" for i in range(1, 5)]
        for label, name in enumerate(names, start=1):
            role = "unknown" if label == 4 else "known"
            sequences = []
            for number, split in enumerate(["unknown"] if role == "unknown" else ["train", "gallery", "probe"]):
                sequence = self.rng.randn(6, joints, 3).astype(np.float32)
                if split == "unknown":
                    sequence = np.concatenate([sequence, sequence])
                if dataset == "WUT":
                    path = self.raw / name / f"seq{number}" / "fusion_master.npy"
                    path.parent.mkdir(parents=True)
                    np.save(path, sequence)
                    format_name = "npy"
                else:
                    path = self.raw / name / f"seq{number}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("\n".join(
                        f"{joint};{x};{y};{z}" for frame in sequence
                        for joint, (x, y, z) in enumerate(frame)
                    ))
                    format_name = "kgbd_txt"
                sequences.append({
                    "name": f"seq{number}", "split": split,
                    "source": path.relative_to(self.raw).as_posix(), "format": format_name,
                    "sha256": protocols.sha256_file(path), "size_bytes": path.stat().st_size,
                })
            identities.append({"name": name, "original_label": label, "role": role,
                               "sequences": sequences})
        return {"schema_version": 1, "dataset": dataset, "trial_id": "synthetic",
                "length": 6, "data_subdir": "6_explicit", "identities": identities,
                "expected_identities": {"known": 3, "unknown": 1},
                "expected_windows": {"train": 3, "gallery": 3, "probe": 3, "unknown": 1}}

    def test_wut_source_holdout_and_provenance(self):
        manifest = self.build_manifest("WUT")
        source_before = {str(path.relative_to(self.raw)): protocols.sha256_file(path)
                         for path in self.raw.rglob("*") if path.is_file()}
        provenance = protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertEqual(provenance["windows"], manifest["expected_windows"])
        self.assertEqual(provenance["known_ids"], [1, 2, 3])
        self.assertEqual(provenance["unknown_original_ids"], [4])
        self.assertEqual(len(provenance["output_sha256"]), 20)
        self.assertEqual(provenance["manifest_sha256"], protocols.manifest_digest(manifest))
        source_after = {str(path.relative_to(self.raw)): protocols.sha256_file(path)
                        for path in self.raw.rglob("*") if path.is_file()}
        self.assertEqual(source_before, source_after)
        known = np.load(self.output / "WUT/6_explicit/train_npy_data/frame_id_WUT_6.npy")
        unknown = np.load(self.output / "WUT/6_explicit/unknown_npy_data/frame_id_WUT_6.npy")
        np.testing.assert_array_equal(known, [1, 2, 3])
        np.testing.assert_array_equal(unknown, [26])
        self.assertEqual(protocols.verify_provenance(manifest, self.output), provenance)
        self.assertEqual(protocols.prepare_trial(manifest, self.raw, self.output), provenance)

    def test_kgbd_text_conversion_and_original_unknown_identity(self):
        manifest = self.build_manifest("KGBD")
        provenance = protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertEqual(provenance["unknown_original_ids"], [4])
        self.assertFalse(any(self.raw.rglob("*.npy")))
        unknown = np.load(self.output / "KGBD/6_explicit/unknown_npy_data/frame_id_KGBD_6.npy")
        np.testing.assert_array_equal(unknown, [165])
        reference = protocols._kgbd_txt_to_fusion(self.raw / "Person001/seq0.txt")[:, :, 0]
        source_x = np.load(self.output / "KGBD/6_explicit/train_npy_data/source_x_KGBD_6.npy")
        np.testing.assert_array_equal(source_x[:6], reference)
        self.assertEqual(provenance["windows"]["unknown"], 1)

    def test_identity_and_sequence_overlap_rejected(self):
        manifest = self.build_manifest("WUT")
        duplicate = copy.deepcopy(manifest)
        duplicate["identities"][-1]["name"] = duplicate["identities"][0]["name"]
        with self.assertRaisesRegex(ValueError, "identity names"):
            protocols.validate_manifest(duplicate)
        leakage = copy.deepcopy(manifest)
        leakage["identities"][0]["sequences"][0]["split"] = "unknown"
        with self.assertRaisesRegex(ValueError, "leaks"):
            protocols.validate_manifest(leakage)
        overlap = copy.deepcopy(manifest)
        overlap["identities"][1]["sequences"][0]["source"] = overlap["identities"][0]["sequences"][0]["source"]
        with self.assertRaisesRegex(ValueError, "assigned more than once"):
            protocols.validate_manifest(overlap)

    def test_source_modification_fails_before_output_is_written(self):
        manifest = self.build_manifest("WUT")
        source = self.raw / manifest["identities"][0]["sequences"][0]["source"]
        array = np.load(source)
        array[0, 0, 0] += 1
        np.save(source, array)
        with self.assertRaisesRegex(ValueError, "Source fingerprint mismatch"):
            protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_prepared_file_modification_and_wrong_manifest_are_rejected(self):
        manifest = self.build_manifest("WUT")
        protocols.prepare_trial(manifest, self.raw, self.output)
        modified = copy.deepcopy(manifest)
        modified["trial_id"] = "different"
        with self.assertRaisesRegex(ValueError, "different manifest"):
            protocols.verify_provenance(modified, self.output)
        target = self.output / "WUT/6_explicit/test_npy_data/probe/t_source_x_WUT_6.npy"
        array = np.load(target)
        array[0, 0] += 1
        np.save(target, array)
        with self.assertRaisesRegex(ValueError, "Prepared output fingerprint mismatch"):
            protocols.verify_provenance(manifest, self.output)

    def test_recorded_count_mismatch_fails_before_output_is_written(self):
        manifest = self.build_manifest("WUT")
        manifest["expected_windows"]["probe"] = 4
        with self.assertRaisesRegex(ValueError, "Historical window count mismatch"):
            protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_historical_log_parser_preserves_person15_exception(self):
        log = (
            "Person ID: 1 | Gallery Seqs: 0, 2| Probe Seq: 1 | Total Seq: 5\n"
            "Person ID: 15 | Gallery Seqs: 0 | Probe Seq: 1 | Total Seq: 3\n"
        )
        result = protocols.parse_preprocess_log(log, "KGBD")
        self.assertEqual(result[1]["gallery_indices"], [0, 2])
        self.assertEqual(result[15]["gallery_indices"], [0])
        self.assertEqual(result[15]["name"], "Person015")
        log = "Julia\n[8 0] [1]\nPerson ID: 3 | Gallery Seqs: 8, 0 | Probe Seq: 1 | Total Seq: 27\n"
        result = protocols.parse_preprocess_log(log, "WUT")
        self.assertEqual(result[3]["total_sequences"], 27)
        self.assertEqual(result[3]["gallery_indices"], [8, 0])

    def test_historical_loader_count_parser_uses_raw_training_windows(self):
        text = (
            "(70278, 19) 11111111\n"
            "[gen_train_data] dataset=WUT | split=probe | T=6\n"
            "[gen_train_data] X_train_J : (22770, 6, 19, 3)\n"
            "[gen_train_data] X_test_J : (777, 6, 19, 3)\n"
            "[gen_train_data] dataset=WUT | split=unknown | T=6\n"
            "[gen_train_data] X_test_J : (376, 6, 19, 3)\n"
        )
        self.assertEqual(protocols._recorded_window_counts(text, "WUT", 6),
                         {"train": 11713, "probe": 777, "unknown": 376})

    def test_missing_probe_windows_are_reported_without_changing_protocol(self):
        manifest = self.build_manifest("WUT")
        seq = manifest["identities"][0]["sequences"][2]
        source = self.raw / seq["source"]
        array = np.load(source)
        array[0, 0, 0] = np.nan
        np.save(source, array)
        seq["sha256"] = protocols.sha256_file(source)
        manifest["expected_windows"]["probe"] = 2
        provenance = protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertEqual(provenance["missing_known_ids"]["probe"], [1])
        self.assertEqual(provenance["windows"]["probe"], 2)


if __name__ == "__main__":
    unittest.main()
