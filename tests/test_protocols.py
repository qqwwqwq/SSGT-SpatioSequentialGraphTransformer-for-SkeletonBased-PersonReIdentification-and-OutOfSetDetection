"""Test raw-data integrity and isolated preparation of SSGT OOS splits."""

import contextlib
import copy
import io
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
                "expected_identities": {"known": 3, "unknown": 1}}

    def test_wut_source_holdout_and_provenance(self):
        manifest = self.build_manifest("WUT")
        source_before = {str(path.relative_to(self.raw)): protocols.sha256_file(path)
                         for path in self.raw.rglob("*") if path.is_file()}
        provenance = protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertEqual(provenance["windows"], {"train": 3, "gallery": 3, "probe": 3, "unknown": 1})
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

    def test_missing_training_identity_fails_before_output_is_written(self):
        manifest = self.build_manifest("WUT")
        sequence = manifest["identities"][0]["sequences"][0]
        source = self.raw / sequence["source"]
        array = np.load(source)
        array[0, 0, 0] = np.nan
        np.save(source, array)
        sequence["sha256"] = protocols.sha256_file(source)
        with self.assertRaisesRegex(ValueError, "cover every known identity"):
            protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_sources_and_outputs_must_stay_in_separate_roots(self):
        manifest = self.build_manifest("WUT")
        with self.assertRaisesRegex(ValueError, "separate, non-nested"):
            protocols.prepare_trial(manifest, self.raw, self.raw / "prepared")
        escaped = copy.deepcopy(manifest)
        escaped["identities"][0]["sequences"][0]["source"] = "../outside.npy"
        with self.assertRaisesRegex(ValueError, "relative and stay within"):
            protocols.prepare_trial(escaped, self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_prepares_a_manifest_for_training(self):
        manifest = self.build_manifest("KGBD")
        filename = self.root / "split.json"
        filename.write_text(json.dumps(manifest))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            protocols.main(["--manifest", str(filename), "--raw-root", str(self.raw),
                            "--prepared-root", str(self.output)])
        summary = json.loads(output.getvalue())
        self.assertEqual(Path(summary["data_root"]), self.output)
        self.assertEqual(summary["windows"], {"train": 3, "gallery": 3, "probe": 3, "unknown": 1})
        protocols.verify_provenance(filename, self.output)

    def test_packaged_manifests_define_five_complete_oos_splits_per_dataset(self):
        config = Path(__file__).resolve().parents[1] / "config" / "trials"
        for dataset, known, unknown, sequence_count in (("wut", 23, 2, 293), ("kgbd", 148, 16, 822)):
            files = sorted((config / dataset).glob("exp*.json"))
            self.assertEqual(len(files), 5)
            for filename in files:
                with self.subTest(manifest=filename.name, dataset=dataset):
                    manifest = protocols.load_manifest(filename)
                    self.assertEqual(protocols.validate_manifest(manifest), {"known": known, "unknown": unknown})
                    self.assertEqual(sum(len(person["sequences"]) for person in manifest["identities"]), sequence_count)
                    self.assertEqual(manifest["length"], 6)

    def test_missing_probe_windows_are_reported_without_changing_protocol(self):
        manifest = self.build_manifest("WUT")
        seq = manifest["identities"][0]["sequences"][2]
        source = self.raw / seq["source"]
        array = np.load(source)
        array[0, 0, 0] = np.nan
        np.save(source, array)
        seq["sha256"] = protocols.sha256_file(source)
        provenance = protocols.prepare_trial(manifest, self.raw, self.output)
        self.assertEqual(provenance["missing_known_ids"]["probe"], [1])
        self.assertEqual(provenance["windows"]["probe"], 2)


if __name__ == "__main__":
    unittest.main()
