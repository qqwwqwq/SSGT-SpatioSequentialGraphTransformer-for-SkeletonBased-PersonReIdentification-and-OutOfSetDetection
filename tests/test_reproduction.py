"""Checks for streamed OOS scoring and independent five-trial aggregation."""

import copy
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce
import evaluation
import evaluation_kgbd
import randomness


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ssgt-reproduce-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rng = np.random.default_rng(12)

    def test_blocked_neighbors_match_full_matrix_for_all_paper_ranks(self):
        known = evaluation._l2n(self.rng.normal(size=(210, 8)).astype(np.float32))
        unknown = evaluation._l2n(self.rng.normal(size=(17, 8)).astype(np.float32))
        ranks = [1, 5, 20, 50, 200]
        for block in (1, 13, 512):
            calibrated = reproduce.neighbor_distances(known, known, ranks, True, block)
            queried = reproduce.neighbor_distances(unknown, known, ranks, False, block)
            for k in ranks:
                expected, _ = evaluation._kth_nn_tau_from_lib(known, k, 0.95)
                np.testing.assert_allclose(calibrated[k], expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(
                    queried[k], evaluation._kth_nn_dist_queries(unknown, known, k),
                    rtol=1e-6, atol=1e-6,
                )

    def test_small_library_rank_clamping_preserves_original_convention(self):
        known = evaluation._l2n(self.rng.normal(size=(7, 4)).astype(np.float32))
        unknown = evaluation._l2n(self.rng.normal(size=(3, 4)).astype(np.float32))
        metric = reproduce.score_features(known, unknown, [1, 5, 20, 200], block_size=2, threads=1)
        for k in (20, 200):
            self.assertEqual(metric[str(k)]["k_calibration"], 6)
            self.assertEqual(metric[str(k)]["k_query"], 7)
        self.assertEqual(metric["20"]["AUROC"], metric["200"]["AUROC"])

    def test_calibration_rejects_different_query_set(self):
        known = np.eye(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            reproduce.neighbor_distances(known[::-1], known, [1], True)
        with self.assertRaises(ValueError):
            reproduce.neighbor_distances(known[:1], known[:1], [1], True)

    def test_paper_defaults(self):
        for module in (evaluation, evaluation_kgbd):
            options = module.get_parser().parse_args([])
            self.assertEqual(options.unk_knn_k, 1)
            self.assertEqual(options.unk_sigma, 0.95)
            self.assertEqual(options.proto_source, "gallery")
            self.assertEqual(options.seed, -1)

    def test_trial_evaluation_uses_persisted_spatial_basis_and_reports_provenance(self):
        import data
        import torch
        from SSGT import SSGT

        model = SSGT(H=8, n_heads=2, L=1, J=20, T=3, enc_k_spa=3, enc_k_time=2,
                     frame_attn_pool=False, learnable_temp=False).eval()
        saved = dict(dataset="KGBD", H=8, n_heads=2, L_transformer=1, length=3,
                     enc_k_spa=3, enc_k_time=2, frame_attn_pool=0, learnable_temp=0)
        encoding = torch.arange(60, dtype=torch.float32).reshape(20, 3) / 60
        encoding_provenance = {"source": "recomputed_same_environment", "captured_from_running_training": False}
        checkpoint = self.root / "spatial.ckpt"
        torch.save({"args": saved, "model_state": model.state_dict(),
                    "spatial_pos_encoding": encoding,
                    "spatial_pos_encoding_provenance": encoding_provenance}, checkpoint)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"dataset": "KGBD", "trial_id": "synthetic", "length": 3}))
        known = self.rng.normal(size=(6, 3, 20, 3)).astype(np.float32)
        unknown = self.rng.normal(size=(3, 3, 20, 3)).astype(np.float32)
        loaded = [known, 0, 0, 0, 0, 0, unknown, 0, 0, 0, 0, 0, 0, 0, np.zeros((20, 3))]
        verified = {"checkpoint_sha256": reproduce.sha256_file(checkpoint),
                    "provenance_sha256": "0" * 64,
                    "provenance": {"known_ids": [1, 2, 3], "unknown_original_ids": [4],
                                   "output_sha256": {"synthetic": "0" * 64}}}
        args = argparse.Namespace(gpu="cpu", manifest=str(manifest), data_root=str(self.root),
                                  checkpoint=str(checkpoint), provenance=None, batch_size=3,
                                  k_values=[1], sigma=0.95, distance_batch_size=2, distance_threads=1,
                                  reference_result=None, save_features=False, output=str(self.root / "result.json"))
        extract = evaluation._extract_seq_feats
        observed = []

        def capture_encoding(*positional, **keywords):
            observed.append(positional[2].copy())
            return extract(*positional, **keywords)

        selected_seed = 2 ** 32 - 1
        with mock.patch.object(reproduce, "verify_trial_inputs", return_value=verified), \
                mock.patch.object(data, "gen_train_data", return_value=loaded) as load_data, \
                mock.patch.object(randomness.secrets, "randbits", return_value=selected_seed) as entropy, \
                mock.patch.object(evaluation, "_extract_seq_feats", side_effect=capture_encoding), \
                contextlib.redirect_stdout(io.StringIO()):
            result = reproduce.evaluate_trial(args)
            self.assertEqual(entropy.call_count, 1)
            self.assertEqual(load_data.call_args[1]["sampling_seed"], selected_seed)
            args.seed = selected_seed
            replay = reproduce.evaluate_trial(args)
            self.assertEqual(entropy.call_count, 1)
            self.assertEqual(load_data.call_args[1]["sampling_seed"], selected_seed)
        self.assertEqual(len(observed), 4)
        for actual in observed:
            np.testing.assert_array_equal(actual, encoding.numpy())
        metadata = result["provenance"]["spatial_position_encoding"]
        self.assertEqual(metadata["used_source"], "checkpoint")
        self.assertEqual(metadata["checkpoint_provenance"], encoding_provenance)
        self.assertEqual(metadata["shape"], [20, 3])
        self.assertEqual(len(metadata["float32_sha256"]), 64)
        self.assertEqual(result["randomness"], {
            "seed": selected_seed, "requested_seed": -1, "selection": "automatic",
        })
        self.assertEqual(result["protocol"]["sampling_seed"], selected_seed)
        self.assertEqual(replay["randomness"]["selection"], "explicit")
        self.assertEqual(replay["metrics"], result["metrics"])

    def test_cli_random_seed_default_and_validation(self):
        arguments = ['evaluate', '--manifest', 'manifest.json', '--data-root', 'data',
                     '--checkpoint', 'model.ckpt', '--output', 'result.json']
        with mock.patch.object(reproduce, 'evaluate_trial') as evaluate:
            reproduce.main(arguments)
            self.assertEqual(evaluate.call_args[0][0].seed, -1)
            reproduce.main(arguments + ['--seed', '0'])
            self.assertEqual(evaluate.call_args[0][0].seed, 0)
        for invalid in ('-2', str(2 ** 32)):
            with mock.patch.object(reproduce, 'evaluate_trial') as evaluate, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    reproduce.main(arguments + ['--seed', invalid])
            evaluate.assert_not_called()

    def test_trial_input_verification_rejects_tampering_and_identity_overlap(self):
        import protocols
        raw = self.root / "raw"
        data_root = self.root / "data"
        identities = []
        for label in range(1, 5):
            sequences = []
            role = "unknown" if label == 4 else "known"
            for split in (["unknown"] if role == "unknown" else ["train", "gallery", "probe"]):
                source = raw / str(label) / (split + ".npy")
                source.parent.mkdir(parents=True, exist_ok=True)
                np.save(source, self.rng.normal(size=(6, 19, 3)).astype(np.float32))
                sequences.append({"name": split, "split": split,
                                  "source": source.relative_to(raw).as_posix(), "format": "npy",
                                  "sha256": reproduce.sha256_file(source), "size_bytes": source.stat().st_size})
            identities.append({"name": str(label), "original_label": label,
                               "role": role, "sequences": sequences})
        checkpoint = self.root / "model.ckpt"
        checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
        manifest = {"schema_version": 1, "dataset": "WUT", "trial_id": "synthetic", "length": 6,
                    "data_subdir": "6", "identities": identities,
                    "checkpoint_sha256": reproduce.sha256_file(checkpoint)}
        protocols.prepare_trial(manifest, raw, data_root)
        verified = reproduce.verify_trial_inputs(manifest, data_root, checkpoint)
        self.assertEqual(verified["checkpoint_sha256"], manifest["checkpoint_sha256"])
        overlapping = copy.deepcopy(manifest)
        overlapping["identities"][-1]["original_label"] = 1
        with self.assertRaises(ValueError):
            reproduce.verify_trial_inputs(overlapping, data_root, checkpoint)
        checkpoint.write_bytes(b"different weights")
        with self.assertRaises(ValueError):
            reproduce.verify_trial_inputs(manifest, data_root, checkpoint)
        checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
        coordinate_file = data_root / "WUT/6/train_npy_data/source_x_WUT_6.npy"
        coordinates = np.load(coordinate_file)
        coordinates[0, 0] += 1.0
        np.save(coordinate_file, coordinates)
        with self.assertRaises(ValueError):
            reproduce.verify_trial_inputs(manifest, data_root, checkpoint)

    def trial_files(self):
        paths = []
        for index in range(5):
            record = {
                "status": "evaluated", "dataset": "WUT", "trial_id": "exp_%02d" % index,
                "provenance": {"checkpoint_sha256": str(index) * 64},
                "protocol": {"length": 6, "sigma": 0.95, "known_score_source": "train", "batch_size": 256},
                "metrics": {"1": {"AUROC": 0.95 + index * 0.01, "FPR95": 0.10 + index * 0.01}},
            }
            path = self.root / ("trial_%d.json" % index)
            path.write_text(json.dumps(record))
            paths.append(path)
        return paths

    def test_five_trial_aggregation_uses_sample_std_and_percent(self):
        paths = self.trial_files()
        targets = {"oos": {"k_values": [1], "WUT": {
            "AUROC_mean": [97.0], "AUROC_sample_std": [1.58],
            "FPR95_mean": [12.0], "FPR95_sample_std": [1.58],
        }}}
        target_path = self.root / "targets.json"
        target_path.write_text(json.dumps(targets))
        summary = reproduce.aggregate_trials(paths, target_path)
        self.assertAlmostEqual(summary["metrics"]["1"]["AUROC_mean"], 97.0)
        self.assertAlmostEqual(summary["metrics"]["1"]["AUROC_sample_std"], np.std([95, 96, 97, 98, 99], ddof=1))
        self.assertAlmostEqual(summary["paper_comparison"]["1"]["FPR95_mean"]["difference_percentage_points"], 0.0)
        self.assertEqual(summary["std_ddof"], 1)
        self.assertEqual(len(summary["trial_files"]), 5)

    def test_aggregation_rejects_incomplete_or_reused_trials(self):
        paths = self.trial_files()
        with self.assertRaises(ValueError):
            reproduce.aggregate_trials(paths[:4])
        with self.assertRaises(ValueError):
            reproduce.aggregate_trials(paths[:4] + [paths[0]])
        last = json.loads(paths[-1].read_text())
        last["provenance"]["checkpoint_sha256"] = "0" * 64
        paths[-1].write_text(json.dumps(last))
        with self.assertRaises(ValueError):
            reproduce.aggregate_trials(paths)


if __name__ == "__main__":
    unittest.main()
