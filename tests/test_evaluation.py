"""Regression checks for checkpoint restoration, OOS scores, and exported IDs."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation
import evaluation_kgbd
import visualization
import randomness
from SSGT import SSGT


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.tmp = tempfile.TemporaryDirectory(prefix="ssgt-evaluation-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, use_time_attn=False, spa_pos_type="learned", dataset="KGBD"):
        """Create a small real model checkpoint without external datasets."""
        args = dict(
            dataset=dataset, length=3, H=8, n_heads=2, L_transformer=1,
            enc_k_spa=3, enc_k_time=2, frame_attn_pool=0, learnable_temp=0,
            use_time_attn=int(use_time_attn), spa_pos_type=spa_pos_type,
            data_subdir="3_variant", pos_enc=1,
        )
        model = SSGT(
            H=8, n_heads=2, L=1, J=evaluation._dataset_joint_num(dataset), T=3,
            enc_k_spa=3, enc_k_time=2,
            frame_attn_pool=False, learnable_temp=False,
            use_time_attn=use_time_attn, spa_pos_type=spa_pos_type,
        ).eval()
        path = self.root / "best.ckpt"
        torch.save({"args": args, "model_state": model.state_dict()}, path)
        return model, args, path

    def test_knn_scores_against_sorted_pairwise_distances(self):
        library = evaluation._l2n(self.rng.normal(size=(20, 8)).astype(np.float32))
        query = evaluation._l2n(self.rng.normal(size=(5, 8)).astype(np.float32))
        distance = np.sqrt(np.maximum(0, 2 - 2 * library @ library.T))
        for k in (1, 3, 100):
            with self.subTest(k=k):
                r, threshold = evaluation._kth_nn_tau_from_lib(library, k, 0.95)
                reference = np.sort(distance, axis=1)[:, min(k, 19)]
                np.testing.assert_allclose(r, reference, atol=1e-6)
                self.assertAlmostEqual(threshold, float(np.quantile(reference, 0.95)), places=6)
                query_distance = np.sqrt(np.maximum(0, 2 - 2 * query @ library.T))
                expected = np.sort(query_distance, axis=1)[:, min(k, 20) - 1]
                np.testing.assert_allclose(
                    evaluation._kth_nn_dist_queries(query, library, k), expected, atol=1e-6,
                )

    def test_metrics_use_known_acceptance_direction(self):
        metrics, _ = evaluation.eval_oos_metrics(
            np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9]), sigma_eval=0.95,
        )
        self.assertEqual(metrics["FPR95"], 0.0)
        self.assertAlmostEqual(metrics["AUROC"], 1.0)

    def test_frame_vote_index_conversion_and_tie_break(self):
        spans = np.array([[0, 2], [1, 3]])
        labels = np.array([0, 2])
        distances = np.array([0.5, 0.1])
        for module in (evaluation, evaluation_kgbd):
            np.testing.assert_array_equal(
                module._majority_vote_per_frame(spans, labels, distances, True, 5),
                [1, 3, 3, 3, 0],
            )
            np.testing.assert_array_equal(
                module._majority_vote_per_frame(spans, labels, distances, False, 5),
                [0, 2, 2, 2, 0],
            )

    def test_original_identity_mapping_preserves_heldout_gaps_and_distances(self):
        for dataset, raw_ids, expected in (
            ('WUT', [1, 13, 25], [0, 12, 24]),
            ('KGBD', [1, 6, 164], [0, 5, 163]),
            ('KS20', [0, 3, 19], [0, 3, 19]),
        ):
            folder = self.root / dataset / '3_variant' / 'train_npy_data'
            folder.mkdir(parents=True)
            np.save(folder / f'ids_{dataset}_3.npy', {i: [j] for j, i in enumerate(raw_ids)})
            with mock.patch.object(evaluation, 'data_root', return_value=self.root):
                identities = evaluation._load_identity_ids(dataset, 3, '3_variant')
            np.testing.assert_array_equal(identities, expected)
            decoded = evaluation._decode_known_labels(np.eye(3), identities)
            np.testing.assert_array_equal(decoded, expected)
            if dataset == 'WUT':
                self.assertEqual([evaluation._idx_to_name(i) for i in decoded],
                                 ['Jacek', 'Malgosia', 'Wojciech'])
            distances = np.array([0.4, 0.1, 0.8])
            original = evaluation._topk_by_identity(distances, np.arange(3), 3)
            mapped = evaluation._topk_by_identity(distances, decoded, 3)
            self.assertEqual(mapped, [(int(identities[i]), d) for i, d in original])
            with self.assertRaises(ValueError):
                evaluation._decode_known_labels(np.zeros((1, 3)), identities)
            with self.assertRaises(ValueError):
                evaluation._decode_known_labels(np.eye(4), identities)
        for one_based in (False, True):
            self.assertEqual(evaluation._parse_groundtruth('Malgosia', 'WUT', one_based),
                             (12, 'Malgosia'))
            self.assertEqual(evaluation._parse_groundtruth(str(12 + one_based), 'WUT', one_based),
                             (12, 'Malgosia'))
            self.assertEqual(evaluation._parse_groundtruth(str(163 + one_based), 'KGBD', one_based),
                             (163, str(163 + one_based)))
        self.assertEqual(evaluation._parse_groundtruth('0', 'WUT', True), (None, '0'))
        self.assertEqual(evaluation._unknown_identity_id('KGBD'), 164)
        self.assertEqual(evaluation._unknown_identity_id('WUT'), 25)

    def test_checkpoint_architecture_and_features_round_trip(self):
        for use_time in (False, True):
            for position in ("learned", "laplacian"):
                with self.subTest(use_time=use_time, position=position):
                    model, args, path = self.checkpoint(use_time, position)
                    restored, saved = evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")
                    self.assertEqual(saved["data_subdir"], "3_variant")
                    self.assertEqual(restored.use_time_attn, use_time)
                    self.assertEqual(restored.spa_pos_type, position)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    windows = self.rng.normal(size=(4, 3, 20, 3)).astype(np.float32)
                    encoding = self.rng.normal(size=(20, 3)).astype(np.float32)
                    original = evaluation._extract_seq_feats(model, windows, encoding, 2, torch.device("cpu"), 3, 20, 8)
                    recovered = evaluation._extract_seq_feats(restored, windows, encoding, 2, torch.device("cpu"), 3, 20, 8)
                    np.testing.assert_array_equal(recovered, original)

    def test_checkpoint_rejects_incompatible_metadata_and_weights(self):
        model, args, path = self.checkpoint()
        with self.assertRaises(ValueError):
            evaluation.load_checkpoint_model(path, torch.device("cpu"), "WUT")
        state = model.state_dict()
        state.pop("in1.weight")
        torch.save({"args": args, "model_state": state}, path)
        with self.assertRaises(RuntimeError):
            evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")
        torch.save({"model_state": model.state_dict()}, path)
        with self.assertRaises(ValueError):
            evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")

    def test_checkpoint_spatial_encoding_override_and_historical_fallback(self):
        model, args, path = self.checkpoint(spa_pos_type="laplacian")
        computed = np.zeros((20, 3), dtype=np.float32)
        stored = torch.arange(60, dtype=torch.float32).reshape(20, 3)
        provenance = {"source": "captured_during_training"}
        original_keys = set(model.state_dict())
        torch.save({"args": args, "model_state": model.state_dict(),
                    "spatial_pos_encoding": stored,
                    "spatial_pos_encoding_provenance": provenance}, path)
        restored, _ = evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")
        resolved = evaluation.resolve_spatial_pos_encoding(restored, computed)
        np.testing.assert_array_equal(resolved, stored.numpy())
        self.assertEqual(set(restored.state_dict()), original_keys)
        self.assertEqual(restored.spatial_pos_encoding_metadata["used_source"], "checkpoint")
        self.assertEqual(restored.spatial_pos_encoding_metadata["checkpoint_provenance"], provenance)
        self.assertEqual(len(restored.spatial_pos_encoding_metadata["float32_sha256"]), 64)

        torch.save({"args": args, "model_state": model.state_dict()}, path)
        historical, _ = evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")
        np.testing.assert_array_equal(evaluation.resolve_spatial_pos_encoding(historical, computed), computed)
        self.assertEqual(historical.spatial_pos_encoding_metadata["used_source"], "recomputed_by_data_loader")
        for invalid in (np.zeros((19, 3)), np.full((20, 3), np.nan), np.full((20, 3), np.inf)):
            with self.subTest(shape=invalid.shape):
                torch.save({"args": args, "model_state": model.state_dict(),
                            "spatial_pos_encoding": invalid}, path)
                with self.assertRaises(ValueError):
                    evaluation.load_checkpoint_model(path, torch.device("cpu"), "KGBD")
                with self.assertRaises(ValueError):
                    evaluation.resolve_spatial_pos_encoding(historical, invalid)

    def test_missing_npy_input_fails_before_loading_checkpoint(self):
        for module in (evaluation, evaluation_kgbd):
            with mock.patch.object(module, "load_checkpoint_model") as load:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        module.main(["--data_source", "npy"])
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()

    def test_paper_oos_defaults_and_checkpoint_data_variant_inheritance(self):
        for module in (evaluation, evaluation_kgbd):
            args = module.get_parser().parse_args([])
            self.assertEqual(args.unk_knn_k, 1)
            self.assertEqual(args.unk_sigma, 0.95)
            self.assertEqual(args.proto_source, 'gallery')
            self.assertIsNone(args.data_subdir)
            self.assertEqual(args.seed, -1)

    def test_random_plot_sampling_can_replay_and_wrap_derived_seeds(self):
        labels = np.repeat(np.arange(3), 20)
        largest_seed = 2 ** 32 - 1
        with mock.patch.object(randomness.secrets, 'randbits', side_effect=[largest_seed, 17]) as entropy:
            first = visualization.subsample_id_oos_indices(
                labels, len(labels), 20, per_id_max=3, oos_max=4)
            second = visualization.subsample_id_oos_indices(
                labels, len(labels), 20, per_id_max=3, oos_max=4)
        self.assertEqual(entropy.call_count, 2)
        replay = visualization.subsample_id_oos_indices(
            labels, len(labels), 20, per_id_max=3, oos_max=4, plot_seed=largest_seed)
        for actual, expected in zip(replay, first):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(first[0], second[0]))
        np.testing.assert_array_equal(first[1], np.random.default_rng(6).choice(20, size=4, replace=False))
        known = self.rng.normal(size=(60, 8)).astype(np.float32)
        unknown = self.rng.normal(size=(20, 8)).astype(np.float32)
        with mock.patch.object(randomness.secrets, 'randbits', return_value=largest_seed) as entropy, \
                mock.patch.object(visualization.plt, 'savefig'):
            generated = visualization.plot_id_clusters_with_oos_2d(
                known, labels, unknown, str(self.root / 'auto.png'), method='pca',
                per_id_max=3, oos_max=4)
            self.assertEqual(entropy.call_count, 1)
            replayed = visualization.plot_id_clusters_with_oos_2d(
                known, labels, unknown, str(self.root / 'replay.png'), method='pca',
                per_id_max=3, oos_max=4, plot_seed=largest_seed)
            self.assertEqual(entropy.call_count, 1)
        self.assertEqual(generated['plot_seed'], largest_seed)
        for field in ('sel_id_idx', 'oos_idx', 'XY_id', 'XY_oos'):
            np.testing.assert_array_equal(generated[field], replayed[field])

    def test_invalid_seed_fails_before_checkpoint_loading(self):
        for module in (evaluation, evaluation_kgbd):
            for invalid in ('-2', str(2 ** 32)):
                with mock.patch.object(module, 'load_checkpoint_model') as load, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        module.main(['--seed', invalid])
                load.assert_not_called()

    def test_tsne_iteration_keyword_compatibility(self):
        class OldTSNE:
            def __init__(self, n_iter=1000):
                self.iterations = n_iter

        class NewTSNE:
            def __init__(self, max_iter=1000):
                self.iterations = max_iter

        for implementation in (OldTSNE, NewTSNE):
            with mock.patch.object(visualization, "TSNE", implementation):
                self.assertEqual(visualization.make_tsne(n_iter=300).iterations, 300)
        known = self.rng.normal(size=(12, 4)).astype(np.float32)
        unknown = self.rng.normal(size=(6, 4)).astype(np.float32)
        known_xy, unknown_xy, _ = visualization.project_id_oos_2d(known, unknown, perplexity=5)
        self.assertEqual(known_xy.shape, (12, 2))
        self.assertEqual(unknown_xy.shape, (6, 2))

    def test_boundary_plot_supports_installed_matplotlib(self):
        known = self.rng.normal(size=(12, 2)).astype(np.float32)
        unknown = self.rng.normal(size=(5, 2)).astype(np.float32) + 3
        for module in (evaluation, evaluation_kgbd):
            output = self.root / (module.__name__ + "_boundary.png")
            threshold = module.plot_knn_boundary_2d(
                known, unknown, sigma=0.95, k=2, path=str(output),
                grid=12, metric="euclidean",
                **({"id_labels": np.repeat(np.arange(3), 4)} if module is evaluation else {}),
            )
            self.assertTrue(output.is_file())
            self.assertTrue(np.isfinite(threshold))

    def test_numpy_pipeline_exports_consistent_window_and_frame_ids(self):
        for dataset in ('WUT', 'KGBD'):
            model, args, path = self.checkpoint(dataset=dataset)
            joints = model.J
            stored_encoding = torch.full((joints, 3), 0.25)
            torch.save({"args": args, "model_state": model.state_dict(),
                        "spatial_pos_encoding": stored_encoding}, path)
            known = self.rng.normal(size=(6, 3, joints, 3)).astype(np.float32)
            labels = np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 2, 2]]
            encoding = self.rng.normal(size=(joints, 3)).astype(np.float32)
            raw_path = self.root / "sequence.npy"
            np.save(raw_path, known[2:4].reshape(6, joints, 3))
            folder = self.root / dataset / '3_variant' / 'train_npy_data'
            folder.mkdir(parents=True)
            # Original identity three occupies compressed one-hot column one.
            np.save(folder / f'ids_{dataset}_3.npy', {1: [0, 1], 3: [2, 3], 5: [4, 5]})
            variant_calls = []
            sampling_calls = []

            def load_split(**kwargs):
                variant_calls.append(kwargs["data_subdir"])
                sampling_calls.append(kwargs["sampling_seed"])
                return (known, 0, 0, 0, 0, labels, known, 0, 0, 0, 0, labels,
                        np.eye(joints), 0, encoding)

            for module in (evaluation, evaluation_kgbd):
                for one_based in (False, True):
                    with self.subTest(dataset=dataset, module=module.__name__, one_based=one_based):
                        prefix = self.root / dataset / module.__name__ / str(one_based) / "prediction"
                        extracted_encodings = []
                        sampling_calls.clear()
                        selected_seed = 37 if one_based else 2 ** 32 - 1
                        extract = module._extract_seq_feats

                        def record_encoding(*args, **kwargs):
                            extracted_encodings.append(args[2].copy())
                            return extract(*args, **kwargs)

                        with contextlib.ExitStack() as stack:
                            stack.enter_context(mock.patch.object(module.process, "gen_train_data", side_effect=load_split))
                            stack.enter_context(mock.patch.object(evaluation, "data_root", return_value=self.root))
                            stack.enter_context(mock.patch.object(module, "_extract_seq_feats", side_effect=record_encoding))
                            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                            entropy = stack.enter_context(mock.patch.object(
                                randomness.secrets, 'randbits', return_value=selected_seed))
                            arguments = [
                                "--dataset", dataset, "--ckpt", str(path), "--gpu", "cpu",
                                "--data_source", "npy", "--npy-path", str(raw_path),
                                "--save_npy_base", str(prefix), "--eval_oos", "0", "--proj2d", "0",
                                "--unk_detect", "0", "--center_root", "0", "--batch_size", "3",
                                "--save_csv", str(prefix) + '.csv',
                                "--one_based_ids", str(int(one_based)),
                                "--groundtruth", 'Julia' if dataset == 'WUT' else str(2 + one_based),
                            ]
                            if one_based:
                                arguments += ['--seed', str(selected_seed)]
                            module.main(arguments)
                        self.assertEqual(entropy.call_count, 0 if one_based else 1)
                        self.assertEqual(sampling_calls, [selected_seed] * 3)
                        window_ids = np.load(str(prefix) + "_win_id.npy")
                        frame_ids = np.load(str(prefix) + "_frame_id.npy")
                        np.testing.assert_array_equal(window_ids, [2 + one_based] * 2)
                        np.testing.assert_array_equal(frame_ids, np.repeat(window_ids, 3))
                        topk = np.load(str(prefix) + "_topk_id.npy")
                        self.assertEqual(topk.shape, (2, 3))
                        self.assertEqual(set(topk[0]), {int(one_based), 2 + one_based, 4 + one_based})
                        with Path(str(prefix) + '.csv').open() as stream:
                            rows = list(csv.DictReader(stream))
                        for row in rows:
                            self.assertEqual(int(row['pred']), 2 + one_based)
                            self.assertEqual(row['pred'], row['gt_id'])
                            self.assertEqual(row['hit_top1'], '1')
                            if dataset == 'WUT':
                                self.assertEqual(row['pred_name'], 'Julia')
                                self.assertEqual(row['gt_name'], 'Julia')
                        summary = json.loads(Path(str(prefix) + "_unk_verdict.json").read_text())
                        self.assertEqual(summary["num_windows"], 2)
                        self.assertEqual(summary["randomness"], {
                            'seed': selected_seed,
                            'requested_seed': selected_seed if one_based else -1,
                            'selection': 'explicit' if one_based else 'automatic',
                        })
                        self.assertEqual(summary["identity_labels"]["known_original_ids"],
                                         [int(one_based), 2 + one_based, 4 + one_based])
                        self.assertEqual(summary["spatial_position_encoding"]["used_source"], "checkpoint")
                        self.assertTrue(extracted_encodings)
                        for encoding_used in extracted_encodings:
                            np.testing.assert_array_equal(encoding_used, stored_encoding.numpy())
            self.assertTrue(variant_calls)
            self.assertEqual(set(variant_calls), {"3_variant"})


if __name__ == "__main__":
    unittest.main()
