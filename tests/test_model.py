"""CPU regression checks for the public encoder and checkpoint evaluation.

All inputs and checkpoints are generated in memory or temporary directories.
These checks do not depend on datasets, original development files, or a GPU.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SSGT import SSGT, TranSG, laplacian_position_encoding_np
import training
from loss_balancing import DynamicLossBalancer, mean_normalized_slope, update_lambda


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_architecture_variants_and_strict_checkpoint_round_trip(self):
        """All position/temporal/QKV variants retain valid checkpoint keys."""
        self.assertIs(TranSG, SSGT)
        for temporal in (False, True):
            for position in ('laplacian', 'learned'):
                for merged in (False, True):
                    with self.subTest(temporal=temporal, position=position, merged=merged):
                        kwargs = dict(
                            J=19, T=4, H=16, n_heads=4, L=1,
                            enc_k_spa=3, enc_k_time=2,
                            use_time_attn=temporal, spa_pos_type=position,
                            qkv_merged=merged,
                        )
                        model = SSGT(**kwargs).eval()
                        inputs = torch.randn(2, 4, 19, 3)
                        position_features = torch.randn(19, 3)
                        with torch.no_grad():
                            expected = model.add_pos_and_attn(
                                model.encode_input(inputs), position_features,
                                train=False, bn_freeze=True,
                            )
                        self.assertEqual(tuple(expected.shape), (2, 4, 19, 16))
                        self.assertTrue(torch.isfinite(expected).all().item())

                        stream = io.BytesIO()
                        torch.save(model.state_dict(), stream)
                        stream.seek(0)
                        restored = SSGT(**kwargs).eval()
                        restored.load_state_dict(
                            torch.load(stream, map_location='cpu', weights_only=True),
                            strict=True,
                        )
                        with torch.no_grad():
                            actual = restored.add_pos_and_attn(
                                restored.encode_input(inputs), position_features,
                                train=False, bn_freeze=True,
                            )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reconstruction_and_contrastive_losses_backpropagate(self):
        """The split modules support a finite gradient through all loss terms."""
        model = SSGT(J=19, T=4, H=16, n_heads=4, L=1,
                     enc_k_spa=3, enc_k_time=2).train()
        inputs = torch.randn(2, 4, 19, 3)
        hidden = model.add_pos_and_attn(
            model.encode_input(inputs), torch.randn(19, 3), train=True,
        )
        target = inputs.reshape(2, 4, 19 * 3)
        joint_mask = torch.arange(19) % 2 == 0
        frame_mask = torch.tensor([True, True, False, False])
        structure_loss = model.structure_prompt_recon(
            hidden, joint_mask, 19, 4, 'l1', target, batch_size=2,
        )
        trajectory_loss = model.trajectory_prompt_recon(
            hidden, frame_mask, 19, 4, 'l1', target, batch_size=2,
        )
        frames = hidden.mean(dim=2)
        prototypes = torch.randn(3, 16)
        labels = torch.tensor([0, 1])
        sequence_loss = F.cross_entropy(
            training.gpc_seq_logits(frames.mean(dim=1), prototypes, t=0.07), labels,
        )
        skeleton_loss = training.gpc_ske_loss(
            frames, prototypes, labels, T=4, H=16, t=14.0,
            gpc_f1=model.gpc_f1, gpc_f2=model.gpc_f2,
        )
        loss = structure_loss + trajectory_loss + sequence_loss + skeleton_loss
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
        self.assertGreater(float(model.in1.weight.grad.abs().sum()), 0.0)

    def test_temporal_encoding_rejects_single_frame(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            laplacian_position_encoding_np(1, 4)

    def test_known_identity_prototypes_use_compressed_labels_and_require_samples(self):
        raw_ids = np.array([1, 1, 4, 4, 8, 8])
        one_hot = label_binarize(raw_ids, classes=[1, 4, 8])
        labels = one_hot.argmax(axis=1)
        features = np.array([[-1, 0], [1, 0], [0, 1], [0, 3],
                             [-2, 0], [0, 0]], dtype=np.float32)
        prototypes = training.make_class_prototypes(
            features, labels, one_hot.shape[1], torch.device('cpu'),
        )
        self.assertEqual(tuple(prototypes.shape), (3, 2))
        # A genuine zero centroid is valid; a zero-sample identity is not.
        torch.testing.assert_close(prototypes, torch.tensor([[0., 0.], [0., 2.], [-1., 0.]]))
        with self.assertRaisesRegex(ValueError, 'No complete-batch training features'):
            training.make_class_prototypes(
                features[:4], labels[:4], one_hot.shape[1], torch.device('cpu'),
            )

    def test_constructor_defaults_match_final_training_preset(self):
        model = SSGT()
        args = training.get_args([])
        self.assertEqual(model.use_time_attn, bool(args.use_time_attn))
        self.assertEqual(model.spa_pos_type, args.spa_pos_type)
        self.assertEqual(model.frame_attn_pool, bool(args.frame_attn_pool))
        self.assertEqual(model.learnable_temp, bool(args.learnable_temp))
        self.assertEqual(model.blocks_spa[0].attn_do.p, args.attn_dropout)
        self.assertEqual(model.blocks_spa[0].use_clamp, bool(args.attn_clamp))
        self.assertFalse(model.logit_scale_seq.requires_grad)
        self.assertFalse(model.logit_scale_ske.requires_grad)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='ssgt_checkpoint_test_')
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.saved_args = training.get_args([
            '--mode', 'Eval',
            '--dataset', 'WUT', '--H', '16', '--n_heads', '4',
            '--L_transformer', '1', '--length', '4', '--enc_k_spa', '3',
            '--enc_k_time', '2', '--use_time_attn', '0', '--qkv_merged', '0',
            '--spa_pos_type', 'learned', '--data_subdir', '4_synthetic',
        ])
        self.model = SSGT(
            H=16, n_heads=4, L=1, J=19, T=4, enc_k_spa=3, enc_k_time=2,
            use_time_attn=False, qkv_merged=False, spa_pos_type='learned',
            pre_norm=True, attn_drop=0.4, attn_clamp=True,
            frame_attn_pool=False, learnable_temp=False,
        )
        self.path = Path(self.temp.name) / 'best.ckpt'
        torch.save({'args': vars(self.saved_args), 'model_state': self.model.state_dict()}, self.path)

    def fake_data(self, **kwargs):
        self.assertEqual(kwargs['time_step'], 4)
        self.assertEqual(kwargs['data_subdir'], '4_synthetic')
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(6, 4, 19, 3)).astype(np.float32)
        labels = np.eye(3, dtype=np.float32)[np.arange(6) % 3]
        output = [None] * 26
        output[0], output[5] = inputs, labels
        output[6], output[11] = inputs + 0.2, labels
        output[12] = np.eye(19, dtype=np.float32)
        output[14] = rng.normal(size=(19, 3)).astype(np.float32)
        output[25] = 3
        return tuple(output)

    def test_evaluation_loads_weights_and_metadata_before_inference(self):
        original_extract = training.extract_seq_feats

        def inspect_model(model, *args, **kwargs):
            self.assertEqual(model.spa_pos_type, 'learned')
            self.assertFalse(model.use_time_attn)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            return original_extract(model, *args, **kwargs)

        with mock.patch.object(training.process, 'gen_train_data', side_effect=self.fake_data) as loader:
            with mock.patch.object(training, 'extract_seq_feats', side_effect=inspect_model) as extractor:
                with contextlib.redirect_stdout(io.StringIO()):
                    training.main(['--mode', 'Eval', '--ckpt', str(self.path),
                                   '--device', 'cpu', '--batch_size', '3'])
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(extractor.call_count, 2)

    def test_runtime_settings_survive_checkpoint_metadata(self):
        args = training.get_args(['--mode', 'Eval', '--ckpt', str(self.path),
                                  '--device', 'cpu', '--gpu', '7', '--batch_size', '3'])
        training._restore_evaluation_args(args)
        self.assertEqual(args.mode, 'Eval')
        self.assertEqual(args.device, 'cpu')
        self.assertEqual(args.gpu, '7')
        self.assertEqual(args.batch_size, 3)
        self.assertEqual(args.length, 4)
        self.assertEqual(args.spa_pos_type, 'learned')
        self.assertEqual(args.data_subdir, '4_synthetic')

    def test_missing_checkpoint_fails_before_loading_data(self):
        with mock.patch.object(training.process, 'gen_train_data') as loader:
            with self.assertRaisesRegex(FileNotFoundError, 'Checkpoint not found'):
                training.main(['--mode', 'Eval', '--ckpt', str(self.path.with_name('missing.ckpt'))])
        loader.assert_not_called()

    def test_incompatible_weights_are_rejected(self):
        state = dict(self.model.state_dict())
        state.pop('in1.weight')
        torch.save({'args': vars(self.saved_args), 'model_state': state}, self.path)
        with mock.patch.object(training.process, 'gen_train_data', side_effect=self.fake_data):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    training.main(['--mode', 'Eval', '--ckpt', str(self.path), '--device', 'cpu'])

    def test_metrics_json_preserves_historical_training_provenance(self):
        saved = dict(vars(self.saved_args))
        saved['gpc_iota'] = 1.2
        saved['conv_check'] = 1
        saved.pop('seed_val')
        saved.pop('fix_seed')
        torch.save({'args': saved, 'model_state': self.model.state_dict()}, self.path)
        output = Path(self.temp.name) / 'metrics.json'
        with mock.patch.object(training.process, 'gen_train_data', side_effect=self.fake_data):
            with contextlib.redirect_stdout(io.StringIO()):
                training.main(['--mode', 'Eval', '--ckpt', str(self.path),
                               '--device', 'cpu', '--metrics_output', str(output)])
        report = json.loads(output.read_text())
        self.assertEqual(len(report['checkpoint']['sha256']), 64)
        self.assertEqual(report['checkpoint']['saved_training_args']['gpc_iota'], 1.2)
        self.assertEqual(report['data']['known_identity_count'], 3)
        self.assertEqual(set(report['metrics']), {'R1', 'R5', 'R10', 'mAP'})
        protocol = report['paper_protocol']
        self.assertFalse(protocol['checkpoint_declares_current_schedule'])
        self.assertFalse(protocol['training_seed_metadata_present'])
        self.assertFalse(protocol['evaluation_performs_training'])
        self.assertEqual(protocol['recorded_differences']['gpc_iota']['checkpoint'], 1.2)

    def test_one_automatic_run_seed_is_recorded_and_shared_by_loader_calls(self):
        output = Path(self.temp.name) / 'seed_metrics.json'
        with mock.patch('randomness.secrets.randbits', return_value=61077) as entropy:
            with mock.patch.object(training.process, 'gen_train_data', side_effect=self.fake_data) as loader:
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    training.main(['--mode', 'Eval', '--ckpt', str(self.path),
                                   '--device', 'cpu', '--metrics_output', str(output)])
        entropy.assert_called_once_with(32)
        self.assertIn('Using seed 61077', stdout.getvalue())
        self.assertTrue(all(call[1]['sampling_seed'] == 61077 for call in loader.call_args_list))
        args = json.loads(output.read_text())['evaluation_config']
        self.assertEqual(args['requested_seed_val'], -1)
        self.assertEqual(args['seed_val'], 61077)
        self.assertEqual(args['seed_selection'], 'automatic')

    def test_explicit_zero_seed_is_honored_without_deterministic_cudnn(self):
        with mock.patch('training.seed_everything', return_value=0) as seed:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(training.set_seed(0, fix=False), 0)
        seed.assert_called_once_with(0, deterministic=False)

    def test_captured_spatial_basis_takes_precedence_over_recomputed_basis(self):
        basis = torch.ones(19, 3)
        torch.save({'args': vars(self.saved_args), 'model_state': self.model.state_dict(),
                    'spatial_pos_encoding': basis}, self.path)
        original_extract = training.extract_seq_feats

        def inspect_encoding(model, inputs, encoding, *args, **kwargs):
            np.testing.assert_array_equal(encoding, basis.numpy())
            return original_extract(model, inputs, encoding, *args, **kwargs)

        with mock.patch.object(training.process, 'gen_train_data', side_effect=self.fake_data):
            with mock.patch.object(training, 'extract_seq_feats', side_effect=inspect_encoding) as extractor:
                with contextlib.redirect_stdout(io.StringIO()):
                    training.main(['--mode', 'Eval', '--ckpt', str(self.path), '--device', 'cpu'])
        self.assertEqual(extractor.call_count, 2)

    def test_malformed_captured_spatial_basis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Spatial position encoding'):
            training.resolve_spatial_pos_encoding(
                {'spatial_pos_encoding': torch.ones(20, 3)}, np.ones((19, 3)), 19, 3,
            )


class PaperLossBalancingTests(unittest.TestCase):
    def test_normalized_slope_retains_direction_and_handles_flat_losses(self):
        self.assertAlmostEqual(mean_normalized_slope([4, 2, 1]), 0.6)
        self.assertAlmostEqual(mean_normalized_slope([1, 2, 4]), -0.6)
        self.assertAlmostEqual(mean_normalized_slope([3, 1]), 0.5)
        self.assertEqual(mean_normalized_slope([0, 0, 0]), 0.0)
        self.assertEqual(mean_normalized_slope([2]), 0.0)
        with self.assertRaises(ValueError):
            mean_normalized_slope([1, float('nan')])

    def test_signed_ratio_fixed_steps_boundaries_and_zero_denominator(self):
        self.assertAlmostEqual(update_lambda(0.5, 0.01, 0.5)[0], 0.55)
        self.assertAlmostEqual(update_lambda(0.5, 0.5, 0.01)[0], 0.45)
        # An absolute-value ratio would choose the opposite update here.
        self.assertAlmostEqual(update_lambda(0.5, -0.5, 0.01)[0], 0.55)
        self.assertEqual(update_lambda(0.5, 0.1, 1.0)[0], 0.5)
        self.assertEqual(update_lambda(0.5, 1.0, 0.1)[0], 0.5)
        self.assertEqual(update_lambda(0.5, 0.0, 0.0), (0.5, 1.0))
        self.assertAlmostEqual(update_lambda(0.5, 1.0, 0.0)[0], 0.45)
        self.assertAlmostEqual(update_lambda(0.5, 0.0, 1.0)[0], 0.55)
        self.assertEqual(update_lambda(0.98, 0.0, 1.0)[0], 1.0)
        self.assertEqual(update_lambda(0.01, 1.0, 0.0)[0], 0.0)

    def test_update_epochs_double_and_step_size_stays_fixed(self):
        balancer = DynamicLossBalancer()
        events = []
        for epoch in range(1, 81):
            event = balancer.observe(1.0, 0.98 ** epoch)
            if event is not None:
                events.append(event)
        self.assertEqual([event['completed_epochs'] for event in events], [10, 20, 40, 80])
        self.assertEqual([event['next_update_epoch'] for event in events], [20, 40, 80, 160])
        for event in events:
            self.assertAlmostEqual(event['lambda_after'] - event['lambda_before'], 0.05)
        self.assertAlmostEqual(balancer.value, 0.7)
        self.assertEqual(balancer.metadata()['initial_lambda'], 0.5)

    def test_training_defaults_follow_final_paper_model(self):
        args = training.get_args([])
        self.assertEqual(args.seed_val, -1)
        self.assertEqual(args.fix_seed, 0)
        self.assertEqual(args.gpc_iota, 10.0)
        self.assertEqual(args.gpc_up_period, 10)
        self.assertEqual(args.gpc_tau, 0.05)
        self.assertEqual(args.GPC_lambda, 0.5)
        for flags in (['--use_time_attn', '0'], ['--spa_pos_type', 'learned'],
                      ['--pos_enc', '0'], ['--frame_attn_pool', '1'],
                      ['--enable_dynamic_gpc', '0'], ['--S_prompt', '0'],
                      ['--T_prompt', '0'], ['--t_1', '0'], ['--t_2', '-1'],
                      ['--t_1', 'nan'], ['--t_2', 'inf'],
                      ['--Tr_mask_num', '0'], ['--Tr_mask_num', '6'],
                      ['--St_mask_num', '0'], ['--St_mask_num', '19']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    training.get_args(flags)
            # Historical checkpoints remain evaluable using their saved architecture.
            training.get_args(['--mode', 'Eval'] + flags)
        for prompt in ('l1', 'l2', 'MSE'):
            training.get_args(['--S_prompt', prompt, '--T_prompt', prompt])
        training.get_args(['--length', '2', '--Tr_mask_num', '1'])
        training.get_args(['--St_prompt_type', '1', '--St_mask_num', '0'])


if __name__ == '__main__':
    unittest.main()
