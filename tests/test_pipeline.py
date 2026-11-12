"""Exercise prepared data, training, checkpoints, and evaluation on CPU.

Synthetic coordinate files follow the published WUT and KGBD layouts. Every
subprocess uses temporary data and output directories; no real data or weights
are read, and no training results are left in the source tree.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class TrainingPipelineTests(unittest.TestCase):
    def write_prepared_data(self, data_root, dataset, joint_count):
        """Create six windows and three identities in each required split."""
        rng = np.random.default_rng(101)
        labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        identities = {
            int(label): np.flatnonzero(labels == label).tolist()
            for label in np.unique(labels)
        }
        dataset_dir = data_root / dataset / '4'
        for split in ('train', 'probe', 'gallery'):
            folder = (dataset_dir / 'train_npy_data' if split == 'train'
                      else dataset_dir / 'test_npy_data' / split)
            folder.mkdir(parents=True)
            coordinates = rng.normal(size=(6, 4, joint_count, 3)).astype(np.float32)
            prefix = 'source' if split == 'train' else 't_source'
            for index, axis in enumerate(('x', 'y', 'z')):
                filename = folder / f'{prefix}_{axis}_{dataset}_4.npy'
                np.save(filename, coordinates[:, :, :, index].reshape(-1, joint_count))
            np.save(folder / f'frame_id_{dataset}_4.npy', labels)
            np.save(folder / f'ids_{dataset}_4.npy', identities)

    def run_entrypoint(self, arguments, working_dir):
        """Run the published command from a directory outside the source tree."""
        environment = dict(os.environ)
        environment.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
        })
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'main.py')] + arguments,
            cwd=working_dir, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        return completed.stdout

    def test_wut_and_kgbd_training_then_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory(prefix='ssgt_pipeline_test_') as temporary:
            root = Path(temporary)
            data_root, model_root = root / 'Datas', root / 'weight'
            paths = ['--data-dir', str(data_root), '--model-dir', str(model_root),
                     '--output-dir', str(root / 'results')]
            for dataset, joints in (('KGBD', 20), ('WUT', 19)):
                with self.subTest(dataset=dataset):
                    self.write_prepared_data(data_root, dataset, joints)
                    training_options = [
                        '--dataset', dataset, '--device', 'cpu', '--length', '4',
                        '--batch_size', '3', '--H', '8', '--n_heads', '2',
                        '--L_transformer', '1', '--enc_k_spa', '3',
                        '--enc_k_time', '2', '--max_epochs', '1', '--patience', '5',
                    ]
                    output = self.run_entrypoint(
                        ['--task', 'train'] + paths + training_options, root,
                    )
                    # The research schedule uses inclusive, zero-based epochs.
                    self.assertIn('[Epoch 0 Summary]', output)
                    self.assertIn('[Epoch 1 Summary]', output)
                    checkpoints = list((model_root / dataset).rglob('best.ckpt'))
                    self.assertEqual(len(checkpoints), 1)
                    checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
                    self.assertTrue(all(
                        torch.isfinite(value).all().item()
                        for value in checkpoint['model_state'].values()
                    ))
                    self.assertEqual(checkpoint['args']['dataset'], dataset)
                    self.assertEqual(checkpoint['args']['length'], 4)
                    chosen_seed = checkpoint['args']['seed_val']
                    self.assertGreaterEqual(chosen_seed, 0)
                    self.assertLess(chosen_seed, 2**32)
                    self.assertEqual(checkpoint['args']['requested_seed_val'], -1)
                    self.assertEqual(checkpoint['args']['seed_selection'], 'automatic')
                    self.assertEqual(checkpoint['paper_protocol']['seed_control']['seed'], chosen_seed)
                    self.assertIn(f'Using seed {chosen_seed}', output)
                    self.assertEqual(checkpoint['known_identity_count'], 3)
                    self.assertEqual(checkpoint['paper_protocol']['prototype_protocol']['identity_count'], 3)
                    self.assertEqual(tuple(checkpoint['spatial_pos_encoding'].shape), (joints, 3))
                    self.assertTrue(torch.isfinite(checkpoint['spatial_pos_encoding']).all().item())
                    self.assertEqual(
                        checkpoint['spatial_pos_encoding_provenance']['source'],
                        'captured_from_training_loader',
                    )
                    # Architecture options are intentionally omitted: Eval must
                    # recover the small model and input layout from metadata.
                    metrics_path = root / (dataset + '_evaluation.json')
                    evaluation = self.run_entrypoint(
                        ['--task', 'evaluate'] + paths + [
                            '--ckpt', str(checkpoints[0]), '--device', 'cpu',
                            '--batch_size', '3', '--metrics_output', str(metrics_path),
                        ], root,
                    )
                    self.assertIn('[Checkpoint] Restored', evaluation)
                    self.assertIn('[Evaluation]', evaluation)
                    report = json.loads(metrics_path.read_text())
                    self.assertEqual(
                        report['data']['spatial_position_encoding']['used_source'], 'checkpoint',
                    )
                    for reported, saved in (('R1', 'r1'), ('R5', 'r5'),
                                            ('R10', 'r10'), ('mAP', 'mAP')):
                        self.assertEqual(report['metrics'][reported], checkpoint['metrics'][saved])
                    if dataset == 'WUT':
                        replay_root = root / 'replay_weight'
                        replay_paths = list(paths)
                        replay_paths[replay_paths.index('--model-dir') + 1] = str(replay_root)
                        self.run_entrypoint(
                            ['--task', 'train'] + replay_paths + training_options +
                            ['--seed_val', str(chosen_seed)], root,
                        )
                        replay_path = next((replay_root / dataset).rglob('best.ckpt'))
                        replay = torch.load(replay_path, map_location='cpu', weights_only=False)
                        self.assertEqual(replay['args']['requested_seed_val'], chosen_seed)
                        self.assertEqual(replay['args']['seed_selection'], 'explicit')
                        self.assertEqual(replay['metrics'], checkpoint['metrics'])
                        for name, value in checkpoint['model_state'].items():
                            torch.testing.assert_close(replay['model_state'][name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
