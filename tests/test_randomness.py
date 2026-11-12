"""Verify automatic run seeds and explicit replay across supported RNGs."""

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from randomness import resolve_seed, seed_everything


class RandomSeedTests(unittest.TestCase):
    def test_automatic_seed_uses_fresh_entropy(self):
        with patch('randomness.secrets.randbits', side_effect=[17, 23]) as entropy:
            self.assertEqual(resolve_seed(), 17)
            self.assertEqual(resolve_seed(-1), 23)
        self.assertEqual(entropy.call_count, 2)

    def test_explicit_seed_does_not_draw_entropy(self):
        with patch('randomness.secrets.randbits') as entropy:
            self.assertEqual(resolve_seed(0), 0)
            self.assertEqual(resolve_seed(np.int64(2 ** 32 - 1)), 2 ** 32 - 1)
        entropy.assert_not_called()

    def test_invalid_seed_is_rejected(self):
        for value in (-2, 2 ** 32, 1.5, '23'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_seed(value)

    def test_recorded_automatic_seed_replays_all_generators(self):
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.get_rng_state()
        deterministic = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        try:
            with patch('randomness.secrets.randbits', return_value=15973):
                selected = seed_everything()
            first = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(selected, 15973)
            self.assertFalse(torch.backends.cudnn.deterministic)
            seed_everything(selected, deterministic=True)
            second = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(first[:2], second[:2])
            self.assertTrue(torch.equal(first[2], second[2]))
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = benchmark


if __name__ == '__main__':
    unittest.main()
