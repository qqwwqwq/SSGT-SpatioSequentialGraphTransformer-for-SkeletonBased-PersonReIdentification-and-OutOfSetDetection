"""Compare optimized retrieval with the original scikit-learn AP protocol."""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import retrieval_metrics


def reference_metrics(distances, query_ids=None, gallery_ids=None,
                      query_cams=None, gallery_cams=None,
                      special_skip_first=False, remove_zero_same_id=True, zero_eps=1e-6):
    """Retain the original sorting/filtering and public sklearn AP call."""
    rows, columns = distances.shape
    query_ids = np.arange(rows) if query_ids is None else np.asarray(query_ids)
    gallery_ids = np.arange(columns) if gallery_ids is None else np.asarray(gallery_ids)
    with_cameras = query_cams is not None and gallery_cams is not None
    if with_cameras:
        query_cams, gallery_cams = np.asarray(query_cams), np.asarray(gallery_cams)
    aps, hits = [], np.zeros(3, dtype=np.int64)
    for index, order in enumerate(np.argsort(distances, axis=1)):
        if special_skip_first:
            order = order[1:]
        matches = gallery_ids[order] == query_ids[index]
        scores = -distances[index][order]
        valid = np.ones_like(matches, dtype=bool)
        if with_cameras:
            valid &= ~(matches & (gallery_cams[order] == query_cams[index]))
        if remove_zero_same_id:
            valid &= ~(matches & (distances[index][order] <= float(zero_eps)))
        matches, scores = matches[valid], scores[valid]
        aps.append(average_precision_score(matches, scores) if np.any(matches) else 0.0)
        filtered = order[valid]
        for position, rank in enumerate((1, 5, 10)):
            hits[position] += query_ids[index] in gallery_ids[filtered[:rank]]
    if not aps:
        raise RuntimeError("No valid query")
    return (float(np.mean(aps)),) + tuple(float(value / rows) for value in hits)


class RetrievalTests(unittest.TestCase):
    def test_random_tied_and_threshold_distances_are_exact(self):
        rng = np.random.RandomState(123)
        query_ids, gallery_ids = rng.randint(0, 8, 29), rng.randint(0, 7, 67)
        query_cams, gallery_cams = rng.randint(0, 3, 29), rng.randint(0, 3, 67)
        for dtype in (np.float32, np.float64):
            for tied in (False, True):
                distances = rng.uniform(0, 5, (29, 67)).astype(dtype)
                if tied:
                    distances = np.round(distances)
                distances[:, :3] = np.asarray([0, 1e-6, np.nextafter(dtype(1e-6), dtype(np.inf))], dtype=dtype)
                before = distances.copy()
                for skip, remove, cameras in itertools.product((False, True), (False, True), (False, True)):
                    options = dict(special_skip_first=skip, remove_zero_same_id=remove,
                                   query_cams=query_cams if cameras else None,
                                   gallery_cams=gallery_cams if cameras else None)
                    expected = reference_metrics(distances, query_ids, gallery_ids, **options)
                    for workers in (1, 2, 8):
                        with self.subTest(dtype=dtype, tied=tied, skip=skip, remove=remove, cameras=cameras, workers=workers):
                            actual = retrieval_metrics(distances, query_ids, gallery_ids, workers=workers, batch_size=4, **options)
                            self.assertEqual(actual, expected)
                np.testing.assert_array_equal(distances, before)

    def test_equal_distances_use_grouped_ap_and_original_cmc_order(self):
        distances = np.ones((1, 40), dtype=np.float32)
        ids = np.arange(40) % 2
        expected = reference_metrics(distances, [0], ids, special_skip_first=True)
        self.assertEqual(retrieval_metrics(distances, [0], ids, special_skip_first=True), expected)
        self.assertEqual(expected[0], float(np.mean(ids[np.argsort(distances[0])[1:]] == 0)))

    def test_nonfinite_behavior_depends_on_retained_positives(self):
        cases = [
            (np.array([[np.nan, np.inf]]), [5], [0, 1], {}),
            (np.array([[np.nan, 2.0]]), [0], [0, 1], {}),
            (np.array([[-np.inf, 2.0]]), [0], [0, 1], {}),
            (np.array([[2.0, np.inf]]), [0], [0, 1], {}),
            (np.array([[np.nan, np.inf]]), [0], [0, 0], {"query_cams": [1], "gallery_cams": [1, 1]}),
            (np.array([[-np.inf, 2.0]]), [0], [0, 1], {"remove_zero_same_id": False}),
        ]
        for distances, query_ids, gallery_ids, options in cases:
            with self.subTest(distances=distances, options=options):
                try:
                    expected = reference_metrics(distances, query_ids, gallery_ids, **options)
                except ValueError:
                    with self.assertRaises(ValueError):
                        retrieval_metrics(distances, query_ids, gallery_ids, **options)
                else:
                    self.assertEqual(retrieval_metrics(distances, query_ids, gallery_ids, **options), expected)

    def test_empty_queries_empty_gallery_default_ids_and_partial_cameras(self):
        with self.assertRaisesRegex(RuntimeError, "No valid query"):
            retrieval_metrics(np.empty((0, 3)))
        self.assertEqual(retrieval_metrics(np.empty((3, 0))), (0.0, 0.0, 0.0, 0.0))
        distances = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        for cameras in ({}, {"query_cams": [0, 0]}, {"gallery_cams": [0, 0]}):
            self.assertEqual(retrieval_metrics(distances, **cameras), reference_metrics(distances, **cameras))
        # The compatibility path preserves unsigned negation in the old API.
        unsigned = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        self.assertEqual(retrieval_metrics(unsigned, [1], [1, 0, 1, 0], remove_zero_same_id=False),
                         reference_metrics(unsigned, [1], [1, 0, 1, 0], remove_zero_same_id=False))


if __name__ == "__main__":
    unittest.main()
