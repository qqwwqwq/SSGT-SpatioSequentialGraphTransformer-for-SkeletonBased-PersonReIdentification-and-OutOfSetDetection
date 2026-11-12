"""Exact retrieval metrics with one NumPy sort per query.

Distance calculation remains with the caller. CMC retains NumPy's default
quicksort tie order, while AP groups equal scores exactly as scikit-learn does.
Small independent row batches bound sorting memory and permit CPU parallelism.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import precision_recall_curve


# Older sklearn versions omit thresholds after full recall is reached. Those
# terms contribute zero, but keeping them can change the final sum's last bit.
# Detect the installed public API's convention once, never once per query.
_TRIM_FULL_RECALL = len(precision_recall_curve([1, 0], [1.0, 0.0])[0]) == 2


# === Average Precision for Already Sorted Distances ===

def _sorted_average_precision(matches, distances):
    """Reproduce scikit-learn's unweighted binary precision-recall integral.

    Inputs have already been filtered and sorted by ascending distance.
    The caller supplies at least one retained positive. Counts and the final
    reversed recall-difference reduction use float64, matching scikit-learn.
    """
    if not np.isfinite(distances).all():
        raise ValueError("Average precision requires finite retained distances.")
    # Integer negation can overflow and change the score order in the original
    # sklearn call. Preserve that unusual API behavior through its reference
    # implementation; normal model distances take the direct floating path.
    if distances.dtype.kind != "f":
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(matches, -distances))
    last = np.r_[np.flatnonzero(np.diff(-distances)), matches.size - 1]
    true_positives = np.cumsum(matches, dtype=np.float64)[last]
    false_positives = 1 + last - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]
    last_full_recall = (true_positives.searchsorted(true_positives[-1])
                        if _TRIM_FULL_RECALL else len(true_positives) - 1)
    reverse = slice(last_full_recall, None, -1)
    return float(-np.sum(np.diff(np.r_[recall[reverse], 0.0]) * precision[reverse]))


# === Shared mAP and CMC Evaluation ===

def retrieval_metrics(
    distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None,
    special_skip_first=False, remove_zero_same_id=True, zero_eps=1e-6,
    workers=4, batch_size=32,
):
    """Compute mAP and Rank-1/5/10 from an unchanged distance matrix.

    Args:
        distmat (np.ndarray): Query-to-gallery distances, shape (Nq, Ng).
        query_ids (array-like or None): Query labels; None uses row indices.
        gallery_ids (array-like or None): Gallery labels; None uses column indices.
        query_cams (array-like or None): Optional query camera labels.
        gallery_cams (array-like or None): Optional gallery camera labels.
        special_skip_first (bool): Drop the first sorted item before filtering.
        remove_zero_same_id (bool): Exclude same-ID distances <= zero_eps.
        zero_eps (float): Duplicate-distance threshold, using the input dtype.
        workers (int): Independent CPU workers; one runs without a thread pool.
        batch_size (int): Query rows sorted together by each worker.

    Returns:
        tuple: (mAP, Rank-1, Rank-5, Rank-10), including zero contributions
        from queries with no retained positive or an empty filtered gallery.

    Raises:
        RuntimeError: If there are no queries.
        ValueError: If retained distances for a query containing a positive are
            nonfinite, matching the reference AP validation convention.

    Notes:
        Both camera arrays must be supplied to enable camera filtering. Equal
        distances form one AP threshold, regardless of their CMC tie order.
        Per-query APs are restored to their original order before np.mean, so
        worker completion order does not change the floating-point reduction.
    """
    distances = np.asarray(distmat)
    m, n = distances.shape
    if m == 0:
        raise RuntimeError("No valid query")
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    query_ids = np.arange(m) if query_ids is None else np.asarray(query_ids)
    gallery_ids = np.arange(n) if gallery_ids is None else np.asarray(gallery_ids)
    has_cam = query_cams is not None and gallery_cams is not None
    if has_cam:
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

    def evaluate_batch(start):
        stop = min(start + batch_size, m)
        # Keep the same default sorting algorithm as np.argsort(distmat, axis=1).
        # Slicing rows changes neither each row's values nor its tie ordering.
        orders = np.argsort(distances[start:stop], axis=1)
        aps = np.zeros(stop - start, dtype=np.float64)
        hits = np.zeros(3, dtype=np.int64)
        for offset, order in enumerate(orders):
            index = start + offset
            if special_skip_first:
                order = order[1:]
            row_distances = distances[index][order]
            matches = gallery_ids[order] == query_ids[index]
            valid = np.ones(matches.shape, dtype=bool)
            if has_cam:
                valid &= ~(matches & (gallery_cams[order] == query_cams[index]))
            if remove_zero_same_id:
                valid &= ~(matches & (row_distances <= float(zero_eps)))
            matches = matches[valid]
            if not np.any(matches):
                continue
            aps[offset] = _sorted_average_precision(matches, row_distances[valid])
            first_positive = int(np.argmax(matches))
            hits += (first_positive < 1, first_positive < 5, first_positive < 10)
        return start, aps, hits

    all_aps = np.empty(m, dtype=np.float64)
    all_hits = np.zeros(3, dtype=np.int64)
    starts = range(0, m, batch_size)

    def collect(results):
        for start, aps, hits in results:
            all_aps[start:start + len(aps)] = aps
            all_hits[:] += hits

    if workers == 1 or m <= batch_size:
        collect(map(evaluate_batch, starts))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(starts))) as pool:
            collect(pool.map(evaluate_batch, starts))
    return (float(np.mean(all_aps)), float(all_hits[0] / m),
            float(all_hits[1] / m), float(all_hits[2] / m))
