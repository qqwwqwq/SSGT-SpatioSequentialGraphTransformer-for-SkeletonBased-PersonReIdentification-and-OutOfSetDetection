"""Reproduce SSGT out-of-set experiments from verified trial data and weights.

Each trial uses identities excluded from training and gallery construction.
The model is encoded once, and all requested neighbor ranks reuse the same
embeddings. Pairwise distances are evaluated in bounded-memory blocks using
NumPy's original float32 formula. Aggregation reports the sample standard
deviation across five independent trials, matching the paper's Figure 5.

Usage:
    python reproduce.py evaluate --manifest trial.json --data-root trial_data \
        --checkpoint best.ckpt --output result.json --gpu 0
    python reproduce.py aggregate trial_*.json --output summary.json
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from randomness import resolve_seed


# =====================================================================
# Source and Prepared-Data Verification
# =====================================================================

def sha256_file(path):
    """Hash a file without loading the entire artifact into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def verify_trial_inputs(manifest, data_root, checkpoint, provenance_path=None):
    """Verify disjoint identities and bind prepared files to a trial manifest.

    Args:
        manifest (dict): Explicit per-identity, per-sequence trial manifest.
        data_root (Path): Root returned by protocols.prepare_trial.
        checkpoint (Path): Model trained for this exact identity split.
        provenance_path (Path): Optional location of preparation provenance.

    Returns:
        dict: Verified preparation provenance and the checkpoint SHA256.

    Raises:
        ValueError: If identities overlap, files changed, or provenance does not
        match the manifest or checkpoint.
    """
    import numpy as np
    import protocols

    protocols.validate_manifest(manifest)
    data_root = Path(data_root).resolve()
    provenance_path = Path(provenance_path) if provenance_path else data_root / "provenance.json"
    provenance = _read_json(provenance_path)
    provenance = protocols.verify_provenance(manifest, data_root, provenance)

    identities = manifest["identities"]
    known = {int(person["original_label"]) for person in identities if person["role"] == "known"}
    unknown = {int(person["original_label"]) for person in identities if person["role"] == "unknown"}
    if not known or not unknown or known & unknown:
        raise ValueError("Known and unknown identity sets must be nonempty and disjoint.")
    dataset = manifest["dataset"]
    length = int(manifest["length"])
    folder = data_root / dataset / str(manifest.get("data_subdir", length))
    for split in ("train_npy_data", "test_npy_data/gallery", "test_npy_data/probe"):
        file = folder / split / ("ids_%s_%d.npy" % (dataset, length))
        labels = {int(label) for label in np.load(file, allow_pickle=True).item()}
        if not labels or labels - known or labels & unknown:
            raise ValueError("Prepared %s contains unexpected identities." % split)
        if split == "train_npy_data" and labels != known:
            raise ValueError("Prepared training identities do not equal the manifest's known set.")
    unknown_file = folder / "unknown_npy_data" / ("ids_%s_%d.npy" % (dataset, length))
    unknown_labels = {int(label) for label in np.load(unknown_file, allow_pickle=True).item()}
    expected_sentinel = 26 if dataset == "WUT" else 165
    if unknown_labels != {expected_sentinel}:
        raise ValueError("Prepared unknown data must use the dataset's unknown sentinel only.")

    checkpoint_sha256 = sha256_file(checkpoint)
    expected_checkpoint = manifest.get("checkpoint_sha256")
    if expected_checkpoint and checkpoint_sha256 != expected_checkpoint:
        raise ValueError("Checkpoint SHA256 does not match the trial's recorded weights.")
    if not expected_checkpoint:
        raise ValueError("A checkpoint_sha256 record is required to bind weights to this identity split.")
    return {"provenance": provenance, "checkpoint_sha256": checkpoint_sha256,
            "provenance_sha256": sha256_file(provenance_path)}


# =====================================================================
# Bounded-Memory OOS Scores
# =====================================================================

def neighbor_distances(queries, library, k_values, calibration=False, block_size=512):
    """Compute the original normalized Euclidean neighbor scores in blocks.

    Args:
        queries (np.ndarray): L2-normalized features with shape (M, H).
        library (np.ndarray): L2-normalized training features with shape (N, H).
        k_values (list[int]): Neighbor ranks to return.
        calibration (bool): Queries are the library itself; skip the self match.
        block_size (int): Maximum query rows held in each distance matrix.

    Returns:
        dict: Requested k mapped to its float32 score vector.

    Notes:
        Calibration retains the inherited sorted-index convention: rank k
        excludes the first self-distance. Oversampled duplicate training
        windows remain in the library, as in the original evaluation.
    """
    import numpy as np

    queries = np.asarray(queries, dtype=np.float32)
    library = np.asarray(library, dtype=np.float32)
    if queries.ndim != 2 or library.ndim != 2 or queries.shape[1] != library.shape[1]:
        raise ValueError("Queries and library must be two feature matrices with equal widths.")
    if not np.isfinite(queries).all() or not np.isfinite(library).all():
        raise ValueError("Feature matrices must contain finite values.")
    if len(queries) == 0 or len(library) < (2 if calibration else 1):
        raise ValueError("Insufficient samples for neighbor scoring.")
    if block_size < 1 or not k_values or any(int(k) < 1 for k in k_values):
        raise ValueError("block_size and all neighbor ranks must be positive.")
    if calibration and (queries.shape != library.shape or not np.array_equal(queries, library)):
        raise ValueError("Calibration queries must equal the training library.")
    ranks = {int(k): min(int(k), len(library) - 1) if calibration
             else min(int(k), len(library)) - 1 for k in k_values}
    positions = sorted(set(ranks.values()))
    lookup = {position: index for index, position in enumerate(positions)}
    scores = {int(k): np.empty(len(queries), dtype=np.float32) for k in k_values}
    for start in range(0, len(queries), block_size):
        stop = min(start + block_size, len(queries))
        similarities = queries[start:stop] @ library.T
        distances = np.sqrt(np.clip(2.0 - 2.0 * similarities, 0.0, None))
        partition = np.partition(distances, kth=positions, axis=1)[:, positions]
        for k, position in ranks.items():
            scores[k][start:stop] = partition[:, lookup[position]]
    return scores


def score_features(known_features, unknown_features, k_values, sigma=0.95,
                   block_size=512, threads=4):
    """Evaluate all neighbor ranks against one fixed feature library."""
    import numpy as np
    from evaluation import _l2n, eval_oos_metrics
    from threadpoolctl import threadpool_limits

    if not 0 < sigma < 1:
        raise ValueError("The acceptance quantile must lie strictly between zero and one.")
    known = _l2n(np.asarray(known_features, dtype=np.float32))
    unknown = _l2n(np.asarray(unknown_features, dtype=np.float32))
    with threadpool_limits(limits=threads):
        calibration = neighbor_distances(known, known, k_values, True, block_size)
        query = neighbor_distances(unknown, known, k_values, False, block_size)
    results = {}
    for k in sorted(set(k_values)):
        metric, _ = eval_oos_metrics(calibration[k], query[k], sigma_eval=sigma)
        results[str(k)] = dict(metric, k_requested=int(k),
                               k_calibration=min(k, len(known) - 1),
                               k_query=min(k, len(known)),
                               AUROC_percent=100.0 * metric["AUROC"],
                               FPR95_percent=100.0 * metric["FPR95"],
                               num_oos_windows=int((query[k] > metric["tau_eval"]).sum()))
    return results


# =====================================================================
# Trial Evaluation and Five-Trial Aggregation
# =====================================================================

def evaluate_trial(args):
    """Run one verified trial and write measured metrics with provenance."""
    requested_seed = getattr(args, "seed", -1)
    run_seed = resolve_seed(requested_seed)
    seed_selection = "automatic" if requested_seed in (None, -1) else "explicit"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu in ("cpu", "-1") else args.gpu
    import numpy as np
    import scipy
    import sklearn
    import torch
    import data
    from evaluation import load_checkpoint_model, _extract_seq_feats, set_seed, resolve_spatial_pos_encoding
    import protocols

    manifest = _read_json(args.manifest)
    verified = verify_trial_inputs(manifest, args.data_root, args.checkpoint, args.provenance)
    dataset = manifest["dataset"]
    trial_id = manifest["trial_id"]
    print("[%s/%s] Verified manifest, prepared files, and checkpoint." % (dataset, trial_id), flush=True)
    if args.gpu not in ("cpu", "-1") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; use --gpu cpu for a CPU run.")
    device = torch.device("cpu" if args.gpu in ("cpu", "-1") else "cuda")
    set_seed(run_seed)
    print("[Seed] Using %d (%s)." % (run_seed, seed_selection), flush=True)
    start = time.monotonic()
    model, saved = load_checkpoint_model(args.checkpoint, device, dataset)
    if model.T != int(manifest["length"]):
        raise ValueError("Checkpoint sequence length differs from the trial manifest.")
    old_root = os.environ.get("SSGT_DATA_DIR")
    os.environ["SSGT_DATA_DIR"] = str(Path(args.data_root).resolve())
    try:
        loaded = data.gen_train_data(
            dataset=dataset, split="unknown", time_step=model.T, nb_nodes=model.J,
            nhood=1, global_att=False, batch_size=args.batch_size,
            enc_k=int(saved.get("enc_k_spa", 10)), data_subdir=manifest.get("data_subdir", ""),
            sampling_seed=run_seed,
        )
    finally:
        if old_root is None:
            os.environ.pop("SSGT_DATA_DIR", None)
        else:
            os.environ["SSGT_DATA_DIR"] = old_root
    known_windows, unknown_windows = loaded[0], loaded[6]
    encoding = resolve_spatial_pos_encoding(model, loaded[14])
    spatial_encoding_metadata = dict(model.spatial_pos_encoding_metadata)
    print("[%s/%s] Encoding %d calibration and %d unknown windows on %s." %
          (dataset, trial_id, len(known_windows), len(unknown_windows), device), flush=True)
    feature_start = time.monotonic()
    known_features = _extract_seq_feats(
        model, known_windows, encoding, args.batch_size, device, model.T, model.J, model.H,
        use_pos=bool(saved.get("pos_enc", 1)), bn_freeze=True,
    )
    unknown_features = _extract_seq_feats(
        model, unknown_windows, encoding, args.batch_size, device, model.T, model.J, model.H,
        use_pos=bool(saved.get("pos_enc", 1)), bn_freeze=True,
    )
    feature_seconds = time.monotonic() - feature_start
    del loaded, known_windows, unknown_windows, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[%s/%s] Features ready in %.1fs; evaluating K=%s." %
          (dataset, trial_id, feature_seconds, args.k_values), flush=True)
    score_start = time.monotonic()
    metrics = score_features(known_features, unknown_features, args.k_values, args.sigma,
                             args.distance_batch_size, args.distance_threads)
    result = {
        "schema_version": 1, "status": "evaluated", "dataset": dataset, "trial_id": trial_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "randomness": {"seed": run_seed, "requested_seed": requested_seed,
                       "selection": seed_selection},
        "protocol": {
            "length": int(manifest["length"]), "sigma": args.sigma,
            "known_score_source": "leave-one-out training calibration library",
            "unknown_score_source": "held-out identities, never enrolled in training or gallery",
            "known_ids": verified["provenance"]["known_ids"],
            "unknown_original_ids": verified["provenance"]["unknown_original_ids"],
            "batch_size": args.batch_size, "sampling_seed": run_seed,
            "missing_known_ids": verified["provenance"].get("missing_known_ids", {}),
            "num_calibration_windows": len(known_features),
            "num_unknown_windows": len(unknown_features),
        },
        "provenance": {
            "spatial_position_encoding": spatial_encoding_metadata,
            "manifest_sha256": protocols.manifest_digest(manifest),
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "preparation_provenance_sha256": verified["provenance_sha256"],
            "source_sha256": verified["provenance"].get("source_sha256", {}),
            "prepared_sha256": verified["provenance"]["output_sha256"],
            "historical_counts_match": verified["provenance"].get("historical_counts_match"),
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__, "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "feature_seconds": feature_seconds,
                    "distance_and_metric_seconds": time.monotonic() - score_start,
                    "total_seconds": time.monotonic() - start},
        "metrics": metrics,
    }
    reference = _read_json(args.reference_result) if args.reference_result else manifest.get("historical_reference")
    if reference and "1" in metrics and "AUROC" in reference and "FPR95" in reference:
        result["historical_k1_comparison"] = {
            key: {"reference": float(reference[key]), "measured": metrics["1"][key],
                  "difference_percentage_points": 100.0 * (metrics["1"][key] - float(reference[key]))}
            for key in ("AUROC", "FPR95")
        }
    if args.save_features:
        folder = Path(args.output).parent
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "calibration_features.npy", known_features)
        np.save(folder / "unknown_features.npy", unknown_features)
    _write_json(args.output, result)
    for k, metric in metrics.items():
        print("[%s/%s] K=%s AUROC=%.6f%% FPR95=%.6f%%" %
              (dataset, trial_id, k, metric["AUROC_percent"], metric["FPR95_percent"]), flush=True)
    print("Saved " + str(args.output), flush=True)
    return result


def aggregate_trials(paths, paper_targets=None):
    """Aggregate exactly five independently verified trials using ddof=1."""
    import numpy as np

    records = [_read_json(path) for path in paths]
    if len(records) != 5 or any(record.get("status") != "evaluated" for record in records):
        raise ValueError("Paper aggregation requires exactly five evaluated trial records.")
    if len({record["trial_id"] for record in records}) != 5:
        raise ValueError("Trial IDs must be distinct.")
    if len({record["provenance"]["checkpoint_sha256"] for record in records}) != 5:
        raise ValueError("Each paper trial must use its own checkpoint.")
    datasets = {record["dataset"] for record in records}
    if len(datasets) != 1:
        raise ValueError("Aggregate one dataset at a time.")
    dataset = records[0]["dataset"]
    k_values = sorted(records[0]["metrics"], key=int)
    signature = (records[0]["protocol"]["length"], records[0]["protocol"]["sigma"],
                 records[0]["protocol"]["known_score_source"], records[0]["protocol"]["batch_size"])
    for record in records:
        current = (record["protocol"]["length"], record["protocol"]["sigma"],
                   record["protocol"]["known_score_source"], record["protocol"]["batch_size"])
        if sorted(record["metrics"], key=int) != k_values or current != signature:
            raise ValueError("All five trials must use the same lengths, quantile, sampling, and K values.")
    summary = {"dataset": dataset, "trials": 5, "units": "percent", "std_ddof": 1,
               "trial_files": [{"name": Path(path).name, "sha256": sha256_file(path)} for path in paths],
               "metrics": {}}
    for k in k_values:
        summary["metrics"][k] = {}
        for metric in ("AUROC", "FPR95"):
            values = np.asarray([record["metrics"][k][metric] * 100 for record in records])
            summary["metrics"][k][metric + "_mean"] = float(values.mean())
            summary["metrics"][k][metric + "_sample_std"] = float(values.std(ddof=1))
            summary["metrics"][k][metric + "_trials"] = values.tolist()
    if paper_targets:
        targets = _read_json(paper_targets)["oos"]
        comparisons = {}
        for k in k_values:
            if int(k) not in targets["k_values"]:
                continue
            index = targets["k_values"].index(int(k))
            comparisons[k] = {}
            for metric in ("AUROC_mean", "AUROC_sample_std", "FPR95_mean", "FPR95_sample_std"):
                target = float(targets[dataset][metric][index])
                measured = summary["metrics"][k][metric]
                comparisons[k][metric] = {"paper": target, "measured": measured,
                                          "difference_percentage_points": measured - target}
        summary["paper_comparison"] = comparisons
        summary["paper_targets_sha256"] = sha256_file(paper_targets)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate one verified trial")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--provenance")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--gpu", default="0", help="CUDA index or cpu")
    evaluate.add_argument("--seed", type=int, default=-1,
                          help="Run seed; -1 generates and records a fresh seed for optional replay")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--distance-batch-size", type=int, default=512)
    evaluate.add_argument("--distance-threads", type=int, default=4)
    evaluate.add_argument("--k-values", default="1,5,20,50,200")
    evaluate.add_argument("--sigma", type=float, default=0.95)
    evaluate.add_argument("--reference-result")
    evaluate.add_argument("--save-features", action="store_true")
    aggregate = commands.add_parser("aggregate", help="Aggregate five trial result files")
    aggregate.add_argument("trials", nargs="+")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--paper-targets", default=str(Path(__file__).parent / "config" / "paper_targets.json"))
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if args.seed < -1 or args.seed >= 2 ** 32:
            parser.error("--seed must be -1 or an integer in [0, 2**32 - 1].")
        try:
            args.k_values = sorted({int(value) for value in args.k_values.split(",")})
        except ValueError:
            parser.error("--k-values must contain comma-separated positive integers.")
        if not args.k_values or min(args.k_values) < 1 or min(args.batch_size, args.distance_batch_size, args.distance_threads) < 1:
            parser.error("Neighbor ranks, batch sizes, and thread counts must be positive.")
        evaluate_trial(args)
    else:
        result = aggregate_trials(args.trials, args.paper_targets)
        _write_json(args.output, result)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
