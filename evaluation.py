# -*- coding: utf-8 -*-
"""SSGT identity retrieval and out-of-set (OOS) evaluation.

This module loads the complete architecture recorded in a training checkpoint,
extracts skeleton sequence embeddings, and rejects unknown identities using
k-nearest-neighbor distances calibrated on the training feature library.

Input Modes:
    - unknown: Read an already prepared unknown split from Datas.
    - npy: Segment an external skeleton sequence and report window/frame IDs.

The known-data reference for OOS metrics is the training calibration library.
Unknown identities must be held out during data preparation; this evaluator
never creates or moves identity splits automatically.

Usage:
    $ python evaluation.py --dataset WUT --ckpt weight/model/best.ckpt
"""

import os
import glob
import argparse
import json
import hashlib
from typing import Tuple, Optional
from pathlib import Path
from paths import data_root, model_root, output_root
from randomness import resolve_seed, seed_everything

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

from sklearn.metrics import roc_curve, auc
try:
    from sklearn.metrics import silhouette_score
except Exception:
    silhouette_score = None
from sklearn.decomposition import PCA
from visualization import make_tsne
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


import data as process
from SSGT import SSGT


CKPT_DEFAULT = ""


def _score_histogram_path(save_plot, save_npy_base=""):
    """Derive a histogram filename distinct from the distance-scatter image."""
    if not save_plot:
        return (save_npy_base.rstrip("/") if save_npy_base else "oos") + "_oos_score_hist.png"
    scatter = Path(save_plot)
    stem = (scatter.stem.replace("_oos_scatter", "_oos_score_hist")
            if "_oos_scatter" in scatter.stem else scatter.stem + "_oos_score_hist")
    return str(scatter.with_name(stem + scatter.suffix))


DICP_WUT = {
    "Jacek": 1, "Aleks": 2, "Julia": 3, "Vibek": 4, "Mikolaj": 5, "Ania": 6, "Kasia": 7, "Martyna": 8,
    "Maria": 9, "MichalD": 10, "MichalJ": 11, "Ercem": 12, "Malgosia": 13, "Moncy": 14, "Juan": 15,
    "Erdem": 16, "Pasko": 17, "Piotr": 18, "Julianna": 19, "Showni": 20, "Weronika": 21,
    "Szykiedans": 22, "Maks": 23, "Kujawinska": 24, "Wojciech": 25, "Unknown": 26,
}
_ID0_TO_NAME = {v - 1: k for k, v in DICP_WUT.items()}
_NAME_LOWER_TO_ID0 = {k.lower(): v - 1 for k, v in DICP_WUT.items()}

# =====================================================================
# Identity Labels and Dataset Metadata
# =====================================================================

def _idx_to_name(idx: int) -> str:
    return _ID0_TO_NAME.get(int(idx), f"ID{int(idx)}")


def _load_identity_ids(dataset, length, data_subdir):
    """Map training one-hot columns to original zero-based dataset identities.

    The loader sorts the prepared training dictionary keys before one-hot
    encoding. Holding identities out leaves gaps in those original keys.
    WUT/KGBD prepared labels are one-based; KS20 prepared labels are zero-based.
    """
    filename = (data_root() / dataset / (data_subdir or str(length)) /
                'train_npy_data' / f'ids_{dataset}_{length}.npy')
    identities = np.load(filename, allow_pickle=True).item()
    raw_ids = sorted(identities)
    if any(not isinstance(value, (int, np.integer)) for value in raw_ids):
        raise ValueError('Prepared identity dictionary keys must be integers.')
    ids = np.asarray(raw_ids, dtype=np.int64) - (0 if dataset == 'KS20' else 1)
    if len(ids) < 3 or np.any(ids < 0) or np.any(ids >= _unknown_identity_id(dataset)):
        raise ValueError(f'Prepared training identities are outside the {dataset} known range.')
    return ids


def _decode_known_labels(labels, identity_ids):
    """Decode known one-hot labels without treating missing classes as ID zero."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != len(identity_ids):
        raise ValueError('Known labels must match the prepared training identity columns.')
    if not np.isfinite(labels).all() or np.any(labels.sum(axis=1) != 1):
        raise ValueError('A known split contains an identity absent from training.')
    return identity_ids[np.argmax(labels, axis=1)]


def _unknown_identity_id(dataset):
    """Return a stable zero-based rejection ID beyond the full known ID range."""
    return {'WUT': 25, 'KGBD': 164, 'KS20': 20}[dataset]


def _parse_groundtruth(value, dataset, one_based):
    """Parse a dataset identity using the same indexing as exported predictions."""
    if value is None or not str(value).strip():
        return None, ''
    text = str(value).strip()
    if dataset == 'WUT' and text.lower() in _NAME_LOWER_TO_ID0:
        identity = _NAME_LOWER_TO_ID0[text.lower()]
    elif text.isdigit():
        identity = int(text) - int(one_based)
        if not 0 <= identity <= _unknown_identity_id(dataset):
            return None, text
    else:
        return None, text
    name = _idx_to_name(identity) if dataset == 'WUT' else str(identity + int(one_based))
    return identity, name


def set_seed(seed=None, deterministic=False):
    """Seed the run and return its explicit or automatically generated seed."""
    return seed_everything(seed, deterministic=deterministic)

def _dataset_joint_num(name: str) -> int:
    name = name.upper()
    if name == 'KS20': return 25
    if name == 'WUT' : return 19
    return 20

def _auto_find_best_ckpt(base_dir: str) -> str:
    pats = ["**/best.ckpt", "**/*_best.ckpt"]
    cands = []
    for p in pats:
        cands += glob.glob(os.path.join(base_dir, p), recursive=True)
    if not cands:
        raise FileNotFoundError(f"No best.ckpt or *_best.ckpt found under {base_dir}")
    cands = sorted(cands, key=lambda f: os.path.getmtime(f), reverse=True)
    return cands[0]


# =====================================================================
# External Skeleton Sequence Preparation
# =====================================================================

def _prepare_npy_as_segments(npy_path: str, J: int, T: int, stride: int, sliding: bool,
                             center_root: bool = True, root_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load a skeleton sequence, remove NaN frames, and build temporal windows.

    Args:
        npy_path (str): Skeleton array with 3D joints, optionally confidence channels.
        J (int): Number of joints in the dataset.
        T (int): Number of frames per model input.
        stride (int): Step between sliding windows.
        sliding (bool): Use sliding rather than non-overlapping windows.
        center_root (bool): Subtract the root joint from every frame.
        root_idx (int): Root joint index.

    Returns:
        tuple: Windows (N, T, J, 3), inclusive spans in the cleaned sequence
        (N, 2), and indices of original windows containing NaN frames.

    Raises:
        ValueError: If the array shape or number of valid frames is invalid."""
    arr = np.load(npy_path, allow_pickle=True)
    arr = np.asarray(arr)

    def to_N_J_3(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 3:
            if a.shape[1] == J and a.shape[2] >= 3: return a[:, :, :3]
            if a.shape[2] == J and a.shape[1] >= 3: return np.transpose(a, (0, 2, 1))[:, :, :3]
        if a.ndim == 2:
            for C in (4, 3):
                if a.shape[1] == J * C: return a.reshape(-1, J, C)[:, :, :3]
        if a.ndim == 1:
            for C in (4, 3):
                if a.size % (J * C) == 0: return a.reshape(-1, J, C)[:, :, :3]
        raise ValueError(f"Cannot reshape {a.shape} to [N,{J},3]")

    seq_raw = to_N_J_3(arr)  # (N_raw, J, 3)


    if center_root:
        if not (0 <= root_idx < J): raise ValueError(f"root_idx={root_idx} is outside [0,{J})")
        seq_raw = seq_raw - seq_raw[:, root_idx:root_idx+1, :]


    nan_frame_mask_raw = np.isnan(seq_raw).any(axis=(1, 2))
    Nf_raw = seq_raw.shape[0]
    if sliding:
        starts_raw = list(range(0, max(0, Nf_raw - T + 1), max(1, stride)))
        ends_raw = [s + T - 1 for s in starts_raw]
    else:
        n_win_raw = Nf_raw // T
        starts_raw = [i * T for i in range(n_win_raw)]
        ends_raw = [s + T - 1 for s in starts_raw]
    nan_win_idxs_raw = []
    for i, (s, e) in enumerate(zip(starts_raw, ends_raw)):
        if e >= Nf_raw: break
        if nan_frame_mask_raw[s:e+1].any():
            nan_win_idxs_raw.append(i)
    if len(nan_win_idxs_raw) > 0:
        print(f"[NaN] Original window indices containing NaNs: {nan_win_idxs_raw} / {len(starts_raw)} total")
    else:
        print(f"[NaN] No NaN windows in the original segmentation (T={T}, sliding={int(sliding)}, stride={stride}).")


    mask_good = ~nan_frame_mask_raw
    n_bad = int((~mask_good).sum())
    if n_bad > 0:
        print(f"[Info] Removed {n_bad} frames containing NaNs before feature extraction.")
    seq = seq_raw[mask_good]
    Nf = seq.shape[0]
    if Nf < T:
        raise ValueError(f"Valid frame count {Nf} is smaller than window length {T}")

    if sliding:
        starts = list(range(0, Nf - T + 1, max(1, stride)))
        ends = [s + T - 1 for s in starts]
    else:
        n_win = Nf // T
        starts = [i * T for i in range(n_win)]
        ends = [s + T - 1 for s in starts]

    X_win, spans = [], []
    for s, e in zip(starts, ends):
        if e >= Nf: break
        X_win.append(seq[s:e + 1])   # (T, J, 3)
        spans.append([s, e])
    if not X_win:
        raise ValueError(f"No windows generated: Nf={Nf}, T={T}, sliding={sliding}, stride={stride}")

    X_win = np.stack(X_win, axis=0)               # (Nw, T, J, 3)
    spans = np.asarray(spans, dtype=np.int64)     # (Nw, 2)
    return X_win, spans, nan_win_idxs_raw


# =====================================================================
# Diagnostic Projection and Visualization
# =====================================================================

def _fisher_plane(Z_id: np.ndarray, Z_oos: np.ndarray, eps: float = 1e-4):
    μ0 = Z_id.mean(axis=0); μ1 = Z_oos.mean(axis=0)
    Σ0 = np.cov(Z_id, rowvar=False); Σ1 = np.cov(Z_oos, rowvar=False)
    d = Z_id.shape[1]
    Σw = Σ0 + Σ1 + eps * np.eye(d)
    w = np.linalg.solve(Σw, (μ0 - μ1))
    n = np.linalg.norm(w);  w = w / max(n, 1e-12)

    Z = np.vstack([Z_id, Z_oos])
    Z_res = Z - (Z @ w[:, None]) * w[None, :]
    _, _, VT = np.linalg.svd(Z_res, full_matrices=False)
    v2 = VT[0, :]
    v2 = v2 - (v2 @ w) * w
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)

    X1 = Z @ w
    X2 = Z @ v2
    XY = np.stack([X1, X2], axis=1)
    return XY, w, v2

def _svm_plane(Z_id: np.ndarray, Z_oos: np.ndarray, C: float = 1.0, seed=None):
    Z = np.vstack([Z_id, Z_oos])
    y = np.hstack([np.zeros(Z_id.shape[0], dtype=np.int32),
                   np.ones( Z_oos.shape[0], dtype=np.int32)])
    clf = LinearSVC(C=C, class_weight='balanced', dual=False, max_iter=20000,
                    random_state=resolve_seed(seed))
    clf.fit(Z, y)
    w_raw = clf.coef_.ravel().astype(np.float64); b_raw = float(clf.intercept_[0])
    n = np.linalg.norm(w_raw);  w = w_raw / max(n, 1e-12)
    b = b_raw / max(n, 1e-12)

    Z_res = Z - (Z @ w[:, None]) * w[None, :]
    _, _, VT = np.linalg.svd(Z_res, full_matrices=False)
    v2 = VT[0, :]
    v2 = v2 - (v2 @ w) * w
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)

    X1 = Z @ w
    X2 = Z @ v2
    XY = np.stack([X1, X2], axis=1)
    return XY, w, b, v2


def _preprocess_for_proj(Z_id: np.ndarray, Z_oos: np.ndarray,
                         standardize: bool = True, whiten: bool = True):
    """Fit standardization and whitening on known identities only.

    Args:
        Z_id (np.ndarray): Known-identity features of shape (N, H).
        Z_oos (np.ndarray): Unknown-identity features of shape (M, H).
        standardize (bool): Apply the known-identity mean and standard deviation.
        whiten (bool): Apply the regularized known-identity covariance transform.

    Returns:
        tuple: Transformed known and unknown feature arrays in float32."""
    Z_id2, Z_oos2 = Z_id.astype(np.float64).copy(), Z_oos.astype(np.float64).copy()
    if standardize:
        scaler = StandardScaler().fit(Z_id2)
        Z_id2 = scaler.transform(Z_id2)
        Z_oos2 = scaler.transform(Z_oos2)
    if whiten:
        C = np.cov(Z_id2, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-6, None)
        W = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)
        Z_id2  = Z_id2  @ W.T
        Z_oos2 = Z_oos2 @ W.T
    return Z_id2.astype(np.float32), Z_oos2.astype(np.float32)


def _project_fisher_2d(Z_id_fit: np.ndarray,
                       Z_unk_fit: np.ndarray,
                       Z_id_q: np.ndarray,
                       Z_unk_q: np.ndarray,
                       standardize: bool = True,
                       whiten: bool = True,
                       eps: float = 1e-4):
    """Fit Fisher directions on the selected samples and project all queries."""
    Z_id_fit_p, Z_unk_fit_p = _preprocess_for_proj(
        Z_id_fit, Z_unk_fit, standardize=standardize, whiten=whiten)
    _, w, v2 = _fisher_plane(Z_id_fit_p, Z_unk_fit_p, eps=eps)
    Z_id_q_p, Z_unk_q_p = _preprocess_for_proj(
        Z_id_q, Z_unk_q, standardize=standardize, whiten=whiten)
    XY_id = np.stack([Z_id_q_p @ w, Z_id_q_p @ v2], axis=1)
    XY_unk = np.stack([Z_unk_q_p @ w, Z_unk_q_p @ v2], axis=1)
    return XY_id, XY_unk


def plot_fisher_id_vs_unknown_2d(
    Z_id: np.ndarray,
    Z_unk: np.ndarray,
    id_fn_mask: np.ndarray,
    unk_fp_mask: np.ndarray,
    path: str,
    id_labels: Optional[np.ndarray] = None,
    standardize: bool = True,
    whiten: bool = True,
    id_plot_max: int = 2000,
    unk_plot_max: int = 2000,
    fit_max_per_class: int = 2500,
    plot_seed: Optional[int] = None,
    auroc: Optional[float] = None,
) -> None:
    """Plot all unknown probes and known samples in the Fisher plane.

    Known samples above the threshold are false negatives; unknown samples
    at or below the threshold are false positives. The plot is a diagnostic
    projection, while reported OOS metrics use the original feature space."""
    n_id, n_unk = Z_id.shape[0], Z_unk.shape[0]
    fit_n = min(fit_max_per_class, n_id, n_unk)
    if fit_n < 2:
        print("[Fisher] Insufficient samples; skipping the all-unknown projection.")
        return

    plot_seed = resolve_seed(plot_seed)
    rng = np.random.default_rng(plot_seed)
    fit_id_idx = rng.choice(n_id, size=fit_n, replace=False) if n_id > fit_n else np.arange(n_id)
    fit_unk_idx = rng.choice(n_unk, size=fit_n, replace=False) if n_unk > fit_n else np.arange(n_unk)

    idx_id = _sample_idx(n_id, id_plot_max, plot_seed)
    idx_unk = _sample_idx(n_unk, unk_plot_max, (plot_seed + 1) % (2 ** 32))

    XY_id, XY_unk = _project_fisher_2d(
        Z_id[fit_id_idx], Z_unk[fit_unk_idx],
        Z_id[idx_id], Z_unk[idx_unk],
        standardize=standardize, whiten=whiten,
    )
    id_fn = id_fn_mask[idx_id]
    unk_fp = unk_fp_mask[idx_unk]
    unk_tn = ~unk_fp

    plt.figure(figsize=(8.5, 6.5))


    m_id_ok = ~id_fn
    if np.any(m_id_ok):
        plt.scatter(XY_id[m_id_ok, 0], XY_id[m_id_ok, 1],
                    s=28, alpha=0.35, marker='o', color='#888888',
                    edgecolors='none', label='ID', zorder=1)

    if np.any(id_fn):
        plt.scatter(XY_id[id_fn, 0], XY_id[id_fn, 1],
                    s=55, alpha=0.85, marker='o', facecolors='none',
                    edgecolors='#D62728', linewidths=1.2, label='ID-FN', zorder=3)


    if np.any(unk_tn):
        plt.scatter(XY_unk[unk_tn, 0], XY_unk[unk_tn, 1],
                    s=52, alpha=0.55, marker='^', color='#4169E1',
                    edgecolors='black', linewidths=0.3, label='OOS-TN', zorder=2)

    if np.any(unk_fp):
        plt.scatter(XY_unk[unk_fp, 0], XY_unk[unk_fp, 1],
                    s=72, alpha=0.95, marker='^', color='#FF7F0E',
                    edgecolors='#D62728', linewidths=0.8, label='OOS-FP', zorder=4)

    title = 'Fisher 2D: ID train lib vs all unknown probe'
    if auroc is not None:
        title += f'  (AUROC={auroc:.4f})'
    plt.title(title, fontsize=12)
    plt.xlabel('Fisher dim 1')
    plt.ylabel('Fisher dim 2')
    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    plt.legend(loc='best', fontsize=9, framealpha=0.92)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=360, bbox_inches='tight')
    plt.close()


def plot_id_clusters_with_oos_2d(
    Z_id: np.ndarray,
    id_labels: np.ndarray,
    Z_oos: np.ndarray,
    path: str,
    method: str = "tsne",
    per_id_max: int = 120,
    oos_max: int = 800,
    feat_mode: str = "l2norm",
    perplexity: float = 30.0,
    name_fn=None,
    plot_seed: Optional[int] = None,
    annotate_centroids: bool = True,
    auroc: Optional[float] = None,
) -> dict:
    """Project known identities and unknown probes into one diagnostic figure.

    Args:
        Z_id (np.ndarray): Known feature matrix of shape (N, H).
        id_labels (np.ndarray): Identity labels aligned with Z_id.
        Z_oos (np.ndarray): Unknown feature matrix of shape (M, H).
        path (str): Destination image path.

    Returns:
        dict: Silhouette statistics and, in the shared plotting helper,
        reusable sampling indices, projected coordinates, and axis limits."""
    id_labels = np.asarray(id_labels).astype(np.int64)
    uniq = np.unique(id_labels)


    plot_seed = resolve_seed(plot_seed)
    rng = np.random.default_rng(plot_seed)
    sel_id_idx = []
    for c in uniq:
        idx_c = np.where(id_labels == c)[0]
        if idx_c.size > per_id_max:
            idx_c = rng.choice(idx_c, size=per_id_max, replace=False)
        sel_id_idx.append(idx_c)
    sel_id_idx = np.concatenate(sel_id_idx)
    Z_id_s = Z_id[sel_id_idx]
    lab_id_s = id_labels[sel_id_idx]


    n_oos = Z_oos.shape[0]
    oos_idx = _sample_idx(n_oos, oos_max, (plot_seed + 7) % (2 ** 32)) if n_oos > 0 else np.array([], dtype=np.int64)
    Z_oos_s = Z_oos[oos_idx] if n_oos > 0 else np.zeros((0, Z_id.shape[1]), dtype=Z_id.dtype)


    if feat_mode == "whiten":
        Z_id_p, Z_oos_p = _preprocess_for_proj(
            Z_id_s, Z_oos_s, standardize=True, whiten=True)
    else:
        def _l2(a):
            if a.shape[0] == 0:
                return a.astype(np.float32)
            n = np.linalg.norm(a, axis=1, keepdims=True)
            return (a / np.clip(n, 1e-12, None)).astype(np.float32)
        Z_id_p = _l2(Z_id_s.astype(np.float32))
        Z_oos_p = _l2(Z_oos_s.astype(np.float32))

    X = np.vstack([Z_id_p, Z_oos_p]) if Z_oos_p.shape[0] > 0 else Z_id_p
    n_id_pts = Z_id_p.shape[0]


    m = method.lower()
    if m == "tsne":
        perp = max(5.0, min(float(perplexity), X.shape[0] - 1))
        lr = max(50.0, X.shape[0] / 12.0)
        reducer = make_tsne(n_components=2, init="pca", learning_rate=lr,
                       perplexity=perp, random_state=plot_seed, n_iter=1000, verbose=0)
        XY = reducer.fit_transform(X)
        method_title = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=plot_seed)
        XY = reducer.fit_transform(X)
        method_title = "PCA"

    XY_id = XY[:n_id_pts]
    XY_oos = XY[n_id_pts:]


    metrics = {}
    if silhouette_score is not None:
        try:
            if len(uniq) >= 2 and XY_id.shape[0] > len(uniq):
                metrics["sil_within_id"] = float(
                    silhouette_score(XY_id, lab_id_s, metric="euclidean"))
        except Exception:
            pass
        try:
            if XY_oos.shape[0] >= 2:
                bin_lab = np.hstack([
                    np.zeros(XY_id.shape[0], dtype=np.int32),
                    np.ones(XY_oos.shape[0], dtype=np.int32)])
                metrics["sil_id_vs_oos"] = float(
                    silhouette_score(XY, bin_lab, metric="euclidean"))
        except Exception:
            pass


    plt.figure(figsize=(10.5, 8.0))
    cmap = plt.get_cmap("tab20", max(2, len(uniq)))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "p"]

    for i, c in enumerate(uniq):
        msk = (lab_id_s == c)
        if not np.any(msk):
            continue
        color_i = cmap(i % cmap.N)
        mk = markers[i % len(markers)]
        lab = name_fn(int(c)) if name_fn is not None else f"ID {int(c)}"
        plt.scatter(XY_id[msk, 0], XY_id[msk, 1],
                    s=42, alpha=0.80, marker=mk, color=color_i,
                    edgecolors="white", linewidths=0.4, label=lab, zorder=3)
        if annotate_centroids:
            cx, cy = XY_id[msk, 0].mean(), XY_id[msk, 1].mean()
            plt.text(cx, cy, lab, fontsize=8, ha="center", va="center",
                     color="black", zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    if XY_oos.shape[0] > 0:
        plt.scatter(XY_oos[:, 0], XY_oos[:, 1],
                    s=46, alpha=0.55, marker="x", color="black",
                    linewidths=1.1, label="OOS (unknown)", zorder=5)

    sub = []
    if auroc is not None:
        sub.append(f"ID-vs-OOS AUROC={auroc:.3f}")
    if "sil_within_id" in metrics:
        sub.append(f"within-ID silhouette={metrics['sil_within_id']:.3f}")
    if "sil_id_vs_oos" in metrics:
        sub.append(f"ID-vs-OOS silhouette={metrics['sil_id_vs_oos']:.3f}")
    title = f"{method_title}: per-identity clusters + OOS overlay"
    if sub:
        title += "\n" + " | ".join(sub)
    plt.title(title, fontsize=12)
    plt.xlabel(f"{method_title} dim 1")
    plt.ylabel(f"{method_title} dim 2")
    plt.grid(alpha=0.22, linestyle="--", linewidth=0.6)

    ax = plt.gca()
    ncol = 2 if len(uniq) > 14 else 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=True, framealpha=0.9, ncol=ncol, borderaxespad=0.3,
              labelspacing=0.3, handletextpad=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=360, bbox_inches="tight")
    plt.close()
    return metrics


def plot_knn_boundary_2d(
    XY_id: np.ndarray,
    XY_oos: np.ndarray,
    id_labels: Optional[np.ndarray],
    sigma: float,
    k: int,
    path: str,
    method_name: str = "PCA/TSNE/FISHER/SVM",
    grid: int = 400,
    metric: str = "euclidean",
    id_core_q: float = 1.0,

    id_plot_max: Optional[int] = 3000,
    oos_plot_max: Optional[int] = 300,
    plot_seed: Optional[int] = None,

    title_size: int = 18,
    axis_label_size: int = 20,
    tick_label_size: int = 20,
    legend_size: int = 20,
    contour_label_size: int = 20,

    fill_color: Optional[str] = "#BDBDBD",
    fill_alpha: float = 0,

    contour_lw: float = 4.0,
    contour_ls: str = "--",
    contour_color: str = '#000000',
    outline: bool = True,
    outline_extra: float = 2.0,
    draw_hull: bool = False,
) -> float:
    """Save a k-nearest-neighbor quantile boundary in a diagnostic 2D plane.

    The projected threshold is independent of the high-dimensional OOS
    threshold used for classification and reported evaluation metrics.

    Returns:
        float: Quantile threshold estimated from known points in this plane."""

    plot_seed = resolve_seed(plot_seed)

    def _subsample_rows(X: np.ndarray, m: Optional[int], seed: int) -> np.ndarray:
        X = np.asarray(X)
        if m is None or m <= 0 or X.shape[0] <= m:
            return X
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=int(m), replace=False)
        return X[idx]

    def _subsample_index(n: int, m: Optional[int], seed: int) -> np.ndarray:
        if m is None or m <= 0 or n <= m:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=int(m), replace=False)


    id_core_mask = None
    if id_core_q < 1.0:
        nbrs_tmp = NearestNeighbors(n_neighbors=max(1, k), metric="euclidean").fit(XY_id)
        d_tmp, _ = nbrs_tmp.kneighbors(XY_id)
        kth_tmp = d_tmp[:, max(1, k) - 1]
        qv = np.quantile(kth_tmp, q=float(id_core_q))
        id_core_mask = (kth_tmp <= qv)
        XY_id = XY_id[id_core_mask]
        if id_labels is not None and len(id_labels) == len(id_core_mask):
            id_labels = np.asarray(id_labels)[id_core_mask]


    nn_kwargs = dict(n_neighbors=max(1, k))
    if metric == "mahalanobis":
        C = np.cov(XY_id, rowvar=False) + 1e-6 * np.eye(2)
        VI = np.linalg.pinv(C)
        nn_kwargs.update(metric="mahalanobis", metric_params={"VI": VI})
    elif metric == "cosine":
        nn_kwargs.update(metric="cosine")
    else:
        nn_kwargs.update(metric="euclidean")

    nbrs_id = NearestNeighbors(**nn_kwargs).fit(XY_id)


    d_id, _ = nbrs_id.kneighbors(XY_id)
    d_id_k = d_id[:, max(1, k) - 1]
    tau2d = float(np.quantile(d_id_k, q=float(sigma)))


    x_min, x_max = float(XY_id[:, 0].min()), float(XY_id[:, 0].max())
    y_min, y_max = float(XY_id[:, 1].min()), float(XY_id[:, 1].max())
    pad_x = 0.08 * max(1e-9, x_max - x_min)
    pad_y = 0.08 * max(1e-9, y_max - y_min)
    xx = np.linspace(x_min - pad_x, x_max + pad_x, int(grid))
    yy = np.linspace(y_min - pad_y, y_max + pad_y, int(grid))
    XX, YY = np.meshgrid(xx, yy)
    G = np.c_[XX.ravel(), YY.ravel()]


    dG, _ = nbrs_id.kneighbors(G)
    dG_k = dG[:, max(1, k) - 1].reshape(XX.shape)
    inside = (dG_k <= tau2d).astype(np.int32)


    id_idx_plot = _subsample_index(XY_id.shape[0], id_plot_max, plot_seed)
    XY_id_plot = XY_id[id_idx_plot]
    if id_labels is not None and len(id_labels) == len(XY_id):
        id_labels_plot = np.asarray(id_labels)[id_idx_plot]
    else:
        id_labels_plot = None
    XY_oos_plot = _subsample_rows(XY_oos, oos_plot_max, (plot_seed + 1) % (2 ** 32)) if (XY_oos is not None and len(XY_oos) > 0) else None


    plt.figure(figsize=(9.2, 6.8))

    oos_color = "#4169E1"

    if XY_oos_plot is not None and len(XY_oos_plot) > 0:
        plt.scatter(XY_oos_plot[:, 0], XY_oos_plot[:, 1],
                    s=82, alpha=0.90, label="OOS", marker="^",
                    color=oos_color, edgecolors="black", linewidths=0.45, zorder=4)

    if id_labels_plot is not None:
        uniq = np.unique(id_labels_plot.astype(np.int64))
        cmap = plt.get_cmap("gist_ncar", max(1, len(uniq)))
        for i, cid in enumerate(uniq):
            m = (id_labels_plot == cid)
            if not np.any(m):
                continue
            color_i = cmap(i)
            plt.scatter(
                XY_id_plot[m, 0], XY_id_plot[m, 1],
                s=50, alpha=0.62, marker="o",
                color=color_i, edgecolors="black", linewidths=0.25,
                label=f"ID {int(cid)}", zorder=2
            )

            pts = XY_id_plot[m]
            if draw_hull and ConvexHull is not None and pts.shape[0] >= 4:
                try:
                    hull = ConvexHull(pts)
                    hp = pts[hull.vertices]
                    hp = np.vstack([hp, hp[0]])
                    plt.plot(hp[:, 0], hp[:, 1], linestyle='-', linewidth=1.0,
                             color=color_i, alpha=0.75, zorder=3)
                except Exception:
                    pass
    else:
        plt.scatter(XY_id_plot[:, 0], XY_id_plot[:, 1],
                    s=50, alpha=0.62, label="ID", marker="o",
                    color="#FF7F0E", edgecolors="black", linewidths=0.25, zorder=2)


    if fill_color is not None and fill_alpha > 0:
        plt.contourf(
            XX, YY, inside,
            levels=[-0.5, 0.5, 1.5],
            colors=[fill_color, fill_color],
            alpha=fill_alpha,
            antialiased=True,
        )


    CS = plt.contour(
        XX, YY, dG_k,
        levels=[tau2d],
        linewidths=contour_lw,
        linestyles=contour_ls,
        colors=[contour_color],
        antialiased=True,
    )
    if outline:
        # Matplotlib 3.8+ exposes the contour set as a single collection.
        artists = [CS] if hasattr(CS, "set_path_effects") else CS.collections
        for col in artists:
            col.set_path_effects([
                pe.Stroke(linewidth=contour_lw + outline_extra, foreground="white"),
                pe.Normal()
            ])


    plt.xlabel("Dim 1", fontsize=axis_label_size)
    plt.ylabel("Dim 2", fontsize=axis_label_size)

    ax = plt.gca()
    ax.tick_params(
        axis="both", which="both",
        labelbottom=False, labelleft=False, labelright=False, labeltop=False,
        labelsize=tick_label_size
    )
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        bbox_transform=ax.transAxes,
        fontsize=min(legend_size, 11),
        frameon=True, framealpha=0.9, fancybox=True,
        borderpad=0.4, borderaxespad=0.4, handletextpad=0.6,
        labelspacing=0.3, markerscale=1.0, ncol=2,
    )

    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    plt.tight_layout(rect=[0.0, 0.0, 0.79, 1.0])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=360, bbox_inches='tight')
    plt.close()

    return tau2d


# =====================================================================
# Feature Distances and OOS Calibration
# =====================================================================

def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _kth_nn_tau_from_lib(Z_lib: np.ndarray, k: int, sigma_q: float):
    """Estimate the training-library kNN distance quantile, excluding self.

    Args:
        Z_lib (np.ndarray): L2-normalized training features of shape (N, H).
        k (int): Neighbor rank, clamped to N - 1.
        sigma_q (float): Quantile in [0, 1] used to calibrate the threshold.

    Returns:
        tuple: Per-sample distances (N,) and the scalar rejection threshold."""
    n = Z_lib.shape[0]
    if n < 2:
        raise ValueError('OOS calibration requires at least two training samples.')
    k_eff = int(max(1, min(k, n - 1)))
    S = Z_lib @ Z_lib.T
    D = np.sqrt(np.clip(2.0 - 2.0 * S, 0.0, None))
    kth_vals = np.partition(D, kth=k_eff, axis=1)[:, k_eff]
    tau = float(np.quantile(kth_vals, q=float(sigma_q)))
    return kth_vals.astype(np.float32), tau

def _kth_nn_dist_queries(Z_q: np.ndarray, Z_lib: np.ndarray, k: int):
    """Measure each query's k-th neighbor distance in the training library.

    Args:
        Z_q (np.ndarray): L2-normalized query features of shape (M, H).
        Z_lib (np.ndarray): L2-normalized training features of shape (N, H).
        k (int): Neighbor rank, clamped to the available library size.

    Returns:
        np.ndarray: One float32 distance per query."""
    n = Z_lib.shape[0]
    if n == 0:
        raise ValueError('The OOS feature library is empty.')
    k_eff = int(max(1, min(k, n)))
    S = Z_q @ Z_lib.T
    D = np.sqrt(np.clip(2.0 - 2.0 * S, 0.0, None))
    kth_vals = np.partition(D, kth=k_eff-1, axis=1)[:, k_eff-1]
    return kth_vals.astype(np.float32)

def eval_oos_metrics(rK_lib: np.ndarray,
                     dK_win: np.ndarray,
                     sigma_eval: float = 0.95):
    """Evaluate unknown acceptance and ROC area against calibration samples.

    Args:
        rK_lib (np.ndarray): Known training-library kNN distances.
        dK_win (np.ndarray): Unknown-probe kNN distances.
        sigma_eval (float): Target known-data acceptance quantile.

    Returns:
        tuple: Metric dictionary and (false-positive rate, true-positive rate)
        arrays. The legacy FPR95 key uses sigma_eval, which defaults to 0.95.

    Notes:
        Known samples are the training calibration library, not a held-out
        known test split. Keep this protocol when comparing reported scores."""
    tau_eval = float(np.quantile(rK_lib, q=float(sigma_eval)))
    tpr_id = float((rK_lib <= tau_eval).mean())
    fpr_oos = float((dK_win <= tau_eval).mean())
    y_true = np.concatenate([np.ones_like(rK_lib), np.zeros_like(dK_win)]).astype(np.int32)
    scores = np.concatenate([-rK_lib, -dK_win]).astype(np.float32)
    fpr, tpr, _ = roc_curve(y_true, scores)
    auroc = auc(fpr, tpr)
    return {"tau_eval": tau_eval, "TPR_ID_at_tau": tpr_id, "FPR95": fpr_oos, "AUROC": auroc}, (fpr, tpr)


# =====================================================================
# Sequence Feature Extraction and Identity Retrieval
# =====================================================================

@torch.no_grad()
def _extract_seq_feats(model: SSGT,
                       X_seq: np.ndarray,
                       L_eig_np: np.ndarray,
                       batch_size: int,
                       device: torch.device,
                       T: int, J: int, H: int,
                       use_pos: bool = True,
                       bn_freeze: bool = True) -> np.ndarray:
    """Extract sequence embeddings without updating batch-normalization state.

    Args:
        model (SSGT): Trained skeleton graph model.
        X_seq (np.ndarray): Skeleton windows of shape (N, T, J, 3).
        L_eig_np (np.ndarray): Spatial positional encodings of shape (J, K).
        batch_size (int): Number of windows processed together.
        device (torch.device): Inference device.
        T (int): Sequence length.
        J (int): Joint count.
        H (int): Embedding width retained for API compatibility.
        use_pos (bool): Enable the checkpoint's positional encoding.
        bn_freeze (bool): Use stored batch-normalization statistics.

    Returns:
        np.ndarray: Sequence features of shape (N, H)."""
    model.eval()
    feats = []
    N = X_seq.shape[0]
    if N == 0:
        raise ValueError('No skeleton windows are available for feature extraction.')
    L_batch = torch.as_tensor(L_eig_np, dtype=torch.float32, device=device)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        xb_np = X_seq[s:e].reshape(-1, J, 3)
        xb = torch.from_numpy(xb_np).float().to(device)
        x = model.encode_input(xb)                # [B*T, J, H]
        B = e - s
        x = x.reshape(B, T, J, model.H)
        h = model.add_pos_and_attn(x, L_batch, use_pos=use_pos, train=False, bn_freeze=bn_freeze)  # [B,T,J,H]
        C_seq = h.mean(dim=-2)                    # [B,T,H]
        if model.frame_attn_pool:
            logits = model.frame_attn(C_seq).squeeze(-1)   # [B,T]
            w = torch.softmax(logits, dim=1).unsqueeze(-1) # [B,T,1]
            seq_ftr = (w * C_seq).sum(dim=1)               # [B,H]
        else:
            seq_ftr = C_seq.mean(dim=1)
        feats.append(seq_ftr.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def _compute_dist_matrix(A: np.ndarray, B: np.ndarray, metric: str, l2norm_feats: bool) -> np.ndarray:
    A_t = torch.from_numpy(A).float()
    B_t = torch.from_numpy(B).float()
    if l2norm_feats or metric == 'cosine':
        A_t = F.normalize(A_t, dim=1)
        B_t = F.normalize(B_t, dim=1)
    if metric == 'cosine':
        sim = A_t @ B_t.t()
        dist = (1.0 - sim).cpu().numpy()
    else:
        m, n = A_t.size(0), B_t.size(0)
        a2 = (A_t*A_t).sum(dim=1, keepdim=True).expand(m, n)
        b2 = (B_t*B_t).sum(dim=1, keepdim=True).expand(n, m).t()
        dist = a2 + b2
        dist.addmm_(A_t, B_t.t(), beta=1.0, alpha=-2.0)
        dist = (dist.clamp(min=1e-12)).sqrt().cpu().numpy()
    return dist

def _topk_by_identity(dist_row: np.ndarray, gal_labels: np.ndarray, topk: int):
    labels = gal_labels.astype(int)
    classes = np.unique(labels)
    min_dists = []
    for c in classes:
        d = dist_row[labels == c]
        if d.size == 0:
            continue
        min_dists.append((int(c), float(d.min())))
    min_dists.sort(key=lambda x: x[1])
    return min_dists[:topk]

def _sample_idx(n: int, k: int, seed: Optional[int] = None):
    seed = resolve_seed(seed)
    if n <= k: return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)

def _majority_vote_per_frame(spans: np.ndarray,
                             win_ids: np.ndarray,
                             win_dists: np.ndarray,
                             one_based: bool,
                             n_frames: int) -> np.ndarray:
    """Vote over overlapping windows and convert label indexing once.

    Args:
        spans (np.ndarray): Inclusive frame intervals, shape (N, 2).
        win_ids (np.ndarray): Zero-based identity labels for each window.
        win_dists (np.ndarray): Matching distances used to break voting ties.
        one_based (bool): Add one to labels of covered frames.
        n_frames (int): Number of frames in the cleaned sequence.

    Returns:
        np.ndarray: Per-frame IDs. Uncovered frames retain zero."""
    covered = np.zeros(n_frames, dtype=bool)
    frame_pred = np.zeros(n_frames, dtype=np.int64)
    for f in range(n_frames):
        mask = (spans[:, 0] <= f) & (spans[:, 1] >= f)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            frame_pred[f] = 0; continue
        covered[f] = True
        ids = win_ids[idxs]; dists = win_dists[idxs]
        uniq, counts = np.unique(ids, return_counts=True)
        maxc = counts.max(); cand = uniq[counts == maxc]
        if cand.size == 1:
            best = cand[0]
        else:
            best = None; best_mean = 1e9
            for c in cand:
                mean_d = dists[ids == c].mean()
                if mean_d < best_mean: best_mean = mean_d; best = c
        frame_pred[f] = best
    if one_based:
        frame_pred = np.where(covered, frame_pred + 1, 0)
    return frame_pred

# =====================================================

# =====================================================================
# Checkpoint Restoration
# =====================================================================

def _validate_spatial_pos_encoding(value, joint_count, encoding_width):
    """Convert a persisted or reconstructed spatial encoding to finite float32."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    encoding = np.asarray(value, dtype=np.float32)
    expected = (int(joint_count), int(encoding_width))
    if encoding.shape != expected or not np.isfinite(encoding).all():
        raise ValueError(
            f'Spatial position encoding must be finite with shape {expected}; '
            f'got {encoding.shape}'
        )
    return encoding.copy()


def resolve_spatial_pos_encoding(model, computed_encoding):
    """Use the saved spatial basis or compute it from the dataset graph.

    Args:
        model (SSGT): Model returned by load_checkpoint_model.
        computed_encoding (np.ndarray): Basis reconstructed by the dataset loader.

    Returns:
        np.ndarray: Validated spatial basis of shape (J, enc_k_spa).

    Notes:
        If no basis is stored, the dataset loader supplies it. Eigenvector
        orientation can vary between numerical-library versions. The selected
        basis is recorded in model.spatial_pos_encoding_metadata.
    """
    persisted = getattr(model, 'checkpoint_spatial_pos_encoding', None)
    encoding = _validate_spatial_pos_encoding(
        computed_encoding if persisted is None else persisted,
        model.J, model.pos_dense.in_features,
    )
    model.spatial_pos_encoding_metadata = {
        'used_source': 'checkpoint' if persisted is not None else 'recomputed_by_data_loader',
        'checkpoint_provenance': getattr(model, 'spatial_pos_encoding_provenance', None),
        'shape': list(encoding.shape),
        'float32_sha256': hashlib.sha256(np.ascontiguousarray(encoding).tobytes()).hexdigest(),
    }
    return encoding


def load_checkpoint_model(ckpt_path, device, dataset):
    """Restore the complete model architecture and require matching state keys.

    Args:
        ckpt_path (str): Path to a trusted training checkpoint.
        device (torch.device): Target inference device.
        dataset (str): Expected dataset; must match saved metadata if present.

    Returns:
        tuple: Model in evaluation mode and its saved argument dictionary.

    Raises:
        ValueError: If checkpoint metadata is missing or names a different dataset.
        RuntimeError: If checkpoint weights do not match the recorded architecture.

    Notes:
        Training checkpoints contain optimizer metadata and use pickle. Load only
        files from a trusted source.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint:
        raise ValueError('Expected a training checkpoint containing model_state and args.')
    saved_args = checkpoint.get('args')
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, dict):
        raise ValueError('Checkpoint args metadata is required to reconstruct the model.')
    saved_args = dict(saved_args)
    checkpoint_dataset = str(saved_args.get('dataset', dataset)).upper()
    if checkpoint_dataset != str(dataset).upper():
        raise ValueError(f'Checkpoint dataset {checkpoint_dataset} does not match --dataset {dataset}.')
    saved_args['dataset'] = checkpoint_dataset
    model = SSGT(
        H=int(saved_args.get('H', 128)),
        n_heads=int(saved_args.get('n_heads', 8)),
        L=int(saved_args.get('L_transformer', 2)),
        p_drop=0.5,
        enc_k_spa=int(saved_args.get('enc_k_spa', 10)),
        enc_k_time=int(saved_args.get('enc_k_time', 4)),
        J=_dataset_joint_num(checkpoint_dataset),
        T=int(saved_args.get('length', 6)),
        pre_norm=bool(saved_args.get('pre_norm', 1)),
        attn_drop=float(saved_args.get('attn_dropout', 0.4)),
        attn_clamp=bool(saved_args.get('attn_clamp', 1)),
        qkv_merged=bool(saved_args.get('qkv_merged', 1)),
        frame_attn_pool=bool(saved_args.get('frame_attn_pool', 0)),
        t1_init=float(saved_args.get('t_1', 0.07)),
        t2_init=float(saved_args.get('t_2', 14.0)),
        learnable_temp=bool(saved_args.get('learnable_temp', 0)),
        use_time_attn=bool(saved_args.get('use_time_attn', 1)),
        spa_pos_type=str(saved_args.get('spa_pos_type', 'laplacian')),
    ).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    persisted = checkpoint.get('spatial_pos_encoding')
    # Plain NumPy attributes preserve the original model_state key set.
    model.checkpoint_spatial_pos_encoding = (
        None if persisted is None else _validate_spatial_pos_encoding(
            persisted, model.J, model.pos_dense.in_features
        )
    )
    model.spatial_pos_encoding_provenance = checkpoint.get('spatial_pos_encoding_provenance')
    model.eval()
    return model, saved_args


# =====================================================================
# Command-Line Configuration
# =====================================================================

def get_parser():
    """Build command-line options for identity retrieval and OOS evaluation."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])


    p.add_argument('--data_source', default='unknown', choices=['unknown','npy'],
                   help='Use a preprocessed unknown split or an external skeleton NPY sequence.')
    p.add_argument('--input', '--npy_path', '--npy-path',
                   default='',
                   type=str, help='Skeleton NPY path; required when --data_source npy is selected.')


    p.add_argument('--dataset', default='WUT', type=str, choices=['KGBD','KS20','WUT'])
    p.add_argument('--probe',   default='probe', type=str,
                   help='Known evaluation split; use probe for WUT, KGBD, or KS20.')
    p.add_argument('--gpu', default='0', type=str, help='CUDA device index, or -1/cpu for CPU.')
    p.add_argument('--seed', default=-1, type=int,
                   help='Run seed; -1 generates and records a fresh seed. Supply that seed to replay sampling and plots.')
    p.add_argument('--ckpt', default=CKPT_DEFAULT, type=str, help='Checkpoint path; when empty, search weight/<dataset> for the latest best.ckpt.')
    p.add_argument('--proto_source', default='gallery', choices=['gallery','train'], help='Feature library used for identity retrieval and Top-k predictions.')


    p.add_argument('--stride', default=6, type=int, help='Frame step for sliding windows.')
    p.add_argument('--sliding', default=0, type=int, help='1: sliding windows; 0: non-overlapping windows.')
    p.add_argument('--center_root', default=1, type=int, help='Subtract the root joint from each input frame.')
    p.add_argument('--root_idx', default=0, type=int, help='Zero-based root joint index.')


    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--topk', default=5, type=int, help='Number of retrieved identities to report.')
    p.add_argument('--metric_override', default='euclidean', choices=['cosine','euclidean'], help='Distance metric for identity retrieval.')
    p.add_argument('--l2norm_features', default=1, type=int, help='L2-normalize identity-retrieval features.')
    p.add_argument('--debug', default=0, type=int, help='Print intermediate diagnostics.')


    p.add_argument('--groundtruth', default='', type=str,
                   help='Optional original dataset identity; numeric IDs follow --one_based_ids. WUT also accepts names.')


    p.add_argument('--save_csv', default='', type=str, help='Optional destination CSV path.')
    p.add_argument('--save_npy_base', default=str(output_root() / 'oos'), type=str,
                   help='Output prefix for NPY, JSON, and diagnostic figures.')
    p.add_argument('--one_based_ids', default=1, type=int, choices=[0,1], help='Use one-based original identity IDs in CSV/NPY exports and numeric ground truth.')


    p.add_argument('--unk_detect', default=1, type=int, choices=[0,1], help='Enable unknown-identity rejection.')
    p.add_argument('--unk_method', default='knn', choices=['knn'])
    p.add_argument('--unk_knn_k', default=1, type=int, help='Neighbor rank used for OOS scoring.')
    p.add_argument('--unk_sigma', default=0.95, type=float, help='Known-data acceptance quantile.')
    p.add_argument('--unk_scope', default='window', choices=['window','sequence'], help='Reject individual windows or use the median sequence score.')


    p.add_argument('--eval_oos', default=1, type=int, choices=[0,1], help='Report unknown acceptance rate and AUROC.')
    p.add_argument('--sigma_eval', default=0.95, type=float, help='Known-data quantile for the legacy FPR95 metric.')
    p.add_argument('--save_plot', default='', type=str, help='Optional 1D distance plot path.')


    p.add_argument('--proj2d', default=1, type=int, choices=[0,1], help='Create diagnostic 2D projections.')
    p.add_argument('--proj2d_method', default='pca', choices=['pca','tsne','fisher','svm'],
                   help='Projection algorithm: PCA, t-SNE, Fisher, or linear SVM.')
    p.add_argument('--svm_C', default=1.0, type=float, help='Linear SVM regularization parameter.')
    p.add_argument('--proj2d_perplexity', default=30.0, type=float, help='t-SNE perplexity.')
    p.add_argument('--proj2d_max_points', default=3000, type=int, help='Maximum total samples in a diagnostic projection.')
    p.add_argument('--save_plot2d', default='', type=str, help='Optional 2D projection image path.')


    p.add_argument('--proj2d_standardize', default=1, type=int, choices=[0,1],
                   help='Standardize using known-data statistics before projection.')
    p.add_argument('--proj2d_whiten', default=1, type=int, choices=[0,1],
                   help='Whiten using the known-data covariance before projection.')
    p.add_argument('--boundary_metric', default='mahalanobis',
                   choices=['euclidean','mahalanobis','cosine'],
                   help='Metric for the diagnostic 2D kNN boundary.')
    p.add_argument('--boundary_id_core_q', default=1.0, type=float,
                   help='Quantile selecting core known points for the 2D boundary.')
    p.add_argument('--boundary_grid', default=500, type=int,
                   help='Resolution of the 2D boundary grid.')
    p.add_argument('--plot_fisher_all', default=1, type=int, choices=[0, 1],
                   help='Plot all unknown samples in the Fisher plane.')
    p.add_argument('--boundary_draw_hull', default=0, type=int, choices=[0, 1],
                   help='Draw convex hulls around known identities.')

    p.add_argument('--plot_id_oos', default=1, type=int, choices=[0, 1],
                   help='Plot known-identity clusters with unknown samples overlaid.')
    p.add_argument('--id_oos_method', default='tsne', choices=['tsne', 'pca'],
                   help='Projection algorithm for the identity/OOS overlay.')
    p.add_argument('--id_oos_per_id_max', default=120, type=int,
                   help='Maximum plotted windows per known identity.')
    p.add_argument('--id_oos_oos_max', default=800, type=int,
                   help='Maximum plotted unknown windows.')
    p.add_argument('--id_oos_feat_mode', default='l2norm', choices=['l2norm', 'whiten'],
                   help='Feature preprocessing used by the identity/OOS overlay.')

    p.add_argument('--data_subdir', default=None, type=str,
                   help='Dataset variant directory; by default inherit checkpoint metadata.')
    return p


# =====================================================================
# Evaluation Pipeline
# =====================================================================

def main(argv=None):
    """Run OOS evaluation from command-line options or a supplied argument list."""
    p = get_parser()
    args = p.parse_args(argv)
    if args.data_source == 'npy' and not args.input:
        p.error('--input is required when --data_source npy is selected.')
    if args.input and args.data_source == 'npy' and not Path(args.input).is_file():
        p.error(f'Skeleton input does not exist: {args.input}')
    if args.batch_size < 1 or args.topk < 1 or args.unk_knn_k < 1 or args.stride < 1:
        p.error('batch_size, topk, unk_knn_k, and stride must be positive.')
    if not 0 <= args.unk_sigma <= 1 or not 0 <= args.sigma_eval <= 1:
        p.error('unk_sigma and sigma_eval must lie in [0, 1].')
    if args.boundary_grid < 2 or not 0 < args.boundary_id_core_q <= 1:
        p.error('boundary_grid must be >= 2 and boundary_id_core_q must lie in (0, 1].')
    if args.proj2d_max_points < 4:
        p.error('proj2d_max_points must be >= 4.')
    try:
        run_seed = resolve_seed(args.seed)
    except ValueError as error:
        p.error(str(error))
    seed_selection = 'automatic' if args.seed == -1 else 'explicit'
    for output in (args.save_npy_base, args.save_csv, args.save_plot, args.save_plot2d):
        if output:
            Path(output.rstrip('/')).parent.mkdir(parents=True, exist_ok=True)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(run_seed)
    print(f'[Seed] Using {run_seed} ({seed_selection}).')
    use_cuda = args.gpu not in ('-1', 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    one_based = bool(int(args.one_based_ids))


    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        base_dir = str(model_root() / args.dataset)
        ckpt_path = _auto_find_best_ckpt(base_dir)
    model, ckpt_args = load_checkpoint_model(ckpt_path, device, args.dataset)
    dataset = ckpt_args['dataset']
    H, J, length_T = model.H, model.J, model.T
    enc_k_spa = int(ckpt_args.get('enc_k_spa', 10))
    data_subdir = ckpt_args.get('data_subdir', '') if args.data_subdir is None else args.data_subdir
    use_pos = bool(ckpt_args.get('pos_enc', 1))


    nb_nodes = J
    T = length_T
    nhood = 1; global_att = False
    batch_size = int(args.batch_size)


    (_X_train_J_tmp, _, _, _, _, _y_train_tmp,
     _X_test_J_tmp,  _, _, _, _, _y_test_tmp,
     _adj_J_tmp,     _,  pos_enc_ori,
     *rest_) = process.gen_train_data(
        dataset=dataset, split=args.probe, time_step=T,
        nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
        batch_size=batch_size, enc_k=int(enc_k_spa), data_subdir=data_subdir,
        sampling_seed=run_seed)
    L_eig_np = resolve_spatial_pos_encoding(model, pos_enc_ori)
    identity_ids = _load_identity_ids(dataset, T, data_subdir)


    if args.proto_source == 'gallery':
        (_X_train_J_g, _, _, _, _, _y_train_g,
         X_gal_J,     _, _, _, _, y_gal,
         _, _, _pos_enc_g, *_) = process.gen_train_data(
            dataset=dataset, split='gallery', time_step=T,
            nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
            batch_size=batch_size, enc_k=int(enc_k_spa), data_subdir=data_subdir,
            sampling_seed=run_seed)
        lib_labels = _decode_known_labels(y_gal, identity_ids)
        lib_feats = _extract_seq_feats(model, X_gal_J, L_eig_np, batch_size, device, T, J, H,
                                       use_pos=use_pos, bn_freeze=True)
        lib_src = 'gallery'
    else:
        (X_train_J, _, _, _, _, y_train,
         _X_test_J_unused,  _, _, _, _, _y_test_unused,
         _adj_J_unused,     _,  _pos_enc_unused,
         *_) = process.gen_train_data(
            dataset=dataset, split=args.probe, time_step=T,
            nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
            batch_size=batch_size, enc_k=int(enc_k_spa), data_subdir=data_subdir,
            sampling_seed=run_seed)
        y_train_int = _decode_known_labels(y_train, identity_ids)
        lib_labels = y_train_int
        lib_feats = _extract_seq_feats(model, X_train_J, L_eig_np, batch_size, device, T, J, H,
                                       use_pos=use_pos, bn_freeze=True)
        lib_src = 'train'


    if args.proto_source == 'gallery':
        (X_train_J_oos, _, _, _, _, _y_train_oos,
         _X_test_J_oos,  _, _, _, _, _y_test_oos,
         _adj_J_oos,     _,  _pos_enc_oos,
         *_) = process.gen_train_data(
            dataset=dataset, split=args.probe, time_step=T,
            nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
            batch_size=batch_size, enc_k=int(enc_k_spa), data_subdir=data_subdir,
            sampling_seed=run_seed)
        oos_lib_feats = _extract_seq_feats(model, X_train_J_oos, L_eig_np, batch_size, device, T, J, H,
                                           use_pos=use_pos, bn_freeze=True)
        oos_lib_labels = _decode_known_labels(_y_train_oos, identity_ids)
    else:
        oos_lib_feats = _extract_seq_feats(model, X_train_J, L_eig_np, batch_size, device, T, J, H,
                                           use_pos=use_pos, bn_freeze=True)
        oos_lib_labels = y_train_int
    oos_lib_src = 'train'


    spans = None
    nan_win_idxs_raw = []
    if args.data_source == 'unknown':

        (_XtrJ, _XtrP, _XtrB, _z0, _z1, _ytr,
         X_unk_J,  _XunkP, _XunkB, _z2, _z3, y_unk,
         _adj_unused, _unusedA, _pos_enc_unused, *_) = process.gen_train_data(
            dataset=dataset, split='unknown', time_step=T,
            nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
            batch_size=batch_size, enc_k=int(enc_k_spa), data_subdir=data_subdir,
            sampling_seed=run_seed)
        X_probe = X_unk_J  # (Nw,T,J,3)
        probe_desc = "split='unknown'"


        if isinstance(y_unk, np.ndarray) and y_unk.ndim > 1:
            row_sum = y_unk.sum(axis=1)
            print(f"[debug] unknown all-zero rows: {(row_sum==0).sum()}/{y_unk.shape[0]}")
        print("[debug] X_probe(shape), y_unk(shape):", X_probe.shape, (y_unk.shape if isinstance(y_unk, np.ndarray) else None))
    else:
        if not args.input:
            raise ValueError("--input is required when data_source=npy")
        X_probe, spans, nan_win_idxs_raw = _prepare_npy_as_segments(
            args.input, J=J, T=T,
            stride=int(args.stride), sliding=bool(int(args.sliding)),
            center_root=bool(int(args.center_root)), root_idx=int(args.root_idx)
        )
        probe_desc = f"npy:{os.path.basename(args.input)}"


    feats = _extract_seq_feats(model, X_probe, L_eig_np, batch_size, device, T, J, H,
                               use_pos=use_pos, bn_freeze=True)  # [Nw,H]
    dist_m = _compute_dist_matrix(feats, lib_feats,
                                  metric=args.metric_override,
                                  l2norm_feats=bool(int(args.l2norm_features)))  # [Nw,Ng]

    use_name_map = (dataset.upper() == 'WUT')
    topk = min(max(1, int(args.topk)), len(np.unique(lib_labels)))

    print(f"\n== Predictions completed: {probe_desc} ==")
    if args.data_source == 'npy':
        print(f"  dataset = {dataset} | J = {J} | T = {T} | cleaned windows = {len(X_probe)} | "
              f"ckpt = {os.path.relpath(ckpt_path)} | retrieval library = {lib_src} | OOS library = {oos_lib_src} | "
              f"metric = {args.metric_override} | center_root={int(args.center_root)}(root_idx={int(args.root_idx)})")
    else:
        print(f"  dataset = {dataset} | J = {J} | T = {T} | unknown windows = {len(X_probe)} | "
              f"ckpt = {os.path.relpath(ckpt_path)} | retrieval library = {lib_src} | OOS library = {oos_lib_src} | "
              f"metric = {args.metric_override}")


    Nw = len(X_probe)
    win_topk_ids = np.zeros((Nw, topk), dtype=np.int64)
    win_topk_dists = np.zeros((Nw, topk), dtype=np.float32)
    win_top1_ids = np.zeros((Nw,), dtype=np.int64)
    win_top1_dists = np.zeros((Nw,), dtype=np.float32)

    lib_labels_int = lib_labels.astype(int)
    for i in range(Nw):
        topk_pairs = _topk_by_identity(dist_m[i], lib_labels_int, topk)
        for kidx, (cid, cd) in enumerate(topk_pairs):
            win_topk_ids[i, kidx] = int(cid)
            win_topk_dists[i, kidx] = float(cd)
        win_top1_ids[i] = int(topk_pairs[0][0])
        win_top1_dists[i] = float(topk_pairs[0][1])


    Z_lib = _l2n(oos_lib_feats.astype(np.float32))
    Z_win = _l2n(feats.astype(np.float32))
    _rK_lib, _tau_sigma_cfg = _kth_nn_tau_from_lib(Z_lib, k=int(args.unk_knn_k), sigma_q=float(args.unk_sigma))
    dK_win = _kth_nn_dist_queries(Z_win, Z_lib, k=int(args.unk_knn_k))  # [Nw]

    win_oos_mask = None
    if int(args.unk_detect) == 1 and args.unk_method == 'knn':
        if args.unk_scope == 'window':
            win_oos_mask = (dK_win > _tau_sigma_cfg)
            print(f"[OOS/window] lib={oos_lib_src}, k={int(args.unk_knn_k)}, σ={float(args.unk_sigma):.3f}, "
                  f"τ={_tau_sigma_cfg:.4f} | median(d^K)={np.median(dK_win):.4f} | "
                  f"unknown_windows={int(win_oos_mask.sum())}/{len(dK_win)}")
        else:
            seq_decision = ('known' if float(np.median(dK_win)) <= float(_tau_sigma_cfg) else 'unknown')
            print(f"[OOS/sequence] lib={oos_lib_src}, k={int(args.unk_knn_k)}, σ={float(args.unk_sigma):.3f}, "
                  f"τ={_tau_sigma_cfg:.4f} | median(d^K)={np.median(dK_win):.4f} -> {seq_decision}")


    for i in range(Nw):
        pairs_txt = ", ".join(
            [f"{_idx_to_name(k)}({int(k) + int(one_based)}):{v:.3f}" for k, v in zip(win_topk_ids[i], win_topk_dists[i])]
        ) if use_name_map else ", ".join([f"{int(k) + int(one_based)}:{v:.3f}" for k, v in zip(win_topk_ids[i], win_topk_dists[i])])
        flag = (" | OOS" if (win_oos_mask is not None and win_oos_mask[i]) else "")
        if spans is not None:
            s, e = spans[i].tolist()
            if use_name_map:
                print(f"[{i:04d}] frames {s:05d}-{e:05d}  =>  pred={_idx_to_name(win_top1_ids[i])}({int(win_top1_ids[i]) + int(one_based)}) "
                      f"| top{topk} (dist): {pairs_txt}{flag}")
            else:
                print(f"[{i:04d}] frames {s:05d}-{e:05d}  =>  pred={int(win_top1_ids[i]) + int(one_based)} "
                      f"| top{topk} (dist): {pairs_txt}{flag}")
        else:
            if use_name_map:
                print(f"[{i:04d}]  pred={_idx_to_name(win_top1_ids[i])}({int(win_top1_ids[i]) + int(one_based)}) "
                      f"| top{topk} (dist): {pairs_txt}{flag}")
            else:
                print(f"[{i:04d}]  pred={int(win_top1_ids[i]) + int(one_based)} | top{topk} (dist): {pairs_txt}{flag}")


    oos_metrics = None
    roc_pair = None
    if int(args.eval_oos) == 1:
        oos_metrics, roc_pair = eval_oos_metrics(_rK_lib, dK_win, sigma_eval=float(args.sigma_eval))
        print("\n== OOS detection metrics ==")
        print(f"  k = {int(args.unk_knn_k)} | σ_eval = {float(args.sigma_eval):.2f}")
        print(f"  tau(σ)   = {oos_metrics['tau_eval']:.6f}")
        print(f"  TPR_ID   = {oos_metrics['TPR_ID_at_tau']*100:.2f}%")
        print(f"  FPR95(OOS)= {oos_metrics['FPR95']*100:.2f}%")
        print(f"  AUROC    = {oos_metrics['AUROC']:.4f}")


    gt_id, gt_human = _parse_groundtruth(args.groundtruth, dataset, one_based)
    if gt_id is not None:
        correct = int((win_top1_ids == int(gt_id)).sum()); total = int(Nw)
        acc = correct / max(1, total)
        if use_name_map:
            print(f"\n== Sequence-level Top-1 accuracy ==\n  GT: {gt_human}({int(gt_id) + int(one_based)}) | "
                  f"Correct/Total = {correct}/{total} | Acc = {acc:.4f}")
        else:
            print(f"\n== Sequence-level Top-1 accuracy ==\n  GT: {int(gt_id) + int(one_based)} | Correct/Total = {correct}/{total} | Acc = {acc:.4f}")
    elif args.groundtruth and args.groundtruth.strip():
        print(f"\n[Info] groundtruth='{args.groundtruth}' cannot be parsed (dataset={dataset}); skipping accuracy evaluation.")


    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            header = []
            if spans is not None:
                header += ['index', 'frame_start', 'frame_end']
            else:
                header += ['index']
            if use_name_map:
                header += ['pred', 'pred_name', 'pred_dist']
                for k in range(topk): header += [f'top{k+1}_id', f'top{k+1}_name', f'top{k+1}_dist']
            else:
                header += ['pred', 'pred_dist']
                for k in range(topk): header += [f'top{k+1}_id', f'top{k+1}_dist']
            if gt_id is not None:
                header += ['gt_id']
                if use_name_map: header += ['gt_name']
                header += ['hit_top1']
            if win_oos_mask is not None: header += ['oos_window']
            w.writerow(header)

            for i in range(Nw):
                row = []
                if spans is not None:
                    s, e = spans[i].tolist(); row += [i, s, e]
                else:
                    row += [i]
                row += [int(win_top1_ids[i]) + int(one_based)]
                if use_name_map: row += [_idx_to_name(win_top1_ids[i])]
                row += [float(win_top1_dists[i])]
                for k in range(topk):
                    cid = int(win_topk_ids[i, k]); cd = float(win_topk_dists[i, k])
                    if use_name_map: row += [cid + int(one_based), _idx_to_name(cid), cd]
                    else:            row += [cid + int(one_based), cd]
                if gt_id is not None:
                    if use_name_map: row += [int(gt_id) + int(one_based), _idx_to_name(gt_id)]
                    else:            row += [int(gt_id) + int(one_based)]
                    row += [int(int(win_top1_ids[i]) == int(gt_id))]
                if win_oos_mask is not None: row += [int(bool(win_oos_mask[i]))]
                w.writerow(row)
        print(f"\nSaved CSV -> {args.save_csv}")


    if args.save_npy_base:
        base = args.save_npy_base.rstrip("/")
        out_dir = os.path.dirname(base)
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        win_ids_to_save = win_top1_ids.copy()
        win_topk_ids_to_save = win_topk_ids.copy()
        if one_based:
            win_ids_to_save = win_ids_to_save + 1
            win_topk_ids_to_save = win_topk_ids_to_save + 1


        if win_oos_mask is not None:
            unk_write_id = _unknown_identity_id(dataset) + int(one_based)
            win_ids_to_save[win_oos_mask] = unk_write_id

        np.save(f"{base}_win_id.npy", win_ids_to_save.astype(np.int64))
        np.save(f"{base}_win_dist.npy", win_top1_dists.astype(np.float32))
        np.save(f"{base}_topk_id.npy", win_topk_ids_to_save.astype(np.int64))
        np.save(f"{base}_topk_dist.npy", win_topk_dists.astype(np.float32))

        if win_oos_mask is not None:
            np.save(f"{base}_win_oos_mask.npy", win_oos_mask.astype(np.uint8))


        if args.data_source == 'npy' and spans is not None:
            Nf = int(spans[:, 1].max() + 1)
            unknown_id0 = _unknown_identity_id(dataset)

            win_ids_for_vote = win_top1_ids.copy()
            if win_oos_mask is not None:
                win_ids_for_vote[win_oos_mask] = unknown_id0

            frame_ids = _majority_vote_per_frame(
                spans=spans,
                win_ids=win_ids_for_vote,
                win_dists=win_top1_dists,
                one_based=one_based,
                n_frames=Nf
            )
            np.save(f"{base}_frame_id.npy", frame_ids.astype(np.int64))

            if win_oos_mask is not None:
                frame_oos = np.zeros(Nf, dtype=np.uint8)
                for i, (s, e) in enumerate(spans):
                    if win_oos_mask[i]: frame_oos[s:e+1] = 1
                np.save(f"{base}_frame_oos_mask.npy", frame_oos)


        summary = {
            "randomness": {"seed": run_seed, "requested_seed": args.seed,
                           "selection": seed_selection},
            "identity_labels": {
                "indexing": "one_based" if one_based else "zero_based",
                "known_original_ids": (identity_ids + int(one_based)).tolist(),
                "unknown_id": _unknown_identity_id(dataset) + int(one_based),
            },
            "spatial_position_encoding": model.spatial_pos_encoding_metadata,
            "data_source": args.data_source,
            "probe_desc": probe_desc,
            "unk_detect_enabled": bool(int(args.unk_detect)),
            "unk_method": args.unk_method,
            "unk_scope": args.unk_scope,
            "oos_lib_source": oos_lib_src,
            "k": int(args.unk_knn_k),
            "sigma_for_oos": float(args.unk_sigma),
            "tau_for_oos": float(_tau_sigma_cfg),
            "num_windows": int(Nw),
            "num_oos_windows": (int(win_oos_mask.sum()) if win_oos_mask is not None else 0),
        }
        if args.data_source == 'npy':
            summary["nan_window_indices_raw"] = nan_win_idxs_raw
        if int(args.eval_oos) == 1 and oos_metrics is not None:
            summary.update({
                "eval_sigma": float(args.sigma_eval),
                "eval_tau": float(oos_metrics["tau_eval"]),
                "TPR_ID_at_tau": float(oos_metrics["TPR_ID_at_tau"]),
                "FPR95": float(oos_metrics["FPR95"]),
                "AUROC": float(oos_metrics["AUROC"]),
            })
        with open(f"{base}_unk_verdict.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] Saved:\n"
              f"  {base}_win_id.npy / _win_dist.npy / _topk_id.npy / _topk_dist.npy\n"
              f"  {base}_win_oos_mask.npy (if OOS detection is enabled)\n"
              f"  {base}_frame_id.npy / _frame_oos_mask.npy (only with data_source=npy)\n"
              f"  {base}_unk_verdict.json")


    if int(args.eval_oos) == 1:
        tau95 = float(np.quantile(_rK_lib, q=float(args.sigma_eval)))
        rng = np.random.default_rng(run_seed)
        x_id  = rng.normal(loc=0.0, scale=0.02, size=_rK_lib.shape[0])
        x_oos = rng.normal(loc=1.0, scale=0.02, size=dK_win.shape[0])

        plt.figure(figsize=(6.0, 4.2))
        plt.scatter(x_id,  _rK_lib, s=8, alpha=0.6, label='ID')
        plt.scatter(x_oos, dK_win,  s=8, alpha=0.6, label='OOS')
        plt.axhline(y=tau95, linestyle='--', linewidth=1.2)
        plt.text(1.02, tau95, f' τ(σ={args.sigma_eval:.2f})', va='center', fontsize=9)
        plt.xticks([0,1], ['ID', 'OOS'])
        plt.ylabel(f'k-NN distance (k={int(args.unk_knn_k)})')
        plt.title('ID vs OOS distance distribution')
        plt.grid(alpha=0.2, axis='y'); plt.legend(loc='best'); plt.tight_layout()

        plot_path = args.save_plot or ((args.save_npy_base.rstrip("/") if args.save_npy_base else "oos") + "_oos_scatter.png")
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        plt.savefig(plot_path, dpi=160); plt.close()
        print(f"[Plot] Saved distance scatter -> {plot_path}")


        s_id  = tau95 - _rK_lib
        s_oos = tau95 - dK_win
        plt.figure(figsize=(6.2, 4.2))
        bins = 48
        plt.hist(s_id,  bins=bins, density=True, alpha=0.55, label='ID')
        plt.hist(s_oos, bins=bins, density=True, alpha=0.55, label='OOS')
        plt.axvline(0.0, linestyle='--', linewidth=1.2, label='decision boundary (s=0)')
        plt.xlabel(r'score  $s=\tau - d^K$'); plt.ylabel('density')
        plt.title(f'ID vs OOS on score (k={int(args.unk_knn_k)}, σ={float(args.sigma_eval):.2f})')
        plt.legend(loc='best'); plt.grid(alpha=0.2); plt.tight_layout()

        score_plot = _score_histogram_path(args.save_plot, args.save_npy_base)
        os.makedirs(os.path.dirname(score_plot) or ".", exist_ok=True)
        plt.savefig(score_plot, dpi=170); plt.close()
        print(f"[Plot] Saved score histogram -> {score_plot}")


    if int(args.proj2d) == 1 and int(args.eval_oos) == 1:
        tau_vis = float(np.quantile(_rK_lib, q=float(args.sigma_eval)))
        auroc_val = float(oos_metrics['AUROC']) if oos_metrics is not None else None


        if int(args.plot_fisher_all) == 1 and str(args.data_source).lower() == 'unknown':
            id_fn_mask = (_rK_lib > tau_vis)
            unk_fp_mask = (dK_win <= tau_vis)
            bnd_base = (args.save_npy_base.rstrip("/") if args.save_npy_base else "oos")
            fisher_all_path = f"{bnd_base}_oos_fisher_all_unknown.png"
            plot_fisher_id_vs_unknown_2d(
                Z_id=oos_lib_feats,
                Z_unk=feats,
                id_fn_mask=id_fn_mask,
                unk_fp_mask=unk_fp_mask,
                path=fisher_all_path,
                standardize=bool(int(args.proj2d_standardize)),
                whiten=bool(int(args.proj2d_whiten)),
                auroc=auroc_val,
                plot_seed=run_seed,
            )
            n_fp = int(unk_fp_mask.sum())
            n_tn = int((~unk_fp_mask).sum())
            n_fn = int(id_fn_mask.sum())
            print(f"[Fisher] Saved all-unknown projection -> {fisher_all_path}")
            print(f"  @τ(σ={float(args.sigma_eval):.2f}): "
                  f"ID-FN={n_fn}/{len(id_fn_mask)}, "
                  f"OOS-FP={n_fp}/{len(unk_fp_mask)}, OOS-TN={n_tn}/{len(unk_fp_mask)}")


        if int(args.plot_id_oos) == 1 and str(args.data_source).lower() == 'unknown':
            from visualization import plot_id_clusters_with_oos_2d as plot_id_oos_shared, save_proj_config
            base = (args.save_npy_base.rstrip("/") if args.save_npy_base else "oos")
            id_oos_path = f"{base}_id_clusters_with_oos_{args.id_oos_method}.png"
            proj_kw = dict(
                method=str(args.id_oos_method),
                per_id_max=int(args.id_oos_per_id_max),
                oos_max=int(args.id_oos_oos_max),
                feat_mode=str(args.id_oos_feat_mode),
                perplexity=float(args.proj2d_perplexity),
                plot_seed=run_seed,
                annotate_centroids=True,
            )
            sep_metrics = plot_id_oos_shared(
                Z_id=oos_lib_feats,
                id_labels=oos_lib_labels,
                Z_oos=feats,
                path=id_oos_path,
                name_fn=(_idx_to_name if use_name_map else None),
                auroc=auroc_val,
                title_prefix="SSGT",
                **proj_kw,
            )
            proj_cfg_path = f"{base}_id_oos_proj_config.json"
            save_proj_config(
                proj_cfg_path,
                sep_metrics["sel_id_idx"],
                sep_metrics["oos_idx"],
                proj_kw,
                sep_metrics["xlim"],
                sep_metrics["ylim"],
                source="evaluation",
                extra={"plot": id_oos_path, "auroc": auroc_val},
            )
            from visualization import save_proj_xy
            xy_path = f"{base}_id_oos_proj_xy.npz"
            save_proj_xy(xy_path, sep_metrics["XY_id"], sep_metrics["XY_oos"], sep_metrics["lab_id_s"])
            print(f"[ID/OOS] Saved identity/unknown overlay -> {id_oos_path}")
            print(f"[ID/OOS] Saved projection configuration -> {proj_cfg_path}")
            print(f"[ID/OOS] Saved projected coordinates -> {xy_path}")
            if sep_metrics:
                msg = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sep_metrics.items()
                    if k not in ("sel_id_idx", "oos_idx", "xlim", "ylim", "XY_id", "XY_oos", "lab_id_s")
                )
                print(f"  Separation: {msg}  (higher within-ID means better identity separation; higher ID-vs-OOS means better unknown separation)")


        oos_mask = (win_oos_mask.astype(bool) if win_oos_mask is not None else None)
        if oos_mask is None or oos_mask.sum() == 0:
            print("[2D] No probe windows were rejected as OOS; skipping the subset projection and boundary.")
        else:
            Z_id  = oos_lib_feats
            Z_oos = feats[oos_mask]


            max_total = int(args.proj2d_max_points)
            n_id, n_oos = Z_id.shape[0], Z_oos.shape[0]
            half = max_total // 2
            take_id  = min(n_id, max(half, max_total - min(n_oos, half)))
            take_oos = min(n_oos, max_total - take_id)
            idx_id  = _sample_idx(n_id, take_id, seed=run_seed)
            idx_oos = _sample_idx(n_oos, take_oos, seed=(run_seed + 1) % (2 ** 32))
            Z_id_s  = Z_id[idx_id]
            Z_oos_s = Z_oos[idx_oos]


            Z_id_s, Z_oos_s = _preprocess_for_proj(
                Z_id_s, Z_oos_s,
                standardize=bool(int(args.proj2d_standardize)),
                whiten=bool(int(args.proj2d_whiten))
            )

            X2 = np.vstack([Z_id_s, Z_oos_s])
            y2 = np.hstack([np.zeros(Z_id_s.shape[0], dtype=np.int32),
                            np.ones( Z_oos_s.shape[0], dtype=np.int32)])

            method = args.proj2d_method.lower()
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=run_seed)
                XY = reducer.fit_transform(X2); method_title = 'PCA'
            elif method == 'tsne':
                perp = float(args.proj2d_perplexity)
                max_perp = max(5.0, min(perp, X2.shape[0]-1))
                reducer = make_tsne(n_components=2, init='pca', learning_rate='auto',
                               perplexity=max_perp, random_state=run_seed, n_iter=1000, verbose=0)
                XY = reducer.fit_transform(X2); method_title = 't-SNE'
            elif method == 'fisher':
                XY, w_f, v2_f = _fisher_plane(Z_id_s, Z_oos_s, eps=1e-4); method_title = 'FISHER'
            elif method == 'svm':
                XY, w_s, b_s, v2_s = _svm_plane(Z_id_s, Z_oos_s, C=float(args.svm_C), seed=run_seed); method_title = 'Linear-SVM'
            else:
                raise ValueError(f"Unknown proj2d_method: {method}")

            XY_id  = XY[y2 == 0]
            XY_oos = XY[y2 == 1]


            plt.figure(figsize=(5.8, 5.0))
            plt.scatter(XY_id[:,0],  XY_id[:,1],  s=8, alpha=0.65, label='ID')
            plt.scatter(XY_oos[:,0], XY_oos[:,1], s=8, alpha=0.65, label='OOS')
            if method == 'svm':
                x0 = -b_s
                yl, yh = XY[:,1].min(), XY[:,1].max()
                plt.plot([x0, x0], [yl, yh], linestyle='--', linewidth=1.4, label='SVM boundary')
            plt.title(f'2D projection ({method_title}) of ID vs OOS (probe)')
            plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
            plt.grid(alpha=0.25); plt.legend(loc='best'); plt.tight_layout()

            plot2d_path = args.save_plot2d or\
                          ( (args.save_npy_base.rstrip("/") if args.save_npy_base else "oos")
                            + f"_oos_proj2d_{method}.png" )
            os.makedirs(os.path.dirname(plot2d_path) or ".", exist_ok=True)
            plt.savefig(plot2d_path, dpi=170); plt.close()
            print(f"[2D] Saved projection -> {plot2d_path}")


            bnd_base = (args.save_npy_base.rstrip("/") if args.save_npy_base else "oos")
            bnd_path = f"{bnd_base}_oos_boundary_{method}.png"
            id_labels_2d = np.asarray(oos_lib_labels)[idx_id]
            tau2d = plot_knn_boundary_2d(
                XY_id=XY_id, XY_oos=XY_oos,
                id_labels=id_labels_2d,
                sigma=float(args.sigma_eval),
                k=int(args.unk_knn_k),
                path=bnd_path,
                method_name=method_title,
                grid=int(args.boundary_grid),
                metric=str(args.boundary_metric),
                id_core_q=float(args.boundary_id_core_q),
                draw_hull=bool(int(args.boundary_draw_hull)),
                plot_seed=run_seed,
            )
            print(f"[2D] Saved decision boundary -> {bnd_path}  (τ2D={tau2d:.4f}, σ={float(args.sigma_eval):.2f})")


            print("Lib classes:", sorted(set(lib_labels.tolist())))
            try:
                if isinstance(y_unk, np.ndarray):
                    if y_unk.ndim > 1:
                        sums = y_unk.sum(axis=1)
                        y_unk_decoded = np.where(sums == 0, -1, identity_ids[np.argmax(y_unk, axis=1)])
                    else:
                        y_unk_decoded = y_unk
                    inter = set(lib_labels.tolist()) & set([int(t) for t in y_unk_decoded.tolist() if int(t) >= 0])
                    print("Intersection with unknown:", inter)
                else:
                    print("Unknown split label check skipped: y_unk is None or not ndarray")
            except Exception as e:
                print("Unknown split label check skipped:", e)


            D = _compute_dist_matrix(feats, lib_feats, metric=args.metric_override,
                                     l2norm_feats=bool(int(args.l2norm_features)))
            d_sorted = np.sort(D, axis=1)
            d1, d2 = d_sorted[:, 0], d_sorted[:, 1]
            print("d1/d2 median:", np.median(d1 / np.maximum(d2, 1e-6)))

if __name__ == '__main__':
    main()
