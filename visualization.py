"""Reusable identity/OOS projection figures and their serialized metadata.

This module preserves the sampling and plotting behavior of the original
id_oos_plot_utils.py without importing application entry points."""

from typing import Optional, Tuple, Dict, Any
import os
import json
import inspect
from randomness import resolve_seed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =====================================================================
# Feature Preparation and Projection
# =====================================================================

def make_tsne(**kwargs):
    """Construct t-SNE across scikit-learn versions with renamed iterations."""
    if "n_iter" in kwargs and "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = kwargs.pop("n_iter")
    return TSNE(**kwargs)

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


def _sample_idx(n: int, k: int, seed: Optional[int] = None):
    seed = resolve_seed(seed)
    if n <= k: return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)


def subsample_id_oos_indices(
    id_labels: np.ndarray,
    n_id: int,
    n_oos: int,
    per_id_max: int = 120,
    oos_max: int = 800,
    plot_seed: Optional[int] = None,
):
    """Select balanced known-identity samples and a seeded unknown subset."""
    plot_seed = resolve_seed(plot_seed)
    id_labels = np.asarray(id_labels).astype(np.int64)
    uniq = np.unique(id_labels)
    rng = np.random.default_rng(plot_seed)
    sel_id_idx = []
    for c in uniq:
        idx_c = np.where(id_labels == c)[0]
        if idx_c.size > per_id_max:
            idx_c = rng.choice(idx_c, size=per_id_max, replace=False)
        sel_id_idx.append(idx_c)
    sel_id_idx = np.concatenate(sel_id_idx)
    oos_idx = _sample_idx(n_oos, oos_max, (plot_seed + 7) % (2 ** 32)) if n_oos > 0 else np.array([], dtype=np.int64)
    return sel_id_idx, oos_idx


def _l2norm_rows(a: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0:
        return a.astype(np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / np.clip(n, 1e-12, None)).astype(np.float32)


def preprocess_features(Z_id_s: np.ndarray, Z_oos_s: np.ndarray, feat_mode: str = "l2norm"):
    if feat_mode == "whiten":
        return _preprocess_for_proj(Z_id_s, Z_oos_s, standardize=True, whiten=True)
    return _l2norm_rows(Z_id_s.astype(np.float32)), _l2norm_rows(Z_oos_s.astype(np.float32))


def project_id_oos_2d(
    Z_id_s: np.ndarray,
    Z_oos_s: np.ndarray,
    method: str = "tsne",
    feat_mode: str = "l2norm",
    perplexity: float = 30.0,
    plot_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Project sampled features using the same preprocessing as OOS evaluation."""
    plot_seed = resolve_seed(plot_seed)
    Z_id_p, Z_oos_p = preprocess_features(Z_id_s, Z_oos_s, feat_mode=feat_mode)
    X = np.vstack([Z_id_p, Z_oos_p]) if Z_oos_p.shape[0] > 0 else Z_id_p
    n_id_pts = Z_id_p.shape[0]

    m = method.lower()
    if m == "tsne":
        perp = max(5.0, min(float(perplexity), X.shape[0] - 1))
        lr = max(50.0, X.shape[0] / 12.0)
        reducer = make_tsne(
            n_components=2, init="pca", learning_rate=lr,
            perplexity=perp, random_state=plot_seed, n_iter=1000, verbose=0,
        )
        XY = reducer.fit_transform(X)
        method_title = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=plot_seed)
        XY = reducer.fit_transform(X)
        method_title = "PCA"

    return XY[:n_id_pts], XY[n_id_pts:], method_title


def compute_axis_limits(XY_id: np.ndarray, XY_oos: np.ndarray, pad_ratio: float = 0.06):
    """Compute shared axis limits from robust point-cloud quantiles."""
    pts = XY_id
    if XY_oos.shape[0] > 0:
        pts = np.vstack([pts, XY_oos])
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * pad_ratio
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


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
    title_prefix: str = "",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    sel_id_idx: Optional[np.ndarray] = None,
    oos_idx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Project known identities and unknown probes into one diagnostic figure.

    Args:
        Z_id (np.ndarray): Known feature matrix of shape (N, H).
        id_labels (np.ndarray): Identity labels aligned with Z_id.
        Z_oos (np.ndarray): Unknown feature matrix of shape (M, H).
        path (str): Destination image path.

    Returns:
        dict: Silhouette statistics and, in the shared plotting helper,
        reusable sampling indices, projected coordinates, and axis limits."""
    plot_seed = resolve_seed(plot_seed)
    id_labels = np.asarray(id_labels).astype(np.int64)
    uniq = np.unique(id_labels)

    if sel_id_idx is None or oos_idx is None:
        sel_id_idx, oos_idx = subsample_id_oos_indices(
            id_labels, Z_id.shape[0], Z_oos.shape[0],
            per_id_max=per_id_max, oos_max=oos_max, plot_seed=plot_seed,
        )

    Z_id_s = Z_id[sel_id_idx]
    lab_id_s = id_labels[sel_id_idx]
    Z_oos_s = Z_oos[oos_idx] if oos_idx.size > 0 else np.zeros((0, Z_id.shape[1]), dtype=Z_id.dtype)

    XY_id, XY_oos, method_title = project_id_oos_2d(
        Z_id_s, Z_oos_s, method=method, feat_mode=feat_mode,
        perplexity=perplexity, plot_seed=plot_seed,
    )

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
                    silhouette_score(
                        np.vstack([XY_id, XY_oos]) if XY_oos.shape[0] > 0 else XY_id,
                        bin_lab, metric="euclidean"))
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

    prefix = f"{title_prefix}: " if title_prefix else ""
    title = f"{prefix}{method_title}: per-identity clusters + OOS overlay"
    if sub:
        title += "\n" + " | ".join(sub)
    plt.title(title, fontsize=12)
    plt.xlabel(f"{method_title} dim 1")
    plt.ylabel(f"{method_title} dim 2")
    plt.grid(alpha=0.22, linestyle="--", linewidth=0.6)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ncol = 2 if len(uniq) > 14 else 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=True, framealpha=0.9, ncol=ncol, borderaxespad=0.3,
              labelspacing=0.3, handletextpad=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=360, bbox_inches="tight")
    plt.close()

    auto_xlim, auto_ylim = compute_axis_limits(XY_id, XY_oos)
    metrics["plot_seed"] = plot_seed
    metrics["xlim"] = list(xlim if xlim is not None else auto_xlim)
    metrics["ylim"] = list(ylim if ylim is not None else auto_ylim)
    metrics["sel_id_idx"] = sel_id_idx.tolist()
    metrics["oos_idx"] = oos_idx.tolist()
    metrics["XY_id"] = XY_id
    metrics["XY_oos"] = XY_oos
    metrics["lab_id_s"] = lab_id_s
    return metrics


def save_proj_xy(path: str, XY_id: np.ndarray, XY_oos: np.ndarray, lab_id_s: np.ndarray) -> str:
    """Save projected coordinates and labels for reproducible figures."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        XY_id=np.asarray(XY_id, dtype=np.float32),
        XY_oos=np.asarray(XY_oos, dtype=np.float32),
        lab_id_s=np.asarray(lab_id_s, dtype=np.int64),
    )
    return path


def save_proj_config(
    path: str,
    sel_id_idx,
    oos_idx,
    proj: dict,
    xlim,
    ylim,
    source: str = "evaluation",
    extra: Optional[dict] = None,
) -> str:
    """Save sampling indices, projection parameters, and axis limits as JSON."""
    cfg = {
        "source": source,
        "sel_id_idx": np.asarray(sel_id_idx, dtype=np.int64).tolist(),
        "oos_idx": np.asarray(oos_idx, dtype=np.int64).tolist(),
        "proj": dict(proj),
        "axis_limits": {
            "xlim": [float(xlim[0]), float(xlim[1])],
            "ylim": [float(ylim[0]), float(ylim[1])],
        },
    }
    if extra:
        cfg.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path

