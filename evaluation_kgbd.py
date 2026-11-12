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
    $ python evaluation_kgbd.py --dataset KGBD --ckpt weight/model/best.ckpt
"""

import os
import glob
import argparse
import json
from typing import Tuple, Optional
from pathlib import Path
from paths import model_root, output_root
from randomness import resolve_seed

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from visualization import make_tsne
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


import data as process
from evaluation import (
    _idx_to_name,
    _load_identity_ids,
    _decode_known_labels,
    _unknown_identity_id,
    _parse_groundtruth,
    set_seed,
    _dataset_joint_num,
    _auto_find_best_ckpt,
    _prepare_npy_as_segments,
    _fisher_plane,
    _svm_plane,
    _preprocess_for_proj,
    _l2n,
    _kth_nn_tau_from_lib,
    _kth_nn_dist_queries,
    eval_oos_metrics,
    _extract_seq_feats,
    _compute_dist_matrix,
    _topk_by_identity,
    _sample_idx,
    _majority_vote_per_frame,
    _score_histogram_path,
    load_checkpoint_model,
    resolve_spatial_pos_encoding,
)


CKPT_DEFAULT = ""


DICP_WUT = {
    "Jacek": 1, "Aleks": 2, "Julia": 3, "Vibek": 4, "Mikolaj": 5, "Ania": 6, "Kasia": 7, "Martyna": 8,
    "Maria": 9, "MichalD": 10, "MichalJ": 11, "Ercem": 12, "Malgosia": 13, "Moncy": 14, "Juan": 15,
    "Erdem": 16, "Pasko": 17, "Piotr": 18, "Julianna": 19, "Showni": 20, "Weronika": 21,
    "Szykiedans": 22, "Maks": 23, "Kujawinska": 24, "Wojciech": 25, "Unknown": 26,
}
_ID0_TO_NAME = {v - 1: k for k, v in DICP_WUT.items()}
_NAME_LOWER_TO_ID0 = {k.lower(): v - 1 for k, v in DICP_WUT.items()}

# =====================================================================
# Diagnostic Projection and Visualization
# =====================================================================


def plot_knn_boundary_2d(
    XY_id: np.ndarray,
    XY_oos: np.ndarray,
    sigma: float,
    k: int,
    path: str,
    method_name: str = "PCA/TSNE/FISHER/SVM",
    grid: int = 400,
    metric: str = "euclidean",
    id_core_q: float = 1.0,
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
) -> float:
    """Save a k-nearest-neighbor quantile boundary in a diagnostic 2D plane.

    The projected threshold is independent of the high-dimensional OOS
    threshold used for classification and reported evaluation metrics.

    Returns:
        float: Quantile threshold estimated from known points in this plane."""
    if id_core_q < 1.0:
        nbrs_tmp = NearestNeighbors(n_neighbors=max(1, k), metric="euclidean").fit(XY_id)
        d_tmp, _ = nbrs_tmp.kneighbors(XY_id)
        kth_tmp = d_tmp[:, max(1, k) - 1]
        qv = np.quantile(kth_tmp, q=float(id_core_q))
        XY_id = XY_id[kth_tmp <= qv]

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

    plt.figure(figsize=(6.8, 5.4))
    oos_color = "#4169E1"
    is_color  = "#FF7F0E"

    if XY_oos is not None and len(XY_oos) > 0:
        plt.scatter(XY_oos[:, 0], XY_oos[:, 1], s=58, alpha=0.75, label="OOS",
                    color=oos_color, edgecolors="white", linewidths=0.6)
    plt.scatter(XY_id[:, 0], XY_id[:, 1], s=58, alpha=0.75, label="IS",
                color=is_color, edgecolors="white", linewidths=0.6)

    if fill_color is not None and fill_alpha > 0:
        plt.contourf(XX, YY, inside, levels=[-0.5, 0.5, 1.5],
                     colors=[fill_color, fill_color], alpha=fill_alpha, antialiased=True)

    CS = plt.contour(XX, YY, dG_k, levels=[tau2d], linewidths=4.0,
                     linestyles="--", colors=[contour_color], antialiased=True)
    if outline:
        # Matplotlib 3.8+ exposes the contour set as a single collection.
        artists = [CS] if hasattr(CS, "set_path_effects") else CS.collections
        for col in artists:
            col.set_path_effects([pe.Stroke(linewidth=6.0, foreground="white"), pe.Normal()])

    plt.xlabel("Dim 1", fontsize=axis_label_size)
    plt.ylabel("Dim 2", fontsize=axis_label_size)
    ax = plt.gca()
    ax.tick_params(axis="both", which="both",
                   labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                   labelsize=tick_label_size)
    ax.legend(loc="upper right", fontsize=legend_size, frameon=True, framealpha=0.9, fancybox=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    return tau2d


# =====================================================

# =====================================================================
# Command-Line Configuration
# =====================================================================

def get_parser():
    """Build command-line options for identity retrieval and OOS evaluation."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])


    p.add_argument('--data_source', default='unknown', choices=['unknown','npy'],
                   help='Use a preprocessed unknown split or an external skeleton NPY sequence.')
    p.add_argument('--input', '--npy_path', '--npy-path', default='', type=str,
                   help='Skeleton NPY path; required when --data_source npy is selected.')


    p.add_argument('--dataset', default='KGBD', type=str, choices=['KGBD','KS20','WUT'])
    p.add_argument('--probe',   default='probe', type=str,
                   help='Known evaluation split; use probe for WUT, KGBD, or KS20.')
    p.add_argument('--gpu', default='0', type=str, help='CUDA device index, or -1/cpu for CPU.')
    p.add_argument('--seed', default=-1, type=int,
                   help='Run seed; -1 generates and records a fresh seed. Supply that seed to replay sampling and plots.')
    p.add_argument('--ckpt', default=CKPT_DEFAULT, type=str,
                   help='Checkpoint path; when empty, search weight/<dataset> for the latest best.ckpt.')
    p.add_argument('--proto_source', default='gallery', choices=['gallery','train'],
                   help='Feature library used for identity retrieval and Top-k predictions.')


    p.add_argument('--stride', default=6, type=int, help='Frame step for sliding windows.')
    p.add_argument('--sliding', default=0, type=int, help='1: sliding windows; 0: non-overlapping windows.')
    p.add_argument('--center_root', default=1, type=int, help='Subtract the root joint from each input frame.')
    p.add_argument('--root_idx', default=0, type=int, help='Zero-based root joint index.')


    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--topk', default=5, type=int, help='Number of retrieved identities to report.')
    p.add_argument('--metric_override', default='cosine', choices=['cosine','euclidean'], help='Distance metric for identity retrieval.')
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
    else:
        oos_lib_feats = _extract_seq_feats(model, X_train_J, L_eig_np, batch_size, device, T, J, H,
                                           use_pos=use_pos, bn_freeze=True)
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
        if int(args.eval_oos) == 1:
            summary.update({
                "eval_sigma": float(args.sigma_eval),
                "eval_tau": float(np.quantile(_rK_lib, q=float(args.sigma_eval))),
                "TPR_ID_at_tau": float(( _rK_lib <= np.quantile(_rK_lib, q=float(args.sigma_eval)) ).mean()),
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


    if int(args.proj2d) == 1:
        oos_mask = (win_oos_mask.astype(bool) if win_oos_mask is not None else None)
        if oos_mask is None or oos_mask.sum() == 0:
            print("[2D] No probe windows were rejected as OOS; skipping the projection.")
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
            tau2d = plot_knn_boundary_2d(
                XY_id=XY_id, XY_oos=XY_oos,
                sigma=float(args.sigma_eval),
                k=int(args.unk_knn_k),
                path=bnd_path,
                method_name=method_title,
                grid=int(args.boundary_grid),
                metric=str(args.boundary_metric),
                id_core_q=float(args.boundary_id_core_q)
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

