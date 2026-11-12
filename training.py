"""SSGT training, prototype construction, and closed-set evaluation.

This module retains the reconstruction/contrastive objectives and implements
the published Algorithm 2 loss-weight schedule. The public entry point accepts an argument list so main.py can combine
JSON configuration with command-line overrides. Evaluation restores checkpoint
metadata and loads all model parameters strictly before extracting features.

Usage:
    $ python main.py --task train --dataset WUT
    $ python main.py --task evaluate --ckpt weight/WUT/.../best.ckpt
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as process
import paths
from SSGT import SSGT, TranSG
from loss_balancing import DynamicLossBalancer, PAPER_PROTOCOL
from retrieval import retrieval_metrics
from randomness import seed_everything


# =====================================================================
# Reproducibility and Configuration
# =====================================================================

def set_seed(seed=None, fix=False):
    """Seed all generators once and return the seed needed to replay the run.

    None or -1 selects fresh entropy. Explicit seeds, including zero, are
    honored regardless of fix, which controls deterministic cuDNN settings.
    """
    resolved = seed_everything(seed, deterministic=bool(fix))
    selection = 'automatic' if seed is None or seed == -1 else 'explicit'
    print(f'[Seed] Using seed {resolved}; selection={selection}; deterministic_cudnn={bool(fix)}.')
    return resolved


def get_args(argv=None):
    """Parse training or checkpoint-evaluation options.

    Args:
        argv (list[str] | None): Arguments without the executable name. None
            reads the process command line.

    Returns:
        argparse.Namespace: Training and evaluation settings.
    """
    p = argparse.ArgumentParser(description='Train SSGT or evaluate a saved checkpoint.')
    p.add_argument('--dataset', default='WUT', type=str, choices=['WUT', 'KGBD', 'KS20'])
    p.add_argument('--length', default=6, type=int, help='Number of frames per skeleton sequence (at least two).')
    p.add_argument('--data_subdir', default='', type=str, help='Dataset variant directory; defaults to the sequence length (for example 6_10fps).')
    p.add_argument('--lr', default=0.00035, type=float)
    p.add_argument('--probe', default='probe', type=str, help='Probe split name; the paper protocols use probe.')
    p.add_argument('--gpu', default='0', type=str, help='CUDA device visibility string; use --device cpu for CPU execution.')
    p.add_argument('--probe_type', default='', type=str)
    p.add_argument('--patience', default=500, type=int)
    p.add_argument('--max_epochs', default=0, type=int, help='Maximum zero-based epoch index (inclusive); 0 uses the original early-stopping schedule.')
    p.add_argument('--mode', default='Train', type=str, choices=['Train', 'Eval'])
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--fix_seed', default=0, type=int, choices=[0, 1], help='Enable deterministic cuDNN settings when 1; an explicit seed is honored in either mode.')
    p.add_argument('--seed_val', default=-1, type=int, help='Run seed in [0, 2**32-1]; -1 selects and records a fresh random seed.')
    p.add_argument('--H', default=128, type=int)
    p.add_argument('--n_heads', default=8, type=int)
    p.add_argument('--L_transformer', default=2, type=int)
    p.add_argument('--seq_lambda', default=0.5, type=float)
    p.add_argument('--prompt_lambda', default=0.5, type=float)
    p.add_argument('--GPC_lambda', default=0.5, type=float)
    p.add_argument('--t_1', default=0.07, type=float)
    p.add_argument('--t_2', default=14.0, type=float)
    p.add_argument('--pos_enc', default=1, type=int)
    p.add_argument('--spa_pos_type', default='laplacian', type=str, choices=['laplacian', 'learned'], help='Spatial position encoding: graph Laplacian or learned joint embeddings.')
    p.add_argument('--enc_k_spa', default=10, type=int)
    p.add_argument('--enc_k_time', default=4, type=int)
    p.add_argument('--rand_flip', default=0, type=int)
    p.add_argument('--St_mask_num', default=10, type=int)
    p.add_argument('--Tr_mask_num', default=2, type=int)
    p.add_argument('--St_prompt_type', default=3, type=int, choices=[1, 2, 3])
    p.add_argument('--T_prompt', default='l1', type=str, choices=['0', 'l1', 'l2', 'MSE'])
    p.add_argument('--S_prompt', default='l1', type=str, choices=['0', 'l1', 'l2', 'MSE'])
    p.add_argument('--pre_norm', default=1, type=int)
    p.add_argument('--attn_dropout', default=0.4, type=float)
    p.add_argument('--attn_clamp', default=1, type=int)
    p.add_argument('--metric', default='euclidean', type=str, choices=['cosine', 'euclidean'], help='Feature distance for gallery/probe retrieval.')
    p.add_argument('--frame_attn_pool', default=0, type=int)
    p.add_argument('--qkv_merged', default=1, type=int)
    p.add_argument('--use_time_attn', default=1, type=int, help='1 enables temporal attention; 0 keeps only spatial attention.')
    p.add_argument('--train_eval_gallery', default='split', type=str, choices=['train', 'split'])
    p.add_argument('--enable_dynamic_gpc', default=1, type=int, help='Enable the paper Algorithm 2 loss balancing; required for training.')
    p.add_argument('--gpc_up_period', default=10, type=int, help='Initial absolute update epoch; doubles after each update (10, 20, 40, ...).')
    p.add_argument('--gpc_iota', default=10.0, type=float, help='Threshold for the signed ratio of GPC and reconstruction loss slopes.')
    p.add_argument('--gpc_tau', default=0.05, type=float, help='Base step for updates of the GPC loss weight.')
    p.add_argument('--amp', default=0, type=int, help='1 enables CUDA automatic mixed precision; 0 disables it.')
    p.add_argument('--learnable_temp', default=0, type=int, help='1 learns contrastive temperatures; 0 uses t_1 and t_2.')
    p.add_argument('--bn_freeze_eval', default=0, type=int)
    p.add_argument('--bn_freeze_proto', default=0, type=int)
    p.add_argument('--bn_freeze_train', default=0, type=int)
    p.add_argument('--meanap_mode', default='tf', type=str, choices=['tf', 'pt'], help='Legacy compatibility selector; both values use the same filtered AP computation.')
    p.add_argument('--save_model', default=1, type=int)
    p.add_argument('--ckpt', default='', help='Checkpoint for Eval; defaults to the matching weight directory.')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                   help='Execution device; auto selects CUDA when available.')
    p.add_argument('--metrics_output', default='', help='Write closed-set Eval metrics and provenance to this JSON path.')
    args = p.parse_args(argv)
    if not -1 <= args.seed_val <= 2**32 - 1:
        p.error('--seed_val must be -1 or an integer in [0, 2**32-1]')
    if args.length < 2:
        p.error('--length must be at least 2')
    if args.batch_size < 1:
        p.error('--batch_size must be positive')
    if args.H <= 0 or args.n_heads <= 0 or args.H % args.n_heads:
        p.error('--H must be positive and divisible by --n_heads')
    if args.mode == 'Train':
        required = {'use_time_attn': 1, 'spa_pos_type': 'laplacian', 'pos_enc': 1,
                    'frame_attn_pool': 0, 'learnable_temp': 0,
                    'enable_dynamic_gpc': 1, 'train_eval_gallery': 'split'}
        for name, expected in required.items():
            if getattr(args, name) != expected:
                p.error(f'Final SSGT training requires --{name}={expected}; historical variants can be evaluated with --mode Eval.')
        if args.S_prompt == '0' or args.T_prompt == '0':
            p.error('Final SSGT training requires both spatial and trajectory reconstruction losses.')
        for name in ('t_1', 't_2'):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                p.error(f'--{name} must be finite and positive.')
        if not 1 <= args.Tr_mask_num < args.length:
            p.error('--Tr_mask_num must retain at least one frame and mask at least one frame.')
        joint_count = {'WUT': 19, 'KGBD': 20, 'KS20': 25}[args.dataset]
        if args.St_prompt_type == 3 and not 1 <= args.St_mask_num < joint_count:
            p.error('--St_mask_num must retain at least one joint and mask at least one joint.')
        try:
            DynamicLossBalancer(args.GPC_lambda, args.gpc_up_period, args.gpc_iota, args.gpc_tau)
        except ValueError as exc:
            p.error(str(exc))
    return args


# =====================================================================
# Graph Prototype Contrastive Losses
# =====================================================================

def gpc_seq_logits(seq_ftr, class_protos, t=None, logit_scale: torch.Tensor = None):
    sf = F.normalize(seq_ftr, dim=-1)
    cp = F.normalize(class_protos, dim=-1)
    logits = sf @ cp.t()
    if logit_scale is not None:
        return logits * logit_scale.exp().clamp(max=100.0)
    else:
        return logits / float(t)


def gpc_ske_loss(C_seq, class_protos, labels, T, H, t=None, gpc_f1=None, gpc_f2=None,
                 logit_scale: torch.Tensor = None):
    B = C_seq.size(0)
    all_ftr_trans = gpc_f1(C_seq)
    cluster_ftr_trans = gpc_f2(class_protos)
    logits = torch.matmul(all_ftr_trans, cluster_ftr_trans.t())
    if logit_scale is not None:
        logits = logits * logit_scale.exp().clamp(max=100.0)
    else:
        logits = logits / float(t)
    label_frames = labels.view(B, 1).expand(B, T).contiguous().view(-1)
    loss = F.cross_entropy(logits.view(B * T, -1), label_frames, reduction='none')
    loss = loss.view(B, T).sum(dim=1).mean()
    return loss


# =====================================================================
# Retrieval Metrics
# =====================================================================

def mean_ap(distmat,
            query_ids=None,
            gallery_ids=None,
            query_cams=None,
            gallery_cams=None,
            special_skip_first=False,
            mode='tf',
            remove_zero_same_id=True,
            zero_eps=1e-6):
    """Compute mean average precision after retrieval-protocol filtering.

    Args:
        distmat (numpy.ndarray): Query-to-gallery distances, shape (Nq, Ng).
        query_ids (numpy.ndarray | None): Identity labels for queries.
        gallery_ids (numpy.ndarray | None): Identity labels for gallery samples.
        query_cams (numpy.ndarray | None): Optional source/camera labels.
        gallery_cams (numpy.ndarray | None): Optional source/camera labels.
        special_skip_first (bool): Remove the nearest gallery item per query.
        mode (str): Retained for compatibility; both choices share this code.
        remove_zero_same_id (bool): Exclude near-zero-distance same-ID items.
        zero_eps (float): Distance threshold used by the duplicate filter.

    Returns:
        float: Mean AP, including AP=0 for queries without positive matches.

    Notes:
        When cameras are given, same-ID/same-camera matches are excluded. The
        near-zero-distance filter applies independently of camera metadata.
    """
    return retrieval_metrics(
        distmat, query_ids=query_ids, gallery_ids=gallery_ids,
        query_cams=query_cams, gallery_cams=gallery_cams,
        special_skip_first=special_skip_first,
        remove_zero_same_id=remove_zero_same_id, zero_eps=zero_eps,
    )[0]


def _euclidean_dist(a, b):
    m, n = a.size(0), b.size(0)
    a2 = (a * a).sum(dim=1, keepdim=True).expand(m, n)
    b2 = (b * b).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = a2 + b2
    dist.addmm_(a, b.t(), beta=1.0, alpha=-2.0)
    return (dist.clamp(min=1e-12)).sqrt()


def _cosine_dist(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = a @ b.t()
    return 1.0 - sim


def rank_metrics(X, y, t_X, t_y, special_skip_first=False, metric='cosine', meanap_mode='tf',
                 query_cams=None, gallery_cams=None,
                 remove_zero_same_id=True, zero_eps=1e-6):
    """Compute filtered mAP and Rank-1/5/10 retrieval accuracy.

    Args:
        X (numpy.ndarray): Gallery features.
        y (numpy.ndarray): Gallery identity labels.
        t_X (numpy.ndarray): Probe features.
        t_y (numpy.ndarray): Probe identity labels.
        metric (str): 'cosine' or 'euclidean'.

    Returns:
        tuple[float, float, float, float]: mAP, Rank-1, Rank-5, and Rank-10.

    Notes:
        Filtering matches mean_ap(). Queries with an empty filtered gallery
        contribute zero to each rank score. Euclidean distances are clamped
        before the square root, so identical vectors have distance 1e-6.
    """
    y = np.asarray(y)
    t_y = np.asarray(t_y)

    a = torch.from_numpy(t_X).reshape(len(t_X), -1).float()
    b = torch.from_numpy(X).reshape(len(X), -1).float()

    if metric == 'cosine':
        dist_m = _cosine_dist(a, b).numpy()
    else:
        dist_m = _euclidean_dist(a, b).numpy()

    return retrieval_metrics(
        dist_m, query_ids=t_y, gallery_ids=y,
        query_cams=query_cams, gallery_cams=gallery_cams,
        special_skip_first=special_skip_first,
        remove_zero_same_id=remove_zero_same_id, zero_eps=zero_eps,
    )


# =====================================================================
# Masks, Prototypes, and Feature Extraction
# =====================================================================

def sample_seq_mask(T, keep_num):
    keep_num = max(1, min(T, keep_num))
    idx = np.random.choice(T, size=keep_num, replace=False)
    m = np.zeros((T,), dtype=bool)
    m[idx] = True
    return m


def sample_node_mask(adj, J, St_prompt_type, St_mask_num):
    if St_prompt_type in [1, 2]:
        adj_M = adj.reshape(J, J)
        node_flag = [i for i in range(J)]
        rand_list = np.random.choice(J, size=J, replace=False)
        select_nodes = []
        cnt = 0
        while np.mean(node_flag) != -1:
            if node_flag[rand_list[cnt]] != -1:
                select_nodes.append(rand_list[cnt])
                node_flag[rand_list[cnt]] = -1
                for k in range(J):
                    if int(adj_M[rand_list[cnt], k]) == 1:
                        node_flag[k] = -1
            cnt += 1
        node_mask = np.ones((J,), dtype=bool)
        if St_prompt_type == 1:
            node_mask[select_nodes] = False
        else:
            rand_list = np.random.choice(J, size=len(select_nodes), replace=False)
            node_mask[rand_list] = False
    else:
        node_mask = np.ones((J,), dtype=bool)
        rand_list = np.random.choice(J, size=min(St_mask_num, J), replace=False)
        node_mask[rand_list] = False
    return node_mask


def make_class_prototypes(features, labels, nb_classes, device):
    """Average complete-batch features for every known training identity.

    Labels must use the contiguous columns of the training one-hot encoding.
    A class without sampled features has no defined centroid and is rejected.
    """
    C = int(nb_classes)
    H = features.shape[1]
    centers = np.zeros((C, H), dtype=np.float32)
    counts = np.zeros((C,), dtype=np.int64)
    for i, lab in enumerate(labels):
        if 0 <= lab < C:
            centers[lab] += features[i]
            counts[lab] += 1
        else:
            raise ValueError(f'Encoded training labels must lie in [0, {C - 1}]; got {lab}.')
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(
            'No complete-batch training features for encoded identity labels '
            f'{missing.tolist()}. Reduce --batch_size, add training sequences, '
            'or adjust sampling so every known identity remains in complete batches.'
        )
    for c in range(C):
        if counts[c] > 0:
            centers[c] /= float(counts[c])
    return torch.from_numpy(centers).to(device)


def frame_attn_pool(C_seq, frame_attn_module, mask_bool_or_none, device, use_attn=True):
    if mask_bool_or_none is None:
        if use_attn:
            logits = frame_attn_module(C_seq).squeeze(-1)
            w = torch.softmax(logits, dim=1).unsqueeze(-1)
            return (w * C_seq).sum(dim=1)
        else:
            return C_seq.mean(dim=1)
    mask_t = torch.as_tensor(mask_bool_or_none, device=device)[None, :]
    if use_attn:
        logits = frame_attn_module(C_seq).squeeze(-1)
        logits = logits.masked_fill(~mask_t, float('-inf'))
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        w = torch.where(mask_t.unsqueeze(-1), w, torch.zeros_like(w))
        denom = (w.sum(dim=1) + 1e-6)
        return (w * C_seq).sum(dim=1) / denom
    else:
        C_used = C_seq[:, mask_t.squeeze(0), :]
        return C_used.mean(dim=1)


@torch.no_grad()
def extract_seq_feats(model, X_seq, L_eig, batch_size, device, T, J, H, use_pos=True, rand_flip=0,
                      bn_freeze=True):
    """Extract sequence descriptors by averaging over joints and frames.

    Args:
        model (SSGT): Encoder with weights already restored or trained.
        X_seq (numpy.ndarray): Skeleton coordinates, shape (N, T, J, 3).
        L_eig (numpy.ndarray): Spatial graph encoding, shape (J, enc_k_spa).
        batch_size (int): Maximum number of sequences per inference batch.
        device (torch.device): Computation device.
        T, J, H (int): Sequence length, joint count, and feature dimension.
        use_pos (bool): Add spatial and temporal position encodings.
        rand_flip (int): Randomly flip Laplacian eigenvector signs when 1.
        bn_freeze (bool): Explicitly use the stored batch-normalization state.

    Returns:
        numpy.ndarray: Float sequence descriptors with shape (N, H).

    Notes:
        Inference retains the original mean pooling even when prototypes use
        learned frame attention during training.
    """
    model.eval()
    feats = []
    N = X_seq.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        xb_np = X_seq[s:e].reshape(-1, J, 3)
        xb = torch.from_numpy(xb_np).float().to(device)
        if rand_flip == 1:
            sign = torch.empty(L_eig.shape[1], device=device).uniform_(0, 1)
            sign = torch.where(sign >= 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
            L_batch = torch.as_tensor(L_eig, dtype=torch.float32, device=device) * sign
        else:
            L_batch = torch.as_tensor(L_eig, dtype=torch.float32, device=device)
        x = model.encode_input(xb)
        B = e - s
        x = x.reshape(B, T, J, H)
        h = model.add_pos_and_attn(x, L_batch, use_pos=use_pos, train=False, bn_freeze=bool(bn_freeze))
        seq_ftr = h.mean(dim=-2).mean(dim=1)
        feats.append(seq_ftr.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def build_prototypes_with_cached_masks(model, X_train_J, y_train_int, L_eig_np, batch_size, device,
                                       T, J, H, adj_J, St_prompt_type, St_mask_num, Tr_mask_num,
                                       nb_classes_total, use_pos=True, bn_freeze=True):
    """Construct identity prototypes and cache masks for the next epoch.

    Only complete training batches are used, preserving the original training
    schedule. The caller must supply at least one complete batch.

    Returns:
        tuple: Detached prototypes, frame masks, joint masks, and batch count.
    """
    model.eval()
    N = X_train_J.shape[0]
    num_batches = N // batch_size
    feats_list, labels_list = [], []
    seq_masks_cache, node_masks_cache = [], []
    L_batch = torch.as_tensor(L_eig_np, dtype=torch.float32, device=device)

    for b in range(num_batches):
        s = b * batch_size
        e = s + batch_size
        keep_f = T - int(Tr_mask_num)
        seq_mask_np = sample_seq_mask(T, keep_f)
        node_mask_np = sample_node_mask(adj_J, J, St_prompt_type, St_mask_num)
        seq_masks_cache.append(seq_mask_np)
        node_masks_cache.append(node_mask_np)

        xb_np = X_train_J[s:e].reshape(-1, J, 3)
        xb = torch.from_numpy(xb_np).float().to(device)
        x = model.encode_input(xb)
        x = x.reshape(batch_size, T, J, H)
        h = model.add_pos_and_attn(x, L_batch, use_pos=use_pos, train=False, bn_freeze=bool(bn_freeze))

        C_seq = h.mean(dim=-2)
        seq_ftr = frame_attn_pool(C_seq, model.frame_attn,
                                  mask_bool_or_none=seq_mask_np,
                                  device=device, use_attn=model.frame_attn_pool)
        feats_list.append(seq_ftr.cpu().numpy())
        labels_list.append(y_train_int[s:e])

    train_feats = np.concatenate(feats_list, axis=0)
    train_labs = np.concatenate(labels_list, axis=0)
    class_protos = make_class_prototypes(train_feats, train_labs, nb_classes=int(nb_classes_total), device=device)
    return class_protos.detach(), np.asarray(seq_masks_cache), np.asarray(node_masks_cache), num_batches


# =====================================================================
# Checkpoint Paths and Restoration
# =====================================================================

def checkpoint_directory(args):
    """Return the original experiment name under the portable model directory."""
    change = '_CME' if args.probe_type else ''
    change += (f'Pytorch_{int(args.length)}_layers_{int(args.L_transformer)}'
               f'_heads_{int(args.n_heads)}_alpha_{float(args.seq_lambda)}'
               f'_beta_{float(args.prompt_lambda)}_lambda_{float(args.GPC_lambda)}')
    if args.spa_pos_type == 'learned':
        change += '_spaPos_learned'
    variant = process.data_variant_tag(args.length, args.data_subdir)
    if variant:
        change += '_' + variant
    return paths.model_root() / args.dataset / (args.probe + change)


def _restore_evaluation_args(args):
    """Load saved metadata before building the dataset and model for evaluation.

    Runtime choices such as device, GPU, checkpoint path, and batch size stay
    under command-line control. Architecture and input-protocol fields follow
    the checkpoint; incompatible or metadata-free files are rejected.

    Returns:
        dict: The checkpoint containing model_state and training args.
    """
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else checkpoint_directory(args) / 'best.ckpt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}. Supply --ckpt or train a model first.')
    # Historical PyTorch 1.13 checkpoints include pickled argument metadata.
    # Load only local checkpoint files whose source you trust.
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('args'), dict) or 'model_state' not in checkpoint:
        raise ValueError('Expected an SSGT checkpoint with model_state and args metadata.')
    saved = checkpoint['args']
    restore_fields = (
        'dataset', 'length', 'data_subdir', 'probe', 'probe_type',
        'H', 'n_heads', 'L_transformer', 'enc_k_spa', 'enc_k_time',
        'pre_norm', 'attn_dropout', 'attn_clamp', 'qkv_merged', 'frame_attn_pool',
        't_1', 't_2', 'learnable_temp', 'use_time_attn', 'spa_pos_type',
        'pos_enc', 'bn_freeze_eval', 'meanap_mode', 'metric',
    )
    for field in restore_fields:
        if field in saved:
            setattr(args, field, saved[field])
    if args.dataset not in ('WUT', 'KGBD', 'KS20'):
        raise ValueError(f'Unsupported checkpoint dataset: {args.dataset}')
    args.ckpt = str(ckpt_path.resolve())
    return checkpoint


def _sha256_file(filename):
    """Hash a file without loading the full data or checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_spatial_pos_encoding(checkpoint, computed_encoding, joint_count, encoding_width):
    """Use the training-time graph basis when it was captured in a checkpoint.

    Eigenvectors have sign and degenerate-subspace ambiguity across numerical
    libraries. Keeping the actual training basis avoids recomputing a different
    input representation when restoring the same neural-network parameters.
    """
    value = checkpoint.get('spatial_pos_encoding') if checkpoint is not None else None
    if value is None:
        value = computed_encoding
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    encoding = np.asarray(value, dtype=np.float32)
    expected = (int(joint_count), int(encoding_width))
    if encoding.shape != expected or not np.isfinite(encoding).all():
        raise ValueError(f'Spatial position encoding must be finite with shape {expected}; got {encoding.shape}')
    return encoding


def evaluation_report(args, checkpoint, device, metrics, split_labels, spatial_pos_encoding):
    """Collect measured retrieval metrics and their checkpoint/data provenance.

    Historical checkpoint arguments are reported literally. Missing training
    metadata is identified as missing, rather than being filled from current
    defaults and presented as evidence of the historical training protocol.
    """
    mean_ap_value, rank1, rank5, rank10 = metrics
    train_labels, gallery_labels, probe_labels = split_labels
    saved_args = checkpoint['args']
    paper_fields = {
        'GPC_lambda': 0.5, 'gpc_up_period': 10, 'gpc_iota': 10.0,
        'gpc_tau': 0.05, 'enable_dynamic_gpc': 1,
        'use_time_attn': 1, 'spa_pos_type': 'laplacian',
        'frame_attn_pool': 0, 'learnable_temp': 0,
    }
    differences = {
        key: {'checkpoint': saved_args[key], 'paper_release': expected}
        for key, expected in paper_fields.items()
        if key in saved_args and saved_args[key] != expected
    }
    for key, expected in (('conv_check', 0), ('gpc_lambda_min', 0.0), ('gpc_lambda_max', 1.0)):
        if key in saved_args and saved_args[key] != expected:
            differences[key] = {'checkpoint': saved_args[key], 'paper_release': expected}
    subdirectory = args.data_subdir or str(args.length)
    dataset_dir = paths.data_root() / args.dataset / subdirectory
    split_names = {'train': 'train_npy_data',
                   'gallery': 'test_npy_data/gallery',
                   'probe': 'test_npy_data/' + args.probe}
    files = []
    identity_sets = {}
    for split, folder in split_names.items():
        prefix = 'source' if split == 'train' else 't_source'
        names = [f'{prefix}_{axis}_{args.dataset}_{args.length}.npy' for axis in ('x', 'y', 'z')]
        names += [f'ids_{args.dataset}_{args.length}.npy',
                  f'frame_id_{args.dataset}_{args.length}.npy']
        for name in names:
            filename = dataset_dir / folder / name
            if filename.is_file():
                files.append({
                    'path': str(filename.relative_to(paths.data_root())),
                    'bytes': filename.stat().st_size,
                    'sha256': _sha256_file(filename),
                })
        labels_path = dataset_dir / folder / f'frame_id_{args.dataset}_{args.length}.npy'
        if labels_path.is_file():
            raw_labels = np.load(labels_path, allow_pickle=True)
            identity_sets[split] = np.unique(raw_labels).tolist()
    config = dict(vars(args))
    checkpoint_protocol = checkpoint.get('paper_protocol')
    return {
        'schema_version': 1,
        'evaluation': 'closed_set_reidentification',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'metrics': {'R1': float(rank1), 'R5': float(rank5),
                    'R10': float(rank10), 'mAP': float(mean_ap_value)},
        'metric_scale': 'fraction in [0, 1]',
        'checkpoint': {
            'path': args.ckpt,
            'sha256': _sha256_file(args.ckpt),
            'epoch': checkpoint.get('epoch'),
            'saved_metrics': checkpoint.get('metrics'),
            'saved_training_args': saved_args,
            'known_identity_count': checkpoint.get('known_identity_count'),
        },
        'evaluation_config': config,
        'paper_protocol': {
            'release_loss_schedule': dict(PAPER_PROTOCOL),
            'checkpoint_loss_schedule': checkpoint_protocol,
            'checkpoint_declares_current_schedule': (
                isinstance(checkpoint_protocol, dict)
                and checkpoint_protocol.get('version') == PAPER_PROTOCOL['version']
            ),
            'recorded_differences': differences,
            'unrecorded_training_fields': [key for key in paper_fields if key not in saved_args],
            'training_seed_metadata_present': all(key in saved_args for key in ('seed_val', 'fix_seed')),
            'evaluation_performs_training': False,
        },
        'data': {
            'root': str(paths.data_root()),
            'dataset': args.dataset,
            'subdirectory': subdirectory,
            'sequence_length': int(args.length),
            'train_sequences_after_sampler': len(train_labels),
            'gallery_sequences': len(gallery_labels),
            'probe_sequences': len(probe_labels),
            'known_label_indices': np.unique(train_labels).tolist(),
            'raw_identity_sets': identity_sets,
            'known_identity_count': len(np.unique(train_labels)),
            'file_checksums': files,
            'spatial_position_encoding': {
                'used_source': ('checkpoint' if checkpoint.get('spatial_pos_encoding') is not None
                                else 'recomputed_by_data_loader'),
                'checkpoint_provenance': checkpoint.get('spatial_pos_encoding_provenance'),
                'shape': list(spatial_pos_encoding.shape),
                'float32_sha256': hashlib.sha256(
                    np.ascontiguousarray(spatial_pos_encoding, dtype=np.float32).tobytes()
                ).hexdigest(),
            },
        },
        'runtime': {
            'python': sys.version.split()[0],
            'pytorch': str(torch.__version__),
            'numpy': np.__version__,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        },
    }


# =====================================================================
# Training and Evaluation Pipeline
# =====================================================================

def main(argv=None):
    """Run training or strict checkpoint evaluation with the supplied options."""
    args = get_args(argv)
    checkpoint = None
    if args.mode == 'Eval':
        checkpoint = _restore_evaluation_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.requested_seed_val = args.seed_val
    args.seed_selection = 'automatic' if args.seed_val == -1 else 'explicit'
    args.seed_val = set_seed(seed=args.seed_val, fix=bool(args.fix_seed))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if args.device == 'auto' else args.device
    )


    nb_nodes = 20
    if args.dataset == 'KS20': nb_nodes = 25
    if args.dataset == 'WUT':  nb_nodes = 19
    if args.dataset == 'KGBD':
        args.lr = 3.5e-4
        args.rand_flip = 0
    # Dataset dimensions and loader settings must match the checkpoint.
    T = int(args.length)
    H = int(args.H)
    batch_size = int(args.batch_size)
    nhood = 1
    global_att = False


    (X_train_J, _, _, _, _, y_train,
     X_test_J, _, _, _, _, y_test,
     adj_J, _, pos_enc_ori,
     _, _, _, _, _, _, _, _, _, _, nb_classes) = process.gen_train_data(
        dataset=args.dataset, split=args.probe, time_step=T,
        nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
        batch_size=batch_size, enc_k=int(args.enc_k_spa),
        data_subdir=args.data_subdir, sampling_seed=args.seed_val)

    joint_num = X_train_J.shape[2]
    L_eig_np = resolve_spatial_pos_encoding(
        checkpoint, pos_enc_ori, joint_num, args.enc_k_spa,
    )
    if y_train.ndim != 2:
        raise ValueError('The training loader must return a one-hot label matrix.')
    # Held-out identities are absent from the encoded label columns and must
    # not become zero-valued negative prototypes in the contrastive losses.
    nb_classes_total = int(y_train.shape[1])


    model = SSGT(H=args.H, n_heads=args.n_heads, L=args.L_transformer, p_drop=0.5,
                   enc_k_spa=int(args.enc_k_spa), enc_k_time=int(args.enc_k_time),
                   J=joint_num, T=T,
                   pre_norm=bool(args.pre_norm),
                   attn_drop=float(args.attn_dropout),
                   attn_clamp=bool(args.attn_clamp),
                   qkv_merged=bool(args.qkv_merged),
                   frame_attn_pool=bool(args.frame_attn_pool),
                   t1_init=args.t_1, t2_init=args.t_2,
                   learnable_temp=bool(args.learnable_temp),
                   use_time_attn=bool(args.use_time_attn),
                   spa_pos_type=str(args.spa_pos_type)).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'], strict=True)
        print(f"[Checkpoint] Restored {args.ckpt}")
    if args.mode == 'Train' and len(X_train_J) < batch_size:
        raise ValueError(
            f'Training requires at least one full batch: {len(X_train_J)} sequences '
            f'available, batch_size={batch_size}. Reduce --batch_size.'
        )
    if args.dataset in ['KGBD', 'KS20', 'WUT']:
        (X_train_J_g, _, _, _, _, y_train_g,
         X_gal_J, _, _, _, _, y_gal,
         _, _, pos_enc_g,
         _, _, _, _, _, _, _, _, _, _, nb_cls_g) = process.gen_train_data(
            dataset=args.dataset, split='gallery', time_step=T,
            nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
            batch_size=batch_size, enc_k=int(args.enc_k_spa),
            data_subdir=args.data_subdir, sampling_seed=args.seed_val)
    y_train_int = np.argmax(y_train, axis=-1) if y_train.ndim > 1 else y_train
    y_test_int = np.argmax(y_test, axis=-1) if y_test.ndim > 1 else y_test
    y_gal_int = np.argmax(y_gal, axis=-1) if y_gal.ndim > 1 else y_gal


    special_list = ['nm.nm', 'cl.cl', 'bg.bg']

    @torch.no_grad()
    def evaluate_train_gallery():
        model.eval()
        gal_src_X = X_train_J if args.train_eval_gallery == 'train' else X_gal_J
        gal_src_y = y_train_int if args.train_eval_gallery == 'train' else y_gal_int
        gal_feats = extract_seq_feats(model, gal_src_X, L_eig_np, batch_size, device, T, joint_num, H,
                                      use_pos=bool(args.pos_enc), rand_flip=0,
                                      bn_freeze=bool(args.bn_freeze_eval))
        pro_feats = extract_seq_feats(model, X_test_J, L_eig_np, batch_size, device, T, joint_num, H,
                                      use_pos=bool(args.pos_enc), rand_flip=0,
                                      bn_freeze=bool(args.bn_freeze_eval))
        is_special = (args.probe_type in special_list)
        return rank_metrics(gal_feats, gal_src_y, pro_feats, y_test_int,
                            special_skip_first=is_special, metric=args.metric,
                            meanap_mode=args.meanap_mode)

    @torch.no_grad()
    def evaluate_gallery_split():
        model.eval()
        gal_feats = extract_seq_feats(model, X_gal_J, L_eig_np, batch_size, device, T, joint_num, H,
                                      use_pos=bool(args.pos_enc), rand_flip=0,
                                      bn_freeze=bool(args.bn_freeze_eval))
        pro_feats = extract_seq_feats(model, X_test_J, L_eig_np, batch_size, device, T, joint_num, H,
                                      use_pos=bool(args.pos_enc), rand_flip=0,
                                      bn_freeze=bool(args.bn_freeze_eval))
        is_special = (args.probe_type in special_list)
        return rank_metrics(gal_feats, y_gal_int, pro_feats, y_test_int,
                            special_skip_first=is_special, metric=args.metric,
                            meanap_mode=args.meanap_mode)

    # Evaluation never runs with newly initialized parameters.
    if args.mode == 'Eval':
        mAP, r1, r5, r10 = evaluate_gallery_split()
        print(
            '[Evaluation] (gallery split; BN_frozen_eval=%d; meanAP=%s) mAP: %.4f | R1: %.4f | R5: %.4f | R10: %.4f'
            % (args.bn_freeze_eval, args.meanap_mode, mAP, r1, r5, r10))
        if args.metrics_output:
            report = evaluation_report(args, checkpoint, device,
                                       (mAP, r1, r5, r10),
                                       (y_train_int, y_gal_int, y_test_int), L_eig_np)
            output = Path(args.metrics_output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            print(f'[Metrics] Wrote {output}')
        return

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp) and device.type == 'cuda'
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        # PyTorch 1.13 retains the CUDA-specific GradScaler API.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


    # Algorithm 2 uses signed slope ratios and a doubling update schedule.
    loss_balancer = DynamicLossBalancer(
        args.GPC_lambda, args.gpc_up_period, args.gpc_iota, args.gpc_tau,
    )
    current_gpc_lambda = loss_balancer.value
    run_protocol = loss_balancer.metadata()
    run_protocol['seed_control'] = {
        'enabled': True, 'seed': int(args.seed_val),
        'requested_seed': int(args.requested_seed_val),
        'selection': args.seed_selection,
        'deterministic_cudnn': bool(args.fix_seed),
        'cross_device_bitwise_reproducibility_verified': False,
    }
    run_protocol['prototype_protocol'] = {
        'identity_count': nb_classes_total,
        'count_source': 'training_one_hot_label_columns',
        'dataset_identity_count_before_holdout': int(nb_classes),
        'scope': 'known training identities only',
        'zero_sample_prototypes': 'rejected',
    }

    base_dir = str(checkpoint_directory(args))
    os.makedirs(base_dir, exist_ok=True)
    csv_log_path = os.path.join(base_dir, 'best_accuracy_log.csv')

    def _ensure_and_get_ckpt_path():
        return os.path.join(base_dir, 'best.ckpt')

    print(
        '----- Start Training (AMP=%s, LearnableTemp=%s, UseTimeAttn=%s, SpaPos=%s) -----' % (
            str(bool(args.amp)), str(bool(args.learnable_temp)), str(bool(args.use_time_attn)),
            str(args.spa_pos_type)))
    print(f'BN freeze: eval={args.bn_freeze_eval}, proto={args.bn_freeze_proto}, train={args.bn_freeze_train}; '
          f'meanAP={args.meanap_mode}; eval_gallery={args.train_eval_gallery}')
    cluster_epochs = 50000
    patience = int(args.patience)


    best_r1 = -1.0
    best_r5 = 0.0
    best_r10 = 0.0
    best_map = 0.0
    best_epoch = -1
    best_lambda = float(current_gpc_lambda)
    cur_pat = 0

    hundred_epoch_time = 0.0

    # Epoch indices and evaluation-before-update order follow the original code.
    for epoch in range(cluster_epochs):

        class_protos, cached_seq_masks, cached_node_masks, num_batches = \
            build_prototypes_with_cached_masks(
                model, X_train_J, y_train_int, L_eig_np, batch_size, device,
                T, joint_num, H, adj_J, args.St_prompt_type, args.St_mask_num, args.Tr_mask_num,
                nb_classes_total=nb_classes_total, use_pos=bool(args.pos_enc),
                bn_freeze=bool(args.bn_freeze_proto)
            )


        mAP, r1, r5, r10 = evaluate_train_gallery()
        eval_tag = '@TrainGallery' if args.train_eval_gallery == 'train' else '@GallerySplit'


        # Select one checkpoint by Rank-1, then mAP/Rank-5/Rank-10 tie-breaks.
        eps_cmp = 1e-12
        is_better_top1 = (r1 > best_r1 + eps_cmp)
        is_same_top1_but_better_aux = (
            abs(r1 - best_r1) <= eps_cmp and
            (
                (mAP > best_map + eps_cmp) or
                (abs(mAP - best_map) <= eps_cmp and r5 > best_r5 + eps_cmp) or
                (abs(mAP - best_map) <= eps_cmp and abs(r5 - best_r5) <= eps_cmp and r10 > best_r10 + eps_cmp)
            )
        )

        if is_better_top1 or is_same_top1_but_better_aux:
            best_r1 = float(r1)
            best_r5 = float(r5)
            best_r10 = float(r10)
            best_map = float(mAP)
            best_epoch = int(epoch)
            best_lambda = float(current_gpc_lambda)
            cur_pat = 0
            ckpt_path = _ensure_and_get_ckpt_path()
            print(f"Saving best Top-1 model to {ckpt_path}")


            if int(args.save_model) == 1:
                torch.save({
                    "epoch": int(epoch),
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scaler_state": (scaler.state_dict() if use_amp else None),
                    "args": vars(args),
                    "known_identity_count": nb_classes_total,
                    "paper_protocol": run_protocol,
                    "spatial_pos_encoding": torch.from_numpy(L_eig_np.copy()),
                    "spatial_pos_encoding_provenance": {
                        "source": "captured_from_training_loader",
                        "dataset": args.dataset,
                        "sequence_length": int(args.length),
                        "data_subdir": args.data_subdir or str(args.length),
                        "encoding_width": int(args.enc_k_spa),
                    },
                    "metrics": {
                        "r1": float(r1),
                        "r5": float(r5),
                        "r10": float(r10),
                        "mAP": float(mAP),
                        "lambda": float(current_gpc_lambda),
                        "selection_rule": "best_top1_with_map_r5_r10_tiebreak"
                    },
                }, ckpt_path)


            try:
                file_exists = os.path.isfile(csv_log_path)
                with open(csv_log_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        header = ['Epoch', 'Rank-1', 'Rank-5', 'Rank-10', 'mAP', 'Lambda(GPC)', 'Selection']
                        writer.writerow(header)

                    row = [
                        epoch,
                        f"{r1:.4f}",
                        f"{r5:.4f}",
                        f"{r10:.4f}",
                        f"{mAP:.4f}",
                        f"{current_gpc_lambda:.4f}",
                        'best_top1'
                    ]
                    writer.writerow(row)
                    print(f">> [Log] Best@Top1 record updated in csv: {csv_log_path}")
            except Exception as e:
                print(f">> [Warning] Failed to write csv: {e}")

        else:
            cur_pat += 1

        print(f'[Probe Evaluation {eval_tag}] Epoch {epoch} | '
              f'Top-1: {r1:.4f} (best@Top1 {best_r1:.4f}) | '
              f'Top-5: {r5:.4f} (best@Top1 {best_r5:.4f}) | '
              f'Top-10: {r10:.4f} (best@Top1 {best_r10:.4f}) | '
              f'mAP: {mAP:.4f} (best@Top1 {best_map:.4f}) | '
              f'best_epoch={best_epoch} | λ_current={current_gpc_lambda:.3f}')

        if cur_pat >= patience: break


        # Reuse the prototype masks throughout this epoch.
        model.train(True)
        epoch_recon_sum = 0.0
        epoch_gpc_sum = 0.0
        counted_batches = 0

        epoch_start_time = time.time()

        for tr_step in range(num_batches):
            s = tr_step * batch_size
            e = s + batch_size
            Xb = X_train_J[s:e].reshape(-1, joint_num, 3)
            xb = torch.from_numpy(Xb).float().to(device)
            B = batch_size

            seq_mask_np = cached_seq_masks[tr_step]
            node_mask_np = cached_node_masks[tr_step]
            seq_mask_t = torch.from_numpy(seq_mask_np).to(device)
            node_mask_t = torch.from_numpy(node_mask_np).to(device)

            if args.rand_flip == 1 and str(args.spa_pos_type) == 'laplacian':
                sign = torch.empty(L_eig_np.shape[1], device=device).uniform_(0, 1)
                sign = torch.where(sign >= 0.5, torch.tensor(1., device=device), torch.tensor(-1., device=device))
                L_batch = torch.as_tensor(L_eig_np, dtype=torch.float32, device=device) * sign
            else:
                L_batch = torch.as_tensor(L_eig_np, dtype=torch.float32, device=device)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                x = model.encode_input(xb)
                h = x.reshape(B, T, joint_num, H)
                h = model.add_pos_and_attn(h, L_batch, use_pos=bool(args.pos_enc),
                                           train=True, bn_freeze=bool(args.bn_freeze_train))

                C_seq = h.mean(dim=-2)
                seq_ftr = frame_attn_pool(C_seq, model.frame_attn,
                                          mask_bool_or_none=seq_mask_np,
                                          device=device, use_attn=model.frame_attn_pool)
                gt_pos = torch.from_numpy(X_train_J[s:e].reshape(B, T, joint_num * 3)).float().to(device)

                G_recon = model.structure_prompt_recon(h, node_mask_t, joint_num, T, args.S_prompt, gt_pos,
                                                       batch_size=B)
                T_recon = model.trajectory_prompt_recon(h, seq_mask_t, joint_num, T, args.T_prompt, gt_pos,
                                                        batch_size=B)
                recon_loss = G_recon * float(args.prompt_lambda) + T_recon * (1.0 - float(args.prompt_lambda))

                labels_b = torch.from_numpy(y_train_int[s:e]).long().to(device)
                if bool(args.learnable_temp):
                    logits_seq = gpc_seq_logits(seq_ftr, class_protos, t=None, logit_scale=model.logit_scale_seq)
                    loss_seq = F.cross_entropy(logits_seq, labels_b, reduction='mean')
                    loss_ske = gpc_ske_loss(C_seq, class_protos, labels_b, T=T, H=H, t=None,
                                            gpc_f1=model.gpc_f1, gpc_f2=model.gpc_f2,
                                            logit_scale=model.logit_scale_ske)
                else:
                    logits_seq = gpc_seq_logits(seq_ftr, class_protos, t=args.t_1, logit_scale=None)
                    loss_seq = F.cross_entropy(logits_seq, labels_b, reduction='mean')
                    loss_ske = gpc_ske_loss(C_seq, class_protos, labels_b, T=T, H=H, t=args.t_2,
                                            gpc_f1=model.gpc_f1, gpc_f2=model.gpc_f2, logit_scale=None)
                H_loss = (1.0 - float(args.seq_lambda)) * loss_ske + float(args.seq_lambda) * loss_seq

                w_final = current_gpc_lambda

                loss = w_final * H_loss + (1.0 - w_final) * recon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)

            scaler.step(optim)
            scaler.update()

            epoch_recon_sum += float(recon_loss.item())
            epoch_gpc_sum += float(H_loss.item())
            counted_batches += 1

            if tr_step % 20 == 0:
                if int(args.learnable_temp) == 1:
                    ls_seq = float(model.logit_scale_seq.detach().cpu())
                    ls_ske = float(model.logit_scale_ske.detach().cpu())
                    eff_t1 = math.exp(-ls_seq)
                    eff_t2 = math.exp(-ls_ske)
                else:
                    eff_t1, eff_t2 = args.t_1, args.t_2
                msg = (f'[{epoch}] Batch {tr_step}/{num_batches} | '
                       f'λ_used={w_final:.3f} | Recon: {recon_loss.item():.5f} '
                       f'(G {G_recon.item():.5f} / T {T_recon.item():.5f}) | '
                       f'GPC: {H_loss.item():.5f} (seq {loss_seq.item():.5f} / ske {loss_ske.item():.5f}) | '
                       f't1~{eff_t1:.4f}, t2~{eff_t2:.4f}')
                print(msg)

        epoch_duration = time.time() - epoch_start_time
        hundred_epoch_time += epoch_duration

        if counted_batches > 0:
            avg_recon = epoch_recon_sum / counted_batches  # STPR
            avg_gpc = epoch_gpc_sum / counted_batches  # GPC
        else:
            avg_recon = avg_gpc = 0.0

        update = loss_balancer.observe(avg_gpc, avg_recon)
        current_gpc_lambda = loss_balancer.value
        if update is not None:
            print('[Lambda update / Algorithm 2] ' + json.dumps(update, sort_keys=True))

        print(f'[Epoch {epoch} Summary] avg_STPR={avg_recon:.6f} | avg_GPC={avg_gpc:.6f} | '
              f'λ_current={current_gpc_lambda:.3f} | Time={epoch_duration:.2f}s')

        max_epochs = int(args.max_epochs)
        if max_epochs > 0 and epoch >= max_epochs:
            print(f'[Stop] Reached max_epochs={max_epochs}')
            break

        if (epoch + 1) % 5000 == 0:
            avg_time_100 = hundred_epoch_time / 5000.0
            print(f'>>>>> [100-Epoch Stats] Average Time per Epoch: {avg_time_100:.4f}s <<<<<')
            hundred_epoch_time = 0.0

    print('Training finished.')
    print(f'[Best @ Top-1] Epoch {best_epoch} | '
          f'R1: {best_r1:.4f} | R5: {best_r5:.4f} | R10: {best_r10:.4f} | '
          f'mAP: {best_map:.4f} | λ_at_best={best_lambda:.4f}')

    if args.train_eval_gallery == 'train':
        mAP, r1, r5, r10 = evaluate_train_gallery()
        final_tag = '@TrainGallery'
    else:
        mAP, r1, r5, r10 = evaluate_gallery_split()
        final_tag = '@GallerySplit'
    print(f'[Last-epoch model {final_tag}] (BN_frozen_eval={args.bn_freeze_eval}; meanAP={args.meanap_mode}) '
          f'mAP: {mAP:.4f} | R1: {r1:.4f} | R5: {r5:.4f} | R10: {r10:.4f}')


if __name__ == '__main__':
    main()
