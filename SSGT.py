"""Skeleton graph encoder used by SSGT.

The joint-level encoder combines spatial and temporal self-attention with
structure and trajectory reconstruction heads.

Tensor Conventions:
    B: Batch size; T: sequence length; J: joint count; H: hidden dimension.
    Inputs use XYZ coordinates and have shape (B, T, J, 3).

Example:
    >>> model = SSGT(J=19, T=6, H=128)
    >>> # Use encode_input() followed by add_pos_and_attn().
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Temporal Position Encoding
# =====================================================================

def laplacian_position_encoding_np(length: int, K: int) -> np.ndarray:
    """Compute nonconstant eigenvectors of a normalized chain-graph Laplacian.

    Args:
        length (int): Number of frames in the sequence (at least two).
        K (int): Requested embedding dimension, capped at length - 1.

    Returns:
        numpy.ndarray: Float32 temporal encoding with shape (length, K_clamped).
    """
    if length < 2:
        raise ValueError('Sequence length must be at least two frames.')
    K = max(1, min(K, length - 1))
    A = np.zeros((length, length), dtype=np.float64)
    for i in range(length - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    deg = np.sum(A, axis=1)
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(length) - D_inv_sqrt @ A @ D_inv_sqrt
    # This chain graph is small and dense. Using a dense symmetric solver
    # avoids ARPACK's independent random initialization, which is not
    # controlled by the recorded training seed.
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)
    v = v[:, idx]
    pos = v[:, 1:K + 1]
    return pos.astype(np.float32)


# =====================================================================
# Normalization
# =====================================================================

class BN4d(nn.Module):
    """Batch normalization for (B, T, J, H) tensors.

    The epsilon and running-statistics momentum retain the original TensorFlow
    convention. Passing update_stats=False uses stored statistics explicitly.
    """
    def __init__(self, H):
        super().__init__()
        self.bn = nn.BatchNorm1d(H, eps=1e-3, momentum=0.01)

    def forward(self, x, update_stats=True):  # x: [B,T,J,H]
        B, T, J, H = x.shape
        y = x.reshape(B * T * J, H)
        if update_stats:
            y = self.bn(y)
        else:
            y = F.batch_norm(
                y, self.bn.running_mean, self.bn.running_var,
                self.bn.weight, self.bn.bias,
                training=False, eps=self.bn.eps
            )
        return y.reshape(B, T, J, H)


# =====================================================================
# Spatial Attention
# =====================================================================

class SpatialMHAttnLayer(nn.Module):
    """Multi-head attention over joints independently for each frame."""
    def __init__(self, H, n_heads, attn_drop=0.0, qkv_merged=True, use_clamp=False):
        super().__init__()
        assert H % n_heads == 0
        self.H, self.nh, self.dk = H, n_heads, H // n_heads
        self.use_clamp = bool(use_clamp)
        self.attn_do = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.qkv_merged = bool(qkv_merged)
        if self.qkv_merged:
            self.qkv = nn.Linear(H, 3 * H, bias=False)
            nn.init.xavier_uniform_(self.qkv.weight)
        else:
            self.W_Q = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            self.W_K = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            self.W_V = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            for l in (self.W_Q + self.W_K + self.W_V):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, x):  # x: [B,T,J,H]
        B, T, J, H = x.shape
        if self.qkv_merged:
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q.view(B, T, J, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            k = k.view(B, T, J, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            v = v.view(B, T, J, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
            if self.use_clamp: att = torch.clamp(att, -5.0, 5.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_do(att)
            y = torch.matmul(att, v)
            y = y.permute(0, 1, 3, 2, 4).contiguous().view(B, T, J, H)
            return y
        heads = []
        for i in range(self.nh):
            Q = self.W_Q[i](x)
            K = self.W_K[i](x)
            V = self.W_V[i](x)
            att = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dk)
            if self.use_clamp: att = torch.clamp(att, -5.0, 5.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_do(att)
            heads.append(torch.matmul(att, V))
        return torch.cat(heads, dim=-1)


# =====================================================================
# Temporal Attention
# =====================================================================

class TemporalMHAttnLayer(nn.Module):
    """Multi-head attention over frames independently for each joint."""
    def __init__(self, H, n_heads, attn_drop=0.0, qkv_merged=True, use_clamp=False):
        super().__init__()
        assert H % n_heads == 0
        self.H, self.nh, self.dk = H, n_heads, H // n_heads
        self.use_clamp = bool(use_clamp)
        self.attn_do = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.qkv_merged = bool(qkv_merged)
        if self.qkv_merged:
            self.qkv = nn.Linear(H, 3 * H, bias=False)
            nn.init.xavier_uniform_(self.qkv.weight)
        else:
            self.W_Q = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            self.W_K = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            self.W_V = nn.ModuleList([nn.Linear(H, self.dk, bias=False) for _ in range(n_heads)])
            for l in (self.W_Q + self.W_K + self.W_V):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, x):  # x: [B,T,J,H]
        B, T, J, H = x.shape
        X = x.transpose(1, 2)  # [B,J,T,H]
        if self.qkv_merged:
            qkv = self.qkv(X)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q.view(B, J, T, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            k = k.view(B, J, T, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            v = v.view(B, J, T, self.nh, self.dk).permute(0, 1, 3, 2, 4)
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
            if self.use_clamp: att = torch.clamp(att, -5.0, 5.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_do(att)
            y = torch.matmul(att, v)

            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B, T, J, H)
            return y
        heads = []
        for i in range(self.nh):
            Q = self.W_Q[i](X)
            K = self.W_K[i](X)
            V = self.W_V[i](X)
            att = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dk)
            if self.use_clamp: att = torch.clamp(att, -5.0, 5.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_do(att)
            heads.append(torch.matmul(att, V))
        Y = torch.cat(heads, dim=-1)
        return Y.transpose(1, 2)


# =====================================================================
# SSGT Encoder
# =====================================================================

class SSGT(nn.Module):
    """Joint-level SSGT encoder with auxiliary reconstruction heads.

    Args:
        H (int): Hidden feature dimension, divisible by n_heads.
        n_heads (int): Attention heads in each spatial or temporal block.
        L (int): Number of attention blocks.
        J (int): Number of joints in each frame.
        T (int): Number of frames in each sequence.
        enc_k_spa (int): Width of the supplied spatial Laplacian encoding.
        enc_k_time (int): Requested temporal encoding dimension.
        use_time_attn (bool): Enable the temporal attention branch.
        spa_pos_type (str): 'laplacian' or learned joint embeddings.

    Notes:
        The research API exposes encode_input(), add_pos_and_attn(), and the
        two reconstruction methods separately; there is no monolithic forward.
        Use the settings stored in checkpoint['args'] when restoring a model.
    """
    def __init__(self, H=128, n_heads=8, L=2, p_drop=0.5, enc_k_spa=10, enc_k_time=4, J=20, T=6,
                 pre_norm=True, attn_drop=0.4, attn_clamp=True, qkv_merged=True,
                 frame_attn_pool=False, t1_init=0.07, t2_init=14.0, learnable_temp=False,
                 use_time_attn=True, spa_pos_type='laplacian'):
        super().__init__()
        self.H = int(H)
        self.J = int(J)
        self.T = int(T)
        self.pre_norm = bool(pre_norm)
        self.frame_attn_pool = bool(frame_attn_pool)
        self.learnable_temp = bool(learnable_temp)
        self.use_time_attn = bool(use_time_attn)
        self.spa_pos_type = str(spa_pos_type)

        self.in1 = nn.Linear(3, self.H, bias=True)
        self.in2 = nn.Linear(self.H, self.H, bias=True)

        self.pos_dense = nn.Linear(int(enc_k_spa), self.H, bias=True)
        if self.spa_pos_type == 'learned':
            self.joint_pos_embed = nn.Embedding(self.J, self.H)
        else:
            self.joint_pos_embed = None

        self.time_k = max(1, min(int(enc_k_time), self.T - 1))
        time_pe_np = laplacian_position_encoding_np(self.T, self.time_k)
        self.register_buffer('time_pe', torch.from_numpy(time_pe_np))
        self.time_pos_dense = nn.Linear(self.time_k, self.H, bias=True)

        self.blocks_spa = nn.ModuleList(
            [SpatialMHAttnLayer(self.H, n_heads, attn_drop, qkv_merged, attn_clamp) for _ in range(L)])

        if self.use_time_attn:
            self.blocks_tmp = nn.ModuleList(
                [TemporalMHAttnLayer(self.H, n_heads, attn_drop, qkv_merged, attn_clamp) for _ in range(L)])
        else:
            self.blocks_tmp = nn.ModuleList([])

        merge_in_dim = 2 * self.H if self.use_time_attn else self.H
        self.blocks_merge = nn.ModuleList([nn.Linear(merge_in_dim, self.H, bias=True) for _ in range(L)])

        self.blocks_norm = nn.ModuleList([nn.LayerNorm(self.H) for _ in range(L)])

        self.drop = nn.Dropout(p_drop)
        self.attn_out = nn.Linear(self.H, self.H, bias=True)
        self.bn1 = BN4d(self.H)
        self.ff1 = nn.Linear(self.H, 2 * self.H, bias=True)
        self.ff2 = nn.Linear(2 * self.H, self.H, bias=True)
        self.bn2 = BN4d(self.H)

        self.frame_attn = nn.Linear(self.H, 1, bias=True)

        self.g_r1 = nn.Linear(self.H, self.H, bias=True)
        self.g_r2 = nn.Linear(self.H, self.J * 3, bias=True)
        self.tr1 = nn.Linear(self.H, self.H // 2, bias=True)
        self.tr2 = nn.Linear(self.H // 2, self.T * 3, bias=True)

        self.gpc_f1 = nn.Linear(self.H, self.H, bias=False)
        self.gpc_f2 = nn.Linear(self.H, self.H, bias=False)

        init_logit_seq = math.log(1.0 / float(t1_init))
        init_logit_ske = math.log(1.0 / float(t2_init))
        self.logit_scale_seq = nn.Parameter(torch.tensor(init_logit_seq))
        self.logit_scale_ske = nn.Parameter(torch.tensor(init_logit_ske))
        if not self.learnable_temp:
            self.logit_scale_seq.requires_grad_(False)
            self.logit_scale_ske.requires_grad_(False)

        for l in [self.in1, self.in2, self.pos_dense, self.time_pos_dense,
                  self.attn_out, self.ff1, self.ff2] + list(self.blocks_merge) + [self.frame_attn]:
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        for l in [self.g_r1, self.g_r2, self.tr1, self.tr2]:
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
        nn.init.xavier_uniform_(self.gpc_f1.weight)
        nn.init.xavier_uniform_(self.gpc_f2.weight)
        if self.joint_pos_embed is not None:
            nn.init.normal_(self.joint_pos_embed.weight, std=0.02)

    def encode_input(self, J_in):
        """Map XYZ coordinates to hidden features along the last dimension."""
        x = F.relu(self.in1(J_in), inplace=True)
        x = self.in2(x)
        return x

    def add_pos_and_attn(self, h, L_eig, use_pos=True, train=True, bn_freeze=False):
        """Encode joint features using position embeddings and attention.

        Args:
            h (torch.Tensor): Features with shape (B, T, J, H).
            L_eig (torch.Tensor | None): Spatial eigenvectors, shape (J, K).
            use_pos (bool): Add the configured position embeddings.
            train (bool): Apply dropout outside the attention modules.
            bn_freeze (bool): Force stored batch-normalization statistics.

        Returns:
            torch.Tensor: Encoded features with shape (B, T, J, H).

        Notes:
            Call model.eval() as well as train=False for inference, because
            attention dropout follows the PyTorch module's training state.
        """
        B, T, J, H = h.shape
        if use_pos:
            if self.spa_pos_type == 'learned' and self.joint_pos_embed is not None:
                idx = torch.arange(J, device=h.device, dtype=torch.long)
                pe_spa = self.joint_pos_embed(idx)
                pe_spa = pe_spa.unsqueeze(0).unsqueeze(0).expand(B, T, J, H)
                h = h + pe_spa
            elif L_eig is not None:
                pe_spa = self.pos_dense(L_eig)
                pe_spa = pe_spa.unsqueeze(0).unsqueeze(0).expand(B, T, J, H)
                h = h + pe_spa

            if self.use_time_attn:
                pe_tmp = self.time_pos_dense(self.time_pe)
                pe_tmp = pe_tmp.unsqueeze(0).unsqueeze(2).expand(B, T, J, H)
                h = h + pe_tmp

        res1 = h

        for l, spa_blk in enumerate(self.blocks_spa):
            h_in = self.blocks_norm[l](h) if self.pre_norm else h

            spa = spa_blk(h_in)

            if self.use_time_attn:
                tmp = self.blocks_tmp[l](h_in)
                merged = torch.cat([spa, tmp], dim=-1)
            else:
                merged = spa

            h = self.blocks_merge[l](merged)

        y = self.drop(h) if train else h
        y = self.attn_out(y)
        h = self.bn1(res1 + y, update_stats=not bn_freeze)

        res2 = h
        y = F.relu(self.ff1(h), inplace=True)
        y = self.drop(y) if train else y
        y = self.ff2(y)
        h = self.bn2(res2 + y, update_stats=not bn_freeze)
        return h

    def structure_prompt_recon(self, h, node_mask, J, T, S_prompt, gt_pos, batch_size):
        """Reconstruct all joint coordinates from the retained joint features."""
        if S_prompt == '0' or node_mask is None:
            return torch.tensor(0.0, device=h.device)
        mask = node_mask.to(h.device)
        h_mask = h[:, :, mask, :]
        if h_mask.shape[2] == 0:
            return torch.tensor(0.0, device=h.device)
        G_h = h_mask.mean(dim=-2)
        G_h = F.relu(self.g_r1(G_h), inplace=True)
        pred = self.g_r2(G_h)
        if S_prompt == 'l1':
            loss = F.l1_loss(pred, gt_pos, reduction='mean')
        elif S_prompt == 'l2':
            loss = 0.5 * F.mse_loss(pred, gt_pos, reduction='mean')
        else:
            loss = F.mse_loss(pred, gt_pos, reduction='mean')
        return loss

    def trajectory_prompt_recon(self, h, seq_mask, J, T, T_prompt, gt_pos, batch_size):
        """Reconstruct each joint trajectory from the retained frame features."""
        if T_prompt == '0' or seq_mask is None:
            return torch.tensor(0.0, device=h.device)
        mask = seq_mask.to(h.device)
        part = h[:, mask, :, :]
        part = part.transpose(1, 2).mean(dim=-2)
        h_seq1 = F.relu(self.tr1(part), inplace=True)
        pred = self.tr2(h_seq1)
        T_gt = gt_pos.reshape(gt_pos.size(0), T, J, 3).transpose(1, 2).reshape(gt_pos.size(0), J, T * 3)
        if T_prompt == 'l1':
            loss = F.l1_loss(pred, T_gt, reduction='mean')
        elif T_prompt == 'l2':
            loss = 0.5 * F.mse_loss(pred, T_gt, reduction='mean')
        else:
            loss = F.mse_loss(pred, T_gt, reduction='mean')
        return loss


# Compatibility name used by the original experiment and inference scripts.
TranSG = SSGT
