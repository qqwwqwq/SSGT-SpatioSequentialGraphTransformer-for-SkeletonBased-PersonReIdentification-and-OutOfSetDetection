"""
Skeleton Dataset Preprocessing Module.

Converts raw WUT, KGBD, and KS20 skeleton sequences into the established coordinate
file format used by data.py. Preserves sequence splitting, windowing, NaN
handling, and unknown-window deduplication from the research implementation.
Input directories are explicit and all generated data use paths.data_root().
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import paths
from randomness import resolve_seed

DATASET_NATIVE_FPS = {"KGBD": 30.0, "WUT": 15.0}
DEFAULT_WUT_SPLIT = Path(__file__).resolve().parent / "config" / "wut_split.json"

# === Input Paths and Dataset Splits ===


def _require_input_dir(input_dir):
    """Resolve an explicitly supplied raw-data directory."""
    if input_dir is None:
        raise ValueError("Provide input_dir explicitly; no raw-data location is assumed.")
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return str(root)


def _load_wut_split(split_file=None):
    """Load zero-based gallery/probe positions in each sorted sequence list."""
    path = Path(split_file) if split_file else DEFAULT_WUT_SPLIT
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    gallery = split["gallery_indices"]
    probe = split["probe_indices"]
    indices = gallery + probe
    if len(gallery) != 2 or len(probe) != 1:
        raise ValueError("The original WUT protocol requires two gallery and one probe index.")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("WUT split indices must be non-negative integers.")
    if len(set(indices)) != len(indices):
        raise ValueError("WUT gallery and probe indices must not overlap.")
    return np.asarray(indices, dtype=np.int64)


def _load_kgbd_fusion(person_dir, seq_txt, fusion_file):
    """Read an existing fusion array, or parse raw text without writing a cache."""
    stem = Path(seq_txt).stem
    fusion_path = Path(person_dir) / stem / fusion_file
    if fusion_path.is_file():
        return np.load(fusion_path)
    txt_path = Path(person_dir) / seq_txt
    if not txt_path.is_file():
        raise FileNotFoundError(f"No sequence found at {txt_path} or {fusion_path}")
    return _kgbd_txt_to_fusion(txt_path)


# === Windowing and Dataset Conversion ===

def _fps_keep_indices(n_frames, src_fps, dst_fps):
    """Select original frames on a nearest-neighbor time grid.

    Args:
        n_frames (int): Number of input frames.
        src_fps (float): Original frame rate.
        dst_fps (float): Target frame rate.

    Returns:
        np.ndarray or None: Sorted indices, or None when no downsampling is
        requested. No new poses are synthesized.
    """
    if n_frames <= 0 or not src_fps or (not dst_fps) or (float(dst_fps) >= float(src_fps)):
        return None
    ratio = float(src_fps) / float(dst_fps)
    n_out = int((n_frames - 1) / ratio) + 1
    idx = np.floor(np.arange(n_out) * ratio + 0.5).astype(int)
    return np.unique(np.clip(idx, 0, n_frames - 1))


def _resample_fps(lines_array, src_fps, dst_fps):
    """Downsample an array of shape (T, J, 3) without synthesizing poses."""
    idx = _fps_keep_indices(len(lines_array), src_fps, dst_fps)
    return lines_array if idx is None else lines_array[idx]


def _lines_to_xyz_blocks(lines_array, fr, drop_nan=True):
    """Split a skeleton sequence into complete non-overlapping windows.

    Args:
        lines_array (np.ndarray): Sequence of shape (T, J, C), with C >= 3.
        fr (int): Window length in frames.
        drop_nan (bool): Drop an entire window when it contains a NaN.

    Returns:
        tuple: Three frame lists (x, y, z); incomplete final windows are
        discarded and retained windows are concatenated in order.
    """
    assert lines_array.ndim == 3 and lines_array.shape[-1] >= 3
    T = lines_array.shape[0]
    (x, y, z) = ([], [], [])
    for s in range(0, T - fr + 1, fr):
        block = lines_array[s:s + fr]
        if drop_nan and np.isnan(block).any():
            continue
        x.extend(block[:, :, 0].tolist())
        y.extend(block[:, :, 1].tolist())
        z.extend(block[:, :, 2].tolist())
    return (x, y, z)


def _win_fingerprint(x_win, y_win, z_win, nd=4):
    """Hash rounded window coordinates to deduplicate unknown windows."""
    arr = np.stack([x_win, y_win, z_win], axis=-1).astype(np.float32)
    arr = np.round(arr, nd)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def process_dataset_WUT(
    save_dir=None, fr=8, clip_num=0, fusion_file='fusion_master.npy',
    out_tag='', src_fps=0, dst_fps=0, input_dir=None, unknown_dir=None,
    data_subdir='', split_file=None,
):
    """Prepare WUT train, gallery, probe, and optional unknown windows.

    Args:
        save_dir: Legacy positional argument, retained for compatibility.
        fr (int): Window length in frames.
        clip_num (int): Original sequence-trimming parameter.
        fusion_file (str): Name of each precomputed skeleton array.
        out_tag (str): Optional suffix used when data_subdir is omitted.
        src_fps (float): Original frame rate; zero disables resampling.
        dst_fps (float): Target frame rate; zero disables resampling.
        input_dir (str): Directory containing known identity directories.
        unknown_dir (str): Optional unknown identity/sequence directory.
        data_subdir (str): Output directory below the dataset name.
        split_file (str): WUT split JSON; unused for KGBD.

    Returns:
        None: Writes coordinate arrays, per-window labels, and identity maps
        below ``paths.data_root() / 'WUT' / data_subdir``.

    Notes:
        WUT uses the explicit split JSON; KGBD keeps the original random
        sequence selection, controlled by the CLI seed. KGBD additionally
        accepts include_unknown to disable OOS preprocessing. Input files
        are read only; generated arrays are written to the data root.
    """
    subdir = data_subdir or (str(fr) if not out_tag else f'{fr}_{out_tag}')
    out_root = paths.data_root() / 'WUT' / subdir
    os.makedirs(out_root, exist_ok=True)
    if dst_fps and src_fps and (float(dst_fps) < float(src_fps)):
        print(f'[FPS] WUT frame-rate downsampling {src_fps} -> {dst_fps} FPS (before windowing)')
    overlap = fr
    time_steps = fr
    a = _require_input_dir(input_dir)
    if not os.path.isdir(a):
        raise FileNotFoundError(f'Input directory not found: {a}')
    p_dir = sorted(os.listdir(a))
    (label, X, Y, Z) = ([], [], [], [])
    (gal_label, gal_X, gal_Y, gal_Z) = ([], [], [], [])
    (pro_label, pro_X, pro_Y, pro_Z) = ([], [], [], [])
    dicp = {'Jacek': 1, 'Aleks': 2, 'Julia': 3, 'Vibek': 4, 'Mikolaj': 5, 'Ania': 6, 'Kasia': 7,
        'Martyna': 8, 'Maria': 9, 'MichalD': 10, 'MichalJ': 11, 'Ercem': 12, 'Malgosia': 13,
        'Moncy': 14, 'Juan': 15, 'Erdem': 16, 'Pasko': 17, 'Piotr': 18, 'Julianna': 19,
        'Showni': 20, 'Weronika': 21, 'Szykiedans': 22, 'Maks': 23, 'Kujawinska': 24,
        'Wojciech': 25}
    rand_seq_nums = _load_wut_split(split_file)
    gal_seq_num = rand_seq_nums[:2]
    pro_seq_num = rand_seq_nums[2:3]
    for p in p_dir:
        if p not in dicp:
            continue
        print(p)
        p_num = dicp[p]
        data_dir = os.path.join(a, p)
        fold_dir = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        print(gal_seq_num, pro_seq_num)
        print('Person ID: %d | Gallery Seqs: %d, %d | Probe Seq: %d | Total Seq: %d' % (p_num, gal_seq_num[0], gal_seq_num[1], pro_seq_num[0], len(fold_dir)))
        for (seq_num, seq_name) in enumerate(fold_dir):
            fpath = os.path.join(a, p, seq_name, fusion_file)
            if not os.path.isfile(fpath):
                continue
            lines = np.load(fpath)
            lines = _resample_fps(lines, src_fps, dst_fps)
            (x, y, z) = _lines_to_xyz_blocks(lines, fr=time_steps)
            if clip_num > 0:
                start = clip_num * fr
                end = None if clip_num == 0 else -clip_num * fr
                x = x[start:end if end != 0 else None]
                y = y[start:end if end != 0 else None]
                z = z[start:end if end != 0 else None]
            if seq_num not in rand_seq_nums:
                X.append(x)
                Y.append(y)
                Z.append(z)
                label.append(p_num)
            elif seq_num in gal_seq_num:
                gal_X.append(x)
                gal_Y.append(y)
                gal_Z.append(z)
                gal_label.append(p_num)
            elif seq_num in pro_seq_num:
                pro_X.append(x)
                pro_Y.append(y)
                pro_Z.append(z)
                pro_label.append(p_num)
    unknown_root = str(unknown_dir) if unknown_dir else ''
    (U_X, U_Y, U_Z, U_label) = ([], [], [], [])
    UNKNOWN_ID = 26
    if os.path.isdir(unknown_root):
        print('[Unknown] Processing Unknown/ ...')
        subdirs = [d for d in sorted(os.listdir(unknown_root)) if os.path.isdir(os.path.join(unknown_root, d))]
        if len(subdirs) == 0:
            subdirs = ['.']
        for person_dir in subdirs:
            seq_parent = unknown_root if person_dir == '.' else os.path.join(unknown_root, person_dir)
            seqs = [s for s in sorted(os.listdir(seq_parent)) if os.path.isdir(os.path.join(seq_parent, s))]
            if len(seqs) == 0 and os.path.isfile(os.path.join(seq_parent, fusion_file)):
                seqs = ['']
            for seq in seqs:
                fpath = os.path.join(seq_parent, fusion_file) if seq == '' else os.path.join(seq_parent, seq, fusion_file)
                if not os.path.isfile(fpath):
                    continue
                lines = np.load(fpath)
                lines = _resample_fps(lines, src_fps, dst_fps)
                (x, y, z) = _lines_to_xyz_blocks(lines, fr=time_steps)
                if clip_num > 0:
                    start = clip_num * fr
                    end = None if clip_num == 0 else -clip_num * fr
                    x = x[start:end if end != 0 else None]
                    y = y[start:end if end != 0 else None]
                    z = z[start:end if end != 0 else None]
                U_X.append(x)
                U_Y.append(y)
                U_Z.append(z)
                U_label.append(UNKNOWN_ID)
        print('[Unknown] Complete: %d unknown sequences.' % len(U_X))
    else:
        print('[Unknown] No unknown directory supplied; skipping OOS preprocessing.')

    def get_npy_data(out_root, X, Y, Z, label, type):
        """Write coordinate arrays and per-window identity metadata.

        Unknown windows are deduplicated within the unknown split only.
        """
        frames_cnt = 0
        ids = {}
        frame_id = []
        (all_wins_x, all_wins_y, all_wins_z, all_wins_pid) = ([], [], [], [])
        for s in range(0, 1):
            for (index, x_row) in enumerate(X):
                y_row = Y[index]
                z_row = Z[index]
                if type == 'XX':
                    t_x = [x_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(x_row) - overlap) // (overlap // 2) + 1)]
                    t_y = [y_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(y_row) - overlap) // (overlap // 2) + 1)]
                    t_z = [z_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(z_row) - overlap) // (overlap // 2) + 1)]
                else:
                    t_x = [x_row[i * overlap:i * overlap + overlap] for i in range(len(x_row) // overlap)]
                    t_y = [y_row[i * overlap:i * overlap + overlap] for i in range(len(y_row) // overlap)]
                    t_z = [z_row[i * overlap:i * overlap + overlap] for i in range(len(z_row) // overlap)]
                person = label[index]
                for k in range(len(t_x)):
                    all_wins_x.append(t_x[k])
                    all_wins_y.append(t_y[k])
                    all_wins_z.append(t_z[k])
                    all_wins_pid.append(person)
        if type == 'unknown':
            seen = set()
            keep = []
            for i in range(len(all_wins_x)):
                fp = _win_fingerprint(np.array(all_wins_x[i]), np.array(all_wins_y[i]), np.array(all_wins_z[i]))
                if fp in seen:
                    continue
                seen.add(fp)
                keep.append(i)
            if len(keep) < len(all_wins_x):
                print(f'[Unknown] Deduplication removed {len(all_wins_x) - len(keep)} duplicate windows.')
            all_wins_x = [all_wins_x[i] for i in keep]
            all_wins_y = [all_wins_y[i] for i in keep]
            all_wins_z = [all_wins_z[i] for i in keep]
            all_wins_pid = [all_wins_pid[i] for i in keep]
            if not all((pid == 26 for pid in all_wins_pid)):
                raise RuntimeError('[LEAK] Unknown windows must use identity label 26.')
        x_source = np.array([frame for window in all_wins_x for frame in window], dtype=np.float32)
        y_source = np.array([frame for window in all_wins_y for frame in window], dtype=np.float32)
        z_source = np.array([frame for window in all_wins_z for frame in window], dtype=np.float32)
        for (i, pid) in enumerate(all_wins_pid):
            if pid not in ids:
                ids[pid] = []
            ids[pid].append(i + frames_cnt)
        frame_id = [pid for pid in all_wins_pid]
        frames_cnt += len(all_wins_pid)
        if type == 'Train':
            outdir = os.path.join(out_root, 'train_npy_data')
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f'source_x_WUT_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f'source_y_WUT_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f'source_z_WUT_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_WUT_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_WUT_{fr}.npy'), np.array(frame_id, dtype=np.int32))
        elif type == 'unknown':
            outdir = os.path.join(out_root, 'unknown_npy_data')
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f't_source_x_WUT_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f't_source_y_WUT_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f't_source_z_WUT_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_WUT_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_WUT_{fr}.npy'), np.array(frame_id, dtype=np.int32))
        else:
            outdir = os.path.join(out_root, 'test_npy_data', type)
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f't_source_x_WUT_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f't_source_y_WUT_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f't_source_z_WUT_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_WUT_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_WUT_{fr}.npy'), np.array(frame_id, dtype=np.int32))
    get_npy_data(out_root, X, Y, Z, label, type='Train')
    get_npy_data(out_root, gal_X, gal_Y, gal_Z, gal_label, type='gallery')
    get_npy_data(out_root, pro_X, pro_Y, pro_Z, pro_label, type='probe')
    if len(U_X) > 0:
        get_npy_data(out_root, U_X, U_Y, U_Z, U_label, type='unknown')


def _kgbd_txt_to_fusion(txt_path: str) -> np.ndarray:
    """Parse a KGBD semicolon-separated coordinate file.

    Args:
        txt_path (str): Input file containing 20 joint records per frame.

    Returns:
        np.ndarray: Skeleton sequence of shape (T, 20, 3), dtype float32.
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    frames = []
    (x_t, y_t, z_t) = ([], [], [])
    cnt = 0
    for line in lines:
        coors = line.strip().split(';')[1:]
        if len(coors) < 3:
            continue
        x_t.append(float(coors[0]))
        y_t.append(float(coors[1]))
        z_t.append(float(coors[2]))
        cnt += 1
        if cnt % 20 == 0:
            frames.append(np.stack([x_t, y_t, z_t], axis=-1))
            (x_t, y_t, z_t) = ([], [], [])
    if not frames:
        return np.zeros((0, 20, 3), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32)


def process_dataset_KGBD(
    save_dir='KGBD', fr=6, clip_num=0, fusion_file='fusion_master.npy',
    include_unknown=True, out_tag='', src_fps=0, dst_fps=0,
    input_dir=None, unknown_dir=None, data_subdir='', split_file=None,
):
    """Prepare KGBD train, gallery, probe, and optional unknown windows.

    Args:
        save_dir: Legacy positional argument, retained for compatibility.
        fr (int): Window length in frames.
        clip_num (int): Original sequence-trimming parameter.
        fusion_file (str): Name of each precomputed skeleton array.
        include_unknown (bool): Include the explicitly supplied unknown data.
        out_tag (str): Optional suffix used when data_subdir is omitted.
        src_fps (float): Original frame rate; zero disables resampling.
        dst_fps (float): Target frame rate; zero disables resampling.
        input_dir (str): Directory containing known identity directories.
        unknown_dir (str): Optional unknown identity/sequence directory.
        data_subdir (str): Output directory below the dataset name.
        split_file (str): WUT split JSON; unused for KGBD.

    Returns:
        None: Writes coordinate arrays, per-window labels, and identity maps
        below ``paths.data_root() / 'KGBD' / data_subdir``.

    Notes:
        WUT uses the explicit split JSON; KGBD keeps the original random
        sequence selection, controlled by the CLI seed. KGBD additionally
        accepts include_unknown to disable OOS preprocessing. Input files
        are read only; generated arrays are written to the data root.
    """
    subdir = data_subdir or (str(fr) if not out_tag else f'{fr}_{out_tag}')
    out_root = paths.data_root() / 'KGBD' / subdir
    os.makedirs(out_root, exist_ok=True)
    if dst_fps and src_fps and (float(dst_fps) < float(src_fps)):
        print(f'[FPS] KGBD frame-rate downsampling {src_fps} -> {dst_fps} FPS (before windowing)')
    overlap = fr
    time_steps = fr
    a = _require_input_dir(input_dir)
    if not os.path.isdir(a):
        raise FileNotFoundError(f'KGBD input directory not found: {a}')
    p_dir = sorted(os.listdir(a))
    (label, X, Y, Z) = ([], [], [], [])
    (gal_label, gal_X, gal_Y, gal_Z) = ([], [], [], [])
    (pro_label, pro_X, pro_Y, pro_Z) = ([], [], [], [])
    for p in p_dir:
        if 'Person' not in p:
            continue
        p_num = int(p.split('Person')[1])
        data_dir = os.path.join(a, p)
        fold_dir = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        if not fold_dir:
            continue
        if p_num == 15:
            rand_seq_nums = np.random.choice(len(fold_dir), 2, replace=False)
            gal_seq_num = rand_seq_nums[:1]
            pro_seq_num = rand_seq_nums[1]
            print('Person ID: %d | Gallery Seqs: %d | Probe Seq: %d | Total Seq: %d' % (p_num, gal_seq_num[0], pro_seq_num, len(fold_dir)))
        else:
            rand_seq_nums = np.random.choice(len(fold_dir), 3, replace=False)
            gal_seq_num = rand_seq_nums[:2]
            pro_seq_num = rand_seq_nums[2]
            print('Person ID: %d | Gallery Seqs: %d, %d| Probe Seq: %d | Total Seq: %d' % (p_num, gal_seq_num[0], gal_seq_num[1], pro_seq_num, len(fold_dir)))
        for (seq_num, txt_ind) in enumerate(fold_dir):
            lines = _load_kgbd_fusion(data_dir, txt_ind, fusion_file)
            lines = _resample_fps(lines, src_fps, dst_fps)
            (x, y, z) = _lines_to_xyz_blocks(lines, fr=time_steps)
            if clip_num > 0:
                start = clip_num * fr
                end = None if clip_num == 0 else -clip_num * fr
                x = x[start:end if end != 0 else None]
                y = y[start:end if end != 0 else None]
                z = z[start:end if end != 0 else None]
            if seq_num not in rand_seq_nums:
                X.append(x)
                Y.append(y)
                Z.append(z)
                label.append(p_num)
            elif seq_num in gal_seq_num:
                gal_X.append(x)
                gal_Y.append(y)
                gal_Z.append(z)
                gal_label.append(p_num)
            elif seq_num == pro_seq_num:
                pro_X.append(x)
                pro_Y.append(y)
                pro_Z.append(z)
                pro_label.append(p_num)
    (U_X, U_Y, U_Z, U_label) = ([], [], [], [])
    if include_unknown:
        UNKNOWN_ID = 165
        unk_dir = str(unknown_dir) if unknown_dir else None
        if unk_dir:
            print(f'\n[Unknown] Processing {unk_dir}/ ...')
            subdirs = [d for d in sorted(os.listdir(unk_dir)) if os.path.isdir(os.path.join(unk_dir, d))]
            if len(subdirs) == 0:
                subdirs = ['.']
            for person_dir in subdirs:
                seq_parent = unk_dir if person_dir == '.' else os.path.join(unk_dir, person_dir)
                seqs = sorted([s for s in os.listdir(seq_parent) if s.endswith('.txt')])
                for seq in seqs:
                    fpath = os.path.join(seq_parent, seq)
                    with open(fpath, 'r') as f:
                        lines = f.readlines()
                        (x, y, z) = ([], [], [])
                        (x_t, y_t, z_t) = ([], [], [])
                        cnt = 0
                        for line in lines:
                            parts = line.strip().split(';')
                            if len(parts) < 4:
                                continue
                            coors = parts[1:]
                            x_t.append(float(coors[0]))
                            y_t.append(float(coors[1]))
                            z_t.append(float(coors[2]))
                            cnt += 1
                            if cnt % 20 == 0:
                                x.append(x_t)
                                y.append(y_t)
                                z.append(z_t)
                                (x_t, y_t, z_t) = ([], [], [])
                    keep = _fps_keep_indices(len(x), src_fps, dst_fps)
                    if keep is not None:
                        x = [x[i] for i in keep]
                        y = [y[i] for i in keep]
                        z = [z[i] for i in keep]
                    start_idx = clip_num if clip_num > 0 else None
                    end_idx = -clip_num if clip_num > 0 else None
                    x = x[start_idx:end_idx]
                    y = y[start_idx:end_idx]
                    z = z[start_idx:end_idx]
                    if len(x) >= fr:
                        U_X.append(x)
                        U_Y.append(y)
                        U_Z.append(z)
                        U_label.append(UNKNOWN_ID)
            print('[Unknown] Complete: %d unknown sequences.\n' % len(U_X))
        else:
            print('\n[Unknown] No unknown directory supplied; skipping OOS preprocessing.\n')
    else:
        print('\n[Unknown] include_unknown=False; skipping OOS preprocessing.\n')

    def get_npy_data(out_root, X, Y, Z, label, data_type):
        """Write coordinate arrays and per-window identity metadata.

        Unknown windows are deduplicated within the unknown split only.
        """
        frames_cnt = 0
        ids = {}
        frame_id = []
        (all_wins_x, all_wins_y, all_wins_z, all_wins_pid) = ([], [], [], [])
        for s in range(0, 1):
            for (index, xx_row) in enumerate(X):
                x_row = copy.deepcopy(xx_row[s:])
                y_row = copy.deepcopy(Y[index][s:])
                z_row = copy.deepcopy(Z[index][s:])
                if data_type == 'XX':
                    t_x = [x_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(x_row) - overlap) // (overlap // 2) + 1)]
                    t_y = [y_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(y_row) - overlap) // (overlap // 2) + 1)]
                    t_z = [z_row[i * overlap // 2:i * overlap // 2 + overlap] for i in range((len(z_row) - overlap) // (overlap // 2) + 1)]
                else:
                    t_x = [x_row[i * overlap:i * overlap + overlap] for i in range(len(x_row) // overlap)]
                    t_y = [y_row[i * overlap:i * overlap + overlap] for i in range(len(y_row) // overlap)]
                    t_z = [z_row[i * overlap:i * overlap + overlap] for i in range(len(z_row) // overlap)]
                person = label[index]
                for k in range(len(t_x)):
                    all_wins_x.append(t_x[k])
                    all_wins_y.append(t_y[k])
                    all_wins_z.append(t_z[k])
                    all_wins_pid.append(person)
        if data_type == 'unknown':
            seen = set()
            keep = []
            for i in range(len(all_wins_x)):
                fp = _win_fingerprint(all_wins_x[i], all_wins_y[i], all_wins_z[i])
                if fp in seen:
                    continue
                seen.add(fp)
                keep.append(i)
            if len(keep) < len(all_wins_x):
                print(f'[Unknown] Deduplication removed {len(all_wins_x) - len(keep)} duplicate windows.')
            all_wins_x = [all_wins_x[i] for i in keep]
            all_wins_y = [all_wins_y[i] for i in keep]
            all_wins_z = [all_wins_z[i] for i in keep]
            all_wins_pid = [all_wins_pid[i] for i in keep]
            if not all((pid == UNKNOWN_ID for pid in all_wins_pid)):
                raise RuntimeError(f'[LEAK] Unknown window labels must all equal {UNKNOWN_ID} .')
        if len(all_wins_x) == 0:
            return
        x_source = np.array([frame for window in all_wins_x for frame in window], dtype=np.float32)
        y_source = np.array([frame for window in all_wins_y for frame in window], dtype=np.float32)
        z_source = np.array([frame for window in all_wins_z for frame in window], dtype=np.float32)
        for (i, pid) in enumerate(all_wins_pid):
            if pid not in ids:
                ids[pid] = []
            ids[pid].append(i + frames_cnt)
        frame_id = [pid for pid in all_wins_pid]
        frames_cnt += len(all_wins_pid)
        if data_type == 'Train':
            outdir = os.path.join(out_root, 'train_npy_data')
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f'source_x_KGBD_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f'source_y_KGBD_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f'source_z_KGBD_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_KGBD_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_KGBD_{fr}.npy'), np.array(frame_id, dtype=np.int32))
        elif data_type == 'unknown':
            outdir = os.path.join(out_root, 'unknown_npy_data')
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f't_source_x_KGBD_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f't_source_y_KGBD_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f't_source_z_KGBD_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_KGBD_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_KGBD_{fr}.npy'), np.array(frame_id, dtype=np.int32))
        else:
            outdir = os.path.join(out_root, 'test_npy_data', data_type)
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f't_source_x_KGBD_{fr}.npy'), x_source)
            np.save(os.path.join(outdir, f't_source_y_KGBD_{fr}.npy'), y_source)
            np.save(os.path.join(outdir, f't_source_z_KGBD_{fr}.npy'), z_source)
            np.save(os.path.join(outdir, f'ids_KGBD_{fr}.npy'), ids)
            np.save(os.path.join(outdir, f'frame_id_KGBD_{fr}.npy'), np.array(frame_id, dtype=np.int32))
    get_npy_data(out_root, X, Y, Z, label, data_type='Train')
    get_npy_data(out_root, gal_X, gal_Y, gal_Z, gal_label, data_type='gallery')
    get_npy_data(out_root, pro_X, pro_Y, pro_Z, pro_label, data_type='probe')
    if len(U_X) > 0:
        get_npy_data(out_root, U_X, U_Y, U_Z, U_label, data_type='unknown')

# === Command-Line Interface ===


def _write_preprocessing_metadata(args, subdir):
    """Persist the chosen seed alongside the prepared coordinate files."""
    metadata = {
        'dataset': args.dataset,
        'sequence_length': args.length,
        'requested_seed': args.requested_seed,
        'resolved_seed': args.seed,
        'seed_selection': 'automatic' if args.requested_seed == -1 else 'explicit',
        'seed_affects_sequence_split': args.dataset == 'KGBD',
        'split_policy': ('seeded_sequence_selection' if args.dataset == 'KGBD'
                         else 'explicit_fixed_protocol'),
        'split_file': args.split_file,
    }
    destination = paths.data_root() / args.dataset / subdir / 'preprocessing_metadata.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'[Seed metadata] {destination}')


def main(argv=None):
    """Preprocess an explicit WUT, KGBD, or KS20 raw-data directory.

    Args:
        argv (list[str] or None): Arguments, or None to read sys.argv.

    Returns:
        None: Writes the dataset under paths.data_root().
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["WUT", "KGBD", "KS20"])
    parser.add_argument("--input_dir", required=True,
                        help="Raw root: WUT names, KGBD PersonXXX directories, or KS20 viewpoint folders.")
    parser.add_argument("--unknown_dir", default=None,
                        help="Optional unknown input root; never inferred from the current directory.")
    parser.add_argument("--length", "--fr", dest="length", type=int, default=6)
    parser.add_argument("--data_subdir", default="",
                        help="Output subdirectory, e.g. 6_10fps; default is the window length.")
    parser.add_argument("--out_tag", default="", help="Legacy suffix used when data_subdir is empty.")
    parser.add_argument("--fusion_file", default="fusion_master.npy")
    parser.add_argument("--split_file", default=None, help="Explicit WUT or KS20 JSON split configuration.")
    parser.add_argument("--clip_num", type=int, default=None,
                        help="Sequence trimming; defaults to 0 for WUT/KGBD and 10 per end for KS20.")
    parser.add_argument("--src_fps", type=float, default=0)
    parser.add_argument("--dst_fps", type=float, default=0)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Run seed in [0, 2**32-1]; -1 selects and records fresh entropy. WUT/KS20 protocol splits stay fixed.")
    parser.add_argument("--skip_unknown", nargs="?", const=1, default=0,
                        type=int, choices=[0, 1],
                        help="Disable unknown preprocessing; accepts a flag or 0/1.")
    args = parser.parse_args(argv)
    if args.length <= 0:
        parser.error("--length must be positive")
    if (args.clip_num is not None and args.clip_num < 0) or args.src_fps < 0 or args.dst_fps < 0:
        parser.error("--clip_num and frame rates must be non-negative")
    subdir = args.data_subdir or (f"{args.length}_{args.out_tag}" if args.out_tag else str(args.length))
    if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
        parser.error("--data_subdir must stay within the dataset output directory")
    if args.unknown_dir is not None and not Path(args.unknown_dir).is_dir():
        parser.error("--unknown_dir does not exist")
    args.requested_seed = args.seed
    try:
        args.seed = resolve_seed(args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    np.random.seed(args.seed)
    print(f'[Seed] Using preprocessing seed {args.seed}; '
          f'selection={"automatic" if args.requested_seed == -1 else "explicit"}.')
    if args.dataset == "KS20":
        if args.unknown_dir is not None or args.src_fps or args.dst_fps:
            parser.error("KS20 supports the closed-set protocol without unknown inputs or FPS resampling.")
        from ks20_preprocessing import process_dataset_KS20
        process_dataset_KS20(
            input_dir=args.input_dir, fr=args.length, data_subdir=subdir,
            split_file=args.split_file, clip_num=10 if args.clip_num is None else args.clip_num,
        )
        _write_preprocessing_metadata(args, subdir)
        print(f"Prepared KS20 data: {paths.data_root() / 'KS20' / subdir}")
        return
    common = dict(
        fr=args.length, clip_num=0 if args.clip_num is None else args.clip_num, fusion_file=args.fusion_file,
        input_dir=args.input_dir,
        unknown_dir=None if args.skip_unknown else args.unknown_dir,
        data_subdir=subdir, split_file=args.split_file,
        src_fps=args.src_fps or DATASET_NATIVE_FPS[args.dataset],
        dst_fps=args.dst_fps,
    )
    if args.dataset == "WUT":
        process_dataset_WUT(**common)
    else:
        process_dataset_KGBD(include_unknown=not args.skip_unknown, **common)
    _write_preprocessing_metadata(args, subdir)
    print(f"Prepared {args.dataset} data: {paths.data_root() / args.dataset / subdir}")


if __name__ == "__main__":
    main()
