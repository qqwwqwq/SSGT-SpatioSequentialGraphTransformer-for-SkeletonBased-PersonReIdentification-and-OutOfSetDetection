"""Convert KS20 MATLAB recordings into train, gallery, and probe arrays.

Each identity uses a distinct gait recording for each split across five views.
The converter trims sequence ends and creates shifted windows for training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import scipy.io as sio

import paths

VIEWS = ("frontal", "left_diagonal", "left_lateral", "right_diagonal", "right_lateral")
BODY_IDS = (1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24)
SETTINGS = frozenset(("gc1", "gc2", "gc3"))
DEFAULT_SPLIT = Path(__file__).resolve().parent / "config" / "ks20_split.json"

# === MATLAB Skeleton Parsing ===

def _to_native(obj):
    """Convert MATLAB structs and object arrays to Python containers."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_to_native(x) for x in obj.flat]
        return obj
    if hasattr(obj, '_fieldnames'):
        d = {}
        for k in obj._fieldnames:
            d[k] = _to_native(getattr(obj, k))
        return d
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj

def _finish_25x3(M, pad_mode='zeros'):
    """Keep three coordinate columns and pad or truncate to 25 joints."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        return None
    if M.shape[1] < 3 and M.shape[0] >= 3:
        M = M.T
    if M.shape[1] >= 3:
        M = M[:, :3]
    else:
        return None
    rows = M.shape[0]
    if rows == 25:
        return M
    if rows > 25:
        return M[:25]
    if pad_mode == 'repeat' and rows > 0:
        pad = np.repeat(M[-1:], 25 - rows, axis=0)
    else:
        pad = np.zeros((25 - rows, 3), dtype=float)
    return np.vstack([M, pad])

def _frame_from_joints_dictlike(frame_like):
    """Extract joint positions by their Kinect v2 type indices."""
    if not isinstance(frame_like, dict):
        return None
    body = frame_like.get('body', frame_like)
    if not isinstance(body, dict):
        return None
    joints = body.get('joints', None)
    if not isinstance(joints, (list, tuple)) or len(joints) == 0:
        return None
    M = np.zeros((25, 3), dtype=float)
    seen = np.zeros(25, dtype=bool)
    for jd in joints:
        if not isinstance(jd, dict):
            continue
        t = jd.get('type', None)
        try:
            idx = int(t)
        except Exception:
            continue
        if not 0 <= idx < 25:
            continue
        pos = jd.get('position', None)
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue
        try:
            (x, y, z) = (float(pos[0]), float(pos[1]), float(pos[2]))
        except Exception:
            continue
        M[idx, 0] = x
        M[idx, 1] = y
        M[idx, 2] = z
        seen[idx] = True
    if not seen.any():
        return None
    return _finish_25x3(M)

def _collect_frames_any(obj):
    """Collect skeleton frames from nested MATLAB objects in stored order."""
    out = []

    def visit(x):
        M = _frame_from_joints_dictlike(x) if isinstance(x, dict) else None
        if M is not None:
            out.append(M)
            return
        if isinstance(x, (list, tuple)):
            for it in x:
                visit(it)
            return
        if isinstance(x, dict):
            for v in x.values():
                visit(v)
            return
        if isinstance(x, np.ndarray) and x.dtype == object:
            for it in x.flat:
                visit(it)
            return
    visit(_to_native(obj))
    return out

def _choose_object_array_var(md, prefer='gc'):
    """Select the named MATLAB object array, or the first object-array fallback."""
    keys = [k for k in md.keys() if not k.startswith('__')]
    if prefer in md and isinstance(md[prefer], np.ndarray) and (md[prefer].dtype == object):
        return prefer
    for k in keys:
        v = md[k]
        if isinstance(v, np.ndarray) and v.dtype == object and (v.size > 0):
            return k
    return None

def extract_frames_from_mat(file_path, var_name='gc', fill_missing_with_prev=True, verbose=False):
    """Read a KS20 MATLAB sequence with the inherited parser.

    Args:
        file_path (str or Path): MATLAB v5/v7 file containing skeleton frames.
        var_name (str): Preferred object-array variable, normally gc.
        fill_missing_with_prev (bool): Fill entirely zero joint rows from the
            previous parsed frame. Missing rows in the first frame stay zero.
        verbose (bool): Print diagnostics when parsing fails.

    Returns:
        np.ndarray or None: Float32 coordinates of shape (T, 25, 3), or None
        when no valid skeleton frames can be extracted.
    """
    try:
        md = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        if verbose:
            print(f'[Warning] Cannot read MATLAB file: {file_path} | {e}')
        return None
    use_key = _choose_object_array_var(md, prefer=var_name)
    if use_key is None:
        if verbose:
            ks = [k for k in md.keys() if not k.startswith('__')]
            print(f"[Warning] No object-array variable (preferred '{var_name}') | File: {file_path} | Available keys: {ks}")
        return None
    obj_arr = np.asarray(md[use_key])
    frames_raw = []
    for item in obj_arr.flat:
        Fs = _collect_frames_any(item)
        if Fs:
            frames_raw.extend(Fs)
            continue
        item_native = _to_native(item)
        if isinstance(item_native, np.ndarray):
            if item_native.ndim == 2 or item_native.ndim == 1:
                M = _finish_25x3(item_native)
                if M is not None:
                    frames_raw.append(M)
                    continue
            if item_native.ndim == 3:
                axis3 = next((i for (i, s) in enumerate(item_native.shape) if s == 3), None)
                if axis3 is not None:
                    B = np.moveaxis(item_native, axis3, -1)[..., :3]
                    if B.ndim == 3:
                        for t in range(B.shape[0]):
                            Mt = _finish_25x3(B[t])
                            if Mt is not None:
                                frames_raw.append(Mt)
    if not frames_raw:
        if verbose:
            print(f"[Warning] No frames found in {file_path} variable '{use_key}'")
        return None
    seq = []
    prev = None
    for M in frames_raw:
        if fill_missing_with_prev and prev is not None:
            missing = np.where(np.all(M == 0.0, axis=1))[0]
            if missing.size > 0:
                M[missing] = prev[missing]
        seq.append(M.astype(np.float32))
        prev = M
    return np.stack(seq, axis=0)


# === Explicit Split and Window Conversion ===

def load_split(split_file=None):
    """Validate per-identity settings and return a label-to-record mapping.

    Args:
        split_file (str or Path or None): Explicit JSON settings. None selects
            config/ks20_split.json.

    Returns:
        dict: Original contiguous labels 0..19 mapped to body IDs and the
        distinct train, gallery, and probe settings for each identity.
    """
    source = Path(split_file) if split_file else DEFAULT_SPLIT
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("dataset") != "KS20" or document.get("schema_version") != 1:
        raise ValueError("Expected a KS20 split document with schema_version=1.")
    if document.get("views") != list(VIEWS):
        raise ValueError("The KS20 protocol requires the five views in their recorded order.")
    records = document.get("identities", [])
    if len(records) != 20:
        raise ValueError("The KS20 split must specify all 20 identities.")
    mapping = {}
    for record in records:
        label = record.get("label")
        if type(label) is not int or not 0 <= label < 20 or label in mapping:
            raise ValueError("KS20 labels must be unique integers from 0 through 19.")
        if type(record.get("body_id")) is not int or record["body_id"] != BODY_IDS[label]:
            raise ValueError("KS20 body IDs must preserve the original sparse-to-contiguous mapping.")
        if {record.get(split) for split in ("train", "gallery", "probe")} != SETTINGS:
            raise ValueError("Each identity must use distinct gc1/gc2/gc3 settings across splits.")
        mapping[label] = record
    return mapping


def _write_split(output, split, sequences, length):
    """Write axis-major windows in the original shift/view/filename order."""
    windows, labels, identities, counts = [], [], {}, {}
    for shift in range(length if split == "train" else 1):
        for relative, label, sequence in sequences:
            identities.setdefault(label, [])
            stride = length // 2 if split == "train" else length
            number = max(0, (len(sequence) - shift - length) // stride + 1)
            first = len(windows)
            identities[label].extend(range(first, first + number))
            labels.extend([label] * number)
            counts[relative] = counts.get(relative, 0) + number
            for index in range(number):
                start = shift + index * stride
                windows.append(sequence[start:start + length])
    # The original parser produced float32 frames, converted them to Python
    # lists, then wrote float64 coordinate files. Preserve that promotion.
    coordinates = np.asarray(windows, dtype=np.float64).reshape(-1, 25, 3)
    directory = output / ("train_npy_data" if split == "train" else "test_npy_data/" + split)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "" if split == "train" else "t_"
    for axis, name in enumerate("xyz"):
        np.save(directory / f"{prefix}source_{name}_KS20_{length}.npy", coordinates[:, :, axis])
    np.save(directory / f"ids_KS20_{length}.npy", identities)
    np.save(directory / f"frame_id_KS20_{length}.npy", np.asarray(labels, dtype=np.int64))
    return {"window_count": len(labels), "sequence_count": len(sequences),
            "windows_by_sequence": counts,
            "missing_identity_windows": [label for label in range(20) if not identities.get(label)]}


def process_dataset_KS20(input_dir, fr=6, data_subdir="", split_file=None, clip_num=10):
    """Convert the complete KS20 collection without modifying source files.

    Args:
        input_dir (str or Path): Raw root containing the five viewpoint folders.
        fr (int): Window length, at least two frames.
        data_subdir (str): Output variant below KS20; defaults to fr.
        split_file (str or Path or None): Explicit per-identity JSON split.
        clip_num (int): Number of frames removed from each sequence end.

    Returns:
        dict: Source hashes, exact split selections, and per-sequence window
        counts, also written to preprocessing_provenance.json beside the data.

    Raises:
        ValueError: If settings, source coverage, or skeleton coordinates are invalid.
        FileNotFoundError: If a required viewpoint or MAT sequence is missing.
    """
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KS20 raw directory not found: {root}")
    if fr < 2 or clip_num < 0:
        raise ValueError("KS20 requires length >= 2 and clip_num >= 0.")
    subdir = Path(data_subdir or str(fr))
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("data_subdir must stay inside the KS20 output directory.")
    output = (paths.data_root() / "KS20" / subdir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("KS20 raw and prepared directories must be separate, non-nested roots.")
    assignments = load_split(split_file)
    by_body = {record["body_id"]: label for label, record in assignments.items()}
    expected = {(view, setting, body) for view in VIEWS for setting in SETTINGS for body in BODY_IDS}
    observed, source_records = set(), {}
    sequences = {split: [] for split in ("train", "gallery", "probe")}
    filename_pattern = re.compile(r"^(gc[123])_body(\d+)\.mat$", re.IGNORECASE)
    for view in VIEWS:
        directory = root / view
        if not directory.is_dir():
            raise FileNotFoundError(f"Required KS20 viewpoint is missing: {directory}")
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            match = filename_pattern.fullmatch(path.name)
            if not path.is_file() or match is None:
                continue
            setting, body = match.group(1).lower(), int(match.group(2))
            if body not in by_body:
                continue
            key = (view, setting, body)
            if key in observed:
                raise ValueError(f"Duplicate KS20 sequence: {view}/{setting}_body{body}")
            observed.add(key)
            label = by_body[body]
            split = next(name for name in sequences if assignments[label][name] == setting)
            coordinates = extract_frames_from_mat(path)
            if coordinates is None or coordinates.ndim != 3 or coordinates.shape[1:] != (25, 3):
                raise ValueError(f"Cannot parse a valid KS20 skeleton sequence: {path}")
            if not np.isfinite(coordinates).all():
                raise ValueError(f"KS20 coordinates must be finite: {path}")
            frames = len(coordinates)
            if clip_num:
                coordinates = coordinates[clip_num:-clip_num] if frames >= 2 * clip_num else coordinates[:0]
            relative = path.relative_to(root).as_posix()
            sequences[split].append((relative, label, coordinates))
            source_records[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size, "identity": label, "body_id": body,
                "split": split, "frames": frames, "frames_after_trim": len(coordinates)}
    missing = sorted(expected - observed)
    if missing:
        raise FileNotFoundError(f"KS20 requires all 300 sequences; missing {len(missing)}, first: {missing[0]}")
    provenance = {"dataset": "KS20", "length": fr, "clip_num": clip_num,
        "views": list(VIEWS), "identities": [assignments[label] for label in range(20)],
        "windowing": {"train_shifts": list(range(fr)), "train_stride": fr // 2,
                      "gallery_probe_shift": 0, "gallery_probe_stride": fr},
        "source_files": source_records, "splits": {}}
    for split in sequences:
        provenance["splits"][split] = _write_split(output, split, sequences[split], fr)
    provenance["output_sha256"] = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*.npy"))}
    (output / "preprocessing_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


# === Command-Line Interface ===

def main(argv=None):
    """Convert KS20 recordings using the configured sequence split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Raw root containing all five viewpoint folders.")
    parser.add_argument("--length", "--fr", dest="length", type=int, default=6)
    parser.add_argument("--data_subdir", default="", help="Variant below KS20; defaults to the window length.")
    parser.add_argument("--split_file", default=None, help="KS20 JSON split; defaults to config/ks20_split.json.")
    parser.add_argument("--clip_num", type=int, default=10, help="Frames removed from each sequence end.")
    args = parser.parse_args(argv)
    if args.length < 2 or args.clip_num < 0:
        parser.error("--length must be at least 2 and --clip_num must be non-negative")
    process_dataset_KS20(args.input_dir, args.length, args.data_subdir, args.split_file, args.clip_num)
    print(f"Prepared KS20 data: {paths.data_root() / 'KS20' / (args.data_subdir or str(args.length))}")


if __name__ == "__main__":
    main()
