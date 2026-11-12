"""Prepare explicit WUT and KGBD splits for SSGT open-set training.

A manifest assigns each identity and source sequence to a split. Input files
are checked against their sizes and SHA-256 values, then read in manifest order.
Prepared coordinates and integrity metadata are written to a separate directory;
raw identity directories are never moved or modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from preprocessing import _kgbd_txt_to_fusion, _lines_to_xyz_blocks
from preprocessing import _resample_fps, _win_fingerprint


JOINTS = {"WUT": 19, "KGBD": 20}
UNKNOWN_LABEL = {"WUT": 26, "KGBD": 165}
SPLITS = ("train", "gallery", "probe", "unknown")


# === Manifest Validation and File Integrity ===


def sha256_file(filename):
    """Return the SHA-256 digest of a file using bounded-memory reads."""
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest):
    """Hash canonical ASCII JSON with sorted keys and compact separators."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest paths must be nonempty relative strings: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest paths must be relative and stay within their root: {value!r}")
    return path


def _source_path(root, relative):
    root = Path(root).resolve()
    path = (root / _relative_path(relative)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Source resolves outside raw_root: {relative}")
    return path


def load_manifest(manifest_or_path):
    """Load a manifest dictionary or JSON file and validate its split structure."""
    if isinstance(manifest_or_path, dict):
        manifest = manifest_or_path
    else:
        with Path(manifest_or_path).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    """Check identity, sequence, label, path, and split disjointness.

    Args:
        manifest (dict): Version-one dataset manifest.

    Returns:
        dict: Counts of known and unknown original identities.

    Raises:
        ValueError: If the manifest is ambiguous, overlaps splits, or omits
        required sources or fingerprints. At least three known identities
        are required by the published multiclass data loader.
    """
    if manifest.get("schema_version") != 1 or manifest.get("dataset") not in JOINTS:
        raise ValueError("Expected schema_version=1 and dataset WUT or KGBD.")
    if type(manifest.get("length")) is not int or manifest["length"] <= 0:
        raise ValueError("Manifest length must be a positive integer.")
    _relative_path(manifest.get("data_subdir", str(manifest["length"])))
    identities = manifest.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("Manifest must contain an identities list.")
    names, labels, sources = set(), set(), set()
    counts = {"known": 0, "unknown": 0}
    for person in identities:
        name, label, role = person.get("name"), person.get("original_label"), person.get("role")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Original identity names must be unique nonempty strings.")
        if type(label) is not int or label < 1 or label in labels:
            raise ValueError("Original identity labels must be unique positive integers.")
        if label >= UNKNOWN_LABEL[manifest["dataset"]]:
            raise ValueError("Original identity labels must use the dataset's known identity range.")
        if role not in counts:
            raise ValueError("Identity role must be known or unknown.")
        names.add(name)
        labels.add(label)
        counts[role] += 1
        sequences = person.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"Identity {name} has no explicit sequences.")
        sequence_names, observed = set(), set()
        for sequence in sequences:
            seq_name, split = sequence.get("name"), sequence.get("split")
            if not isinstance(seq_name, str) or not seq_name or seq_name in sequence_names:
                raise ValueError(f"Identity {name} has a duplicate or invalid sequence name.")
            sequence_names.add(seq_name)
            allowed = {"unknown"} if role == "unknown" else {"train", "gallery", "probe"}
            if split not in allowed:
                raise ValueError(f"Identity {name} leaks between known and unknown splits.")
            observed.add(split)
            source = str(_relative_path(sequence.get("source")))
            if source in sources:
                raise ValueError(f"A source is assigned more than once: {source}")
            sources.add(source)
            if sequence.get("format") not in {"npy", "kgbd_txt"}:
                raise ValueError("Sequence format must be npy or kgbd_txt.")
            if manifest["dataset"] == "WUT" and sequence["format"] != "npy":
                raise ValueError("WUT sequences must use npy fusion arrays.")
            if not re.fullmatch(r"[0-9a-f]{64}", sequence.get("sha256", "")):
                raise ValueError(f"A SHA-256 fingerprint is required for {source}.")
            if type(sequence.get("size_bytes")) is not int or sequence["size_bytes"] < 0:
                raise ValueError(f"A non-negative byte count is required for {source}.")
        if role == "known" and observed != {"train", "gallery", "probe"}:
            raise ValueError(f"Known identity {name} must occur in train, gallery, and probe.")
    if counts["known"] < 3:
        raise ValueError("At least three known identities are required.")
    expected = manifest.get("expected_identities", counts)
    if expected != counts:
        raise ValueError(f"Identity counts do not match: expected {expected}, found {counts}.")
    settings = manifest.get("preprocessing", {})
    if settings.get("clip_num", 0) < 0 or settings.get("src_fps", 0) < 0 or settings.get("dst_fps", 0) < 0:
        raise ValueError("Preprocessing clipping and frame rates must be non-negative.")
    return counts


def verify_provenance(manifest_or_path, prepared_root, provenance=None):
    """Verify prepared output hashes and their binding to an explicit manifest."""
    manifest = load_manifest(manifest_or_path)
    root = Path(prepared_root).resolve()
    if provenance is None:
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("Prepared provenance belongs to a different manifest.")
    if provenance.get("dataset") != manifest["dataset"]:
        raise ValueError("Prepared provenance has a different dataset.")
    subdir = manifest.get("data_subdir", str(manifest["length"]))
    if provenance.get("data_subdir") != subdir:
        raise ValueError("Prepared provenance has a different data subdirectory.")
    hashes = provenance.get("output_sha256", {})
    if not hashes:
        raise ValueError("Prepared provenance contains no output hashes.")
    for relative, expected in hashes.items():
        path = _source_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Prepared output fingerprint mismatch: {relative}")
    known = sorted(p["original_label"] for p in manifest["identities"] if p["role"] == "known")
    if provenance.get("known_ids") != known:
        raise ValueError("Prepared known identity labels do not match the manifest.")
    dataset, length = manifest["dataset"], manifest["length"]
    expected_files = set()
    for split in SPLITS:
        count = provenance.get("windows", {}).get(split)
        if type(count) is not int or count < 0:
            raise ValueError("Prepared provenance requires non-negative counts for every split.")
        if not count:
            continue
        base = Path(dataset) / subdir
        if split == "train":
            target, prefix = base / "train_npy_data", "source"
        elif split == "unknown":
            target, prefix = base / "unknown_npy_data", "t_source"
        else:
            target, prefix = base / "test_npy_data" / split, "t_source"
        filenames = [f"{prefix}_{axis}_{dataset}_{length}.npy" for axis in "xyz"]
        filenames += [f"frame_id_{dataset}_{length}.npy", f"ids_{dataset}_{length}.npy"]
        expected_files.update((target / filename).as_posix() for filename in filenames)
        ids = np.load(root / target / filenames[-1], allow_pickle=True).item()
        labels = np.load(root / target / filenames[-2], allow_pickle=False)
        if len(labels) != count or sorted(ids) != sorted(set(labels.tolist())):
            raise ValueError(f"Prepared {split} labels/counts do not match its identity map.")
        if split == "train" and sorted(ids) != known:
            raise ValueError("Prepared training identities do not match the known manifest labels.")
        if split in {"gallery", "probe"} and not set(ids).issubset(known):
            raise ValueError(f"Prepared {split} contains a held-out identity.")
        if split == "unknown" and set(ids) != {UNKNOWN_LABEL[dataset]}:
            raise ValueError("Prepared unknown labels do not use the dataset unknown sentinel.")
    if set(hashes) != expected_files:
        raise ValueError("Prepared provenance does not fingerprint the complete expected file set.")
    return provenance


# === Non-destructive Prepared-Data Generation ===


def _sequence_windows(sequence, person, manifest, raw_root):
    path = _source_path(raw_root, sequence["source"])
    if not path.is_file() or path.stat().st_size != sequence["size_bytes"]:
        raise ValueError(f"Source file size mismatch: {sequence['source']}")
    if sha256_file(path) != sequence["sha256"]:
        raise ValueError(f"Source fingerprint mismatch: {sequence['source']}")
    array = np.load(path, allow_pickle=False) if sequence["format"] == "npy" else _kgbd_txt_to_fusion(path)
    dataset, length = manifest["dataset"], manifest["length"]
    if array.ndim != 3 or array.shape[1] != JOINTS[dataset] or array.shape[2] < 3:
        raise ValueError(f"Invalid skeleton shape for {sequence['source']}: {array.shape}")
    settings = manifest.get("preprocessing", {})
    source_fps, target_fps = settings.get("src_fps", 0), settings.get("dst_fps", 0)
    array = _resample_fps(array, source_fps, target_fps)
    clip = int(settings.get("clip_num", 0))
    if dataset == "KGBD" and person["role"] == "unknown":
        # KGBD unknown sequences use frame-based clipping and retain complete
        # windows without applying the known-sequence NaN filter.
        if clip:
            array = array[clip:-clip]
        end = len(array) // length * length
        windows = array[:end, :, :3].reshape(-1, length, JOINTS[dataset], 3)
    else:
        x, y, z = _lines_to_xyz_blocks(array, fr=length)
        if clip:
            x, y, z = x[clip * length:-clip * length], y[clip * length:-clip * length], z[clip * length:-clip * length]
        if not x:
            return np.empty((0, length, JOINTS[dataset], 3), dtype=np.float32)
        windows = np.stack([x, y, z], axis=-1).reshape(-1, length, JOINTS[dataset], 3)
    return windows.astype(np.float32)


def prepare_trial(manifest_or_path, raw_root, prepared_root):
    """Prepare one explicit split without modifying raw files or global paths.

    Args:
        manifest_or_path (dict or Path): Validated manifest or its JSON file.
        raw_root (str or Path): Root used to resolve manifest source filenames.
        prepared_root (str or Path): Dedicated output root for this trial.

    Returns:
        dict: Provenance containing manifest/source/output SHA-256 values,
        original identity labels, prepared window counts, and data_root.

    Raises:
        FileExistsError: If an existing dataset has no matching provenance.
        ValueError: If source hashes, sequence shapes, identity coverage, or
        output fingerprints fail validation. Numerical arrays are written
        after all raw sources and prepared split coverage have been checked.
    """
    manifest = load_manifest(manifest_or_path)
    raw, output = Path(raw_root).resolve(), Path(prepared_root).resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("Prepared and raw roots must be separate, non-nested directories.")
    dataset = manifest["dataset"]
    subdir = manifest.get("data_subdir", str(manifest["length"]))
    destination = output / dataset / subdir
    if (output / "provenance.json").exists():
        return verify_provenance(manifest, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data without provenance: {destination}")
    values = {split: [] for split in SPLITS}
    labels = {split: [] for split in SPLITS}
    seen_unknown = set()
    source_hashes = {}
    sequence_counts = []
    for person in manifest["identities"]:
        for sequence in person["sequences"]:
            windows = _sequence_windows(sequence, person, manifest, raw)
            source_hashes[sequence["source"]] = sequence["sha256"]
            split = sequence["split"]
            before = len(windows)
            if split == "unknown":
                retained = []
                for window in windows:
                    fingerprint = _win_fingerprint(window[:, :, 0], window[:, :, 1], window[:, :, 2])
                    if fingerprint not in seen_unknown:
                        seen_unknown.add(fingerprint)
                        retained.append(window)
                windows = np.asarray(retained, dtype=np.float32)
            count = len(windows)
            sequence_counts.append({"identity": person["name"], "sequence": sequence["name"],
                                    "split": split, "windows_before_deduplication": before,
                                    "windows": count})
            if count:
                values[split].append(windows)
                label = UNKNOWN_LABEL[dataset] if split == "unknown" else person["original_label"]
                labels[split].extend([label] * count)
    counts = {split: len(labels[split]) for split in SPLITS}
    known_ids = sorted(p["original_label"] for p in manifest["identities"] if p["role"] == "known")
    if sorted(set(labels["train"])) != known_ids:
        raise ValueError("Prepared training data does not cover every known identity.")
    for split in ("gallery", "probe"):
        if not labels[split] or not set(labels[split]).issubset(known_ids):
            raise ValueError(f"Prepared {split} must contain only known identities and be nonempty.")
    output_hashes = {}
    length = manifest["length"]
    for split in SPLITS:
        if not counts[split]:
            continue
        if split == "train":
            target, prefix = destination / "train_npy_data", "source"
        elif split == "unknown":
            target, prefix = destination / "unknown_npy_data", "t_source"
        else:
            target, prefix = destination / "test_npy_data" / split, "t_source"
        target.mkdir(parents=True, exist_ok=False)
        windows = np.concatenate(values[split], axis=0)
        identities = {}
        for index, label in enumerate(labels[split]):
            identities.setdefault(label, []).append(index)
        arrays = {f"{prefix}_{axis}_{dataset}_{length}.npy": windows[:, :, :, index].reshape(-1, JOINTS[dataset])
                  for index, axis in enumerate("xyz")}
        arrays[f"frame_id_{dataset}_{length}.npy"] = np.asarray(labels[split], dtype=np.int32)
        arrays[f"ids_{dataset}_{length}.npy"] = identities
        for filename, array in arrays.items():
            path = target / filename
            np.save(path, array)
            output_hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    provenance = {
        "schema_version": 1, "manifest_sha256": manifest_digest(manifest),
        "data_root": str(output), "dataset": dataset, "data_subdir": subdir,
        "trial_id": manifest.get("trial_id"), "known_ids": known_ids,
        "unknown_original_ids": sorted(p["original_label"] for p in manifest["identities"] if p["role"] == "unknown"),
        "windows": counts,
        "missing_known_ids": {split: sorted(set(known_ids) - set(labels[split]))
                              for split in ("train", "gallery", "probe")},
        "source_sha256": source_hashes, "output_sha256": output_hashes,
        "sequence_windows": sequence_counts,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return verify_provenance(manifest, output, provenance)


# === Command-Line Interface ===


def main(argv=None):
    """Prepare an explicit split in a dedicated data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON identity and sequence split manifest.")
    parser.add_argument("--raw-root", required=True, help="Root containing the manifest's relative source files.")
    parser.add_argument("--prepared-root", required=True, help="Separate output root to pass to main.py --data-dir.")
    args = parser.parse_args(argv)
    provenance = prepare_trial(args.manifest, args.raw_root, args.prepared_root)
    print(json.dumps({"data_root": provenance["data_root"], "windows": provenance["windows"]}, indent=2))


if __name__ == "__main__":
    main()
