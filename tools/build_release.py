"""Build source archives from an explicit, reviewable file manifest.

Usage:
    python tools/build_release.py --output-dir ../dist

Only files listed in RELEASE_FILES.json are included. This prevents datasets,
checkpoints, virtual environments, and experiment outputs from entering a
release even when they are present in a developer's working directory.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def release_files(root=ROOT):
    """Validate the manifest and return relative paths in stable order."""
    names = json.loads((root / "RELEASE_FILES.json").read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError("Release manifest must contain unique relative paths.")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid release path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Release file is missing or is a symlink: {name}")
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Release file escapes the source tree: {name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpected large source file: {name}")
    return sorted(names)


def build(output_dir):
    """Create deterministic ZIP/TAR.GZ archives and SHA-256 checksums.

    Args:
        output_dir (Path): Directory receiving the generated release artifacts.

    Returns:
        list[Path]: The two archive paths and the checksum file.
    """
    names = release_files()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "SSGT-source.zip"
    tar_path = output_dir / "SSGT-source.tar.gz"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            entry = zipfile.ZipInfo("SSGT/" + name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (ROOT / name).read_bytes())
    with tar_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in names:
                    payload = (ROOT / name).read_bytes()
                    entry = tarfile.TarInfo("SSGT/" + name)
                    entry.size = len(payload)
                    entry.mode = 0o644
                    entry.mtime = 0
                    archive.addfile(entry, io.BytesIO(payload))
    checksums = output_dir / "SHA256SUMS"
    checksums.write_text("".join(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
        for path in (zip_path, tar_path)
    ), encoding="utf-8")
    return [zip_path, tar_path, checksums]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "dist")
    for artifact in build(parser.parse_args().output_dir):
        print(artifact)
