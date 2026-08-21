#!/usr/bin/env python3
"""Stage private checkpoints through rclone and verify size and SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "manifests" / "models.toml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')!r}")
    return manifest


def verify(path: Path, model: dict[str, Any]) -> None:
    actual_size = path.stat().st_size
    expected_size = int(model["bytes"])
    if actual_size != expected_size:
        raise RuntimeError(
            f"{model['id']}: size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != model["sha256"]:
        raise RuntimeError(
            f"{model['id']}: SHA-256 mismatch: expected {model['sha256']}, got {actual_hash}"
        )


def stage(model: dict[str, Any], root: Path, verify_only: bool) -> Path:
    destination = root / model["id"] / model["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        if verify_only:
            raise FileNotFoundError(f"Missing staged artifact: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if temporary.exists():
            temporary.unlink()
        subprocess.run(
            ["rclone", "copyto", model["remote"], str(temporary), "--progress"],
            check=True,
        )
        verify(temporary, model)
        os.replace(temporary, destination)

    verify(destination, model)
    print(f"verified {model['id']}: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("rclone") is None and not args.verify_only:
        raise RuntimeError("rclone is required to stage remote artifacts")

    manifest = load_manifest(args.manifest)
    root = args.artifact_root
    if root is None:
        configured = os.environ.get("CLOUDSEG_ARTIFACT_ROOT")
        base = Path(configured) if configured else REPOSITORY_ROOT / ".artifacts"
        root = base / "checkpoints"

    requested = set(args.model)
    models = manifest["models"]
    known = {model["id"] for model in models}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown model IDs: {', '.join(sorted(unknown))}")

    selected = [model for model in models if not requested or model["id"] in requested]
    for model in selected:
        stage(model, root, args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

