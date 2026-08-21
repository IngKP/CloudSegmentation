#!/usr/bin/env python3
"""Verify local ONNX artifacts against the tracked optimized-model manifest."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> list[Path]:
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported schema: {manifest.get('schema_version')!r}")
    root = Path(manifest["artifact_root"])
    verified = []
    for artifact in manifest["artifacts"]:
        path = root / artifact["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        verified.append(path)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/optimized_models.toml"),
    )
    args = parser.parse_args()
    for path in verify_manifest(args.manifest):
        print(f"verified {path}")


if __name__ == "__main__":
    main()
