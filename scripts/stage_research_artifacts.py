#!/usr/bin/env python3
"""Stage private checkpoints and deterministic demo samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - export environment on Python 3.10
    import tomli as tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "manifests" / "models.toml"
DEFAULT_SAMPLE_MANIFEST = REPOSITORY_ROOT / "manifests" / "demo_samples.toml"


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


def load_sample_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the manifest for private, prepared sample bundles."""
    manifest = load_manifest(path)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample manifest must contain at least one sample")
    for sample in samples:
        for field in ("id", "filename", "bytes", "sha256"):
            if field not in sample:
                raise ValueError(f"sample {sample.get('id', '<unknown>')} lacks {field}")
        if sample.get("shape") != [5, 512, 512] or sample.get("label_shape") != [512, 512]:
            raise ValueError(f"sample {sample['id']}: unsupported shape")
    return manifest


def verify_sample(path: Path, sample: dict[str, Any]) -> None:
    """Verify a staged bundle's bytes, hash, and array contract."""
    verify(path, sample)
    import numpy as np

    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"image", "label", "metadata"}:
            raise RuntimeError(f"{sample['id']}: invalid bundle members")
        if bundle["image"].shape != (5, 512, 512) or bundle["image"].dtype != np.float32:
            raise RuntimeError(f"{sample['id']}: image must be float32[5,512,512]")
        if bundle["label"].shape != (512, 512) or bundle["label"].dtype != np.int8:
            raise RuntimeError(f"{sample['id']}: label must be int8[512,512]")
        metadata = json.loads(str(bundle["metadata"].item()))
        if metadata.get("sample_id") != sample["id"]:
            raise RuntimeError(f"{sample['id']}: metadata sample ID mismatch")


def _bundle_bytes(image: Any, label: Any, metadata: dict[str, Any]) -> bytes:
    import numpy as np

    output = io.BytesIO()
    np.savez_compressed(
        output,
        image=np.asarray(image, dtype=np.float32),
        label=np.asarray(label, dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    return output.getvalue()


def stage_samples(
    manifest: dict[str, Any], research_root: Path, root: Path, sample_ids: list[str] | None,
    verify_only: bool,
) -> None:
    """Prepare and verify fixed research samples in the export environment."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    from model_pipeline import load_two_samples, verify_research_commit

    verify_research_commit(research_root, manifest["source_commit"])
    records = {sample["id"]: sample for sample in manifest["samples"]}
    selected_ids = sample_ids or list(records)
    unknown = set(selected_ids) - records.keys()
    if unknown:
        raise ValueError(f"Unknown sample IDs: {', '.join(sorted(unknown))}")
    if verify_only:
        for sample_id in selected_ids:
            verify_sample(root / records[sample_id]["filename"], records[sample_id])
        return
    loaded = load_two_samples(research_root, selected_ids)
    for sample_id in selected_ids:
        item = loaded[sample_id]
        image = item["image"].detach().cpu().numpy()
        label = item["label"].detach().cpu().numpy()
        metadata = {
            "sample_id": sample_id,
            "source_commit": manifest["source_commit"],
            "target_semantics": manifest.get("target_semantics", "surface water mask"),
            "channels": ["VV", "VH", "DEM", "HAND", "Water"],
            "image_dtype": "float32",
            "label_dtype": "int8",
            "invalid_label": -1,
        }
        payload = _bundle_bytes(image, label, metadata)
        record = records[sample_id]
        destination = root / record["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        verify_sample(destination, record)
        print(f"staged {sample_id}: {destination}")


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
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--research-root", type=Path, default=Path("../SARfloodEstimation"))
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_mode:
        manifest = load_sample_manifest(args.sample_manifest)
        root = args.artifact_root or REPOSITORY_ROOT / ".artifacts" / "samples"
        stage_samples(manifest, args.research_root, root, args.sample_ids, args.verify_only)
        return
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
