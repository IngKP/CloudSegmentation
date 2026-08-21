from __future__ import annotations

import hashlib
import io
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def verify_file(path: Path, expected_size: int, expected_hash: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch: {path}")
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch: {path}")


def selected_model(manifest: dict[str, Any], model_path: Path) -> dict[str, Any]:
    variant = manifest["deployment_variant"]
    artifact = next(
        item for item in manifest["artifacts"] if item["variant"] == variant
    )
    verify_file(model_path, int(artifact["bytes"]), artifact["sha256"])
    return artifact


@dataclass(frozen=True)
class DemoSample:
    sample_id: str
    country: str
    split: str
    tensor: np.ndarray
    label: np.ndarray


def load_samples(manifest: dict[str, Any], root: Path) -> dict[str, DemoSample]:
    loaded: dict[str, DemoSample] = {}
    for record in manifest.get("samples", []):
        sample_id = record["id"]
        if sample_id in loaded:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        path = root / record["filename"]
        verify_file(path, int(record["bytes"]), record["sha256"])
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {"image", "label", "metadata"}:
                raise ValueError(f"unexpected arrays in sample bundle: {path}")
            tensor = np.array(bundle["image"], copy=True)
            label = np.array(bundle["label"], copy=True)
            metadata = json.loads(str(bundle["metadata"].item()))
        if tensor.shape != (5, 512, 512) or tensor.dtype != np.float32:
            raise ValueError(f"invalid input tensor in {path}")
        if label.shape != (512, 512) or label.dtype != np.int8:
            raise ValueError(f"invalid reference label in {path}")
        if not np.isfinite(tensor).all():
            raise ValueError(f"non-finite input tensor in {path}")
        if not np.isin(label, (-1, 0, 1)).all():
            raise ValueError(f"invalid reference classes in {path}")
        if metadata.get("sample_id") != sample_id:
            raise ValueError(f"sample metadata ID mismatch in {path}")
        if metadata.get("channels") != ["VV", "VH", "DEM", "HAND", "Water"]:
            raise ValueError(f"sample channel order mismatch in {path}")
        loaded[sample_id] = DemoSample(
            sample_id=sample_id,
            country=record.get("country", sample_id.split("_", 1)[0]),
            split=record.get("split", "test"),
            tensor=tensor,
            label=label,
        )
    if not loaded:
        raise ValueError("sample manifest contains no samples")
    return loaded


def encode_score_map(score_map: np.ndarray) -> bytes:
    scores = np.asarray(score_map, dtype=np.float32)
    if scores.shape != (512, 512):
        raise ValueError(f"unexpected score-map shape: {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("score map is outside [0, 1]")
    quantized = np.rint(scores * 65535.0).astype(np.uint16)
    output = io.BytesIO()
    Image.fromarray(quantized).save(output, format="PNG", optimize=False)
    return output.getvalue()


def encode_score_preview(score_map: np.ndarray) -> bytes:
    """Encode scores as a browser-friendly lossless 8-bit PNG preview."""
    scores = np.asarray(score_map, dtype=np.float32)
    if scores.shape != (512, 512):
        raise ValueError(f"unexpected score-map shape: {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("score map is outside [0, 1]")
    quantized = np.rint(scores * 255.0).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(quantized).save(output, format="PNG", optimize=False)
    return output.getvalue()


def encode_sar_visualization(tensor: np.ndarray) -> bytes:
    """Encode normalized VV/VH channels as a neutral grayscale SAR composite."""
    values = np.asarray(tensor, dtype=np.float32)
    if values.shape != (5, 512, 512):
        raise ValueError(f"unexpected input tensor shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("input tensor contains non-finite values")
    vv = np.clip(values[0], 0.0, 1.0)
    vh = np.clip(values[1], 0.0, 1.0)
    intensity = (vv + vh) / 2.0
    composite = np.repeat(intensity[..., np.newaxis], 3, axis=-1)
    output = io.BytesIO()
    Image.fromarray(np.rint(composite * 255.0).astype(np.uint8)).save(
        output, format="PNG", optimize=False
    )
    return output.getvalue()


def encode_reference(tensor: np.ndarray) -> bytes:
    """Encode labels losslessly as 0=land, 1=water, and 255=invalid PNG."""
    labels = np.asarray(tensor)
    if labels.shape != (512, 512):
        raise ValueError(f"unexpected reference shape: {labels.shape}")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("reference contains invalid classes")
    encoded = np.where(labels == -1, 255, labels).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(encoded).save(output, format="PNG", optimize=False)
    return output.getvalue()
