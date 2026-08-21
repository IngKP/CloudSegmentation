"""Small, dependency-light bridge to the read-only SARfloodEstimation checkout.

The serving environment does not install torch or rasterio.  Consequently those
packages are imported only by the functions that need them, while manifest and
provenance helpers remain usable from ordinary tests and export tooling.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.10 export environment
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in export env
    import tomli as tomllib


DEFAULT_CHANNELS = ("VV", "VH", "DEM", "HAND", "Water")


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and minimally validate the project TOML model manifest."""
    with open(path, "rb") as handle:
        manifest = tomllib.load(handle)
    if not manifest.get("source_commit") or not manifest.get("models"):
        raise ValueError("manifest must contain source_commit and at least one model")
    for model in manifest["models"]:
        for field in ("id", "architecture", "channels", "filename", "sha256"):
            if field not in model:
                raise ValueError(f"model {model.get('id', '<unknown>')} lacks {field}")
    return manifest


def verify_research_commit(root: str | os.PathLike[str], expected: str) -> str:
    """Return the checkout HEAD and fail closed when it is not the expected commit."""
    result = subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"research checkout commit mismatch: expected {expected}, got {actual}")
    return actual


def resolve_checkpoint(model: dict[str, Any], artifact_root: str | os.PathLike[str]) -> Path:
    """Resolve, size-check, and hash-check a privately staged checkpoint."""
    path = Path(artifact_root) / model["id"] / model["filename"]
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = model.get("bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"checkpoint size mismatch for {model['id']}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != model["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch for {model['id']}")
    return path


def _normalise(image: Any) -> Any:
    """Apply the research BenchmarkNormalize transform without importing it."""
    import torch

    image = image.float()
    out = torch.empty_like(image)
    out[0:2] = (torch.clamp(image[0:2], -30.0, 0.0) + 30.0) / 30.0
    out[2] = (torch.clamp(image[2], -50.0, 3000.0) + 50.0) / 3050.0
    out[3] = torch.clamp(image[3], 0.0, 100.0) / 100.0
    out[4] = image[4]
    return torch.nan_to_num(out, nan=0.0)


def select_channels(image: Any, channels: list[str] | tuple[str, ...]) -> Any:
    """Select manifest channel names from the canonical five-band stack."""
    indices = [DEFAULT_CHANNELS.index(name) for name in channels]
    return image[indices]


def load_two_samples(
    research_root: str | os.PathLike[str], sample_ids: list[str] | tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Read fixed HandLabeled 5-band chips and labels using research paths."""
    import numpy as np
    import rasterio
    import torch

    root = (
        Path(research_root)
        / "data/sen1floods11/v1.1/data/flood_events/HandLabeled"
    )
    result: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        stack_path = root / "S1HandStacked" / f"{sample_id}_S1HandStacked.tif"
        label_path = root / "LabelHand" / f"{sample_id}_LabelHand.tif"
        if not stack_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing sample {sample_id}: {stack_path} / {label_path}")
        with rasterio.open(stack_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(label_path) as src:
            label = src.read(1)
        label = np.nan_to_num(label, nan=-1.0)
        label[np.isnan(image[0])] = -1
        image[0] = np.where(np.isnan(image[0]), -30.0, image[0])
        image[1] = np.where(np.isnan(image[1]), -30.0, image[1])
        image[2] = np.where(np.isnan(image[2]), -50.0, image[2])
        image[3:] = np.where(np.isnan(image[3:]), 0.0, image[3:])
        result[sample_id] = {
            "image": _normalise(torch.from_numpy(image)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "source_name": sample_id,
        }
    return result


def build_and_load_model(
    model_spec: dict[str, Any], checkpoint_path: str | os.PathLike[str],
    research_root: str | os.PathLike[str], device: str = "cpu",
) -> Any:
    """Construct the exact research model with pretrained encoders disabled."""
    import torch

    research = Path(research_root).resolve()
    if str(research) not in sys.path:
        sys.path.insert(0, str(research))
    create_model = importlib.import_module("src.models").create_model
    config = dict(model_spec.get("model_config", {}))
    config.update({
        "architecture": model_spec["architecture"],
        "in_channels": len(model_spec["channels"]),
        "out_channels": 1,
        "pretrained": False,
        "encoder_weights": None,
    })
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if state and next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.to(device).eval()
    return model
