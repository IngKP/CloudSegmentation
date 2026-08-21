"""Export the selected segmentation model to a fixed-shape FP32 ONNX graph.

The research/model loading code intentionally lives in :mod:`model_pipeline`; this
module keeps ONNX-specific work small and importable in the serving environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_metadata(path: str | Path, **provenance: Any) -> dict[str, Any]:
    target = Path(path)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        **provenance,
    }


def mask_agreement(left: Any, right: Any, threshold: float = 0.5) -> float:
    import numpy as np

    a = np.asarray(left) >= threshold
    b = np.asarray(right) >= threshold
    return float(np.mean(a == b))


def export_model(model: Any, output: str | Path, channels: int, opset: int = 18) -> Path:
    import onnx
    import torch

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, channels, 512, 512), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(target),
            opset_version=opset,
            input_names=["input"],
            output_names=["score_map"],
            dynamic_axes=None,
            external_data=False,
        )
    onnx.checker.check_model(str(target))
    return target


def validate_onnx(
    path: str | Path, samples: list[Any], pytorch_outputs: list[Any]
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    agreements: list[float] = []
    max_abs: list[float] = []
    for sample, reference in zip(samples, pytorch_outputs, strict=True):
        value = np.asarray(sample, dtype=np.float32)
        actual = session.run(["score_map"], {name: value})[0]
        expected = np.asarray(reference)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("ONNX output differs from PyTorch beyond atol=rtol=1e-4")
        agreements.append(mask_agreement(actual, expected))
        max_abs.append(float(np.max(np.abs(actual - expected))))
    if agreements and min(agreements) < 0.9999:
        raise ValueError("ONNX thresholded mask agreement is below 99.99%")
    return {
        "max_abs_error": max(max_abs, default=0.0),
        "mask_agreement": min(agreements, default=1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/models.toml"))
    parser.add_argument("--research-root", type=Path, default=Path("../SARfloodEstimation"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path(".artifacts/checkpoints"))
    parser.add_argument("--artifact-root", type=Path, default=Path(".artifacts/onnx"))
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        default=None,
    )
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    sample_ids = args.sample_ids or ["Sri-Lanka_713926", "India_80221"]

    from model_pipeline import (
        build_and_load_model,
        load_manifest,
        load_two_samples,
        resolve_checkpoint,
        select_channels,
        verify_research_commit,
    )

    manifest = load_manifest(args.manifest)
    verify_research_commit(args.research_root, manifest["source_commit"])
    try:
        record = next(model for model in manifest["models"] if model["id"] == args.model_id)
    except StopIteration as exc:
        raise ValueError(f"unknown model ID: {args.model_id}") from exc
    checkpoint = resolve_checkpoint(record, args.checkpoint_root)
    model = build_and_load_model(record, checkpoint, args.research_root, "cpu")
    samples = load_two_samples(args.research_root, sample_ids)
    tensors = [
        select_channels(item["image"], record["channels"]).unsqueeze(0)
        for item in samples.values()
    ]
    import torch

    outputs = []
    with torch.no_grad():
        for tensor in tensors:
            outputs.append(model(tensor).detach().cpu().numpy())
    output = args.artifact_root / args.model_id / "model.fp32.onnx"
    export_model(model, output, len(record["channels"]))
    validation = validate_onnx(
        output, [tensor.detach().cpu().numpy() for tensor in tensors], outputs
    )
    metadata = artifact_metadata(
        output,
        model_id=args.model_id,
        variant="fp32",
        opset=18,
        input_name="input",
        output_name="score_map",
        input_shape=[1, len(record["channels"]), 512, 512],
        channels=record["channels"],
        samples=list(samples),
        source_commit=manifest["source_commit"],
        source_checkpoint_sha256=record["sha256"],
        release_policy=record["release_policy"],
        validation=validation,
    )
    target = args.metadata or output.with_suffix(".json")
    target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
