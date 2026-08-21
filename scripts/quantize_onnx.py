"""Experimental INT8/INT4 feasibility checks for a fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .export_onnx import artifact_metadata, mask_agreement
except ImportError:  # pragma: no cover - direct script execution
    from export_onnx import artifact_metadata, mask_agreement


def eligible_int4_nodes(graph: Any) -> list[str]:
    """Return constant-weight MatMul/Gather node names supported by ORT INT4."""
    initializer_names = {
        getattr(value, "name", value.get("name", "") if isinstance(value, dict) else "")
        for value in getattr(graph, "initializer", [])
    }
    found: list[str] = []
    for node in getattr(graph, "node", graph if isinstance(graph, list) else []):
        op = getattr(node, "op_type", node.get("op_type") if isinstance(node, dict) else "")
        inputs = list(
            getattr(node, "input", node.get("input", []) if isinstance(node, dict) else [])
        )
        if op not in {"MatMul", "Gather"}:
            continue
        constant = (
            op == "MatMul" and len(inputs) > 1 and inputs[1] in initializer_names
        ) or (op == "Gather" and inputs and inputs[0] in initializer_names)
        if constant:
            found.append(
                getattr(node, "name", node.get("name", "") if isinstance(node, dict) else "")
            )
    return found


def summarize_outputs(reference: Any, actual: Any, elapsed: list[float]) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(actual)
    ref = np.asarray(reference)
    return {
        "finite": bool(np.isfinite(values).all()),
        "range": [float(values.min()), float(values.max())],
        "max_abs_error": float(np.max(np.abs(values - ref))),
        "mask_agreement": mask_agreement(values, ref),
        "mean_ms": float(np.mean(elapsed) * 1000),
        "p50_ms": float(np.percentile(elapsed, 50) * 1000),
        "p95_ms": float(np.percentile(elapsed, 95) * 1000),
    }


def attempt_int8(model_path: Path, output: Path, samples: list[Any]) -> dict[str, Any]:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._items = iter({"input": sample} for sample in samples)

        def get_next(self) -> dict[str, Any] | None:
            return next(self._items, None)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            str(model_path),
            str(output),
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv"],
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )
        ort.InferenceSession(
            str(output), sess_options=_cpu_options(), providers=["CPUExecutionProvider"]
        )
        return {"status": "ok", **artifact_metadata(output, variant="int8")}
    except Exception as exc:  # experimental path must not block FP32 delivery
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def attempt_int4(model_path: Path, output: Path, nodes: list[str]) -> dict[str, Any]:
    try:
        import onnxruntime as ort
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
        from onnxruntime.quantization.quant_utils import QuantFormat

        config = DefaultWeightOnlyQuantConfig(
            block_size=128,
            is_symmetric=True,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul", "Gather"),
            quant_axes=(("MatMul", 0), ("Gather", 1)),
            bits=4,
        )
        quantizer = MatMulNBitsQuantizer(
            str(model_path), nodes_to_include=nodes, algo_config=config
        )
        quantizer.process()
        output.parent.mkdir(parents=True, exist_ok=True)
        quantizer.model.save_model_to_file(str(output), use_external_data_format=False)
        ort.InferenceSession(
            str(output), sess_options=_cpu_options(), providers=["CPUExecutionProvider"]
        )
        return {"status": "ok", **artifact_metadata(output, variant="int4")}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _cpu_options() -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _benchmark_session(
    session: Any, samples: list[Any], warmups: int, runs: int
) -> tuple[list[Any], list[float]]:
    outputs: list[Any] = []
    times: list[float] = []
    for sample in samples:
        for _ in range(warmups):
            session.run(None, {"input": sample})
        output = None
        for _ in range(runs):
            start = time.perf_counter()
            output = session.run(None, {"input": sample})[0]
            times.append(time.perf_counter() - start)
        outputs.append(output)
    return outputs, times


def _output_summary(
    outputs: list[Any], references: list[Any], times: list[float]
) -> dict[str, Any]:
    import numpy as np

    return {
        "finite": bool(all(np.isfinite(value).all() for value in outputs)),
        "range": [
            float(min(np.min(value) for value in outputs)),
            float(max(np.max(value) for value in outputs)),
        ],
        "max_abs_error": float(
            max(np.max(np.abs(actual - expected)) for actual, expected in zip(
                outputs, references, strict=True
            ))
        ),
        "mask_agreement": float(
            min(mask_agreement(actual, expected) for actual, expected in zip(
                outputs, references, strict=True
            ))
        ),
        "mean_ms": float(np.mean(times) * 1000),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "observations": len(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/models.toml"))
    parser.add_argument("--research-root", type=Path, default=Path("../SARfloodEstimation"))
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        default=None,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    import onnx
    import onnxruntime as ort
    from model_pipeline import load_manifest, load_two_samples, select_channels

    sample_ids = args.sample_ids or ["Sri-Lanka_713926", "India_80221"]
    manifest = load_manifest(args.manifest)
    try:
        record = next(model for model in manifest["models"] if model["id"] == args.model_id)
    except StopIteration as exc:
        raise ValueError(f"unknown model ID: {args.model_id}") from exc
    loaded = load_two_samples(args.research_root, sample_ids)
    samples = [
        select_channels(item["image"], record["channels"]).unsqueeze(0).numpy()
        for item in loaded.values()
    ]
    graph = onnx.load(str(args.model)).graph
    output_dir = args.output_dir or args.model.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model": str(args.model),
        "model_id": args.model_id,
        "samples": sample_ids,
        "protocol": {
            "cpu_threads": 1,
            "warmups_per_sample": args.warmups,
            "runs_per_sample": args.runs,
            "result_label": "two-image latency estimate",
        },
        "int4": {"status": "not_applicable_no_supported_ops"},
    }
    nodes = eligible_int4_nodes(graph)
    if nodes:
        int4_path = output_dir / "model.int4.onnx"
        result["int4"] = attempt_int4(args.model, int4_path, nodes)
        result["int4"]["eligible_nodes"] = nodes
    int8_path = output_dir / "model.int8.onnx"
    result["int8"] = attempt_int8(args.model, int8_path, samples)
    session = ort.InferenceSession(
        str(args.model), sess_options=_cpu_options(), providers=["CPUExecutionProvider"]
    )
    outputs, times = _benchmark_session(session, samples, args.warmups, args.runs)
    result["fp32"] = _output_summary(outputs, outputs, times)
    if result["int8"].get("status") == "ok":
        int8_session = ort.InferenceSession(
            str(int8_path), sess_options=_cpu_options(), providers=["CPUExecutionProvider"]
        )
        int8_outputs, int8_times = _benchmark_session(
            int8_session, samples, args.warmups, args.runs
        )
        result["int8"].update(_output_summary(int8_outputs, outputs, int8_times))
        result["int8"]["promising"] = bool(
            result["int8"]["p50_ms"] <= 0.9 * result["fp32"]["p50_ms"]
            and result["int8"]["mask_agreement"] >= 0.99
            and result["int8"]["bytes"] < args.model.stat().st_size
        )
    if result["int4"].get("status") == "ok":
        int4_session = ort.InferenceSession(
            result["int4"]["path"],
            sess_options=_cpu_options(),
            providers=["CPUExecutionProvider"],
        )
        int4_outputs, int4_times = _benchmark_session(
            int4_session, samples, args.warmups, args.runs
        )
        result["int4"].update(_output_summary(int4_outputs, outputs, int4_times))
    target = args.metadata or output_dir / "quantization.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
