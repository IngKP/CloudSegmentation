"""Two-chip Phase 1 model audit (run in the cloudseg_export environment)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_pipeline import (
    build_and_load_model,
    load_manifest,
    load_two_samples,
    resolve_checkpoint,
    select_channels,
    verify_research_commit,
)

DEFAULT_SAMPLES = ("Sri-Lanka_713926",)


def archived_iou(research_root: str | os.PathLike[str], model_id: str) -> dict[str, Any]:
    """Read the authoritative repeated IoU table, preserving its provenance."""
    path = Path(research_root) / "weekly_report/wacv_2027/generated/architecture_results.csv"
    rows = []
    base_id = model_id.rsplit("_seed", 1)[0]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["job_id"].rsplit("_seed", 1)[0] == base_id:
                rows.append(row)
    if not rows and base_id == "unet_ch_01":
        # The 2-channel row is intentionally maintained in per-job evidence.
        job_root = Path(research_root) / "experiments/publication/jobs"
        for seed in (42, 1337, 2026):
            metrics = job_root / f"{base_id}_seed{seed}" / "test_metrics.json"
            if not metrics.is_file():
                raise FileNotFoundError(metrics)
            data = json.loads(metrics.read_text())
            value = data.get("test_iou", data.get("iou"))
            if value is not None:
                rows.append({
                    "job_id": metrics.parent.name, "iou": str(value),
                    "seed": str(data.get("seed", "")),
                })
    if not rows:
        raise FileNotFoundError(f"no authoritative IoU evidence for {model_id}")
    values = [float(row["iou"]) for row in rows]
    source = "weekly_report/wacv_2027/generated/architecture_results.csv"
    if base_id == "unet_ch_01":
        source = "experiments/publication/jobs/unet_ch_01_seed*/test_metrics.json"
    return {"mean": statistics.fmean(values), "values": values, "source": source, "rows": rows}


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1])


def select_model(results: list[dict[str, Any]], margin: float = 0.03) -> str:
    eligible = [r for r in results if r.get("error") is None and "cpu" in r]
    if not eligible:
        raise RuntimeError("no model completed audit")
    best_iou = max(r["archived_iou_mean"] for r in eligible)
    eligible = [r for r in eligible if r["archived_iou_mean"] >= best_iou - margin]
    two = next((r for r in eligible if r["id"] == "unet_ch_01_seed42"), None)
    fastest_p50 = min(r["cpu"]["p50_ms"] for r in eligible)
    near_fastest = [r for r in eligible if r["cpu"]["p50_ms"] <= 1.1 * fastest_p50]
    fastest = min(
        near_fastest,
        key=lambda r: (len(r.get("channels", [])), r.get("parameters", float("inf"))),
    )
    chosen = two if two and two["cpu"]["p50_ms"] <= 1.5 * fastest["cpu"]["p50_ms"] else fastest
    return chosen["id"]


def _time_model(
    model: Any, samples: dict[str, dict[str, Any]], channels: list[str],
    device: str, warmups: int, runs: int,
) -> dict[str, float]:
    import torch

    model = model.to(device).eval()
    inputs = [
        select_channels(item["image"], channels).unsqueeze(0).to(device)
        for item in samples.values()
    ]
    with torch.inference_mode():
        for x in inputs:
            for _ in range(warmups):
                model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    values = []
    with torch.inference_mode():
        for x in inputs:
            for _ in range(runs):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(x)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                values.append((time.perf_counter() - start) * 1000)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("model produced non-finite output")
    values.sort()
    result = {
        "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "observations": len(values),
    }
    if device.startswith("cuda"):
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return result


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    import psutil
    import torch

    manifest = load_manifest(args.manifest)
    verify_research_commit(args.research_root, manifest["source_commit"])
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    samples = load_two_samples(args.research_root, args.samples)
    results = []
    requested = set(args.model_ids or [])
    specs = [spec for spec in manifest["models"] if not requested or spec["id"] in requested]
    unknown = requested - {spec["id"] for spec in manifest["models"]}
    if unknown:
        raise ValueError(f"unknown model IDs: {', '.join(sorted(unknown))}")
    for spec in specs:
        print(f"auditing {spec['id']}...", flush=True)
        row: dict[str, Any] = {
            "id": spec["id"], "architecture": spec["architecture"],
            "channels": spec["channels"], "seed": spec["seed"],
        }
        model = None
        try:
            checkpoint = resolve_checkpoint(spec, manifest.get("artifact_root", args.artifact_root))
            model = build_and_load_model(spec, checkpoint, args.research_root, "cpu")
            row["checkpoint"] = {
                "bytes": checkpoint.stat().st_size,
                "sha256": spec["sha256"],
            }
            row["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            gc.collect()
            row["resident_bytes"] = psutil.Process().memory_info().rss
            row["archived_iou_mean"] = archived_iou(args.research_root, spec["id"])["mean"]
            row["archived_iou"] = archived_iou(args.research_root, spec["id"])
            if not args.skip_cpu:
                row["cpu"] = _time_model(
                    model, samples, spec["channels"], "cpu", args.cpu_warmups, args.cpu_runs,
                )
                print(f"  cpu p50={row['cpu']['p50_ms']:.1f} ms", flush=True)
            if not args.skip_gpu and torch.cuda.is_available():
                row["gpu"] = _time_model(
                    model, samples, spec["channels"], "cuda", args.gpu_warmups, args.gpu_runs,
                )
                print(f"  gpu p50={row['gpu']['p50_ms']:.1f} ms", flush=True)
        except Exception as exc:  # audit all candidates and report failures
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  error: {row['error']}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(row)
    selected = (
        select_model(results)
        if any("cpu" in row and row.get("error") is None for row in results)
        else None
    )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (Python 3.10)
        "source_commit": manifest["source_commit"],
        "samples": list(args.samples),
        "target_semantics": "Sen1Floods11 released all-water mask",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "logical_cpus": psutil.cpu_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "protocol": {
            "input_shape": [1, "channels", 512, 512],
            "batch_size": 1,
            "cpu_threads": args.threads,
            "cpu_warmups_per_sample": args.cpu_warmups,
            "cpu_runs_per_sample": args.cpu_runs,
            "gpu_warmups_per_sample": args.gpu_warmups,
            "gpu_runs_per_sample": args.gpu_runs,
            "inference_only": True,
            "cpu_result_label": f"{len(args.samples)}-image latency estimate",
        },
        "models": results,
        "selected_model": selected,
    }


def run_isolated_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Run each requested model in a fresh process and merge the JSON reports."""
    manifest = load_manifest(args.manifest)
    requested = set(args.model_ids or [])
    specs = [
        spec for spec in manifest["models"]
        if not requested or spec["id"] in requested
    ]
    unknown = requested - {spec["id"] for spec in manifest["models"]}
    if unknown:
        raise ValueError(f"unknown model IDs: {', '.join(sorted(unknown))}")
    reports = []
    with tempfile.TemporaryDirectory(prefix="cloudseg-audit-") as temp_dir:
        for spec in specs:
            child_output = Path(temp_dir) / f"{spec['id']}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest", str(args.manifest),
                "--research-root", str(args.research_root),
                "--artifact-root", str(args.artifact_root),
                "--output", str(child_output),
                "--model-id", spec["id"],
                "--samples", *args.samples,
                "--threads", str(args.threads),
                "--cpu-warmups", str(args.cpu_warmups),
                "--cpu-runs", str(args.cpu_runs),
                "--gpu-warmups", str(args.gpu_warmups),
                "--gpu-runs", str(args.gpu_runs),
            ]
            if args.skip_cpu:
                command.append("--skip-cpu")
            if args.skip_gpu:
                command.append("--skip-gpu")
            subprocess.run(command, check=True)
            reports.append(json.loads(child_output.read_text()))
    combined = reports[0]
    combined["models"] = [report["models"][0] for report in reports]
    combined["selected_model"] = (
        select_model(combined["models"])
        if any(
            "cpu" in row and row.get("error") is None
            for row in combined["models"]
        )
        else None
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/models.toml")
    parser.add_argument("--research-root", default="../SARfloodEstimation")
    parser.add_argument("--artifact-root", default=".artifacts/checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--cpu-warmups", type=int, default=1)
    parser.add_argument("--cpu-runs", type=int, default=2)
    parser.add_argument("--gpu-warmups", type=int, default=10)
    parser.add_argument("--gpu-runs", type=int, default=50)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    output = (
        run_audit(args)
        if args.model_ids and len(args.model_ids) == 1
        else run_isolated_audit(args)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
