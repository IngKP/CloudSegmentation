from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    sample_root: Path
    model_manifest_path: Path
    sample_manifest_path: Path
    ort_threads: int = 1

    @classmethod
    def from_environment(cls) -> Settings:
        threads = int(os.environ.get("CLOUDSEG_ORT_THREADS", "1"))
        if threads < 1:
            raise ValueError("CLOUDSEG_ORT_THREADS must be at least 1")
        return cls(
            model_path=Path(
                os.environ.get(
                    "CLOUDSEG_MODEL_PATH",
                    ".artifacts/onnx/garg_benchmark_seed42/model.fp32.onnx",
                )
            ),
            sample_root=Path(os.environ.get("CLOUDSEG_SAMPLE_ROOT", ".artifacts/samples")),
            model_manifest_path=REPOSITORY_ROOT / "manifests/optimized_models.toml",
            sample_manifest_path=REPOSITORY_ROOT / "manifests/demo_samples.toml",
            ort_threads=threads,
        )
