#!/usr/bin/env python3
"""Report reproducibility-relevant host and runtime facts as JSON."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = (
    "fastapi",
    "google-cloud-storage",
    "numpy",
    "onnxruntime",
    "pillow",
    "pytest",
    "ruff",
    "uvicorn",
)


def command_output(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def package_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for package in PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not-installed"
    return found


def main() -> None:
    research_root = Path(os.environ.get("SAR_RESEARCH_ROOT", "../SARfloodEstimation"))
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "tools": {
            tool: command_output(tool, "--version")
            for tool in ("git", "uv", "rclone", "docker", "gcloud")
        },
        "research_root": str(research_root.resolve()),
        "research_root_exists": research_root.is_dir(),
        "artifact_root": str(
            Path(os.environ.get("CLOUDSEG_ARTIFACT_ROOT", ".artifacts")).resolve()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

