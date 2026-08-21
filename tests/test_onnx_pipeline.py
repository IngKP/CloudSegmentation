"""Pure helper tests; heavyweight export dependencies are intentionally optional."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.export_onnx import artifact_metadata, mask_agreement
from scripts.quantize_onnx import eligible_int4_nodes, summarize_outputs


def test_artifact_metadata_hash_and_size(tmp_path):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"onnx-test")
    expected = hashlib.sha256(b"onnx-test").hexdigest()
    metadata = artifact_metadata(target, model_id="demo")
    assert metadata["bytes"] == 9
    assert metadata["sha256"] == expected
    assert metadata["model_id"] == "demo"


def test_mask_agreement_and_output_summary():
    reference = np.array([[[[0.1, 0.9]]]], dtype=np.float32)
    actual = np.array([[[[0.10001, 0.89999]]]], dtype=np.float32)
    assert mask_agreement(reference, actual) == 1.0
    summary = summarize_outputs(reference, actual, [0.001, 0.003])
    assert summary["finite"] is True
    assert summary["mask_agreement"] == 1.0
    assert summary["p50_ms"] == 2.0


def test_int4_eligibility_only_constant_weight_matmul_and_gather():
    graph = SimpleNamespace(
        initializer=[SimpleNamespace(name="linear_weight"), SimpleNamespace(name="embedding")],
        node=[
            SimpleNamespace(op_type="Conv", name="conv", input=["x", "conv_weight"]),
            SimpleNamespace(op_type="MatMul", name="dynamic", input=["x", "dynamic_weight"]),
            SimpleNamespace(op_type="MatMul", name="linear", input=["x", "linear_weight"]),
            SimpleNamespace(op_type="Gather", name="gather", input=["embedding", "indices"]),
        ]
    )
    assert eligible_int4_nodes(graph) == ["linear", "gather"]
