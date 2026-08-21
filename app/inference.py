from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

SessionFactory = Callable[[Path, int], Any]


def create_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class InferenceEngine:
    def __init__(
        self,
        model_path: Path,
        manifest: dict[str, Any],
        threads: int,
        session_factory: SessionFactory = create_session,
    ) -> None:
        self._session = session_factory(model_path, threads)
        self._input_name = manifest["input_name"]
        self._output_name = manifest["output_name"]
        self._input_shape = list(manifest["input_shape"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != self._input_name:
            raise RuntimeError("ONNX input does not match optimized-model manifest")
        if list(inputs[0].shape) != self._input_shape:
            raise RuntimeError("ONNX input shape does not match optimized-model manifest")
        if len(outputs) != 1 or outputs[0].name != self._output_name:
            raise RuntimeError("ONNX output does not match optimized-model manifest")
        if list(outputs[0].shape) != [1, 1, 512, 512]:
            raise RuntimeError("ONNX output shape must be [1, 1, 512, 512]")
        self._lock = threading.Lock()

    def predict(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        value = np.ascontiguousarray(tensor[np.newaxis], dtype=np.float32)
        with self._lock:
            started = time.perf_counter()
            output = self._session.run([self._output_name], {self._input_name: value})[0]
            latency_ms = (time.perf_counter() - started) * 1000.0
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape != (1, 1, 512, 512):
            raise RuntimeError(f"unexpected inference output shape: {scores.shape}")
        return scores[0, 0], latency_ms
