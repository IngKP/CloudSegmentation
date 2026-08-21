from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.artifacts import (
    encode_reference,
    encode_sar_visualization,
    encode_score_map,
    encode_score_preview,
    load_samples,
    load_toml,
    selected_model,
)
from app.config import Settings
from app.inference import InferenceEngine, SessionFactory, create_session
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SampleResponse,
)

logger = logging.getLogger("cloudsegmentation")
FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"


def create_app(
    settings: Settings | None = None,
    session_factory: SessionFactory = create_session,
) -> FastAPI:
    resolved = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        model_manifest = load_toml(resolved.model_manifest_path)
        sample_manifest = load_toml(resolved.sample_manifest_path)
        artifact = selected_model(model_manifest, resolved.model_path)
        samples = load_samples(sample_manifest, resolved.sample_root)
        visuals = {
            sample_id: {
                "sar": encode_sar_visualization(sample.tensor),
                "reference": encode_reference(sample.label),
            }
            for sample_id, sample in samples.items()
        }
        engine = InferenceEngine(
            resolved.model_path, model_manifest, resolved.ort_threads, session_factory
        )
        application.state.runtime = {
            "model_manifest": model_manifest,
            "artifact": artifact,
            "samples": samples,
            "engine": engine,
            "visuals": visuals,
        }
        logger.info(
            "service_ready model_id=%s variant=%s sample_count=%d",
            model_manifest["selected_model"],
            artifact["variant"],
            len(samples),
        )
        yield

    application = FastAPI(
        title="CloudSegmentation API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")

    def runtime() -> dict[str, Any]:
        return application.state.runtime

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(FRONTEND_ROOT / "index.html")

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        state = runtime()
        manifest = state["model_manifest"]
        return HealthResponse(
            status="ok",
            model_id=manifest["selected_model"],
            variant=state["artifact"]["variant"],
            source_commit=manifest["source_commit"],
            target_semantics=manifest["target_semantics"],
            sample_count=len(state["samples"]),
        )

    @application.get("/samples", response_model=list[SampleResponse])
    def samples() -> list[SampleResponse]:
        return [
            SampleResponse(
                id=sample.sample_id,
                country=sample.country,
                split=sample.split,
                has_reference=True,
            )
            for sample in runtime()["samples"].values()
        ]

    @application.get("/samples/{sample_id}/sar.png")
    def sar_visualization(sample_id: str) -> Response:
        visual = runtime()["visuals"].get(sample_id)
        if visual is None:
            raise HTTPException(status_code=404, detail="unknown sample ID")
        return Response(
            content=visual["sar"],
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.get("/samples/{sample_id}/reference.png")
    def reference_visualization(sample_id: str) -> Response:
        visual = runtime()["visuals"].get(sample_id)
        if visual is None:
            raise HTTPException(status_code=404, detail="unknown sample ID")
        return Response(
            content=visual["reference"],
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @application.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state = runtime()
        sample = state["samples"].get(request.sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="unknown sample ID")
        try:
            score_map, latency_ms = state["engine"].predict(sample.tensor)
            encoded = base64.b64encode(encode_score_map(score_map)).decode("ascii")
            preview = base64.b64encode(encode_score_preview(score_map)).decode("ascii")
        except Exception:
            logger.exception(
                "prediction_failed sample_id=%s model_id=%s",
                request.sample_id,
                state["model_manifest"]["selected_model"],
            )
            raise
        logger.info(
            "prediction_complete sample_id=%s model_id=%s latency_ms=%.3f",
            request.sample_id,
            state["model_manifest"]["selected_model"],
            latency_ms,
        )
        return PredictResponse(
            sample_id=request.sample_id,
            model_id=state["model_manifest"]["selected_model"],
            model_variant=state["artifact"]["variant"],
            latency_ms=latency_ms,
            score_map=encoded,
            score_map_encoding="png16-base64; score=uint16/65535",
            score_map_shape=[512, 512],
            score_map_preview=preview,
            score_map_preview_encoding="png8-base64; score=rint(score*255)/255",
        )

    return application


app = create_app()
