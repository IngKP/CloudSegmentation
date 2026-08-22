from __future__ import annotations

import base64
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
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
from app.rate_limit import process_prediction_rate_limiter
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SampleResponse,
)
from app.staging import resolve_artifact_paths

logger = logging.getLogger("cloudsegmentation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(stream_handler)
logger.propagate = False
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
        try:
            artifact_paths = resolve_artifact_paths(
                resolved.model_path,
                resolved.sample_root,
                model_manifest,
                sample_manifest,
            )
        except Exception as exc:
            logger.exception("startup_artifact_staging_failed")
            raise RuntimeError(f"startup artifact staging failed: {exc}") from exc
        artifact = selected_model(model_manifest, artifact_paths.model_path)
        samples = load_samples(sample_manifest, artifact_paths.sample_root)
        visuals = {
            sample_id: {
                "sar": encode_sar_visualization(sample.tensor),
                "reference": encode_reference(sample.label),
            }
            for sample_id, sample in samples.items()
        }
        engine = InferenceEngine(
            artifact_paths.model_path, model_manifest, resolved.ort_threads, session_factory
        )
        application.state.runtime = {
            "model_manifest": model_manifest,
            "artifact": artifact,
            "artifact_paths": artifact_paths,
            "samples": samples,
            "engine": engine,
            "visuals": visuals,
            "prediction_cache": {},
            "prediction_cache_lock": Lock(),
            "prediction_rate_limiter": process_prediction_rate_limiter(
                resolved.prediction_rate_limit
            ),
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
        retry_after = state["prediction_rate_limiter"].acquire()
        if retry_after is not None:
            logger.info("prediction_rate_limit_exceeded")
            raise HTTPException(
                status_code=429,
                detail="prediction rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        with state["prediction_cache_lock"]:
            cached = state["prediction_cache"].get(request.sample_id)
            if cached is not None:
                logger.info(
                    "prediction_cache_hit sample_id=%s model_id=%s",
                    request.sample_id,
                    state["model_manifest"]["selected_model"],
                )
                return cached
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
            prediction = PredictResponse(
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
            state["prediction_cache"][request.sample_id] = prediction
        logger.info(
            "prediction_complete sample_id=%s model_id=%s latency_ms=%.3f",
            request.sample_id,
            state["model_manifest"]["selected_model"],
            latency_ms,
        )
        return prediction

    return application


app = create_app()
