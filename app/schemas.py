from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: str
    model_id: str
    variant: str
    source_commit: str
    target_semantics: str
    sample_count: int


class SampleResponse(StrictModel):
    id: str
    country: str
    split: str
    has_reference: bool


class PredictRequest(StrictModel):
    sample_id: str


class PredictResponse(StrictModel):
    sample_id: str
    model_id: str
    model_variant: str
    latency_ms: float
    score_map: str
    score_map_encoding: str
    score_map_shape: list[int]
