from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Domain = Literal[
    "traffic",
    "drugs",
    "public_order",
    "health",
    "civil_defense",
    "security",
]

GeoArea = Literal[
    "Central",
    "East",
    "North",
    "West",
    "NorthEast",
    "NorthWest",
    "South",
    "SouthEast",
    "SouthWest",
]


class PredictRequest(BaseModel):
    window_start: datetime = Field(
        ..., description="ISO timestamp for the start of the prediction window"
    )
    window_minutes: int = Field(60, ge=15, le=360)
    geo_area: GeoArea | None = Field(None, description="Optional filter")
    domain: Domain | None = Field(None, description="Optional domain filter")


class RiskByDomain(BaseModel):
    traffic: float
    drugs: float
    public_order: float
    health: float
    civil_defense: float
    security: float


class PredictionRecord(BaseModel):
    window_start: datetime
    geo_area: GeoArea
    center_lat: float
    center_lon: float
    radius_meters: int = 900
    risk_by_domain: RiskByDomain
    risk_score: float
    recommended_responding_sectors: list[str]
    confidence_overlay: dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    generated_at: datetime
    approval_status: Literal["approved", "warning", "blocked"]
    reasons: list[str]
    confidence_score: float
    predictions: list[PredictionRecord]


class LatestPredictionsResponse(BaseModel):
    model_name: str
    window_start: datetime
    approval_status: Literal["approved", "warning", "blocked"] | None = None
    reasons: list[str] | None = None
    confidence_score: float | None = None
    predictions: list[PredictionRecord]


class AreaRiskResponse(BaseModel):
    geo_area: GeoArea
    window_start: datetime
    radius_meters: int
    center_lat: float
    center_lon: float
    risk_by_domain: RiskByDomain
    risk_score: float
    recommended_responding_sectors: list[str]
    confidence_overlay: dict[str, float]


class DepartmentViewResponse(BaseModel):
    domain: Domain
    window_start: datetime
    predictions: list[dict[str, Any]] = Field(
        ..., description="Domain-specific filtered prediction view"
    )
