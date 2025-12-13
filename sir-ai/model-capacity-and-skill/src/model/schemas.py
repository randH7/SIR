from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IncidentInput(BaseModel):
    incident_id: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(default=None)
    geo_area: Optional[str] = Field(default=None)
    latitude: float
    longitude: float
    incident_domain: str
    incident_type: Optional[str] = None
    severity: int = Field(ge=1, le=5, default=3)


class UnitCandidate(BaseModel):
    unit_id: str
    sector: str
    ministry: str
    latitude: float
    longitude: float
    geo_area: Optional[str] = None

    unit_type: Optional[str] = None
    max_capacity_people: int = 0

    is_on_shift: bool = True
    available_people_count: int = 0
    fatigue_score: float = 0.0
    last_dispatch_minutes_ago: float = 9999.0

    skills: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class UnitAssignment(BaseModel):
    unit_id: str
    sector: str
    ministry: str
    latitude: float
    longitude: float
    geo_area: Optional[str] = None

    distance_km: float
    estimated_response_time_minutes: float

    available_people_count: int
    fatigue_score: float

    skills_matched: List[str] = Field(default_factory=list)
    skill_match_score: float = 0.0

    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_tags: List[str] = Field(default_factory=list)


class SectorBreakdown(BaseModel):
    sector: str
    units_selected: int
    people_selected: int


class CheckLayerResult(BaseModel):
    approval_status: str
    reasons: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)


class DeploymentPlan(BaseModel):
    incident_id: str
    timestamp: datetime
    geo_area: Optional[str] = None
    incident_domain: str
    latitude: float
    longitude: float
    severity: int

    selected_units: List[UnitAssignment]
    sector_breakdown: List[SectorBreakdown]
    estimated_response_time_minutes: float

    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_tags: List[str] = Field(default_factory=list)

    check: CheckLayerResult
    meta: Dict[str, Any] = Field(default_factory=dict)


class CapacityRow(BaseModel):
    geo_area: str
    sector: str
    utilization_ratio: float
    capacity_status: str
    recommendation: str


class CapacityView(BaseModel):
    timestamp: datetime
    rows: List[CapacityRow]


class ClosestUnitsResponse(BaseModel):
    timestamp: datetime
    incident_domain: str
    latitude: float
    longitude: float
    units: List[UnitAssignment]


class SectorReadinessResponse(BaseModel):
    timestamp: datetime
    sector: str
    geo_area: Optional[str] = None
    total_units_on_shift: int
    total_people_on_shift: int
    utilization_ratio: float
    overload_flag: bool


class ModelHealth(BaseModel):
    status: str
    model_loaded: bool
    last_trained_at: Optional[datetime] = None
    version: str


class ModelMetrics(BaseModel):
    evaluation_report: Dict[str, Any] = Field(default_factory=dict)
    scenario_test_results: Dict[str, Any] = Field(default_factory=dict)
    capacity_stress_test: Dict[str, Any] = Field(default_factory=dict)
