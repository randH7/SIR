from __future__ import annotations

from fastapi import APIRouter

from .controllers import (
    area_risk,
    department_view,
    health,
    latest_predictions,
    model_check_latest,
    model_health,
    model_metrics,
    predict,
)


router = APIRouter()

router.get("/health")(health)
router.post("/predict")(predict)
router.get("/predictions/latest")(latest_predictions)
router.get("/areas/{geo_area}/risk")(area_risk)
router.get("/departments/{domain}/view")(department_view)

# Optional but recommended for monitoring dashboards
router.get("/model/health")(model_health)
router.get("/model/metrics")(model_metrics)
router.get("/model/check/latest")(model_check_latest)
