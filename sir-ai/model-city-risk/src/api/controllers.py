from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, Request

from ..model.features import FeatureBuildConfig, DOMAINS, build_prediction_frame
from ..model.check_layer import check_and_gate_predictions, fallback_rule_based_predictions
from ..model.predict import predict_records
from ..model.schemas import AreaRiskResponse, DepartmentViewResponse, LatestPredictionsResponse, PredictRequest, PredictResponse
from ..model.utils import get_paths, load_joblib, read_json


def _load_runtime(request: Request) -> tuple[dict[str, Any], Path]:
    cfg: dict[str, Any] = request.app.state.cfg
    repo_root: Path = request.app.state.repo_root
    return cfg, repo_root


def health() -> dict[str, Any]:
    return {"status": "ok"}


def _get_model(cfg: dict[str, Any], repo_root: Path):
    paths = get_paths(repo_root, cfg)
    model_path = paths.models_dir / "city_risk_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run scripts/run_train.sh")
    return load_joblib(model_path)


def _get_metrics(cfg: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    paths = get_paths(repo_root, cfg)
    p = paths.metrics_dir / "evaluation_report.json"
    return read_json(p) if p.exists() else {}


def predict(req: PredictRequest, request: Request) -> PredictResponse:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)

    model = _get_model(cfg, repo_root)
    metrics = _get_metrics(cfg, repo_root)

    fcfg = FeatureBuildConfig(
        window_minutes=int(req.window_minutes),
        lags=list(cfg["features"]["lags"]),
        rolling_windows=list(cfg["features"]["rolling_windows"]),
        themes=list(cfg["features"]["themes"]),
    )

    try:
        X_win = build_prediction_frame(paths.input_dir, fcfg, req.window_start, window_minutes=req.window_minutes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if X_win.empty:
        raise HTTPException(status_code=404, detail="No data for requested window_start")

    # Optional filters.
    if req.geo_area:
        X_win = X_win[X_win["geo_area"] == req.geo_area]
    if X_win.empty:
        raise HTTPException(status_code=404, detail="No rows after filtering")

    records = predict_records(model, X_win, cfg, metrics=metrics)

    if req.domain:
        # Keep full record (per-domain + overall), but sort by the requested domain.
        records.sort(key=lambda r: r.get("risk_by_domain", {}).get(req.domain, 0.0), reverse=True)

    gate = check_and_gate_predictions(repo_root, records)
    if gate.approval_status == "blocked":
        records = fallback_rule_based_predictions(cfg, X_win, metrics=metrics)

    return PredictResponse(
        model_name=cfg["project"]["model_name"],
        generated_at=datetime.now(tz=timezone.utc),
        approval_status=gate.approval_status,
        reasons=gate.reasons,
        confidence_score=gate.confidence_score,
        predictions=records,
    )


def latest_predictions(request: Request) -> LatestPredictionsResponse:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    p = paths.output_dir / "risk_predictions_latest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No latest predictions found. Run src/model/predict.py")

    records = json.loads(p.read_text(encoding="utf-8"))
    if not records:
        raise HTTPException(status_code=404, detail="Latest predictions file is empty")

    window_start = pd.to_datetime(records[0]["window_start"]).to_pydatetime().replace(tzinfo=timezone.utc)
    return LatestPredictionsResponse(
        model_name=cfg["project"]["model_name"],
        window_start=window_start,
        approval_status=None,
        reasons=None,
        confidence_score=None,
        predictions=records,
    )


def area_risk(geo_area: str, request: Request) -> AreaRiskResponse:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    p = paths.output_dir / "risk_predictions_latest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No latest predictions found")

    records = json.loads(p.read_text(encoding="utf-8"))
    for r in records:
        if r["geo_area"] == geo_area:
            return AreaRiskResponse(**r)
    raise HTTPException(status_code=404, detail="Geo area not found in latest predictions")


def department_view(domain: str, request: Request) -> DepartmentViewResponse:
    if domain not in DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")

    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    p = paths.output_dir / "risk_predictions_latest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No latest predictions found")

    records = json.loads(p.read_text(encoding="utf-8"))
    if not records:
        raise HTTPException(status_code=404, detail="Latest predictions file is empty")

    window_start = pd.to_datetime(records[0]["window_start"]).to_pydatetime().replace(tzinfo=timezone.utc)

    view = []
    for r in records:
        view.append(
            {
                "window_start": r["window_start"],
                "geo_area": r["geo_area"],
                "center_lat": r["center_lat"],
                "center_lon": r["center_lon"],
                "radius_meters": r["radius_meters"],
                "risk": r.get("risk_by_domain", {}).get(domain, 0.0),
                "risk_score": r.get("risk_score", 0.0),
                "recommended_responding_sectors": r.get("recommended_responding_sectors", []),
                "confidence": r.get("confidence_overlay", {}).get(domain, 0.0),
            }
        )

    view.sort(key=lambda x: x["risk"], reverse=True)
    return DepartmentViewResponse(domain=domain, window_start=window_start, predictions=view)


def model_health(request: Request) -> dict[str, Any]:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    model_ok = (paths.models_dir / "city_risk_model.pkl").exists()
    data_ok = (paths.input_dir / "incidents_all_911.csv").exists()
    preds_ok = (paths.output_dir / "risk_predictions_latest.json").exists()
    check_ok = (paths.metrics_dir / "check_latest.json").exists()
    return {
        "model_trained": model_ok,
        "datasets_present": data_ok,
        "latest_predictions_present": preds_ok,
        "check_layer_report_present": check_ok,
    }


def model_metrics(request: Request) -> dict[str, Any]:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    out: dict[str, Any] = {}
    for name in [
        "evaluation_report.json",
        "offline_quality_report.json",
        "scenario_test_results.json",
        "capacity_stress_test.json",
    ]:
        p = paths.metrics_dir / name
        out[name] = read_json(p) if p.exists() else None
    return out


def model_check_latest(request: Request) -> dict[str, Any]:
    cfg, repo_root = _load_runtime(request)
    paths = get_paths(repo_root, cfg)
    p = paths.metrics_dir / "check_latest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No check layer report found. Run training.")
    return read_json(p)
