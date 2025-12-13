from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import DOMAINS, FeatureBuildConfig, build_prediction_frame
from .check_layer import check_and_gate_predictions, fallback_rule_based_predictions
from .utils import ensure_dirs, get_paths, load_config, load_joblib, read_json, setup_logging


LOGGER = logging.getLogger(__name__)


def _area_centers() -> dict[str, tuple[float, float]]:
    # Representative centers (synthetic but Riyadh-like)
    return {
        "Central": (24.711, 46.675),
        "North": (24.860, 46.670),
        "South": (24.580, 46.670),
        "East": (24.720, 46.860),
        "West": (24.720, 46.545),
        "NorthEast": (24.860, 46.860),
        "NorthWest": (24.860, 46.545),
        "SouthEast": (24.580, 46.860),
        "SouthWest": (24.580, 46.545),
    }


def _confidence_overlay(metrics: dict[str, Any], domain: str) -> float:
    m = metrics.get("per_domain", {}).get(domain, {})
    v = m.get("avg_precision")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = m.get("roc_auc")
    try:
        return float(v)
    except Exception:  # pragma: no cover
        return 0.0


def _recommended_sectors(cfg: dict[str, Any], risks: dict[str, float]) -> list[str]:
    min_prob = float(cfg["prediction"]["min_prob_for_sector"])
    mapping: dict[str, list[str]] = cfg.get("sectors_by_domain", {})
    out: set[str] = set()
    for d, p in risks.items():
        if p >= min_prob:
            out.update(mapping.get(d, []))
    return sorted(out)


def predict_records(
    model,
    X_win: pd.DataFrame,
    cfg: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if metrics is None:
        metrics = {}

    centers = _area_centers()
    radius = int(cfg["prediction"]["radius_meters"])

    prob_list = model.predict_proba(X_win)
    probs = {d: prob_list[i][:, 1] for i, d in enumerate(DOMAINS)}

    records: list[dict[str, Any]] = []
    for idx, row in X_win.reset_index(drop=True).iterrows():
        geo = row["geo_area"]
        window_start = pd.to_datetime(row["window_start"]).to_pydatetime().replace(tzinfo=timezone.utc)
        risk_by_domain = {d: float(np.clip(probs[d][idx], 0.0, 1.0)) for d in DOMAINS}
        risk_score = float(np.mean(list(risk_by_domain.values())))

        records.append(
            {
                "window_start": window_start.isoformat(),
                "geo_area": geo,
                "center_lat": float(centers.get(geo, (24.711, 46.675))[0]),
                "center_lon": float(centers.get(geo, (24.711, 46.675))[1]),
                "radius_meters": radius,
                "risk_by_domain": risk_by_domain,
                "risk_score": risk_score,
                "recommended_responding_sectors": _recommended_sectors(cfg, risk_by_domain),
                "confidence_overlay": {d: _confidence_overlay(metrics, d) for d in DOMAINS},
            }
        )

    records.sort(key=lambda r: r["risk_score"], reverse=True)
    top_k = int(cfg["prediction"]["top_k_areas"])
    return records[:top_k]


def write_latest_predictions(repo_root: Path, window_start: datetime | None = None) -> tuple[Path, Path]:
    setup_logging(repo_root)
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)
    ensure_dirs(paths.output_dir)

    model = load_joblib(paths.models_dir / "city_risk_model.pkl")
    metrics_path = paths.metrics_dir / "evaluation_report.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}

    fcfg = FeatureBuildConfig(
        window_minutes=int(cfg["features"]["window_minutes"]),
        lags=list(cfg["features"]["lags"]),
        rolling_windows=list(cfg["features"]["rolling_windows"]),
        themes=list(cfg["features"]["themes"]),
    )

    if window_start is None:
        weather = pd.read_csv(paths.input_dir / "weather_conditions.csv")
        wmax = pd.to_datetime(weather["timestamp"], utc=True).max().tz_convert(None)
        window_start = wmax.floor(f"{fcfg.window_minutes}min").to_pydatetime()

    X_win = build_prediction_frame(paths.input_dir, fcfg, window_start, window_minutes=fcfg.window_minutes)
    if X_win.empty:
        raise ValueError("No feature rows found for requested window_start")

    records = predict_records(model, X_win, cfg, metrics=metrics)

    # Pre-deployment gate: block/warn/approve BEFORE persisting outputs.
    gate = check_and_gate_predictions(repo_root, records)
    if gate.approval_status == "blocked":
        LOGGER.warning("Check layer blocked output (%s). Falling back to rule-based plan.", gate.reasons)
        records = fallback_rule_based_predictions(cfg, X_win, metrics=metrics)

    csv_path = paths.output_dir / "risk_predictions_latest.csv"
    json_path = paths.output_dir / "risk_predictions_latest.json"

    flat = []
    for r in records:
        row = {
            "window_start": r["window_start"],
            "geo_area": r["geo_area"],
            "center_lat": r["center_lat"],
            "center_lon": r["center_lon"],
            "radius_meters": r["radius_meters"],
            "risk_score": r["risk_score"],
            "recommended_responding_sectors": json.dumps(r["recommended_responding_sectors"], ensure_ascii=False),
        }
        for d in DOMAINS:
            row[f"risk_{d}"] = r["risk_by_domain"][d]
            row[f"conf_{d}"] = r["confidence_overlay"][d]
        flat.append(row)

    pd.DataFrame(flat).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s and %s", csv_path, json_path)
    return csv_path, json_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    write_latest_predictions(repo_root)


if __name__ == "__main__":
    main()
