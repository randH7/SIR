from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .features import DOMAINS
from .utils import dump_json, get_paths, load_config


ApprovalStatus = Literal["approved", "warning", "blocked"]


@dataclass(frozen=True)
class CheckLayerResult:
    approval_status: ApprovalStatus
    reasons: list[str]
    confidence_score: float


def _parse_sector_list(s: str) -> list[str]:
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def _load_all_incidents(input_dir: Path) -> pd.DataFrame:
    files = [
        "incidents_all_911.csv",
        "incidents_national_guard.csv",
        "incidents_narcotics_control.csv",
        "incidents_general_investigation.csv",
        "incidents_ambulance.csv",
    ]
    frames = [pd.read_csv(input_dir / f) for f in files]
    inc = pd.concat(frames, ignore_index=True)
    inc["timestamp"] = pd.to_datetime(inc["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    inc["responding_sectors_list"] = inc["responding_sectors"].astype(str).map(_parse_sector_list)
    return inc


def _windowize(ts: pd.Series, window_minutes: int) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None).dt.floor(f"{window_minutes}min")


def _mean_confidence_overlay(pred: dict[str, Any]) -> float:
    overlay = pred.get("confidence_overlay") or {}
    if not overlay:
        return 0.0
    vals = [float(v) for v in overlay.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else 0.0


def _compute_policy_violation_rate(cfg: dict[str, Any], predictions: list[dict[str, Any]]) -> float:
    mandatory = cfg.get("check_layer", {}).get("mandatory_sectors_by_domain", {})
    if not predictions:
        return 0.0

    total_checks = 0
    violations = 0

    for p in predictions:
        sectors = set(p.get("recommended_responding_sectors") or [])
        risk_by_domain = p.get("risk_by_domain") or {}

        for d, required in mandatory.items():
            # Only enforce if domain risk looks operationally relevant.
            if float(risk_by_domain.get(d, 0.0) or 0.0) >= float(cfg["prediction"]["min_prob_for_sector"]):
                total_checks += 1
                if not set(required).issubset(sectors):
                    violations += 1

    return float(violations / max(1, total_checks))


def _sector_utilization_predictions(cfg: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
    cap: dict[str, float] = cfg.get("check_layer", {}).get("sector_capacity_per_window", {})
    # IMPORTANT: Predictions are produced per-geo-area, not per-call. Summing across areas
    # will massively over-estimate utilization. We therefore compute a bounded utilization
    # score in [0, 1] based on *average* risk among areas recommending the sector.
    sums: dict[str, float] = {k: 0.0 for k in cap.keys()}
    counts: dict[str, int] = {k: 0 for k in cap.keys()}

    for p in predictions:
        risk = float(p.get("risk_score", 0.0) or 0.0)
        for s in p.get("recommended_responding_sectors") or []:
            if s in sums:
                sums[s] += risk
                counts[s] += 1

    util: dict[str, float] = {}
    for s in cap.keys():
        if counts[s] == 0:
            util[s] = 0.0
        else:
            util[s] = float(np.clip(sums[s] / counts[s], 0.0, 1.0))
    return util


def check_and_gate_predictions(repo_root: Path, predictions: list[dict[str, Any]]) -> CheckLayerResult:
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)

    reasons: list[str] = []

    # Policy compliance
    policy_violation_rate = _compute_policy_violation_rate(cfg, predictions)
    if policy_violation_rate > float(cfg["check_layer"]["max_policy_violation_rate"]):
        reasons.append("policy_violation")

    # Capacity / overload
    util = _sector_utilization_predictions(cfg, predictions)
    max_util = max(util.values()) if util else 0.0
    if max_util > float(cfg["check_layer"]["max_sector_utilization_allowed"]):
        reasons.append("sector_overload_risk")

    # Confidence gate
    conf = float(np.mean([_mean_confidence_overlay(p) for p in predictions]) if predictions else 0.0)
    if conf < float(cfg["check_layer"]["min_confidence_for_auto_dispatch"]):
        reasons.append("low_confidence")

    # Reliability gates (from latest evaluation artifacts if present)
    scen_path = paths.metrics_dir / "scenario_test_results.json"
    if scen_path.exists():
        scen = json.loads(scen_path.read_text(encoding="utf-8"))
        scen_score = float(scen.get("scenario_consistency_score", 1.0))
        if scen_score < float(cfg["check_layer"]["min_scenario_consistency_score"]):
            reasons.append("scenario_inconsistency")

    # Decision
    if "policy_violation" in reasons or "sector_overload_risk" in reasons:
        status: ApprovalStatus = "blocked"
    elif reasons:
        status = "warning"
    else:
        status = "approved"

    return CheckLayerResult(approval_status=status, reasons=reasons, confidence_score=conf)


def fallback_rule_based_predictions(
    cfg: dict[str, Any],
    X_win: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Conservative rule-based fallback using recent lags/rollings and signal sums."""
    if metrics is None:
        metrics = {}

    radius = int(cfg["prediction"]["radius_meters"])
    min_prob = float(cfg["prediction"]["min_prob_for_sector"])
    sectors_by_domain: dict[str, list[str]] = cfg.get("sectors_by_domain", {})

    def _centers(geo: str) -> tuple[float, float]:
        centers = {
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
        return centers.get(geo, (24.711, 46.675))

    records: list[dict[str, Any]] = []

    for _, row in X_win.reset_index(drop=True).iterrows():
        geo = str(row.get("geo_area"))
        ws = pd.to_datetime(row.get("window_start"), utc=True, errors="coerce").to_pydatetime().replace(tzinfo=timezone.utc)

        # Risk heuristic per domain.
        risks: dict[str, float] = {}
        for d in DOMAINS:
            lag = float(row.get(f"inc_count_{d}_lag1", 0.0) or 0.0)
            roll = float(row.get(f"inc_count_{d}_roll8", 0.0) or 0.0)
            sig = float(row.get("conf_sum_social_lag1", 0.0) or 0.0) + float(row.get("conf_sum_news_lag1", 0.0) or 0.0)
            ev = float(row.get("conf_sum_events_lag1", 0.0) or 0.0)
            weather = float(row.get("conf_sum_weather", 0.0) or 0.0)

            raw = 0.10 * lag + 0.03 * roll + 0.02 * sig + 0.02 * ev
            if d == "civil_defense":
                raw += 0.10 * weather
            # Squash to [0,1]
            risks[d] = float(1.0 - np.exp(-raw))

        risk_score = float(np.mean(list(risks.values())))
        sectors: set[str] = set()
        for d, p in risks.items():
            if p >= min_prob:
                sectors.update(sectors_by_domain.get(d, []))

        lat, lon = _centers(geo)
        records.append(
            {
                "window_start": ws.isoformat(),
                "geo_area": geo,
                "center_lat": float(lat),
                "center_lon": float(lon),
                "radius_meters": radius,
                "risk_by_domain": risks,
                "risk_score": risk_score,
                "recommended_responding_sectors": sorted(sectors),
                "confidence_overlay": {d: float(metrics.get("per_domain", {}).get(d, {}).get("avg_precision") or 0.0) for d in DOMAINS},
            }
        )

    records.sort(key=lambda r: r["risk_score"], reverse=True)
    return records[: int(cfg["prediction"]["top_k_areas"])]


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    x = np.array(values, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)


def run_offline_training_evaluation(
    repo_root: Path,
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    window_minutes: int,
) -> dict[str, Any]:
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)

    inc = _load_all_incidents(paths.input_dir)
    inc["window_start"] = _windowize(inc["timestamp"], window_minutes)

    # Actual sectors in next window (T+1) by area.
    actual = (
        inc.groupby(["window_start", "geo_area"], observed=True)
        .agg(
            actual_sectors=("responding_sectors_list", lambda x: sorted({s for lst in x for s in lst})),
            response_time_mean=("response_time_minutes", "mean"),
        )
        .reset_index()
    )
    actual["window_start"] = pd.to_datetime(actual["window_start"])

    # Predict risks for X_test and derive recommended sectors.
    from .predict import predict_records

    # Use a representative slice to avoid huge memory when producing explanations.
    preds = predict_records(model, X_test, cfg, metrics={})

    # Join actual sectors for next window.
    dfp = pd.DataFrame(preds)
    dfp["window_start"] = pd.to_datetime(dfp["window_start"], utc=True).dt.tz_convert(None)
    dfp["window_start_next"] = dfp["window_start"] + pd.Timedelta(minutes=window_minutes)

    joined = dfp.merge(
        actual,
        left_on=["window_start_next", "geo_area"],
        right_on=["window_start", "geo_area"],
        how="left",
        suffixes=("", "_actual"),
    )

    def _as_list(v: Any) -> list[Any]:
        if isinstance(v, list):
            return v
        if v is None:
            return []
        # pandas NaN handling
        try:
            if isinstance(v, float) and np.isnan(v):
                return []
        except Exception:
            pass
        return []

    def _topk_match(row: pd.Series) -> int:
        rec = set(row.get("recommended_responding_sectors") or [])
        act = set(_as_list(row.get("actual_sectors")))
        return int(len(rec & act) > 0)

    topk_acc = float(joined.apply(_topk_match, axis=1).mean()) if len(joined) else 0.0

    # Random baseline for lift.
    all_sectors = sorted({s for v in cfg.get("sectors_by_domain", {}).values() for s in v})
    rng = np.random.default_rng(42)

    def _rand_match(row: pd.Series) -> int:
        k = max(1, len(row.get("recommended_responding_sectors") or []))
        pick = set(rng.choice(all_sectors, size=min(k, len(all_sectors)), replace=False))
        act = set(_as_list(row.get("actual_sectors")))
        return int(len(pick & act) > 0)

    rand_acc = float(joined.apply(_rand_match, axis=1).mean()) if len(joined) else 0.0
    success_lift = float(topk_acc - rand_acc)

    # Skill match: alignment of risks with sector-domain mapping (weighted coverage).
    sector_to_domains: dict[str, set[str]] = {}
    for d, sectors in cfg.get("sectors_by_domain", {}).items():
        for s in sectors:
            sector_to_domains.setdefault(s, set()).add(d)

    def _skill_score(row: pd.Series) -> float:
        risks = row.get("risk_by_domain") or {}
        rec = row.get("recommended_responding_sectors") or []
        if not rec:
            return 0.0
        # Score = average of (sum of covered domain risks) per recommended sector.
        scores = []
        for s in rec:
            covered = sector_to_domains.get(s, set())
            scores.append(float(sum(float(risks.get(d, 0.0) or 0.0) for d in covered)))
        return float(np.mean(scores))

    skill = float(joined.apply(_skill_score, axis=1).mean()) if len(joined) else 0.0

    # Response-time estimates: use per-geo historical mean as a naive estimator.
    rt_hist = (
        inc.groupby(["geo_area"], observed=True)["response_time_minutes"].mean().to_dict()
        if len(inc)
        else {}
    )
    joined["rt_estimate"] = joined["geo_area"].map(lambda g: float(rt_hist.get(g, 0.0)))
    joined["rt_actual"] = joined["response_time_mean"].fillna(0.0)

    rt_error = float(np.mean(np.abs(joined["rt_estimate"] - joined["rt_actual"]))) if len(joined) else 0.0
    rt_mean = float(np.mean(joined["rt_estimate"])) if len(joined) else 0.0
    rt_p90 = float(np.quantile(joined["rt_estimate"], 0.90)) if len(joined) else 0.0
    rt_p95 = float(np.quantile(joined["rt_estimate"], 0.95)) if len(joined) else 0.0

    # Capacity/load: gini of predicted utilization.
    util = _sector_utilization_predictions(cfg, preds)
    load_balance_gini = _gini(list(util.values()))

    # Policy violations
    policy_violation_rate = _compute_policy_violation_rate(cfg, preds)

    return {
        "matching_recommendation_quality": {
            "topk_match_accuracy": topk_acc,
            "skill_match_score": skill,
            "historical_success_lift_vs_random": success_lift,
        },
        "response_time_metrics": {
            "mean_estimated_response_time": rt_mean,
            "p90_estimated_response_time": rt_p90,
            "p95_estimated_response_time": rt_p95,
            "mean_abs_error_estimated_vs_historical": rt_error,
        },
        "capacity_load_metrics": {
            "sector_utilization": util,
            "max_sector_utilization": float(max(util.values()) if util else 0.0),
        },
        "optimization_quality": {
            "load_balance_gini": load_balance_gini,
        },
        "constraint_compliance": {
            "policy_violation_rate": policy_violation_rate,
        },
    }


def run_scenario_based_stress_tests(
    repo_root: Path,
    model,
    X_sample: pd.DataFrame,
) -> dict[str, Any]:
    cfg = load_config(repo_root)

    # Perturbations: small input changes should not cause *large* output changes.
    base = X_sample.copy()
    if base.empty:
        return {"scenario_consistency_score": 0.0, "scenarios": []}

    # Choose a few numeric signal columns to perturb.
    perturb_cols = [c for c in base.columns if c.startswith("conf_sum_") or "theme_conf" in c]
    perturb_cols = perturb_cols[:10]

    def _predict_mean(model_, X_) -> dict[str, float]:
        prob_list = model_.predict_proba(X_)
        return {d: float(np.mean(prob_list[i][:, 1])) for i, d in enumerate(DOMAINS)}

    base_mean = _predict_mean(model, base)

    scenarios = []
    # We measure stability as bounded average absolute change.
    deltas: list[float] = []

    # Scenario 1: +10% signals -> risks should not drop too much.
    s1 = base.copy()
    for c in perturb_cols:
        if pd.api.types.is_numeric_dtype(s1[c]):
            s1[c] = s1[c] * 1.10
    s1_mean = _predict_mean(model, s1)
    for d in DOMAINS:
        deltas.append(abs(s1_mean[d] - base_mean[d]))

    scenarios.append({"name": "signal_increase_10pct", "base": base_mean, "after": s1_mean})

    # Scenario 2: storm proxy increase -> civil_defense should not drop.
    s2 = base.copy()
    if "conf_sum_weather" in s2.columns:
        s2["conf_sum_weather"] = s2["conf_sum_weather"] + 1.0
    s2_mean = _predict_mean(model, s2)
    for d in DOMAINS:
        deltas.append(abs(s2_mean[d] - base_mean[d]))
    scenarios.append({"name": "storm_weather_spike", "base": base_mean, "after": s2_mean})

    mean_abs_delta = float(np.mean(deltas)) if deltas else 0.0
    # Convert to a [0, 1] score where 0.0 is very unstable and 1.0 is stable.
    # The 0.20 scale is a conservative “large change” bound for probabilities.
    score = float(np.clip(1.0 - (mean_abs_delta / 0.20), 0.0, 1.0))
    return {
        "scenario_consistency_score": score,
        "mean_abs_delta": mean_abs_delta,
        "scenarios": scenarios,
    }


def run_capacity_stress_test(repo_root: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = load_config(repo_root)
    util = _sector_utilization_predictions(cfg, predictions)
    max_util = float(max(util.values()) if util else 0.0)

    # Stress by scaling risk_score demand.
    stressed = []
    for p in predictions:
        q = dict(p)
        q["risk_score"] = float(min(1.0, float(q.get("risk_score", 0.0) or 0.0) * 3.0))
        stressed.append(q)

    util_stress = _sector_utilization_predictions(cfg, stressed)
    max_util_stress = float(max(util_stress.values()) if util_stress else 0.0)

    threshold = float(cfg["check_layer"]["max_sector_utilization_allowed"])

    return {
        "base": {"max_sector_utilization": max_util, "sector_utilization": util},
        "stressed": {"max_sector_utilization": max_util_stress, "sector_utilization": util_stress},
        "threshold": threshold,
        "would_block_under_stress": bool(max_util_stress > threshold),
    }


def write_explanations(repo_root: Path, model, X_win: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)

    # Global feature importances per domain (best-effort). If unavailable, produce heuristic summary.
    explanations: list[dict[str, Any]] = []

    top_n = int(cfg.get("check_layer", {}).get("explainability", {}).get("top_n_features", 6))

    # Try to extract feature names from preprocessing.
    feature_names: list[str] = []
    importances_by_domain: dict[str, list[tuple[str, float]]] = {}

    try:
        pre = model.named_steps["preprocess"]
        feature_names = list(pre.get_feature_names_out())
        estimators = model.named_steps["clf"].estimators_
        for i, d in enumerate(DOMAINS):
            est = estimators[i]
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
                pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:top_n]
                importances_by_domain[d] = [(n, float(v)) for n, v in pairs]
    except Exception:
        pass

    for idx, row in X_win.reset_index(drop=True).iterrows():
        pred = predictions[idx] if idx < len(predictions) else {}
        risks = pred.get("risk_by_domain") or {}
        top_domain = max(risks.items(), key=lambda kv: kv[1])[0] if risks else "traffic"

        contrib = []
        if importances_by_domain.get(top_domain):
            for fname, imp in importances_by_domain[top_domain]:
                # Try to attach the raw value if present.
                base_name = fname
                val = row.get(base_name) if base_name in row else None
                contrib.append({"feature": fname, "importance": imp, "value": None if pd.isna(val) else float(val) if isinstance(val, (int, float, np.number)) else str(val)})
        else:
            # Heuristic: show strongest lag/rolling signals.
            candidates = [c for c in row.index if "_lag" in c or "_roll" in c or c.startswith("conf_sum")]
            cand = sorted(candidates, key=lambda c: float(row.get(c, 0.0) or 0.0), reverse=True)[:top_n]
            for c in cand:
                contrib.append({"feature": c, "importance": None, "value": float(row.get(c, 0.0) or 0.0)})

        tags: list[str] = []
        # Policy mandated tag if mandatory sectors are present.
        mandatory = cfg.get("check_layer", {}).get("mandatory_sectors_by_domain", {})
        rec_sectors = set(pred.get("recommended_responding_sectors") or [])
        for d, req in mandatory.items():
            if float(risks.get(d, 0.0) or 0.0) >= float(cfg["prediction"]["min_prob_for_sector"]) and set(req).issubset(rec_sectors):
                tags.append("policy_mandated")
                break
        if pred.get("recommended_responding_sectors"):
            tags.append("highest_skill_match")

        explanations.append(
            {
                "window_start": pred.get("window_start"),
                "geo_area": pred.get("geo_area"),
                "feature_contributions": contrib,
                "reasoning_tags": sorted(set(tags)),
                "confidence_score": float(_mean_confidence_overlay(pred)),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "explanations": explanations,
    }

    dump_json(paths.metrics_dir / "explanations.json", payload)
    return payload


def write_check_artifacts(
    repo_root: Path,
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    X_sample: pd.DataFrame,
    latest_predictions: list[dict[str, Any]],
    window_minutes: int,
) -> None:
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)

    offline = run_offline_training_evaluation(repo_root, model, X_test, y_test, window_minutes=window_minutes)
    dump_json(paths.metrics_dir / "offline_quality_report.json", offline)

    scenario = run_scenario_based_stress_tests(repo_root, model, X_sample)
    dump_json(paths.metrics_dir / "scenario_test_results.json", scenario)

    capacity = run_capacity_stress_test(repo_root, latest_predictions)
    dump_json(paths.metrics_dir / "capacity_stress_test.json", capacity)

    # Compute gate result and store latest.
    gate = check_and_gate_predictions(repo_root, latest_predictions)
    dump_json(
        paths.metrics_dir / "check_latest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "approval_status": gate.approval_status,
            "reasons": gate.reasons,
            "confidence_score": gate.confidence_score,
            "policy_violation_rate": offline.get("constraint_compliance", {}).get("policy_violation_rate"),
            "max_sector_utilization": offline.get("capacity_load_metrics", {}).get("max_sector_utilization"),
            "scenario_consistency_score": scenario.get("scenario_consistency_score"),
        },
    )

    # Explanations for latest predictions.
    # Note: explanations are best-effort (global importances or heuristic signals).
    write_explanations(repo_root, model, X_sample, latest_predictions[: len(X_sample)])
