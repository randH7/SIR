from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .check_layer import evaluate_and_gate, persist_latest_check, to_check_layer_result
from .optimizer import optimize_deployment
from .schemas import (
    CapacityRow,
    CapacityView,
    ClosestUnitsResponse,
    DeploymentPlan,
    IncidentInput,
    SectorBreakdown,
    UnitAssignment,
)
from .scorer import ScoringModel
from .utils import ensure_dir, load_config, load_yaml, now_utc_iso, parse_ts


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest_per_key(df: pd.DataFrame, key: str, ts_col: str = "timestamp") -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return df
    tmp = df.copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], errors="coerce", utc=True)
    tmp = tmp.sort_values(ts_col)
    return tmp.groupby(key, as_index=False).tail(1)


def _latest_per_sector_area(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")
        if "geo_area" in df.columns:
            return df.groupby(["sector", "geo_area"], as_index=False).tail(1)
        return df.groupby(["sector"], as_index=False).tail(1)
    return df


def _deployment_plan_to_rows(plan: DeploymentPlan) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for u in plan.selected_units:
        rows.append(
            {
                "incident_id": plan.incident_id,
                "timestamp": plan.timestamp.isoformat(),
                "geo_area": plan.geo_area,
                "incident_domain": plan.incident_domain,
                "severity": plan.severity,
                "unit_id": u.unit_id,
                "sector": u.sector,
                "ministry": u.ministry,
                "latitude": u.latitude,
                "longitude": u.longitude,
                "distance_km": u.distance_km,
                "estimated_response_time_minutes": u.estimated_response_time_minutes,
                "available_people_count": u.available_people_count,
                "fatigue_score": u.fatigue_score,
                "skills_matched": json.dumps(u.skills_matched),
                "skill_match_score": u.skill_match_score,
                "confidence": u.confidence,
                "reasoning_tags": json.dumps(u.reasoning_tags),
                "approval_status": plan.check.approval_status,
                "check_reasons": json.dumps(plan.check.reasons),
            }
        )
    return rows


def _sector_breakdown(selected: List[UnitAssignment]) -> List[SectorBreakdown]:
    by_sector: Dict[str, Dict[str, int]] = {}
    for u in selected:
        s = u.sector
        if s not in by_sector:
            by_sector[s] = {"units": 0, "people": 0}
        by_sector[s]["units"] += 1
        by_sector[s]["people"] += int(u.available_people_count)
    return [
        SectorBreakdown(sector=s, units_selected=v["units"], people_selected=v["people"])
        for s, v in sorted(by_sector.items())
    ]


def _fallback_rule_based(
    incident: IncidentInput,
    units_master: pd.DataFrame,
    policies_cfg: Dict[str, Any],
    max_units: int = 5,
) -> List[UnitAssignment]:
    # Simple fallback: closest units from primary sector first, then any sector.
    if units_master.empty:
        return []

    dom = (policies_cfg.get("domains") or {}).get(incident.incident_domain, {})
    primary = str(dom.get("primary_sector", "Police"))

    df = units_master.copy()
    # use home coords
    df["unit_lat"] = df.get("home_latitude").astype(float)
    df["unit_lon"] = df.get("home_longitude").astype(float)
    df["distance_km"] = (
        (df["unit_lat"] - float(incident.latitude)) ** 2 + (df["unit_lon"] - float(incident.longitude)) ** 2
    ) ** 0.5
    df = df.sort_values("distance_km")

    picked: List[UnitAssignment] = []
    for _, r in df[df["sector"] == primary].head(max_units).iterrows():
        picked.append(
            UnitAssignment(
                unit_id=str(r["unit_id"]),
                sector=str(r["sector"]),
                ministry=str(r.get("ministry", "")),
                latitude=float(r["unit_lat"]),
                longitude=float(r["unit_lon"]),
                geo_area=str(r.get("home_base_geo_area")) if r.get("home_base_geo_area") is not None else None,
                distance_km=float(r["distance_km"]),
                estimated_response_time_minutes=float(r.get("distance_km", 0.0)) * 2.0 + 5.0,
                available_people_count=int(r.get("max_capacity_people", 0) or 0),
                fatigue_score=0.0,
                skills_matched=[],
                skill_match_score=0.0,
                confidence=0.0,
                reasoning_tags=["fallback_rule_based"],
            )
        )
    if len(picked) < max_units:
        for _, r in df.head(max_units - len(picked)).iterrows():
            if str(r["unit_id"]) in {u.unit_id for u in picked}:
                continue
            picked.append(
                UnitAssignment(
                    unit_id=str(r["unit_id"]),
                    sector=str(r["sector"]),
                    ministry=str(r.get("ministry", "")),
                    latitude=float(r["unit_lat"]),
                    longitude=float(r["unit_lon"]),
                    geo_area=str(r.get("home_base_geo_area")) if r.get("home_base_geo_area") is not None else None,
                    distance_km=float(r["distance_km"]),
                    estimated_response_time_minutes=float(r.get("distance_km", 0.0)) * 2.0 + 5.0,
                    available_people_count=int(r.get("max_capacity_people", 0) or 0),
                    fatigue_score=0.0,
                    skills_matched=[],
                    skill_match_score=0.0,
                    confidence=0.0,
                    reasoning_tags=["fallback_rule_based"],
                )
            )
    return picked


def create_deployment_plan(incident: IncidentInput, root: str | Path = ".") -> DeploymentPlan:
    cfg, paths = load_config(root)
    policies_cfg = load_yaml(paths.root / "configs" / "policies.yaml")

    data_dir = paths.root / cfg["paths"]["data_dir"]

    units_master = _read_csv_if_exists(data_dir / "workforce_units.csv")
    skills = _read_csv_if_exists(data_dir / "workforce_skills.csv")
    on_shift = _read_csv_if_exists(data_dir / "on_shift_units.csv")
    locs = _read_csv_if_exists(data_dir / "unit_locations.csv")
    sector_capacity = _read_csv_if_exists(data_dir / "sector_capacity.csv")
    policy_constraints = _read_csv_if_exists(data_dir / "policy_constraints.csv")
    cross_sector_rules = _read_csv_if_exists(data_dir / "cross_sector_rules.csv")

    on_shift_latest = _latest_per_key(on_shift, "unit_id") if not on_shift.empty else pd.DataFrame()
    locs_latest = _latest_per_key(locs, "unit_id") if not locs.empty else pd.DataFrame()
    sector_capacity_latest = _latest_per_sector_area(sector_capacity) if not sector_capacity.empty else pd.DataFrame()

    scorer = ScoringModel.load(paths.models_dir)

    incident_id = incident.incident_id or f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    ts = incident.timestamp or datetime.now(timezone.utc)

    selected, diag = optimize_deployment(
        incident=incident,
        units_master=units_master,
        skills=skills,
        on_shift_latest=on_shift_latest,
        locations_latest=locs_latest,
        sector_capacity_latest=sector_capacity_latest,
        policy_constraints=policy_constraints,
        cross_sector_rules=cross_sector_rules,
        scoring_model=scorer,
        config=cfg,
        policies_cfg=policies_cfg,
    )

    decision = evaluate_and_gate(selected_units=selected, optimizer_diagnostics=diag, config=cfg)

    # Fallback if blocked
    if decision.status == "blocked":
        selected = _fallback_rule_based(
            incident=incident,
            units_master=units_master,
            policies_cfg=policies_cfg,
            max_units=int(cfg.get("optimization", {}).get("max_units_to_dispatch", 5)),
        )
        diag["fallback_used"] = True

    check_res = to_check_layer_result(decision)

    plan = DeploymentPlan(
        incident_id=incident_id,
        timestamp=ts,
        geo_area=incident.geo_area,
        incident_domain=incident.incident_domain,
        latitude=float(incident.latitude),
        longitude=float(incident.longitude),
        severity=int(incident.severity),
        selected_units=selected,
        sector_breakdown=_sector_breakdown(selected),
        estimated_response_time_minutes=float(min([u.estimated_response_time_minutes for u in selected]) if selected else 0.0),
        confidence_score=float(check_res.confidence_score),
        reasoning_tags=sorted({t for u in selected for t in u.reasoning_tags}),
        check=check_res,
        meta={"model_loaded": scorer.loaded, "optimizer": diag},
    )

    # Persist outputs
    ensure_dir(paths.output_dir)
    (paths.output_dir / "deployment_plan_latest.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    rows = _deployment_plan_to_rows(plan)
    if rows:
        pd.DataFrame(rows).to_csv(paths.output_dir / "deployment_plan_latest.csv", index=False)

    persist_latest_check(
        paths.metrics_dir,
        {
            "approval_status": plan.check.approval_status,
            "reasons": plan.check.reasons,
            "confidence_score": plan.check.confidence_score,
            "model_loaded": scorer.loaded,
            "incident_id": plan.incident_id,
        },
    )

    return plan


def capacity_view(root: str | Path = ".") -> CapacityView:
    cfg, paths = load_config(root)
    data_dir = paths.root / cfg["paths"]["data_dir"]
    df = _read_csv_if_exists(data_dir / "sector_capacity.csv")
    ts = datetime.now(timezone.utc)

    if df.empty:
        return CapacityView(timestamp=ts, rows=[])

    latest = _latest_per_sector_area(df)
    max_util_allowed = float(cfg.get("thresholds", {}).get("max_sector_utilization_allowed", 0.85))

    rows: List[CapacityRow] = []
    for _, r in latest.iterrows():
        util = float(r.get("utilization_ratio", 0.0) or 0.0)
        if util < max_util_allowed * 0.8:
            status = "green"
            rec = ""
        elif util <= max_util_allowed:
            status = "amber"
            rec = "pre_position_units"
        else:
            status = "red"
            rec = "activate_cross_sector_backup"

        rows.append(
            CapacityRow(
                geo_area=str(r.get("geo_area", "unknown")),
                sector=str(r.get("sector", "unknown")),
                utilization_ratio=util,
                capacity_status=status,
                recommendation=rec,
            )
        )

    # Persist capacity map
    ensure_dir(paths.output_dir)
    pd.DataFrame([x.model_dump() for x in rows]).to_csv(paths.output_dir / "capacity_risk_map.csv", index=False)

    return CapacityView(timestamp=ts, rows=rows)


def closest_units(
    *,
    latitude: float,
    longitude: float,
    incident_domain: str,
    limit: int = 10,
    root: str | Path = ".",
) -> ClosestUnitsResponse:
    cfg, paths = load_config(root)
    policies_cfg = load_yaml(paths.root / "configs" / "policies.yaml")

    data_dir = paths.root / cfg["paths"]["data_dir"]
    units_master = _read_csv_if_exists(data_dir / "workforce_units.csv")
    skills = _read_csv_if_exists(data_dir / "workforce_skills.csv")

    if units_master.empty:
        return ClosestUnitsResponse(
            timestamp=datetime.now(timezone.utc),
            incident_domain=incident_domain,
            latitude=float(latitude),
            longitude=float(longitude),
            units=[],
        )

    dom = (policies_cfg.get("domains") or {}).get(incident_domain, {})
    primary = str(dom.get("primary_sector", "Police"))
    allowed = set([primary] + list(dom.get("allowed_support_sectors", [])))

    df = units_master.copy()
    df = df[df["sector"].isin(allowed)]
    df["unit_lat"] = df.get("home_latitude").astype(float)
    df["unit_lon"] = df.get("home_longitude").astype(float)
    df["distance_km"] = (
        (df["unit_lat"] - float(latitude)) ** 2 + (df["unit_lon"] - float(longitude)) ** 2
    ) ** 0.5
    df = df.sort_values("distance_km").head(max(1, int(limit)))

    # Simple skill match via domain required list
    skills_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not skills.empty:
        for _, r in skills.iterrows():
            uid = str(r["unit_id"])
            skills_map.setdefault(uid, {})[str(r["skill_name"])] = {
                "proficiency_level": float(r.get("proficiency_level", 0.0)),
                "years_experience": float(r.get("years_experience", 0.0)),
                "incidents_handled_count": float(r.get("incidents_handled_count", 0.0)),
                "success_rate": float(r.get("success_rate", 0.0)),
            }

    units: List[UnitAssignment] = []
    from .features import skill_match_score, estimate_response_time_minutes

    for _, r in df.iterrows():
        uid = str(r["unit_id"])
        sm, matched = skill_match_score(incident_domain, skills_map.get(uid, {}))
        dist = float(r["distance_km"])
        units.append(
            UnitAssignment(
                unit_id=uid,
                sector=str(r.get("sector", "")),
                ministry=str(r.get("ministry", "")),
                latitude=float(r["unit_lat"]),
                longitude=float(r["unit_lon"]),
                geo_area=str(r.get("home_base_geo_area")) if r.get("home_base_geo_area") is not None else None,
                distance_km=dist,
                estimated_response_time_minutes=float(estimate_response_time_minutes(dist)),
                available_people_count=int(r.get("max_capacity_people", 0) or 0),
                fatigue_score=0.0,
                skills_matched=matched,
                skill_match_score=float(sm),
                confidence=float(sm),
                reasoning_tags=["closest_unit"],
            )
        )

    return ClosestUnitsResponse(
        timestamp=datetime.now(timezone.utc),
        incident_domain=incident_domain,
        latitude=float(latitude),
        longitude=float(longitude),
        units=units,
    )
