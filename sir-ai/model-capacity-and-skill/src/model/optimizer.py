from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .features import compute_candidate_features, estimate_response_time_minutes, skill_match_score
from .schemas import IncidentInput, UnitAssignment
from .scorer import ScoringModel
from .utils import gini, haversine_km, safe_json_loads


@dataclass
class Policy:
    primary_sector: str
    allowed_support_sectors: List[str]
    mandatory_dual_sector: bool


def load_policy_for_domain(policies_cfg: Dict[str, Any], domain: str) -> Policy:
    dom = (policies_cfg.get("domains") or {}).get(domain) or {}
    return Policy(
        primary_sector=str(dom.get("primary_sector", "Police")),
        allowed_support_sectors=list(dom.get("allowed_support_sectors", [])),
        mandatory_dual_sector=bool(dom.get("mandatory_dual_sector", False)),
    )


def _sector_load_lookup(sector_capacity_latest: pd.DataFrame, sector: str, geo_area: Optional[str]) -> float:
    if sector_capacity_latest is None or sector_capacity_latest.empty:
        return 0.0
    df = sector_capacity_latest
    if geo_area is not None and "geo_area" in df.columns:
        df = df[(df["sector"] == sector) & (df["geo_area"] == geo_area)]
        if not df.empty:
            return float(df.iloc[0].get("utilization_ratio", 0.0))
    df = sector_capacity_latest[sector_capacity_latest["sector"] == sector]
    if df.empty:
        return 0.0
    return float(df.iloc[0].get("utilization_ratio", 0.0))


def _cross_sector_priority(cross_rules: pd.DataFrame, domain: str, severity: int, unit_sector: str) -> float:
    if cross_rules is None or cross_rules.empty:
        return 0.0
    df = cross_rules
    df = df[df["incident_domain"] == domain]
    if df.empty:
        return 0.0
    # Heuristic: if severity above threshold and this sector is backup, add priority
    try:
        threshold = float(df["severity_threshold"].median())
    except Exception:
        threshold = 3.0
    if severity < threshold:
        return 0.0
    backup_sectors = set(df["backup_sector"].astype(str).unique().tolist())
    if unit_sector in backup_sectors:
        # prioritize higher priority_level
        try:
            return float(df["priority_level"].max()) / 5.0
        except Exception:
            return 0.5
    return 0.0


def _policy_eligible(policy: Policy, unit_sector: str) -> int:
    if unit_sector == policy.primary_sector:
        return 1
    if unit_sector in policy.allowed_support_sectors:
        return 1
    return 0


def _required_people(severity: int) -> int:
    return {1: 4, 2: 8, 3: 12, 4: 18, 5: 25}.get(int(severity), 12)


def optimize_deployment(
    *,
    incident: IncidentInput,
    units_master: pd.DataFrame,
    skills: pd.DataFrame,
    on_shift_latest: pd.DataFrame,
    locations_latest: pd.DataFrame,
    sector_capacity_latest: pd.DataFrame,
    policy_constraints: pd.DataFrame,
    cross_sector_rules: pd.DataFrame,
    scoring_model: ScoringModel,
    config: Dict[str, Any],
    policies_cfg: Dict[str, Any],
) -> Tuple[List[UnitAssignment], Dict[str, Any]]:
    """Return selected units and optimizer diagnostics."""

    opt_cfg = config.get("optimization", {})
    thresholds = config.get("thresholds", {})

    max_units = int(opt_cfg.get("max_units_to_dispatch", 5))
    top_k = int(opt_cfg.get("top_k", 10))
    max_dist_km = float(opt_cfg.get("max_distance_km", 50))
    min_people = int(opt_cfg.get("min_available_people_per_unit", 2))

    max_fatigue = float(thresholds.get("max_fatigue_score_for_dispatch", 0.7))
    max_util = float(thresholds.get("max_sector_utilization_allowed", 0.85))

    policy = load_policy_for_domain(policies_cfg, incident.incident_domain)

    # Latest status merge
    df = units_master.copy()
    if not on_shift_latest.empty:
        df = df.merge(on_shift_latest, on="unit_id", how="left")
    if not locations_latest.empty:
        df = df.merge(locations_latest, on="unit_id", how="left", suffixes=("", "_live"))

    df["is_on_shift"] = df.get("is_on_shift").fillna(False).astype(bool)
    df["available_people_count"] = df.get("available_people_count").fillna(0).astype(int)
    df["fatigue_score"] = df.get("fatigue_score").fillna(1.0).astype(float)
    df["last_dispatch_minutes_ago"] = df.get("last_dispatch_minutes_ago").fillna(9999.0).astype(float)

    # location preference: live if present, else home
    df["unit_lat"] = df["latitude"].fillna(df.get("home_latitude"))
    df["unit_lon"] = df["longitude"].fillna(df.get("home_longitude"))
    df["unit_geo_area"] = df.get("geo_area").fillna(df.get("home_base_geo_area"))

    # Basic filters
    df = df[df["is_on_shift"]]
    df = df[df["available_people_count"] >= min_people]
    df = df[df["fatigue_score"] <= max_fatigue]

    # Distance filter
    df["distance_km"] = df.apply(
        lambda r: haversine_km(float(incident.latitude), float(incident.longitude), float(r["unit_lat"]), float(r["unit_lon"])),
        axis=1,
    )
    df = df[df["distance_km"] <= max_dist_km]

    if df.empty:
        return [], {"reason": "no_candidates_after_filters"}

    # Skills: pre-group per unit
    skills_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    if not skills.empty:
        for _, row in skills.iterrows():
            uid = str(row["unit_id"])
            s = str(row["skill_name"])
            skills_map[uid][s] = {
                "proficiency_level": float(row.get("proficiency_level", 0.0)),
                "years_experience": float(row.get("years_experience", 0.0)),
                "incidents_handled_count": float(row.get("incidents_handled_count", 0.0)),
                "success_rate": float(row.get("success_rate", 0.0)),
            }

    # Policy constraints: allow per-domain overrides if present in csv
    # (domain, primary_sector, allowed_support_sectors, mandatory_dual_sector_flag)
    if policy_constraints is not None and not policy_constraints.empty:
        pc = policy_constraints[policy_constraints["domain"] == incident.incident_domain]
        if not pc.empty:
            first = pc.iloc[0].to_dict()
            policy.primary_sector = str(first.get("primary_sector", policy.primary_sector))
            allowed = safe_json_loads(first.get("allowed_support_sectors"), policy.allowed_support_sectors)
            if isinstance(allowed, list):
                policy.allowed_support_sectors = [str(x) for x in allowed]
            policy.mandatory_dual_sector = bool(first.get("mandatory_dual_sector_flag", policy.mandatory_dual_sector))

    feature_dicts: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        uid = str(r["unit_id"])
        sector = str(r.get("sector", "Unknown"))
        eligible = _policy_eligible(policy, sector)

        geo_area = incident.geo_area or str(r.get("unit_geo_area"))
        sector_load = _sector_load_lookup(sector_capacity_latest, sector, geo_area)
        cross_prio = _cross_sector_priority(cross_sector_rules, incident.incident_domain, int(incident.severity), sector)

        feats = compute_candidate_features(
            incident_lat=float(incident.latitude),
            incident_lon=float(incident.longitude),
            incident_domain=incident.incident_domain,
            incident_severity=int(incident.severity),
            unit_lat=float(r["unit_lat"]),
            unit_lon=float(r["unit_lon"]),
            unit_sector=sector,
            unit_type=str(r.get("unit_type")) if r.get("unit_type") is not None else None,
            max_capacity_people=int(r.get("max_capacity_people", 0) or 0),
            available_people_count=int(r.get("available_people_count", 0) or 0),
            is_on_shift=bool(r.get("is_on_shift")),
            fatigue_score=float(r.get("fatigue_score", 0.0) or 0.0),
            last_dispatch_minutes_ago=float(r.get("last_dispatch_minutes_ago", 9999.0) or 9999.0),
            unit_skills=skills_map.get(uid, {}),
            sector_load_ratio=float(sector_load),
            policy_eligibility_flag=int(eligible),
            cross_sector_priority=float(cross_prio),
            mobility_status=str(r.get("mobility_status")) if r.get("mobility_status") is not None else None,
        )

        rows.append(
            {
                "unit_id": uid,
                "sector": sector,
                "ministry": str(r.get("ministry", "")),
                "unit_lat": float(r["unit_lat"]),
                "unit_lon": float(r["unit_lon"]),
                "geo_area": str(r.get("unit_geo_area")) if r.get("unit_geo_area") is not None else None,
                "available_people_count": int(r.get("available_people_count", 0) or 0),
                "fatigue_score": float(r.get("fatigue_score", 0.0) or 0.0),
                "max_capacity_people": int(r.get("max_capacity_people", 0) or 0),
                "skills": skills_map.get(uid, {}),
                "sector_load_ratio": float(sector_load),
                "policy_eligible": int(eligible),
                "mobility_status": str(r.get("mobility_status")) if r.get("mobility_status") is not None else None,
                "distance_km": float(feats["distance_km"]),
                "eta_minutes": float(feats["eta_minutes"]),
                "skill_match_score": float(feats["skill_match_score"]),
            }
        )
        feature_dicts.append(feats)

    proba = scoring_model.predict_proba(feature_dicts)
    for i, p in enumerate(proba):
        rows[i]["score"] = float(p)

    # Filter out ineligible (policy)
    rows = [r for r in rows if r.get("policy_eligible", 0) == 1]
    if not rows:
        return [], {"reason": "no_candidates_after_policy"}

    # Sort by score then eta
    rows.sort(key=lambda x: (-float(x.get("score", 0.0)), float(x.get("eta_minutes", 1e9))))

    # Selection with constraints + load balancing
    required_people = _required_people(int(incident.severity))
    selected: List[UnitAssignment] = []
    people_acc = 0
    sectors_used: set[str] = set()

    for r in rows[: max(50, top_k)]:
        if len(selected) >= max_units:
            break
        # Avoid pushing overloaded sectors unless necessary
        load = float(r.get("sector_load_ratio", 0.0))
        if load > max_util and people_acc >= required_people * 0.5:
            continue

        uid = r["unit_id"]
        skills_score, matched = skill_match_score(incident.incident_domain, r.get("skills", {}))
        reasoning: List[str] = []
        if r["distance_km"] <= sorted([x["distance_km"] for x in rows[: min(10, len(rows))]])[0] + 0.1:
            reasoning.append("closest_unit")
        if skills_score >= 0.6:
            reasoning.append("highest_skill_match")
        if load <= max_util:
            reasoning.append("low_sector_load")
        if r["sector"] == policy.primary_sector:
            reasoning.append("policy_mandated")
        if float(r.get("cross_sector_priority", 0.0)) > 0.0:
            reasoning.append("backup_required")

        assignment = UnitAssignment(
            unit_id=uid,
            sector=r["sector"],
            ministry=r["ministry"],
            latitude=float(r["unit_lat"]),
            longitude=float(r["unit_lon"]),
            geo_area=r.get("geo_area"),
            distance_km=float(r["distance_km"]),
            estimated_response_time_minutes=float(r["eta_minutes"]),
            available_people_count=int(r["available_people_count"]),
            fatigue_score=float(r["fatigue_score"]),
            skills_matched=matched,
            skill_match_score=float(skills_score),
            confidence=float(r.get("score", 0.0)),
            reasoning_tags=reasoning,
        )

        selected.append(assignment)
        sectors_used.add(r["sector"])
        people_acc += int(r["available_people_count"])
        if people_acc >= required_people and len(selected) >= 1:
            break

    # Enforce mandatory dual sector if required
    if policy.mandatory_dual_sector and len(sectors_used) < 2:
        # Try to add one support-sector unit
        for r in rows:
            if r["sector"] != policy.primary_sector and r["sector"] in policy.allowed_support_sectors:
                if r["unit_id"] in {u.unit_id for u in selected}:
                    continue
                assignment = UnitAssignment(
                    unit_id=r["unit_id"],
                    sector=r["sector"],
                    ministry=r["ministry"],
                    latitude=float(r["unit_lat"]),
                    longitude=float(r["unit_lon"]),
                    geo_area=r.get("geo_area"),
                    distance_km=float(r["distance_km"]),
                    estimated_response_time_minutes=float(r["eta_minutes"]),
                    available_people_count=int(r["available_people_count"]),
                    fatigue_score=float(r["fatigue_score"]),
                    skills_matched=skill_match_score(incident.incident_domain, r.get("skills", {}))[1],
                    skill_match_score=float(r.get("skill_match_score", 0.0)),
                    confidence=float(r.get("score", 0.0)),
                    reasoning_tags=["backup_required"],
                )
                selected.append(assignment)
                sectors_used.add(r["sector"])
                break

    diagnostics = {
        "required_people": required_people,
        "people_selected": people_acc,
        "sectors_used": sorted(sectors_used),
        "load_balance_gini": gini([
            float(x.get("sector_load_ratio", 0.0)) for x in rows[: min(50, len(rows))]
        ]),
        "policy": {
            "primary_sector": policy.primary_sector,
            "allowed_support_sectors": policy.allowed_support_sectors,
            "mandatory_dual_sector": policy.mandatory_dual_sector,
        },
    }

    return selected, diagnostics
