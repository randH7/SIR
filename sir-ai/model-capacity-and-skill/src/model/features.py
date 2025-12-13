from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .utils import haversine_km


DOMAIN_REQUIRED_SKILLS: Dict[str, List[str]] = {
    "traffic": ["traffic_control", "crowd_control"],
    "drugs": ["drug_raid", "weapons", "investigation"],
    "health": ["medical_emergency", "triage"],
    "security": ["tactical", "weapons", "negotiation"],
    "public_order": ["crowd_control", "negotiation"],
    "civil_defense": ["fire_response", "rescue", "medical_emergency"],
}


def skill_match_score(
    incident_domain: str,
    unit_skills: Dict[str, Dict[str, float]],
) -> Tuple[float, List[str]]:
    required = DOMAIN_REQUIRED_SKILLS.get(incident_domain, [])
    if not required:
        return 0.0, []

    matched: List[str] = []
    scores: List[float] = []
    for s in required:
        if s in unit_skills:
            matched.append(s)
            meta = unit_skills[s]
            # Weighted blend of proficiency + success rate + experience signal
            prof = float(meta.get("proficiency_level", 0.0)) / 5.0
            sr = float(meta.get("success_rate", 0.0))
            exp = min(1.0, float(meta.get("years_experience", 0.0)) / 10.0)
            scores.append(0.5 * prof + 0.35 * sr + 0.15 * exp)
        else:
            scores.append(0.0)

    # Normalize to [0,1]
    return (sum(scores) / max(1, len(required))), matched


def estimate_response_time_minutes(distance_km: float, mobility_status: Optional[str] = None) -> float:
    # Simple speed model: faster if already patrolling
    base_speed_kmh = 45.0
    if mobility_status == "patrolling":
        base_speed_kmh = 55.0
    elif mobility_status == "enroute":
        base_speed_kmh = 60.0
    # Minimum handling/dispatch overhead
    dispatch_overhead = 4.0
    travel = (distance_km / max(1e-6, base_speed_kmh)) * 60.0
    return dispatch_overhead + travel


def compute_candidate_features(
    *,
    incident_lat: float,
    incident_lon: float,
    incident_domain: str,
    incident_severity: int,
    unit_lat: float,
    unit_lon: float,
    unit_sector: str,
    unit_type: str | None,
    max_capacity_people: int,
    available_people_count: int,
    is_on_shift: bool,
    fatigue_score: float,
    last_dispatch_minutes_ago: float,
    unit_skills: Dict[str, Dict[str, float]],
    sector_load_ratio: float,
    policy_eligibility_flag: int,
    cross_sector_priority: float,
    mobility_status: Optional[str] = None,
) -> Dict[str, float | int | str]:
    dist_km = haversine_km(incident_lat, incident_lon, unit_lat, unit_lon)
    sm_score, _ = skill_match_score(incident_domain, unit_skills)

    # Availability is proportion of capacity currently usable
    cap = max(1, int(max_capacity_people) if max_capacity_people else 1)
    avail = max(0, int(available_people_count))
    availability_score = min(1.0, avail / cap)

    # Fatigue and recent dispatch penalize selection
    fatigue_penalty = float(fatigue_score)
    recency_penalty = 1.0 / max(1.0, float(last_dispatch_minutes_ago))

    eta_min = estimate_response_time_minutes(dist_km, mobility_status)

    return {
        "distance_km": float(dist_km),
        "eta_minutes": float(eta_min),
        "skill_match_score": float(sm_score),
        "availability_score": float(availability_score),
        "fatigue_score": float(fatigue_penalty),
        "recency_penalty": float(recency_penalty),
        "sector_load_ratio": float(sector_load_ratio),
        "policy_eligible": int(policy_eligibility_flag),
        "cross_sector_priority": float(cross_sector_priority),
        "severity": int(incident_severity),
        # categorical as strings (vectorizer will one-hot)
        "sector": unit_sector,
        "unit_type": unit_type or "unknown",
        "domain": incident_domain,
        "on_shift": int(bool(is_on_shift)),
    }
