from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SECTORS = [
    "Police",
    "National Guard",
    "Narcotics Control",
    "Ambulance",
    "Civil Defense",
    "General Investigation",
]

MINISTRY_BY_SECTOR = {
    "Police": "Ministry of Interior",
    "National Guard": "Ministry of Interior",
    "Narcotics Control": "Ministry of Interior",
    "Ambulance": "Ministry of Health",
    "Civil Defense": "Ministry of Interior",
    "General Investigation": "Ministry of Interior",
}

DOMAINS = ["traffic", "drugs", "health", "security", "public_order", "civil_defense"]

DOMAIN_PRIMARY = {
    "traffic": "Police",
    "drugs": "Narcotics Control",
    "health": "Ambulance",
    "security": "National Guard",
    "public_order": "Police",
    "civil_defense": "Civil Defense",
}

DOMAIN_SUPPORT = {
    "traffic": ["National Guard", "Ambulance", "Civil Defense"],
    "drugs": ["Police", "National Guard"],
    "health": ["Civil Defense", "Police"],
    "security": ["Police", "Civil Defense"],
    "public_order": ["National Guard", "Civil Defense"],
    "civil_defense": ["Ambulance", "Police"],
}

DOMAIN_MANDATORY_DUAL = {
    "traffic": False,
    "drugs": True,
    "health": False,
    "security": True,
    "public_order": True,
    "civil_defense": False,
}

SKILLS = [
    "drug_raid",
    "crowd_control",
    "medical_emergency",
    "hostage",
    "traffic_control",
    "weapons",
    "negotiation",
    "investigation",
    "triage",
    "tactical",
    "fire_response",
    "rescue",
]

UNIT_TYPES = ["patrol", "tactical", "medical", "investigation", "traffic"]

DOMAIN_REQUIRED_SKILLS = {
    "traffic": ["traffic_control", "crowd_control"],
    "drugs": ["drug_raid", "weapons", "investigation"],
    "health": ["medical_emergency", "triage"],
    "security": ["tactical", "weapons", "negotiation"],
    "public_order": ["crowd_control", "negotiation"],
    "civil_defense": ["fire_response", "rescue", "medical_emergency"],
}

SECTOR_SKILL_PRIORS = {
    "Police": ["traffic_control", "crowd_control", "negotiation", "weapons", "investigation"],
    "National Guard": ["tactical", "weapons", "crowd_control", "negotiation", "rescue"],
    "Narcotics Control": ["drug_raid", "weapons", "investigation", "negotiation"],
    "Ambulance": ["medical_emergency", "triage", "rescue"],
    "Civil Defense": ["fire_response", "rescue", "medical_emergency"],
    "General Investigation": ["investigation", "negotiation", "weapons"],
}


def _geo_areas(n: int = 50) -> list[str]:
    return [f"AREA_{i:03d}" for i in range(n)]


def _area_centers(areas: list[str], seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    # Rough bbox around Riyadh-ish for synthetic realism
    lats = rng.uniform(24.2, 25.2, size=len(areas))
    lons = rng.uniform(46.1, 47.3, size=len(areas))
    return {a: (float(lat), float(lon)) for a, lat, lon in zip(areas, lats, lons)}


def _ts_series(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    mins = rng.integers(0, 60 * 24 * 30, size=n)  # within last 30 days
    return [(now - timedelta(minutes=int(m))).isoformat() for m in mins]


def generate(rows: int, out_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    areas = _geo_areas(60)
    centers = _area_centers(areas, seed)

    # --- workforce_units.csv (>= rows)
    unit_ids = np.array([f"U{idx:06d}" for idx in range(rows)])
    sectors = rng.choice(SECTORS, size=rows, replace=True)
    ministries = np.vectorize(MINISTRY_BY_SECTOR.get)(sectors)
    home_areas = rng.choice(areas, size=rows, replace=True)

    base_lat = np.array([centers[a][0] for a in home_areas])
    base_lon = np.array([centers[a][1] for a in home_areas])
    home_lat = base_lat + rng.normal(0, 0.03, size=rows)
    home_lon = base_lon + rng.normal(0, 0.03, size=rows)

    unit_type = rng.choice(UNIT_TYPES, size=rows)
    max_cap = rng.integers(4, 26, size=rows)

    df_units = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "sector": sectors,
            "ministry": ministries,
            "home_base_geo_area": home_areas,
            "home_latitude": home_lat,
            "home_longitude": home_lon,
            "unit_type": unit_type,
            "max_capacity_people": max_cap,
        }
    )
    df_units.to_csv(out_dir / "workforce_units.csv", index=False)

    # --- workforce_skills.csv (>= rows)
    # 2 skills per unit => 2*rows rows
    skill_rows = rows * 2
    skill_unit = rng.choice(unit_ids, size=skill_rows, replace=True)
    sector_by_unit = df_units.set_index("unit_id")["sector"].astype(str).to_dict()
    # Choose skills conditional on sector to create learnable signal
    skill_name = []
    for uid in skill_unit:
        sec = sector_by_unit.get(uid, "Police")
        priors = SECTOR_SKILL_PRIORS.get(sec, SKILLS)
        # mix: mostly sector-prior skills, sometimes any skill
        if rng.random() < 0.8:
            skill_name.append(str(rng.choice(priors)))
        else:
            skill_name.append(str(rng.choice(SKILLS)))
    skill_name = np.array(skill_name, dtype=object)
    proficiency = rng.integers(1, 6, size=skill_rows)
    years_exp = rng.uniform(0, 15, size=skill_rows).round(2)
    handled = rng.integers(0, 400, size=skill_rows)
    success_rate = np.clip(rng.normal(0.75, 0.15, size=skill_rows), 0, 1).round(3)

    df_skills = pd.DataFrame(
        {
            "unit_id": skill_unit,
            "skill_name": skill_name,
            "proficiency_level": proficiency,
            "years_experience": years_exp,
            "incidents_handled_count": handled,
            "success_rate": success_rate,
        }
    )
    df_skills.to_csv(out_dir / "workforce_skills.csv", index=False)

    # Build unit -> skillset map for dispatch selection
    unit_skillset = {}
    for uid, grp in df_skills.groupby("unit_id"):
        unit_skillset[str(uid)] = set(grp["skill_name"].astype(str).tolist())

    # --- on_shift_units.csv (>= rows)
    ts = _ts_series(rows, seed + 1)
    on_shift = rng.random(rows) < 0.72
    # available people correlated with max capacity
    cap_lookup = df_units.set_index("unit_id")["max_capacity_people"].to_dict()
    caps = np.array([cap_lookup[uid] for uid in unit_ids])
    available = (caps * rng.uniform(0, 1, size=rows)).astype(int)
    fatigue = np.clip(rng.beta(2, 5, size=rows), 0, 1).round(3)
    last_dispatch = rng.integers(0, 600, size=rows)

    df_shift = pd.DataFrame(
        {
            "timestamp": ts,
            "unit_id": unit_ids,
            "is_on_shift": on_shift,
            "available_people_count": available,
            "fatigue_score": fatigue,
            "last_dispatch_minutes_ago": last_dispatch,
        }
    )
    df_shift.to_csv(out_dir / "on_shift_units.csv", index=False)

    # --- unit_locations.csv (>= rows)
    ts_loc = _ts_series(rows, seed + 2)
    mobility = rng.choice(["stationary", "patrolling", "enroute"], size=rows, p=[0.4, 0.45, 0.15])
    loc_lat = home_lat + rng.normal(0, 0.02, size=rows)
    loc_lon = home_lon + rng.normal(0, 0.02, size=rows)

    df_locs = pd.DataFrame(
        {
            "timestamp": ts_loc,
            "unit_id": unit_ids,
            "latitude": loc_lat,
            "longitude": loc_lon,
            "geo_area": home_areas,
            "mobility_status": mobility,
        }
    )
    df_locs.to_csv(out_dir / "unit_locations.csv", index=False)

    # --- sector_capacity.csv (>= rows)
    ts_cap = _ts_series(rows, seed + 3)
    cap_sector = rng.choice(SECTORS, size=rows)
    cap_area = rng.choice(areas, size=rows)
    total_units = rng.integers(10, 180, size=rows)
    total_people = rng.integers(40, 1200, size=rows)
    active_incidents = rng.integers(0, 120, size=rows)
    # utilization correlated with active incidents per people
    util = np.clip((active_incidents / np.maximum(1, total_people / 10)) + rng.normal(0.2, 0.15, size=rows), 0, 1)
    util = np.round(util, 3)
    overload = (util > 0.85).astype(int)

    df_cap = pd.DataFrame(
        {
            "timestamp": ts_cap,
            "sector": cap_sector,
            "geo_area": cap_area,
            "total_units": total_units,
            "total_people_on_shift": total_people,
            "active_incidents": active_incidents,
            "utilization_ratio": util,
            "overload_flag": overload,
        }
    )
    df_cap.to_csv(out_dir / "sector_capacity.csv", index=False)

    # --- policy_constraints.csv (>= rows)
    dom = rng.choice(DOMAINS, size=rows)
    primary = np.array([DOMAIN_PRIMARY[d] for d in dom])
    allowed = [json.dumps(DOMAIN_SUPPORT[d]) for d in dom]
    max_rt = rng.integers(8, 45, size=rows)
    mandatory = np.array([int(DOMAIN_MANDATORY_DUAL[d]) for d in dom])

    df_policy = pd.DataFrame(
        {
            "domain": dom,
            "primary_sector": primary,
            "allowed_support_sectors": allowed,
            "max_response_time_minutes": max_rt,
            "escalation_rules": ["standard" for _ in range(rows)],
            "mandatory_dual_sector_flag": mandatory,
        }
    )
    df_policy.to_csv(out_dir / "policy_constraints.csv", index=False)

    # --- cross_sector_rules.csv (>= rows)
    dom2 = rng.choice(DOMAINS, size=rows)
    sev_thr = rng.integers(3, 6, size=rows)
    trig = np.array([DOMAIN_PRIMARY[d] for d in dom2])
    backup = np.array([rng.choice(DOMAIN_SUPPORT[d]) for d in dom2])
    delay = rng.integers(3, 25, size=rows)
    prio = rng.integers(1, 6, size=rows)

    df_cross = pd.DataFrame(
        {
            "incident_domain": dom2,
            "severity_threshold": sev_thr,
            "trigger_sector": trig,
            "backup_sector": backup,
            "backup_activation_delay": delay,
            "priority_level": prio,
        }
    )
    df_cross.to_csv(out_dir / "cross_sector_rules.csv", index=False)

    # --- shift_rosters.csv (>= rows)
    # 1 row per unit (rows)
    start_dates = [datetime.now(timezone.utc).date() - timedelta(days=int(d)) for d in rng.integers(0, 30, size=rows)]
    shift_start = rng.choice(["06:00", "14:00", "22:00"], size=rows)
    shift_end = np.where(shift_start == "06:00", "14:00", np.where(shift_start == "14:00", "22:00", "06:00"))
    planned_people = np.maximum(1, (caps * rng.uniform(0.5, 1.0, size=rows)).astype(int))
    overtime_allowed = rng.random(rows) < 0.3

    df_roster = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "date": [d.isoformat() for d in start_dates],
            "shift_start": shift_start,
            "shift_end": shift_end,
            "planned_people": planned_people,
            "overtime_allowed": overtime_allowed,
        }
    )
    df_roster.to_csv(out_dir / "shift_rosters.csv", index=False)

    # --- incidents_history.csv (>= rows)
    inc_ids = np.array([f"INC{idx:07d}" for idx in range(rows)])
    inc_ts = _ts_series(rows, seed + 4)
    inc_area = rng.choice(areas, size=rows)
    inc_lat = np.array([centers[a][0] for a in inc_area]) + rng.normal(0, 0.035, size=rows)
    inc_lon = np.array([centers[a][1] for a in inc_area]) + rng.normal(0, 0.035, size=rows)
    inc_domain = rng.choice(DOMAINS, size=rows)
    inc_type = np.array([f"{d}_type_{int(x)}" for d, x in zip(inc_domain, rng.integers(1, 6, size=rows))])
    severity = rng.integers(1, 6, size=rows)

    # units dispatched: 1–3 units, mostly from same geo_area and policy-eligible sectors,
    # selected by a suitability score (distance + skill match + noise).
    units_by_sector = {s: df_units[df_units["sector"] == s]["unit_id"].astype(str).to_numpy() for s in SECTORS}
    units_by_area = {
        a: df_units[df_units["home_base_geo_area"] == a]["unit_id"].astype(str).to_numpy() for a in areas
    }
    sector_by_unit = df_units.set_index("unit_id")["sector"].astype(str).to_dict()
    lat_by_unit = df_units.set_index("unit_id")["home_latitude"].astype(float).to_dict()
    lon_by_unit = df_units.set_index("unit_id")["home_longitude"].astype(float).to_dict()

    dispatched_list = []
    sectors_involved_list = []
    responders = []
    res_time = []
    success = []
    escalation = []

    for a, d, sev, ilat, ilon in zip(inc_area, inc_domain, severity, inc_lat, inc_lon):
        primary_sector = DOMAIN_PRIMARY[d]
        support = DOMAIN_SUPPORT[d]
        n_units = int(rng.integers(1, 4))

        eligible_sectors = set([primary_sector] + list(support))
        # Prefer local area pool ~80% of the time for realism.
        local_pool = units_by_area.get(a, np.array([], dtype=object))
        if len(local_pool) > 0 and rng.random() < 0.8:
            pool = np.array([uid for uid in local_pool if sector_by_unit.get(uid) in eligible_sectors], dtype=object)
        else:
            pool = np.array(
                [uid for uid, sec in sector_by_unit.items() if sec in eligible_sectors],
                dtype=object,
            )
        if len(pool) == 0:
            pool = df_units["unit_id"].astype(str).to_numpy()

        required = DOMAIN_REQUIRED_SKILLS.get(d, [])
        pool_list = pool.tolist()
        # Downsample candidate pool for speed
        if len(pool_list) > 250:
            pool_list = rng.choice(np.array(pool_list, dtype=object), size=250, replace=False).astype(str).tolist()

        def _suitability(uid: str) -> float:
            sec = sector_by_unit.get(uid, "")
            # approximate distance (not haversine, but consistent)
            ulat = float(lat_by_unit.get(uid, 0.0))
            ulon = float(lon_by_unit.get(uid, 0.0))
            dist = float(np.sqrt((ulat - float(ilat)) ** 2 + (ulon - float(ilon)) ** 2))
            # skill match fraction
            sset = unit_skillset.get(uid, set())
            sm = 0.0
            if required:
                sm = len([x for x in required if x in sset]) / float(len(required))
            primary_bonus = 0.15 if sec == primary_sector else 0.0
            # Score: higher skill, lower distance
            return float(1.2 * sm - 2.0 * dist + primary_bonus + rng.normal(0, 0.05))

        scored = sorted([(uid, _suitability(uid)) for uid in pool_list], key=lambda x: x[1], reverse=True)
        chosen = [uid for uid, _ in scored[: min(n_units, len(scored))]]

        dispatched_list.append(json.dumps(chosen))
        sectors_involved_list.append(
            json.dumps(sorted(set([sector_by_unit.get(u, primary_sector) for u in chosen])))
        )

        # responders count rough
        responders.append(int(np.clip(rng.normal(10 + sev * 4, 4), 2, 40)))
        rt = float(np.clip(rng.normal(35 + sev * 8, 10), 8, 180))
        res_time.append(round(rt, 2))
        succ = 1 if rng.random() < (0.8 - (sev - 3) * 0.08) else 0
        success.append(succ)
        escalation.append(1 if (sev >= 4 and rng.random() < 0.25) else 0)

    df_inc = pd.DataFrame(
        {
            "incident_id": inc_ids,
            "timestamp": inc_ts,
            "geo_area": inc_area,
            "latitude": inc_lat,
            "longitude": inc_lon,
            "incident_domain": inc_domain,
            "incident_type": inc_type,
            "severity": severity,
            "sectors_involved": sectors_involved_list,
            "units_dispatched": dispatched_list,
            "responders_count": responders,
            "resolution_time_minutes": res_time,
            "success_flag": success,
            "escalation_flag": escalation,
        }
    )
    df_inc.to_csv(out_dir / "incidents_history.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50000)
    p.add_argument("--out", type=str, default=str(Path("data") / "input_datasets"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    generate(rows=int(args.rows), out_dir=Path(args.out), seed=int(args.seed))


if __name__ == "__main__":
    main()
