#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DOMAINS = ["traffic", "drugs", "public_order", "health", "civil_defense", "security"]
GEO_AREAS = [
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

THEMES = ["crowding", "threat", "weapons", "drifting", "storm", "road_block", "harassment"]
PLATFORMS = ["x", "instagram", "tiktok", "snapchat", "telegram"]
NEWS_SOURCES = [
    "SPA",
    "Al Riyadh",
    "Okaz",
    "Al Arabiya",
    "Saudi Gazette",
]

INCIDENT_TYPES_BY_DOMAIN = {
    "traffic": [
        "collision_minor",
        "collision_major",
        "run_over",
        "road_block",
        "drifting",
        "intentional_crash",
    ],
    "drugs": ["usage", "selling", "trafficking", "raid", "possession"],
    "public_order": [
        "fight",
        "harassment",
        "illegal_gathering",
        "crowd_incident",
        "noise_complaint",
        "domestic_violence",
    ],
    "health": ["injury", "heatstroke", "medical_emergency", "poisoning"],
    "civil_defense": ["fire", "storm_damage", "hazmat", "rescue"],
    "security": [
        "theft",
        "robbery",
        "burglary",
        "suspicious_package",
        "weapons_report",
        "threat_report",
    ],
}

SECTORS = [
    "Traffic Police",
    "Ambulance",
    "Civil Defense",
    "Public Security",
    "National Guard",
    "Narcotics Control",
    "General Investigation",
    "Municipality",
    "Ministry of Health",
]

REPORTING_CHANNELS = ["911", "internal", "ministry", "app", "patrol"]
RESOLUTION_STATUSES = ["resolved", "pending", "false_alarm", "escalated"]

NEIGHBORHOODS = [
    "Al Olaya",
    "Al Malaz",
    "Al Nakheel",
    "Al Yasmin",
    "Al Rawdah",
    "Al Rabwah",
    "King Fahd",
    "Al Wurud",
    "Al Shifa",
    "Al Aziziyah",
]

LANDMARKS = [
    "Kingdom Centre",
    "Al Faisaliah Tower",
    "King Saud University",
    "Riyadh Front",
    "Boulevard Riyadh City",
    "Diriyah Gate",
    "King Khalid International Airport",
    "Imam Mohammad Ibn Saud University",
]

WEATHER_CONDITIONS = ["clear", "hazy", "dust", "cloudy", "light_rain", "storm"]


@dataclass(frozen=True)
class GeoBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Rough Riyadh bounding boxes per area (synthetic but realistic looking).
GEO_BOXES: dict[str, GeoBox] = {
    "Central": GeoBox(24.66, 24.75, 46.66, 46.76),
    "North": GeoBox(24.78, 24.92, 46.60, 46.78),
    "South": GeoBox(24.52, 24.64, 46.60, 46.78),
    "East": GeoBox(24.62, 24.82, 46.78, 46.95),
    "West": GeoBox(24.62, 24.82, 46.45, 46.64),
    "NorthEast": GeoBox(24.78, 24.92, 46.78, 46.95),
    "NorthWest": GeoBox(24.78, 24.92, 46.45, 46.64),
    "SouthEast": GeoBox(24.52, 24.64, 46.78, 46.95),
    "SouthWest": GeoBox(24.52, 24.64, 46.45, 46.64),
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_timestamps(rng: np.random.Generator, start: datetime, end: datetime, n: int) -> pd.DatetimeIndex:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    ts = rng.integers(start_ts, end_ts, size=n, endpoint=False)
    return pd.to_datetime(ts, unit="s", utc=True)


def _choose_geo_areas(rng: np.random.Generator, n: int) -> np.ndarray:
    weights = np.array([0.18, 0.12, 0.14, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08])
    weights = weights / weights.sum()
    return rng.choice(GEO_AREAS, size=n, p=weights)


def _coords_for_areas(rng: np.random.Generator, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lats = np.empty(len(areas))
    lons = np.empty(len(areas))
    for i, a in enumerate(areas):
        box = GEO_BOXES[a]
        lats[i] = rng.uniform(box.lat_min, box.lat_max)
        lons[i] = rng.uniform(box.lon_min, box.lon_max)
    return lats, lons


def _theme_conf(rng: np.random.Generator, n: int, base: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for t in THEMES:
        # Skewed small, with occasional spikes.
        v = rng.beta(1.3, 6.5, size=n) * base
        spikes = rng.random(size=n) < 0.07
        v[spikes] = np.clip(v[spikes] + rng.uniform(0.4, 0.9, size=spikes.sum()), 0.0, 1.0)
        out[f"theme_conf_{t}"] = np.round(v, 3)
    return out


def generate_weather(out_path: Path, start: datetime, end: datetime, seed: int) -> None:
    rng = _rng(seed)
    # Hourly weather.
    ts = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    n = len(ts)

    # Riyadh-like: hot summers, mild winters.
    day_of_year = ts.dayofyear.values
    seasonal = 10 * np.sin(2 * np.pi * (day_of_year - 160) / 365.25)  # peak summer
    temp = 30 + seasonal + rng.normal(0, 2.5, size=n)

    # Rain is rare; wind/dust events exist.
    # Rare rain, but keep enough storm windows to learn civil_defense coupling.
    rain = (rng.random(size=n) < 0.05).astype(float) * rng.uniform(0.2, 12.0, size=n)
    wind = np.clip(rng.normal(12, 5, size=n), 0, 40)

    condition = np.array(["clear"] * n, dtype=object)
    dust = (wind > 18) & (rng.random(size=n) < 0.25)
    condition[dust] = "dust"
    haze = (rng.random(size=n) < 0.08)
    condition[haze] = "hazy"
    cloudy = (rng.random(size=n) < 0.10)
    condition[cloudy] = "cloudy"
    condition[rain > 0] = "light_rain"
    storm = (rain > 6.0) & (wind > 19)
    condition[storm] = "storm"

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": np.round(temp, 2),
            "rain_mm": np.round(rain, 2),
            "wind_kph": np.round(wind, 2),
            "condition": condition,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _build_event_window_index(
    events_df: pd.DataFrame, start: datetime, end: datetime, window_minutes: int
) -> pd.DataFrame:
    """Return per-window per-geo_area event intensity in [0, ~]."""
    if events_df.empty:
        return pd.DataFrame(columns=["window_start", "geo_area", "event_intensity"])

    df = events_df.copy()
    df["timestamp_start"] = pd.to_datetime(df["timestamp_start"], utc=True).dt.tz_convert(None)
    df["timestamp_end"] = pd.to_datetime(df["timestamp_end"], utc=True).dt.tz_convert(None)
    df["expected_size"] = pd.to_numeric(df["expected_size"], errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5)

    rows = []
    for r in df.itertuples(index=False):
        ws = pd.Timestamp(r.timestamp_start).floor(f"{window_minutes}min")
        we = pd.Timestamp(r.timestamp_end).ceil(f"{window_minutes}min")
        windows = pd.date_range(ws, we, freq=f"{window_minutes}min", inclusive="left")
        # intensity ~ log crowd size * confidence
        intensity = float(np.log1p(max(0.0, float(r.expected_size))) * float(r.confidence))
        for w in windows:
            rows.append({"window_start": w, "geo_area": r.geo_area, "event_intensity": intensity})

    out = pd.DataFrame(rows)
    out = (
        out.groupby(["window_start", "geo_area"], observed=True)["event_intensity"]
        .sum()
        .reset_index()
    )
    return out


def _hour_profile(hour: np.ndarray, peaks: list[tuple[int, int]]) -> np.ndarray:
    """Simple piecewise profile: 1.0 baseline, + bumps during peak ranges."""
    prof = np.ones_like(hour, dtype=float)
    for a, b in peaks:
        prof += ((hour >= a) & (hour <= b)).astype(float) * 0.8
    return prof


def _make_hotspots(
    rng: np.random.Generator, n_steps: int, n_areas: int, n_domains: int
) -> np.ndarray:
    """AR(1) hotspots -> multiplicative factors around 1.0."""
    phi = 0.92
    eps = rng.normal(0, 0.20, size=(n_steps, n_areas, n_domains))
    h = np.zeros((n_steps, n_areas, n_domains), dtype=float)
    for t in range(1, n_steps):
        h[t] = phi * h[t - 1] + eps[t]
    # Convert to multiplier, clipped.
    mult = np.clip(np.exp(h), 0.5, 3.0)
    return mult


def _domain_bias_for_dataset(dataset_name: str) -> dict[str, float]:
    # Make department datasets more realistic in what they see.
    if "narcotics_control" in dataset_name:
        return {"drugs": 3.0, "security": 1.1, "public_order": 0.9, "traffic": 0.7, "health": 0.6, "civil_defense": 0.6}
    if "general_investigation" in dataset_name:
        return {"security": 2.6, "public_order": 1.4, "drugs": 1.1, "traffic": 0.7, "health": 0.6, "civil_defense": 0.6}
    if "national_guard" in dataset_name:
        return {"public_order": 2.0, "security": 1.8, "traffic": 0.9, "drugs": 0.8, "health": 0.7, "civil_defense": 0.7}
    if "ambulance" in dataset_name:
        return {"health": 2.6, "traffic": 1.3, "public_order": 0.8, "security": 0.7, "drugs": 0.6, "civil_defense": 0.9}
    # all_911
    return {d: 1.0 for d in DOMAINS}


def _incident_type_weighted(
    rng: np.random.Generator,
    domain: str,
    hour: int,
    is_weekend: bool,
    event_intensity: float,
    stormy: bool,
) -> str:
    types = INCIDENT_TYPES_BY_DOMAIN[domain]
    w = np.ones(len(types), dtype=float)
    for i, t in enumerate(types):
        if domain == "traffic" and t == "drifting":
            w[i] *= 2.2 if (hour >= 20 or hour <= 2) else 0.6
        if domain == "traffic" and t == "road_block":
            w[i] *= 1.6 if event_intensity > 0 else 1.0
        if domain == "public_order" and t in {"illegal_gathering", "crowd_incident"}:
            w[i] *= 2.0 if (event_intensity > 0 or is_weekend) else 0.8
        if domain == "civil_defense" and t in {"storm_damage"}:
            w[i] *= 2.4 if stormy else 0.4
        if domain == "security" and t in {"weapons_report", "threat_report"}:
            w[i] *= 1.5 if (is_weekend or event_intensity > 0) else 1.0
        if domain == "drugs" and t in {"trafficking"}:
            w[i] *= 1.6 if (hour >= 18 or hour <= 3) else 0.9
    w = w / w.sum()
    return str(rng.choice(types, p=w))


def _incident_row_text(domain: str, incident_type: str, geo_area: str) -> str:
    templates = {
        "traffic": [
            "Traffic incident reported near {area}; possible {itype}.",
            "Multiple calls about {itype} in {area}.",
        ],
        "drugs": [
            "Report of suspected drug {itype} activity in {area}.",
            "Tip received regarding {itype} case, {area}.",
        ],
        "public_order": [
            "Public order issue: {itype} reported in {area}.",
            "Caller reports {itype} disturbance, {area}.",
        ],
        "health": [
            "Medical dispatch: {itype} case in {area}.",
            "Ambulance requested for {itype}, {area}.",
        ],
        "civil_defense": [
            "Civil defense call: {itype} incident in {area}.",
            "Emergency response needed for {itype}, {area}.",
        ],
        "security": [
            "Security report: {itype} suspected in {area}.",
            "Alert received about {itype}, {area}.",
        ],
    }
    return np.random.choice(templates[domain]).format(area=geo_area, itype=incident_type)


def generate_incidents(
    out_path: Path,
    start: datetime,
    end: datetime,
    n_rows: int,
    seed: int,
    channel_bias: str | None = None,
    *,
    weather_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    window_minutes: int = 60,
) -> None:
    rng = _rng(seed)

    # Build hourly windows.
    windows = pd.date_range(start=start, end=end, freq=f"{window_minutes}min", inclusive="left", tz="UTC").tz_convert(None)
    n_steps = len(windows)
    areas_list = GEO_AREAS
    n_areas = len(areas_list)

    # Weather multipliers per window (city-wide)
    weather_mult = np.ones((n_steps, len(DOMAINS)), dtype=float)
    stormy = np.zeros(n_steps, dtype=bool)
    if weather_df is not None and not weather_df.empty:
        w = weather_df.copy()
        w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True).dt.tz_convert(None)
        w = w.set_index(w["timestamp"].dt.floor(f"{window_minutes}min")).sort_index()
        # Align to windows; forward fill within day.
        w = w.reindex(windows, method="nearest")
        rain = pd.to_numeric(w["rain_mm"], errors="coerce").fillna(0.0).to_numpy()
        wind = pd.to_numeric(w["wind_kph"], errors="coerce").fillna(0.0).to_numpy()
        temp = pd.to_numeric(w["temp_c"], errors="coerce").fillna(30.0).to_numpy()
        stormy = (rain > 6.0) | (wind > 22.0)

        # Effects
        # traffic increases with rain/wind; civil_defense increases when stormy; health increases with heat and storms.
        idx = {d: i for i, d in enumerate(DOMAINS)}
        weather_mult[:, idx["traffic"]] *= 1.0 + 0.04 * np.clip(rain, 0, 12) + 0.01 * np.clip(wind - 15, 0, 30)
        weather_mult[:, idx["civil_defense"]] *= 1.0 + 0.35 * stormy.astype(float)
        weather_mult[:, idx["health"]] *= 1.0 + 0.02 * np.clip(temp - 38, 0, 10) + 0.10 * stormy.astype(float)

    # Event intensity per window x area
    event_intensity = np.zeros((n_steps, n_areas), dtype=float)
    if events_df is not None and not events_df.empty:
        ev_idx = _build_event_window_index(events_df, start, end, window_minutes)
        if not ev_idx.empty:
            ev_idx["window_start"] = pd.to_datetime(ev_idx["window_start"])
            map_w = {w: i for i, w in enumerate(windows)}
            map_a = {a: i for i, a in enumerate(areas_list)}
            for r in ev_idx.itertuples(index=False):
                wi = map_w.get(r.window_start)
                ai = map_a.get(r.geo_area)
                if wi is not None and ai is not None:
                    event_intensity[wi, ai] += float(r.event_intensity)

    # Base time-of-day patterns per domain
    hours = windows.hour.to_numpy()
    dow = windows.dayofweek.to_numpy()
    is_weekend = (dow >= 4)
    month = windows.month.to_numpy()

    base_time = np.ones((n_steps, len(DOMAINS)), dtype=float)
    idx = {d: i for i, d in enumerate(DOMAINS)}
    base_time[:, idx["traffic"]] *= _hour_profile(hours, [(7, 9), (16, 19)])
    base_time[:, idx["drugs"]] *= _hour_profile(hours, [(20, 23), (0, 2)])
    base_time[:, idx["public_order"]] *= (1.0 + 0.7 * is_weekend.astype(float)) * _hour_profile(hours, [(19, 23)])
    base_time[:, idx["health"]] *= (1.0 + 0.2 * (month >= 6).astype(float))  # summer
    base_time[:, idx["civil_defense"]] *= 1.0 + 0.25 * (stormy.astype(float))
    base_time[:, idx["security"]] *= (1.0 + 0.25 * is_weekend.astype(float))

    # Hotspots (persistent, per area x domain). Increase predictability for
    # lower-frequency domains to make the baseline learnable.
    hotspots = _make_hotspots(rng, n_steps, n_areas, len(DOMAINS))
    hotspots[:, :, idx["drugs"]] = np.clip(hotspots[:, :, idx["drugs"]] ** 1.35, 0.5, 4.0)
    hotspots[:, :, idx["security"]] = np.clip(hotspots[:, :, idx["security"]] ** 1.25, 0.5, 4.0)
    hotspots[:, :, idx["civil_defense"]] = np.clip(hotspots[:, :, idx["civil_defense"]] ** 1.50, 0.5, 4.5)

    # Dataset bias
    bias = _domain_bias_for_dataset(out_path.name)
    bias_vec = np.array([bias[d] for d in DOMAINS], dtype=float)

    # Domain weights per window-area (n_steps, n_areas, n_domains)
    dom_w = (
        base_time[:, None, :] * weather_mult[:, None, :] * hotspots * bias_vec[None, None, :]
    )
    # Event effects: increase traffic/public_order/health around events.
    ev = event_intensity[:, :, None]
    dom_w[:, :, idx["traffic"]] *= 1.0 + 0.15 * ev[:, :, 0]
    dom_w[:, :, idx["public_order"]] *= 1.0 + 0.22 * ev[:, :, 0]
    dom_w[:, :, idx["health"]] *= 1.0 + 0.08 * ev[:, :, 0]
    dom_w[:, :, idx["security"]] *= 1.0 + 0.06 * ev[:, :, 0]
    dom_w[:, :, idx["drugs"]] *= 1.0 + 0.03 * ev[:, :, 0]

    # Flatten window-area weights to sample rows.
    total_w = dom_w.sum(axis=2)  # (n_steps, n_areas)
    flat_w = total_w.reshape(-1)
    flat_w = flat_w / flat_w.sum()

    choices = rng.choice(np.arange(flat_w.size), size=n_rows, replace=True, p=flat_w)
    win_idx = choices // n_areas
    area_idx = choices % n_areas
    areas = np.array([areas_list[i] for i in area_idx], dtype=object)
    lats, lons = _coords_for_areas(rng, areas)

    # Sample domains conditional on chosen window/area.
    domains = np.empty(n_rows, dtype=object)
    for i in range(n_rows):
        wv = dom_w[win_idx[i], area_idx[i], :]
        wv = wv / wv.sum()
        domains[i] = rng.choice(DOMAINS, p=wv)

    # Timestamp within the window.
    base_ts = np.array([windows[i] for i in win_idx], dtype="datetime64[ns]")
    secs = rng.integers(0, window_minutes * 60, size=n_rows)
    ts = pd.to_datetime(base_ts) + pd.to_timedelta(secs, unit="s")

    # Incident type depends on context (event/weather/hour).
    incident_types = np.empty(n_rows, dtype=object)
    severity = np.empty(n_rows, dtype=int)
    for i, d in enumerate(domains):
        h = int(pd.Timestamp(base_ts[i]).hour)
        wknd = bool(is_weekend[win_idx[i]])
        ev_i = float(event_intensity[win_idx[i], area_idx[i]])
        st = bool(stormy[win_idx[i]])
        incident_types[i] = _incident_type_weighted(rng, d, h, wknd, ev_i, st)
        # Severity rises with event intensity, storm and hotspot.
        base = 2.0 + (d in {"security", "civil_defense"}) * 0.5
        sev = base + 0.25 * np.log1p(ev_i) + (1.0 if st and d in {"traffic", "civil_defense"} else 0.0)
        sev += float(np.clip(hotspots[win_idx[i], area_idx[i], idx[d]] - 1.0, -0.5, 1.5))
        severity[i] = int(np.clip(np.round(sev + rng.normal(0, 0.7)), 1, 5))

    # Response time: depends on severity.
    response_time = np.clip(rng.normal(18 - 2 * severity, 6, size=n_rows), 2, 90)
    responders = np.clip((severity + rng.integers(0, 3, size=n_rows)), 1, 12)

    channels = rng.choice(REPORTING_CHANNELS, size=n_rows)
    if channel_bias:
        channels = np.where(rng.random(size=n_rows) < 0.65, channel_bias, channels)

    # Sector assignment: multi-sector based on domain and severity.
    sectors = []
    for d, s in zip(domains, severity):
        base = {
            "traffic": ["Traffic Police", "Ambulance"],
            "drugs": ["Narcotics Control", "General Investigation"],
            "public_order": ["Public Security", "National Guard"],
            "health": ["Ambulance", "Ministry of Health"],
            "civil_defense": ["Civil Defense", "Municipality"],
            "security": ["Public Security", "General Investigation"],
        }[d]
        pick = list(dict.fromkeys(base))
        if s >= 4 and rng.random() < 0.45:
            pick.append("National Guard")
        if d == "traffic" and rng.random() < 0.35:
            pick.append("Civil Defense")
        sectors.append(json.dumps(sorted(set(pick)), ensure_ascii=False))

    status = rng.choice(RESOLUTION_STATUSES, size=n_rows, p=[0.70, 0.18, 0.06, 0.06])

    neighborhoods = rng.choice(NEIGHBORHOODS, size=n_rows)
    landmarks = rng.choice(LANDMARKS, size=n_rows)

    df = pd.DataFrame(
        {
            "incident_id": [f"INC-{seed}-{i:07d}" for i in range(n_rows)],
            "timestamp": ts,
            "incident_domain": domains,
            "incident_type": incident_types,
            "severity": severity,
            "description_text": [
                _incident_row_text(d, t, a) for d, t, a in zip(domains, incident_types, areas)
            ],
            "reporting_channel": channels,
            "responding_sectors": sectors,
            "responders_dispatched_count": responders,
            "resolution_status": status,
            "response_time_minutes": np.round(response_time, 2),
            "latitude": np.round(lats, 6),
            "longitude": np.round(lons, 6),
            "geo_area": areas,
            "neighborhood_name": neighborhoods,
            "nearest_landmark": landmarks,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def generate_social(
    out_path: Path,
    start: datetime,
    end: datetime,
    n_rows: int,
    seed: int,
    *,
    incidents_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    window_minutes: int = 60,
) -> None:
    rng = _rng(seed)
    # Build sampling weights from incident density + events.
    windows = pd.date_range(start=start, end=end, freq=f"{window_minutes}min", inclusive="left", tz="UTC").tz_convert(None)
    areas_list = GEO_AREAS
    n_steps = len(windows)
    n_areas = len(areas_list)

    base_w = np.ones((n_steps, n_areas), dtype=float) * 0.2
    if incidents_df is not None and not incidents_df.empty:
        inc = incidents_df.copy()
        inc["timestamp"] = pd.to_datetime(inc["timestamp"], utc=True).dt.tz_convert(None)
        inc["window_start"] = inc["timestamp"].dt.floor(f"{window_minutes}min")
        agg = inc.groupby(["window_start", "geo_area"], observed=True).size().reset_index(name="count")
        map_w = {w: i for i, w in enumerate(windows)}
        map_a = {a: i for i, a in enumerate(areas_list)}
        for r in agg.itertuples(index=False):
            wi = map_w.get(r.window_start)
            ai = map_a.get(r.geo_area)
            if wi is not None and ai is not None:
                base_w[wi, ai] += float(r.count)

    if events_df is not None and not events_df.empty:
        ev_idx = _build_event_window_index(events_df, start, end, window_minutes)
        if not ev_idx.empty:
            ev_idx["window_start"] = pd.to_datetime(ev_idx["window_start"])
            map_w = {w: i for i, w in enumerate(windows)}
            map_a = {a: i for i, a in enumerate(areas_list)}
            for r in ev_idx.itertuples(index=False):
                wi = map_w.get(r.window_start)
                ai = map_a.get(r.geo_area)
                if wi is not None and ai is not None:
                    base_w[wi, ai] += 0.6 * float(r.event_intensity)

    flat = base_w.reshape(-1)
    flat = flat / flat.sum()
    choices = rng.choice(np.arange(flat.size), size=n_rows, replace=True, p=flat)
    win_idx = choices // n_areas
    area_idx = choices % n_areas

    areas = np.array([areas_list[i] for i in area_idx], dtype=object)
    lats, lons = _coords_for_areas(rng, areas)
    base_ts = np.array([windows[i] for i in win_idx], dtype="datetime64[ns]")
    ts = pd.to_datetime(base_ts) + pd.to_timedelta(rng.integers(0, window_minutes * 60, size=n_rows), unit="s")

    platform = rng.choice(PLATFORMS, size=n_rows)

    # Theme confidences driven by incidents + events + weather.
    theme_cols: dict[str, np.ndarray] = {}
    # Default low baseline.
    for t in THEMES:
        theme_cols[f"theme_conf_{t}"] = rng.beta(1.1, 7.5, size=n_rows) * 0.6

    # Build incident domain proportions for each sampled row (best-effort).
    dom_counts = None
    if incidents_df is not None and not incidents_df.empty:
        inc = incidents_df.copy()
        inc["timestamp"] = pd.to_datetime(inc["timestamp"], utc=True).dt.tz_convert(None)
        inc["window_start"] = inc["timestamp"].dt.floor(f"{window_minutes}min")
        dom_counts = (
            inc.groupby(["window_start", "geo_area", "incident_domain"], observed=True)
            .size()
            .unstack(fill_value=0)
        )

    # Weather storm signal for storm theme.
    storm_sig = np.zeros(n_rows, dtype=float)
    if weather_df is not None and not weather_df.empty:
        w = weather_df.copy()
        w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True).dt.tz_convert(None)
        w = w.set_index(w["timestamp"].dt.floor(f"{window_minutes}min")).sort_index()
        w = w.reindex(windows, method="nearest")
        rain = pd.to_numeric(w["rain_mm"], errors="coerce").fillna(0.0).to_numpy()
        wind = pd.to_numeric(w["wind_kph"], errors="coerce").fillna(0.0).to_numpy()
        st = ((rain > 6.0) | (wind > 22.0)).astype(float)
        storm_sig = st[win_idx]

    for i in range(n_rows):
        ev_i = 0.0
        # Approximate event intensity per sampled row
        if events_df is not None and not events_df.empty:
            # reuse precomputed index quickly
            pass
        ws = pd.Timestamp(base_ts[i])
        ga = areas[i]
        counts = {d: 0.0 for d in DOMAINS}
        if dom_counts is not None and (ws, ga) in dom_counts.index:
            rowc = dom_counts.loc[(ws, ga)]
            for d in DOMAINS:
                if d in rowc.index:
                    counts[d] = float(rowc[d])
        tot = sum(counts.values()) + 1e-6
        traffic = counts["traffic"] / tot
        sec = counts["security"] / tot
        po = counts["public_order"] / tot
        cd = counts["civil_defense"] / tot
        drugs = counts["drugs"] / tot

        theme_cols["theme_conf_drifting"][i] = float(np.clip(0.15 + 0.9 * traffic + rng.normal(0, 0.08), 0, 1))
        theme_cols["theme_conf_road_block"][i] = float(np.clip(0.12 + 0.8 * traffic + rng.normal(0, 0.08), 0, 1))
        # Map drugs activity into threat/weapons chatter (raids/arrests) to make signals informative.
        theme_cols["theme_conf_threat"][i] = float(np.clip(0.10 + 0.85 * sec + 0.25 * drugs + rng.normal(0, 0.08), 0, 1))
        theme_cols["theme_conf_weapons"][i] = float(np.clip(0.10 + 0.75 * sec + 0.20 * po + 0.15 * drugs + rng.normal(0, 0.08), 0, 1))
        theme_cols["theme_conf_crowding"][i] = float(np.clip(0.12 + 0.9 * po + rng.normal(0, 0.08), 0, 1))
        theme_cols["theme_conf_harassment"][i] = float(np.clip(0.10 + 0.7 * po + rng.normal(0, 0.08), 0, 1))
        theme_cols["theme_conf_storm"][i] = float(np.clip(0.10 + 0.8 * cd + 0.3 * storm_sig[i] + rng.normal(0, 0.08), 0, 1))

    extracted = []
    texts = []
    for i in range(n_rows):
        active = [t for t in THEMES if float(theme_cols[f"theme_conf_{t}"][i]) > 0.55]
        if not active:
            # pick the top-1 theme
            best = sorted(THEMES, key=lambda t: float(theme_cols[f"theme_conf_{t}"][i]), reverse=True)[:1]
            active = best if best else [rng.choice(THEMES)]
        extracted.append(json.dumps(active, ensure_ascii=False))
        texts.append(
            f"Reports mention {', '.join(active)} around {areas[i]}. Stay alert." if rng.random() < 0.7 else f"Noise about {active[0]} near {areas[i]}"
        )

    has_coords = rng.random(size=n_rows) < 0.25
    lat = np.where(has_coords, lats, np.nan)
    lon = np.where(has_coords, lons, np.nan)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "platform": platform,
            "text": texts,
            "extracted_themes": extracted,
            **theme_cols,
            "geo_area": areas,
            "latitude": np.round(lat, 6),
            "longitude": np.round(lon, 6),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def generate_news(
    out_path: Path,
    start: datetime,
    end: datetime,
    n_rows: int,
    seed: int,
    *,
    incidents_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    window_minutes: int = 60,
) -> None:
    rng = _rng(seed)
    # News is more correlated with higher-severity / security / public-order signals.
    windows = pd.date_range(start=start, end=end, freq=f"{window_minutes}min", inclusive="left", tz="UTC").tz_convert(None)
    areas_list = GEO_AREAS
    n_steps = len(windows)
    n_areas = len(areas_list)

    base_w = np.ones((n_steps, n_areas), dtype=float) * 0.15
    if incidents_df is not None and not incidents_df.empty:
        inc = incidents_df.copy()
        inc["timestamp"] = pd.to_datetime(inc["timestamp"], utc=True).dt.tz_convert(None)
        inc["window_start"] = inc["timestamp"].dt.floor(f"{window_minutes}min")
        inc["severity"] = pd.to_numeric(inc["severity"], errors="coerce").fillna(2.0)
        inc["w"] = inc["severity"] * (inc["incident_domain"].isin(["security", "public_order"]).astype(float) + 1.0)
        agg = inc.groupby(["window_start", "geo_area"], observed=True)["w"].sum().reset_index(name="w")
        map_w = {w: i for i, w in enumerate(windows)}
        map_a = {a: i for i, a in enumerate(areas_list)}
        for r in agg.itertuples(index=False):
            wi = map_w.get(r.window_start)
            ai = map_a.get(r.geo_area)
            if wi is not None and ai is not None:
                base_w[wi, ai] += float(r.w)

    flat = base_w.reshape(-1)
    flat = flat / flat.sum()
    choices = rng.choice(np.arange(flat.size), size=n_rows, replace=True, p=flat)
    win_idx = choices // n_areas
    area_idx = choices % n_areas

    areas = np.array([areas_list[i] for i in area_idx], dtype=object)
    lats, lons = _coords_for_areas(rng, areas)
    base_ts = np.array([windows[i] for i in win_idx], dtype="datetime64[ns]")
    ts = pd.to_datetime(base_ts) + pd.to_timedelta(rng.integers(0, window_minutes * 60, size=n_rows), unit="s")

    source = rng.choice(NEWS_SOURCES, size=n_rows)
    # Use the same theme shaping as social, but slightly higher confidence.
    tmp_path = out_path.parent / (out_path.stem + "__tmp_social_like.csv")
    generate_social(
        tmp_path,
        start,
        end,
        n_rows=n_rows,
        seed=seed + 999,
        incidents_df=incidents_df,
        events_df=events_df,
        weather_df=weather_df,
        window_minutes=window_minutes,
    )
    tmp = pd.read_csv(tmp_path)
    tmp_path.unlink(missing_ok=True)
    theme_cols = {
        c: np.clip(tmp[c].to_numpy() + rng.normal(0, 0.04, size=n_rows), 0.0, 1.0)
        for c in tmp.columns
        if c.startswith("theme_conf_")
    }

    headlines = []
    bodies = []
    extracted = []
    for i in range(n_rows):
        active = [t for t in THEMES if float(theme_cols[f"theme_conf_{t}"][i]) > 0.60]
        if not active:
            active = [rng.choice(THEMES)]
        extracted.append(json.dumps(active, ensure_ascii=False))
        headlines.append(f"Update: {active[0].replace('_', ' ').title()} situation reported in {areas[i]}")
        bodies.append(
            f"Authorities reported signals related to {', '.join(active)} in {areas[i]}. Situation under monitoring."
        )

    has_coords = rng.random(size=n_rows) < 0.15
    lat = np.where(has_coords, lats, np.nan)
    lon = np.where(has_coords, lons, np.nan)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "source_name": source,
            "headline": headlines,
            "body_snippet": bodies,
            "extracted_themes": extracted,
            **theme_cols,
            "geo_area": areas,
            "latitude": np.round(lat, 6),
            "longitude": np.round(lon, 6),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def generate_events(out_path: Path, start: datetime, end: datetime, n_rows: int, seed: int) -> None:
    rng = _rng(seed)
    areas = _choose_geo_areas(rng, n_rows)
    lats, lons = _coords_for_areas(rng, areas)

    # Events have explicit start/end.
    starts = _random_timestamps(rng, start, end, n_rows).tz_convert(None)
    durations = rng.integers(60, 6 * 60 + 1, size=n_rows)
    ends = pd.to_datetime(starts) + pd.to_timedelta(durations, unit="m")

    event_type = rng.choice(
        ["football", "concert", "festival", "conference", "boxing", "esports"],
        size=n_rows,
        p=[0.30, 0.14, 0.18, 0.18, 0.10, 0.10],
    )

    venue = rng.choice(
        [
            "Mrsool Park",
            "King Fahd Stadium",
            "Riyadh Front Arena",
            "Boulevard City",
            "Diriyah Arena",
            "KAFD Conference Center",
        ],
        size=n_rows,
    )

    expected_size = np.clip(rng.lognormal(mean=8.2, sigma=0.55, size=n_rows).astype(int), 200, 120000)
    confidence = np.round(np.clip(rng.beta(5, 2, size=n_rows), 0.2, 1.0), 3)

    df = pd.DataFrame(
        {
            "timestamp_start": starts,
            "timestamp_end": ends,
            "event_type": event_type,
            "expected_size": expected_size,
            "venue_name": venue,
            "geo_area": areas,
            "latitude": np.round(lats, 6),
            "longitude": np.round(lons, 6),
            "event_present": True,
            "confidence": confidence,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic datasets for model-city-risk")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, default="2025-01-01")
    p.add_argument("--end", type=str, default="2025-12-01")
    p.add_argument("--rows-incidents", type=int, default=50000)
    p.add_argument("--rows-signals", type=int, default=50000)
    p.add_argument("--rows-events", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    input_dir = repo_root / "data" / "input_datasets"
    input_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    generate_weather(input_dir / "weather_conditions.csv", start, end, seed=args.seed)
    generate_events(
        input_dir / "events_sports.csv",
        start,
        end,
        n_rows=max(50000, args.rows_events),
        seed=args.seed + 12,
    )

    weather_df = pd.read_csv(input_dir / "weather_conditions.csv")
    events_df = pd.read_csv(input_dir / "events_sports.csv")

    # Incidents (>= 50k each) — now coupled to weather/events/hotspots
    generate_incidents(
        input_dir / "incidents_all_911.csv",
        start,
        end,
        n_rows=max(50000, args.rows_incidents),
        seed=args.seed + 1,
        channel_bias="911",
        weather_df=weather_df,
        events_df=events_df,
    )
    generate_incidents(
        input_dir / "incidents_national_guard.csv",
        start,
        end,
        n_rows=max(50000, args.rows_incidents),
        seed=args.seed + 2,
        channel_bias="internal",
        weather_df=weather_df,
        events_df=events_df,
    )
    generate_incidents(
        input_dir / "incidents_narcotics_control.csv",
        start,
        end,
        n_rows=max(50000, args.rows_incidents),
        seed=args.seed + 3,
        channel_bias="ministry",
        weather_df=weather_df,
        events_df=events_df,
    )
    generate_incidents(
        input_dir / "incidents_general_investigation.csv",
        start,
        end,
        n_rows=max(50000, args.rows_incidents),
        seed=args.seed + 4,
        channel_bias="internal",
        weather_df=weather_df,
        events_df=events_df,
    )
    generate_incidents(
        input_dir / "incidents_ambulance.csv",
        start,
        end,
        n_rows=max(50000, args.rows_incidents),
        seed=args.seed + 5,
        channel_bias="911",
        weather_df=weather_df,
        events_df=events_df,
    )

    # Build signal sources correlated with incidents (>= 50k)
    inc_all = pd.concat(
        [
            pd.read_csv(input_dir / "incidents_all_911.csv"),
            pd.read_csv(input_dir / "incidents_national_guard.csv"),
            pd.read_csv(input_dir / "incidents_narcotics_control.csv"),
            pd.read_csv(input_dir / "incidents_general_investigation.csv"),
            pd.read_csv(input_dir / "incidents_ambulance.csv"),
        ],
        ignore_index=True,
    )
    generate_social(
        input_dir / "social_media_signals.csv",
        start,
        end,
        n_rows=max(50000, args.rows_signals),
        seed=args.seed + 10,
        incidents_df=inc_all,
        events_df=events_df,
        weather_df=weather_df,
    )
    generate_news(
        input_dir / "news_signals.csv",
        start,
        end,
        n_rows=max(50000, args.rows_signals),
        seed=args.seed + 11,
        incidents_df=inc_all,
        events_df=events_df,
        weather_df=weather_df,
    )

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "start": args.start,
        "end": args.end,
        "rows_incidents": max(50000, args.rows_incidents),
        "rows_signals": max(50000, args.rows_signals),
        "rows_events": max(50000, args.rows_events),
        "files": [
            "weather_conditions.csv",
            "incidents_all_911.csv",
            "incidents_national_guard.csv",
            "incidents_narcotics_control.csv",
            "incidents_general_investigation.csv",
            "incidents_ambulance.csv",
            "social_media_signals.csv",
            "news_signals.csv",
            "events_sports.csv",
        ],
    }
    (input_dir / "mock_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote datasets to: {input_dir}")


if __name__ == "__main__":
    main()
