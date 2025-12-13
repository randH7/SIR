from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ["traffic", "drugs", "public_order", "health", "civil_defense", "security"]
THEME_PREFIX = "theme_conf_"


@dataclass(frozen=True)
class FeatureBuildConfig:
    window_minutes: int
    lags: list[int]
    rolling_windows: list[int]
    themes: list[str]


def _floor_to_window(ts: pd.Series, window_minutes: int) -> pd.Series:
    # Normalize timestamps to UTC then drop timezone to keep merge keys consistent
    # across datasets (some are stored with explicit +00:00 offsets).
    dt = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)
    return dt.dt.floor(f"{window_minutes}min")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def _aggregate_incidents(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    df = df.copy()
    df["window_start"] = _floor_to_window(df["timestamp"], window_minutes)

    # Basic per-domain counts.
    g = (
        df.groupby(["window_start", "geo_area", "incident_domain"], observed=True)
        .agg(
            count=("incident_id", "count"),
            severity_mean=("severity", "mean"),
            resp_time_mean=("response_time_minutes", "mean"),
        )
        .reset_index()
    )

    wide_count = g.pivot_table(
        index=["window_start", "geo_area"],
        columns="incident_domain",
        values="count",
        fill_value=0,
        aggfunc="sum",
        observed=True,
    )
    wide_sev = g.pivot_table(
        index=["window_start", "geo_area"],
        columns="incident_domain",
        values="severity_mean",
        fill_value=0,
        aggfunc="mean",
        observed=True,
    )
    wide_resp = g.pivot_table(
        index=["window_start", "geo_area"],
        columns="incident_domain",
        values="resp_time_mean",
        fill_value=0,
        aggfunc="mean",
        observed=True,
    )

    wide_count.columns = [f"inc_count_{c}" for c in wide_count.columns]
    wide_sev.columns = [f"inc_severity_mean_{c}" for c in wide_sev.columns]
    wide_resp.columns = [f"inc_resp_time_mean_{c}" for c in wide_resp.columns]

    out = pd.concat([wide_count, wide_sev, wide_resp], axis=1).reset_index()

    # Total counts.
    out["inc_count_total"] = out[[c for c in out.columns if c.startswith("inc_count_") and c != "inc_count_total"]].sum(axis=1)
    return out


def _aggregate_signals(df: pd.DataFrame, window_minutes: int, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["window_start"] = _floor_to_window(df["timestamp"], window_minutes)

    theme_cols = [c for c in df.columns if c.startswith(THEME_PREFIX)]

    agg = (
        df.groupby(["window_start", "geo_area"], observed=True)
        .agg(
            conf_sum=(theme_cols[0], "size"),
            **{c: (c, "sum") for c in theme_cols},
        )
        .reset_index()
    )

    # conf_sum here is count; replace with sum of confidences.
    agg["conf_sum"] = agg[theme_cols].sum(axis=1)

    rename = {"conf_sum": f"conf_sum_{prefix}"}
    rename.update({c: f"{prefix}_{c}" for c in theme_cols})
    return agg.rename(columns=rename)


def _expand_events_to_windows(df_events: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    df = df_events.copy()
    df["timestamp_start"] = pd.to_datetime(df["timestamp_start"])
    df["timestamp_end"] = pd.to_datetime(df["timestamp_end"])

    # Expand each event to the set of windows it overlaps.
    rows = []
    for r in df.itertuples(index=False):
        start = pd.Timestamp(r.timestamp_start).floor(f"{window_minutes}min")
        end = pd.Timestamp(r.timestamp_end).ceil(f"{window_minutes}min")
        windows = pd.date_range(start, end, freq=f"{window_minutes}min", inclusive="left")
        for w in windows:
            rows.append(
                {
                    "window_start": w,
                    "geo_area": r.geo_area,
                    "event_type": r.event_type,
                    "expected_size": float(r.expected_size),
                    "confidence": float(r.confidence),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["window_start", "geo_area", "event_type", "expected_size", "confidence"])  # pragma: no cover

    ex = pd.DataFrame(rows)
    agg = (
        ex.groupby(["window_start", "geo_area", "event_type"], observed=True)
        .agg(
            events_count=("event_type", "count"),
            expected_size_sum=("expected_size", "sum"),
            confidence_mean=("confidence", "mean"),
        )
        .reset_index()
    )

    # Wide event types for one-hot-like numeric signals.
    wide_count = agg.pivot_table(
        index=["window_start", "geo_area"],
        columns="event_type",
        values="events_count",
        fill_value=0,
        aggfunc="sum",
        observed=True,
    )
    wide_size = agg.pivot_table(
        index=["window_start", "geo_area"],
        columns="event_type",
        values="expected_size_sum",
        fill_value=0,
        aggfunc="sum",
        observed=True,
    )

    wide_count.columns = [f"events_count_{c}" for c in wide_count.columns]
    wide_size.columns = [f"events_expected_size_sum_{c}" for c in wide_size.columns]

    out = pd.concat([wide_count, wide_size], axis=1).reset_index()
    out["conf_sum_events"] = out[[c for c in out.columns if c.startswith("events_expected_size_sum_")]].sum(axis=1) / 10000.0
    return out


def _aggregate_weather(df_weather: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    df = df_weather.copy()
    df["window_start"] = _floor_to_window(df["timestamp"], window_minutes)
    agg = (
        df.groupby(["window_start"], observed=True)
        .agg(
            temp_c_mean=("temp_c", "mean"),
            rain_mm_sum=("rain_mm", "sum"),
            wind_kph_mean=("wind_kph", "mean"),
        )
        .reset_index()
    )
    # Weather is city-wide; later joined on window_start.
    agg["conf_sum_weather"] = (agg["rain_mm_sum"] > 0).astype(float) + (agg["wind_kph_mean"] > 18).astype(float)
    return agg


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["window_start"])
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 4).astype(int)  # Fri/Sat weekend in KSA

    # Simple Ramadan proxy: approximate month-based flag (synthetic use; no external services).
    out["is_ramadan"] = out["month"].isin([3]).astype(int)

    # Holiday proxy: Saudi National Day (Sep 23) and a few synthetic holiday days.
    out["is_holiday"] = ((dt.dt.month == 9) & (dt.dt.day == 23)).astype(int)
    return out


def _add_lags_and_rollings(df: pd.DataFrame, group_cols: list[str], value_cols: list[str], lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    out = df.sort_values(group_cols + ["window_start"]).copy()
    g = out.groupby(group_cols, observed=True)

    for col in value_cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = g[col].shift(lag).fillna(0)
        for w in rolling_windows:
            out[f"{col}_roll{w}"] = (
                g[col].rolling(window=w, min_periods=1).sum().reset_index(level=group_cols, drop=True)
            )

    return out


def build_training_frame(input_dir: Path, cfg: FeatureBuildConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return X, y where y is multi-label (columns per domain) for next window incident occurrence."""
    # Load and concatenate incident datasets.
    inc_files = [
        "incidents_all_911.csv",
        "incidents_national_guard.csv",
        "incidents_narcotics_control.csv",
        "incidents_general_investigation.csv",
        "incidents_ambulance.csv",
    ]
    inc = pd.concat([_safe_read_csv(input_dir / f) for f in inc_files], ignore_index=True)

    weather = _safe_read_csv(input_dir / "weather_conditions.csv")
    social = _safe_read_csv(input_dir / "social_media_signals.csv")
    news = _safe_read_csv(input_dir / "news_signals.csv")
    events = _safe_read_csv(input_dir / "events_sports.csv")

    df_inc = _aggregate_incidents(inc, cfg.window_minutes)
    df_weather = _aggregate_weather(weather, cfg.window_minutes)
    df_social = _aggregate_signals(social, cfg.window_minutes, prefix="social")
    df_news = _aggregate_signals(news, cfg.window_minutes, prefix="news")
    df_events = _expand_events_to_windows(events, cfg.window_minutes)

    # Build base grid (all windows x geo areas) to avoid missing rows.
    all_windows = pd.date_range(
        start=df_inc["window_start"].min(),
        end=df_inc["window_start"].max() + pd.Timedelta(minutes=cfg.window_minutes),
        freq=f"{cfg.window_minutes}min",
        inclusive="left",
    )
    all_areas = sorted(inc["geo_area"].dropna().unique().tolist())
    grid = pd.MultiIndex.from_product([all_windows, all_areas], names=["window_start", "geo_area"]).to_frame(index=False)

    df = grid.merge(df_inc, on=["window_start", "geo_area"], how="left")
    df = df.merge(df_social, on=["window_start", "geo_area"], how="left")
    df = df.merge(df_news, on=["window_start", "geo_area"], how="left")
    df = df.merge(df_events, on=["window_start", "geo_area"], how="left")
    df = df.merge(df_weather, on=["window_start"], how="left")

    df = df.fillna(0)
    df = _add_calendar_features(df)

    # Domain-specific incident count columns; ensure all exist.
    for d in DOMAINS:
        c = f"inc_count_{d}"
        if c not in df.columns:
            df[c] = 0

    # Create labels: whether next window has any incidents for that domain.
    df = df.sort_values(["geo_area", "window_start"]).reset_index(drop=True)
    y = pd.DataFrame({d: df.groupby("geo_area", observed=True)[f"inc_count_{d}"].shift(-1).fillna(0).gt(0).astype(int) for d in DOMAINS})

    # Lags/rollings on incident counts and signal sums.
    value_cols = ["inc_count_total"] + [f"inc_count_{d}" for d in DOMAINS] + [
        "conf_sum_social",
        "conf_sum_news",
        "conf_sum_events",
        "conf_sum_weather",
    ]
    for c in value_cols:
        if c not in df.columns:
            df[c] = 0

    df = _add_lags_and_rollings(
        df,
        group_cols=["geo_area"],
        value_cols=value_cols,
        lags=cfg.lags,
        rolling_windows=cfg.rolling_windows,
    )

    # Feature set: drop raw per-window counts to reduce leakage (keep lags/rollings instead).
    drop_cols = ["inc_count_total"] + [f"inc_count_{d}" for d in DOMAINS]
    X = df.drop(columns=drop_cols)

    return X, y


def build_prediction_frame(
    input_dir: Path,
    cfg: FeatureBuildConfig,
    window_start: datetime,
    window_minutes: int,
) -> pd.DataFrame:
    X, _y = build_training_frame(input_dir, cfg)
    # Use the row(s) matching the requested window.
    X["window_start"] = pd.to_datetime(X["window_start"], utc=True, errors="coerce").dt.tz_convert(None)
    target = pd.to_datetime(window_start, utc=True).tz_convert(None).floor(f"{window_minutes}min")
    return X[X["window_start"] == target].copy()
