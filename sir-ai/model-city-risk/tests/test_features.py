from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model.features import FeatureBuildConfig, DOMAINS, build_training_frame


def _write_minimal_inputs(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    # Weather hourly
    weather = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=48, freq="1H"),
            "temp_c": [25.0] * 48,
            "rain_mm": [0.0] * 48,
            "wind_kph": [10.0] * 48,
            "condition": ["clear"] * 48,
        }
    )
    weather.to_csv(input_dir / "weather_conditions.csv", index=False)

    # Incidents: small but with required columns.
    base_inc = pd.DataFrame(
        {
            "incident_id": [f"INC-{i}" for i in range(200)],
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="15min"),
            "incident_domain": [DOMAINS[i % len(DOMAINS)] for i in range(200)],
            "incident_type": ["collision_minor"] * 200,
            "severity": [3] * 200,
            "description_text": ["synthetic"] * 200,
            "reporting_channel": ["911"] * 200,
            "responding_sectors": ["[]"] * 200,
            "responders_dispatched_count": [2] * 200,
            "resolution_status": ["resolved"] * 200,
            "response_time_minutes": [10.0] * 200,
            "latitude": [24.71] * 200,
            "longitude": [46.67] * 200,
            "geo_area": ["Central"] * 200,
            "neighborhood_name": ["Al Olaya"] * 200,
            "nearest_landmark": ["Kingdom Centre"] * 200,
        }
    )

    for name in [
        "incidents_all_911.csv",
        "incidents_national_guard.csv",
        "incidents_narcotics_control.csv",
        "incidents_general_investigation.csv",
        "incidents_ambulance.csv",
    ]:
        base_inc.to_csv(input_dir / name, index=False)

    # Social/news with theme_conf_*
    signal = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="20min"),
            "platform": ["x"] * 200,
            "text": ["synthetic"] * 200,
            "extracted_themes": ["[]"] * 200,
            "theme_conf_crowding": [0.1] * 200,
            "theme_conf_threat": [0.0] * 200,
            "theme_conf_weapons": [0.0] * 200,
            "theme_conf_drifting": [0.1] * 200,
            "theme_conf_storm": [0.0] * 200,
            "theme_conf_road_block": [0.0] * 200,
            "theme_conf_harassment": [0.0] * 200,
            "geo_area": ["Central"] * 200,
            "latitude": [float("nan")] * 200,
            "longitude": [float("nan")] * 200,
        }
    )
    signal.to_csv(input_dir / "social_media_signals.csv", index=False)

    news = signal.drop(columns=["platform"]).copy()
    news.insert(1, "source_name", "SPA")
    news.insert(2, "headline", "synthetic")
    news.insert(3, "body_snippet", "synthetic")
    news.to_csv(input_dir / "news_signals.csv", index=False)

    # Events
    events = pd.DataFrame(
        {
            "timestamp_start": ["2025-01-01T10:00:00"] * 5,
            "timestamp_end": ["2025-01-01T12:00:00"] * 5,
            "event_type": ["football"] * 5,
            "expected_size": [10000] * 5,
            "venue_name": ["Mrsool Park"] * 5,
            "geo_area": ["Central"] * 5,
            "latitude": [24.71] * 5,
            "longitude": [46.67] * 5,
            "event_present": [True] * 5,
            "confidence": [0.9] * 5,
        }
    )
    events.to_csv(input_dir / "events_sports.csv", index=False)


def test_build_training_frame_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "input_datasets"
    _write_minimal_inputs(input_dir)

    cfg = FeatureBuildConfig(window_minutes=60, lags=[1, 2], rolling_windows=[8], themes=["crowding"])  # themes list is accepted
    X, y = build_training_frame(input_dir, cfg)

    assert not X.empty
    assert not y.empty
    assert set(y.columns) == set(DOMAINS)
    assert "hour" in X.columns
    assert "inc_count_traffic_lag1" in X.columns
    assert "conf_sum_social_lag1" in X.columns
