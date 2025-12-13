from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.model.features import FeatureBuildConfig, DOMAINS, build_training_frame
from src.model.predict import predict_records

from test_features import _write_minimal_inputs


def test_predict_records_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "input_datasets"
    _write_minimal_inputs(input_dir)

    fcfg = FeatureBuildConfig(window_minutes=60, lags=[1, 2], rolling_windows=[8], themes=["crowding"])
    X, y = build_training_frame(input_dir, fcfg)

    # Tiny model for speed
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, random_state=0))
    # Drop non-numeric columns except geo_area/window_start which model can ignore because it's not preprocessed here.
    # For this unit test we use only numeric features.
    X_num = X.drop(columns=["geo_area", "window_start"], errors="ignore")
    clf.fit(X_num, y)

    # Predict on same frame (smoke)
    records = predict_records(
        model=type("Wrap", (), {"predict_proba": lambda self, X: clf.predict_proba(X.drop(columns=["geo_area", "window_start"], errors="ignore"))})(),
        X_win=X.head(3),
        cfg={"prediction": {"radius_meters": 900, "min_prob_for_sector": 0.35, "top_k_areas": 200}, "sectors_by_domain": {}},
        metrics={"per_domain": {d: {"avg_precision": 0.5} for d in DOMAINS}},
    )

    assert len(records) == 3
    for r in records:
        assert r["radius_meters"] == 900
        assert set(r["risk_by_domain"].keys()) == set(DOMAINS)
        assert 0.0 <= r["risk_score"] <= 1.0
