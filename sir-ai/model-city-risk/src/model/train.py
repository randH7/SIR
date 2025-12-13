from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FeatureBuildConfig, DOMAINS, build_training_frame
from .check_layer import write_check_artifacts
from .utils import dump_json, ensure_dirs, get_paths, load_config, save_joblib, setup_logging


LOGGER = logging.getLogger(__name__)


def _time_based_split(X: pd.DataFrame, y: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = X.copy()
    X["window_start"] = pd.to_datetime(X["window_start"])
    max_ts = X["window_start"].max()
    cutoff = max_ts - pd.Timedelta(days=test_days)

    train_idx = X["window_start"] < cutoff
    test_idx = ~train_idx

    X_train = X.loc[train_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def _time_based_split_train_val_test(
    X: pd.DataFrame, y: pd.DataFrame, test_days: int, val_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = X.copy()
    X["window_start"] = pd.to_datetime(X["window_start"])
    max_ts = X["window_start"].max()
    cutoff_test = max_ts - pd.Timedelta(days=test_days)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_days)

    train_idx = X["window_start"] < cutoff_val
    val_idx = (X["window_start"] >= cutoff_val) & (X["window_start"] < cutoff_test)
    test_idx = X["window_start"] >= cutoff_test

    return (
        X.loc[train_idx].reset_index(drop=True),
        y.loc[train_idx].reset_index(drop=True),
        X.loc[val_idx].reset_index(drop=True),
        y.loc[val_idx].reset_index(drop=True),
        X.loc[test_idx].reset_index(drop=True),
        y.loc[test_idx].reset_index(drop=True),
    )


def build_pipeline(cfg: dict) -> Pipeline:
    cat_features = ["geo_area"]

    pre = ColumnTransformer(
        transformers=[
            # datetime columns (e.g., window_start) are not numeric and will be excluded.
            ("num", "passthrough", make_column_selector(dtype_include="number")),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf_params = cfg["model"]["random_forest"]
    base = RandomForestClassifier(**rf_params)
    clf = MultiOutputClassifier(base)

    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    setup_logging(repo_root)
    cfg = load_config(repo_root)
    paths = get_paths(repo_root, cfg)
    ensure_dirs(paths.models_dir, paths.metrics_dir, paths.output_dir)

    fcfg = FeatureBuildConfig(
        window_minutes=int(cfg["features"]["window_minutes"]),
        lags=list(cfg["features"]["lags"]),
        rolling_windows=list(cfg["features"]["rolling_windows"]),
        themes=list(cfg["features"]["themes"]),
    )

    LOGGER.info("Building training frame from input datasets...")
    X, y = build_training_frame(paths.input_dir, fcfg)

    test_days = int(cfg["model"]["train_test_split"]["test_days"])
    thr_cfg = cfg.get("model", {}).get("evaluation", {}).get("threshold_tuning", {})
    tune_enabled = bool(thr_cfg.get("enabled", False))
    val_days = int(thr_cfg.get("val_days", 0) or 0)

    if tune_enabled and val_days > 0:
        X_train, y_train, X_val, y_val, X_test, y_test = _time_based_split_train_val_test(
            X, y, test_days=test_days, val_days=val_days
        )
        LOGGER.info(
            "Train rows=%s | Val rows=%s | Test rows=%s",
            len(X_train),
            len(X_val),
            len(X_test),
        )
    else:
        X_train, y_train, X_test, y_test = _time_based_split(X, y, test_days)
        X_val, y_val = X_test.head(0), y_test.head(0)
        LOGGER.info("Train rows=%s | Test rows=%s", len(X_train), len(X_test))

    pipe = build_pipeline(cfg)
    LOGGER.info("Training baseline multi-label model...")
    pipe.fit(X_train, y_train)

    model_path = paths.models_dir / "city_risk_model.pkl"
    enc_path = paths.models_dir / "encoders.pkl"
    # Refit on train+val (if enabled) after tuning to ship a stronger final model.
    if tune_enabled and not X_val.empty:
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = pd.concat([y_train, y_val], ignore_index=True)
        pipe.fit(X_full, y_full)

    save_joblib(model_path, pipe)
    # For compatibility with contract requirement: store encoder separately too.
    save_joblib(enc_path, pipe.named_steps["preprocess"])

    # Evaluate and write report.
    from .evaluate import evaluate_model, tune_thresholds

    threshold = float(
        cfg.get("model", {}).get("evaluation", {}).get("threshold", cfg["prediction"]["min_prob_for_sector"])
    )
    per_domain_thresholds = None
    grid = thr_cfg.get("grid") if tune_enabled else None
    if tune_enabled and not X_val.empty:
        per_domain_thresholds = tune_thresholds(pipe, X_val, y_val, domains=DOMAINS, grid=grid)
        report = evaluate_model(pipe, X_test, y_test, domains=DOMAINS, threshold=per_domain_thresholds)
        report["overall"]["thresholding"] = "per_domain_tuned_on_validation"
        report["overall"]["tuning_val_days"] = val_days
    else:
        report = evaluate_model(pipe, X_test, y_test, domains=DOMAINS, threshold=threshold)
        report["overall"]["thresholding"] = "fixed"
    dump_json(paths.metrics_dir / "evaluation_report.json", report)

    # Model card.
    from .evaluate import write_model_card

    write_model_card(paths.metrics_dir / "model_card.md", cfg, report)

    # Dedicated quality/safety layer artifacts (offline + scenario + capacity + gate + explanations).
    # Use a small sample window slice for scenario/explanations to keep runtime reasonable.
    X_sample = X_test.head(300).copy()
    # Create a set of latest-like predictions from the same sample.
    from .predict import predict_records

    sample_preds = predict_records(pipe, X_sample, cfg, metrics=report)
    write_check_artifacts(
        repo_root=repo_root,
        model=pipe,
        X_test=X_test,
        y_test=y_test,
        X_sample=X_sample,
        latest_predictions=sample_preds,
        window_minutes=int(cfg["features"]["window_minutes"]),
    )

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Saved metrics to %s", paths.metrics_dir / "evaluation_report.json")


if __name__ == "__main__":
    main()
