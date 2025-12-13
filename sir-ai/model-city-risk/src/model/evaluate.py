from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_thresholds(
    model,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    domains: list[str],
    grid: list[float] | None = None,
) -> dict[str, float]:
    """Tune per-domain thresholds on validation by maximizing F1."""
    if grid is None:
        grid = [i / 100 for i in range(5, 96, 5)]

    prob_list = model.predict_proba(X_val)
    thresholds: dict[str, float] = {}

    for i, d in enumerate(domains):
        p = prob_list[i][:, 1]
        y = y_val[d].to_numpy()
        best_thr = float(grid[0])
        best_f1 = -1.0
        for thr in grid:
            y_hat = (p >= thr).astype(int)
            f1 = float(f1_score(y, y_hat, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        thresholds[d] = best_thr

    return thresholds


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    domains: list[str],
    threshold: float | dict[str, float],
) -> dict[str, Any]:
    # MultiOutputClassifier returns a list of prob arrays, one per label.
    prob_list = model.predict_proba(X_test)

    metrics: dict[str, Any] = {"per_domain": {}, "overall": {}}
    aucs = []
    aps = []
    precs = []
    recs = []
    f1s = []
    accs = []

    for i, d in enumerate(domains):
        p = prob_list[i][:, 1]
        y = y_test[d].to_numpy()
        thr = float(threshold[d]) if isinstance(threshold, dict) else float(threshold)
        y_hat = (p >= thr).astype(int)

        # Guard against all-zeros in synthetic slices.
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

        prec = float(precision_score(y, y_hat, zero_division=0))
        rec = float(recall_score(y, y_hat, zero_division=0))
        f1 = float(f1_score(y, y_hat, zero_division=0))
        acc = float(accuracy_score(y, y_hat))

        metrics["per_domain"][d] = {
            "roc_auc": auc,
            "avg_precision": ap,
            "threshold": float(thr),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "positive_rate": float(np.mean(y)),
        }
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(ap):
            aps.append(ap)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        accs.append(acc)

    metrics["overall"] = {
        "roc_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "avg_precision_mean": float(np.mean(aps)) if aps else float("nan"),
        "threshold": None if isinstance(threshold, dict) else float(threshold),
        "precision_mean": float(np.mean(precs)) if precs else float("nan"),
        "recall_mean": float(np.mean(recs)) if recs else float("nan"),
        "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
        "n_test_rows": int(len(X_test)),
    }
    return metrics


def write_model_card(path: Path, cfg: dict, report: dict[str, Any]) -> None:
    domains = ", ".join(cfg["model"]["domains"])
    window_minutes = cfg["features"]["window_minutes"]
    thr = report.get("overall", {}).get("threshold")
    thr_mode = report.get("overall", {}).get("thresholding")

    content = f"""# Model Card — City Risk Prediction Model (Baseline)\n\n## Use case\nPredict next-interval (T+1) risk by geo area and time window for multiple incident domains: **{domains}**.\n\n## Input data sources (synthetic)\n- Incidents: 911 + multiple responding departments\n- Weather conditions\n- Social media signals\n- Trusted news signals\n- Sports/major events signals\n\n## Features\n- Lag counts and rolling counts per domain (lags: {cfg['features']['lags']}, rolling: {cfg['features']['rolling_windows']})\n- Calendar features (hour/day/month/weekend + synthetic holiday/Ramadan flags)\n- Weather aggregates (temp/rain/wind)\n- Signal aggregates (social/news/events/weather confidence sums)\n- One-hot geo area\n\n## Model type\nMulti-label classifier using **RandomForest** with a **MultiOutput** wrapper (one classifier per domain).\n\n## Evaluation\nTime-based split; test window = last {cfg['model']['train_test_split']['test_days']} days.\nThresholding mode: {thr_mode}\n\nMetrics (per domain):\n"""

    for d, m in report.get("per_domain", {}).items():
        content += (
            f"- {d}: ROC-AUC={m['roc_auc']:.3f} | AP={m['avg_precision']:.3f} | "
            f"P/R/F1/Acc@{m.get('threshold')}: {m.get('precision'):.3f}/{m.get('recall'):.3f}/{m.get('f1'):.3f}/{m.get('accuracy'):.3f} | "
            f"positive_rate={m['positive_rate']:.3f}\n"
        )

    overall = report.get("overall", {})
    content += f"\nOverall mean ROC-AUC={overall.get('roc_auc_mean')} | mean AP={overall.get('avg_precision_mean')}\n"
    content += f"Overall mean P/R/F1/Acc@{thr}: {overall.get('precision_mean')}/{overall.get('recall_mean')}/{overall.get('f1_mean')}/{overall.get('accuracy_mean')}\n"

    content += """\n\n## Limitations\n- This model is trained on **synthetic data** and is for engineering integration only.\n- Probability calibration and real-world validation are not performed.\n- Spatial effects are simplified (geo areas are coarse).\n\n## Privacy\nNo personal data is used. All text, locations, and IDs are synthetic.\n"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
