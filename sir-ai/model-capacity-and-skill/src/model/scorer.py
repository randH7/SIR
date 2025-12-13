from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class ScoringArtifacts:
    vectorizer: DictVectorizer
    model: LogisticRegression


class ScoringModel:
    """ML-based unit scoring model.

    - Trains a simple logistic regression on incident-unit feature dicts.
    - Falls back to a heuristic probability if artifacts are missing.
    """

    def __init__(self, artifacts: Optional[ScoringArtifacts] = None):
        self._artifacts = artifacts

    @property
    def loaded(self) -> bool:
        return self._artifacts is not None

    @staticmethod
    def _heuristic_prob(features: Dict[str, Any]) -> float:
        # Distance lower is better; skill and availability higher are better; sector load and fatigue lower.
        dist = float(features.get("distance_km", 50.0))
        eta = float(features.get("eta_minutes", 60.0))
        skill = float(features.get("skill_match_score", 0.0))
        avail = float(features.get("availability_score", 0.0))
        fatigue = float(features.get("fatigue_score", 1.0))
        load = float(features.get("sector_load_ratio", 1.0))
        eligible = int(features.get("policy_eligible", 0))

        if eligible <= 0:
            return 0.0

        # Squash into [0,1]
        x = (
            1.2 * skill
            + 0.9 * avail
            + 0.3 * (1.0 - min(1.0, fatigue))
            + 0.3 * (1.0 - min(1.0, load))
            - 0.015 * min(200.0, eta)
            - 0.01 * min(200.0, dist)
        )
        return float(1.0 / (1.0 + np.exp(-x)))

    def predict_proba(self, feature_dicts: List[Dict[str, Any]]) -> np.ndarray:
        if not feature_dicts:
            return np.array([], dtype=float)

        if self._artifacts is None:
            return np.array([self._heuristic_prob(f) for f in feature_dicts], dtype=float)

        X = self._artifacts.vectorizer.transform(feature_dicts)
        return self._artifacts.model.predict_proba(X)[:, 1]

    @classmethod
    def load(cls, models_dir: str | Path) -> "ScoringModel":
        models_dir = Path(models_dir)
        model_path = models_dir / "resource_match_model.pkl"
        enc_path = models_dir / "encoders.pkl"
        if not model_path.exists() or not enc_path.exists():
            return cls(None)
        model = joblib.load(model_path)
        vectorizer = joblib.load(enc_path)
        return cls(ScoringArtifacts(vectorizer=vectorizer, model=model))

    @classmethod
    def train(
        cls,
        X_dicts: List[Dict[str, Any]],
        y: np.ndarray,
        seed: int = 42,
    ) -> Tuple["ScoringModel", Dict[str, Any]]:
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dicts)

        model = LogisticRegression(
            max_iter=600,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
        )
        model.fit(X, y)

        proba = model.predict_proba(X)[:, 1]
        metrics: Dict[str, Any] = {}
        try:
            metrics["train_roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            metrics["train_roc_auc"] = None

        return cls(ScoringArtifacts(vectorizer=vec, model=model)), metrics

    def save(self, models_dir: str | Path) -> None:
        if self._artifacts is None:
            return
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._artifacts.model, models_dir / "resource_match_model.pkl")
        joblib.dump(self._artifacts.vectorizer, models_dir / "encoders.pkl")
