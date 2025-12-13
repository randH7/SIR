from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import CheckLayerResult, UnitAssignment
from .utils import ensure_dir, now_utc_iso


@dataclass
class CheckDecision:
    status: str  # approved | warning | blocked
    reasons: List[str]
    confidence: float
    warnings: List[str]
    fallback_used: bool = False


def evaluate_and_gate(
    *,
    selected_units: List[UnitAssignment],
    optimizer_diagnostics: Dict[str, Any],
    config: Dict[str, Any],
) -> CheckDecision:
    """Hard validation gate deciding whether to approve, warn, or block output."""

    thresholds = config.get("thresholds", {})
    check_cfg = config.get("check_layer", {})

    max_util_allowed = float(thresholds.get("max_sector_utilization_allowed", 0.85))
    min_conf_auto = float(thresholds.get("min_confidence_score_for_auto_dispatch", 0.60))

    warn_band = check_cfg.get("warning_band", {})
    warn_min_conf = float(warn_band.get("min_confidence", 0.45))
    warn_max_util = float(warn_band.get("max_sector_utilization", 0.90))

    reasons: List[str] = []
    warnings: List[str] = []

    if not selected_units:
        return CheckDecision(
            status="blocked",
            reasons=["no_units_selected"],
            confidence=0.0,
            warnings=[],
            fallback_used=True,
        )

    confidences = [float(u.confidence) for u in selected_units]
    conf = float(sum(confidences) / max(1, len(confidences)))

    # Policy violation rate (should be near-zero). Here: optimizer should have filtered.
    policy_violation_rate = float(optimizer_diagnostics.get("policy_violation_rate", 0.0))
    if policy_violation_rate > float(thresholds.get("max_policy_violation_rate", 0.0)):
        reasons.append("policy_violation")

    # Sector overload risk (approx) based on diagnostics load balance and selected unit tags
    max_sector_util_seen = float(optimizer_diagnostics.get("max_sector_utilization", optimizer_diagnostics.get("sector_utilization", 0.0) or 0.0))
    # Some pipelines won't populate this; treat missing as safe.

    if max_sector_util_seen > max_util_allowed:
        reasons.append("sector_overload_risk")

    if conf < min_conf_auto:
        reasons.append("low_confidence")

    # Decision logic
    if reasons:
        # Warning band: allow output with warning if not egregious
        if conf >= warn_min_conf and max_sector_util_seen <= warn_max_util and "policy_violation" not in reasons:
            warnings.append("check_layer_warning")
            return CheckDecision(status="warning", reasons=reasons, confidence=conf, warnings=warnings)
        return CheckDecision(status="blocked", reasons=reasons, confidence=conf, warnings=warnings, fallback_used=True)

    return CheckDecision(status="approved", reasons=[], confidence=conf, warnings=[])


def to_check_layer_result(decision: CheckDecision) -> CheckLayerResult:
    return CheckLayerResult(
        approval_status=decision.status,
        reasons=decision.reasons,
        confidence_score=float(decision.confidence),
        warnings=decision.warnings,
    )


def persist_latest_check(metrics_dir: str | Path, payload: Dict[str, Any]) -> None:
    metrics_dir = Path(metrics_dir)
    ensure_dir(metrics_dir)
    out = metrics_dir / "latest_check.json"
    payload = dict(payload)
    payload.setdefault("timestamp", now_utc_iso())
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
