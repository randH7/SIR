import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_ts(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = [max(0.0, float(v)) for v in values]
    s = sum(vals)
    if s == 0:
        return 0.0
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    cum = 0.0
    for i, v in enumerate(vals_sorted, start=1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


@dataclass(frozen=True)
class Paths:
    root: Path
    config: Dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return self.root / self.config["paths"]["data_dir"]

    @property
    def output_dir(self) -> Path:
        return self.root / self.config["paths"]["output_dir"]

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.config["paths"]["artifacts_dir"]

    @property
    def models_dir(self) -> Path:
        return self.root / self.config["paths"]["models_dir"]

    @property
    def metrics_dir(self) -> Path:
        return self.root / self.config["paths"]["metrics_dir"]


def load_config(root: str | os.PathLike = ".") -> Tuple[Dict[str, Any], Paths]:
    root_path = Path(root).resolve()
    cfg = load_yaml(root_path / "configs" / "config.yaml")
    return cfg, Paths(root=root_path, config=cfg)
