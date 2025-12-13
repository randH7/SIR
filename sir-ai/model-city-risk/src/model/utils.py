from __future__ import annotations

import json
import logging
import logging.config
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    input_dir: Path
    output_dir: Path
    artifacts_dir: Path
    models_dir: Path
    metrics_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(repo_root: Path) -> dict[str, Any]:
    return load_yaml(repo_root / "configs" / "config.yaml")


def setup_logging(repo_root: Path) -> None:
    cfg_path = repo_root / "configs" / "logging.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO)


def get_paths(repo_root: Path, cfg: dict[str, Any]) -> ProjectPaths:
    paths = cfg["paths"]
    data_dir = repo_root / paths["data_dir"]
    input_dir = repo_root / paths["input_dir"]
    output_dir = repo_root / paths["output_dir"]
    artifacts_dir = repo_root / paths["artifacts_dir"]
    models_dir = repo_root / paths["models_dir"]
    metrics_dir = repo_root / paths["metrics_dir"]
    return ProjectPaths(
        root=repo_root,
        data_dir=data_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
    )


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compress to keep artifacts GitHub-friendly (and faster to download).
    # Note: joblib compression is deterministic for our use case here.
    joblib.dump(obj, path, compress=3)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}
