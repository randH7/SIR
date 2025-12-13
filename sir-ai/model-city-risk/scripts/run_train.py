#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "yaml",
    "numpy",
    "pandas",
    "sklearn",
    "joblib",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_deps(root: Path) -> None:
    missing = []
    for m in REQUIRED_IMPORTS:
        try:
            __import__(m)
        except Exception:
            missing.append(m)

    if not missing:
        return

    print(f"[run_train] Missing deps: {missing}. Installing from requirements.txt...")
    req = root / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", str(req)]
    subprocess.check_call(cmd)


def generate_data_if_missing(root: Path) -> None:
    sentinel = root / "data" / "input_datasets" / "incidents_all_911.csv"
    if sentinel.exists():
        return

    print("[run_train] Input datasets missing. Generating synthetic data...")
    cmd = [sys.executable, str(root / "scripts" / "generate_mock_data.py"), "--repo-root", str(root)]
    subprocess.check_call(cmd)


def run_training(root: Path) -> None:
    print("[run_train] Training baseline model + writing artifacts...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cmd = [sys.executable, "-m", "src.model.train"]
    subprocess.check_call(cmd, cwd=str(root), env=env)


def main() -> None:
    root = repo_root()
    ensure_deps(root)
    generate_data_if_missing(root)
    run_training(root)

    # Quick success check
    model_path = root / "artifacts" / "models" / "city_risk_model.pkl"
    check_path = root / "artifacts" / "metrics" / "check_latest.json"
    if model_path.exists() and check_path.exists():
        print("[run_train] SUCCESS ✅")
        print(f"[run_train] Model: {model_path}")
        print(f"[run_train] Check layer: {check_path}")
    else:
        print("[run_train] Training finished, but expected artifacts are missing.")
        print(f"[run_train] Missing? model={model_path.exists()} check={check_path.exists()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
