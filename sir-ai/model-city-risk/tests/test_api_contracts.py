from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


def test_contract_files_are_valid_json_and_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # JSON schemas
    schemas_dir = repo_root / "contracts" / "schemas"
    for name in [
        "request_predict.json",
        "response_predict.json",
        "response_area_risk.json",
        "response_department_view.json",
    ]:
        schema = json.loads((schemas_dir / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    # OpenAPI
    openapi = yaml.safe_load((repo_root / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
    assert "paths" in openapi
    for p in [
        "/health",
        "/predict",
        "/predictions/latest",
        "/areas/{geo_area}/risk",
        "/departments/{domain}/view",
        "/model/health",
        "/model/metrics",
        "/model/check/latest",
    ]:
        assert p in openapi["paths"]


def test_sample_request_validates_against_schema() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (repo_root / "contracts" / "schemas" / "request_predict.json").read_text(encoding="utf-8")
    )
    v = Draft202012Validator(schema)

    sample = {"window_start": "2025-11-15T10:00:00Z", "window_minutes": 60, "geo_area": "Central"}
    errors = sorted(v.iter_errors(sample), key=lambda e: e.path)
    assert not errors
