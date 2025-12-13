## model-resource-deployment

Backend-only **Resource Readiness, Capacity & Deployment Optimization Model**.

This repository contains **ONLY ONE MODEL**. It does **not** predict incidents. Instead, it answers:

- **Who should respond?**
- **From which sector / ministry?**
- **From where (location)?**
- **With how many people?**
- **Under skills, experience, availability, proximity, and policy constraints**
- **And whether we risk overload + what pre-positioning actions to take**

This repo is designed to be **standalone** today, and **plug-and-play** later inside a multi-model orchestration engine.

---

## Model type & libraries

### Model type
- **Scoring (ML)**: **Logistic Regression** unit suitability model trained on synthetic incident→unit pairs (`scikit-learn`).
- **Optimization (rules + constraints)**: policy-constrained ranking/selection that balances ETA, skill fit, and load (`src/model/optimizer.py`).
- **Safety gate (hard checks)**: check-layer approval/warning/block with fallback plan (`src/model/check_layer.py`).

### Core libraries
- **Python**: 3.12+
- **Data/ML**: `pandas`, `numpy`, `scikit-learn`, `joblib`
- **API/Contracts**: `fastapi`, `pydantic`, `jsonschema`
- **Config**: `PyYAML`

---

## What this model does (and does not do)

### In scope
- **Candidate scoring**: rank units for an incident using an ML scoring model (plus heuristics if the model isn’t trained yet).
- **Constraint handling**: enforce domain → sector responsibility rules and cross-sector support rules.
- **Optimization**: pick a small set of units that jointly satisfy constraints and operational goals.
- **Capacity view**: compute utilization-based capacity status and recommendations by sector/area.
- **Safety gate (Check Layer)**: block or downgrade outputs that violate thresholds (confidence, utilization, policy compliance, missing data).

### Out of scope
- Incident forecasting / incident risk prediction (that is a separate model/repo).
- A live dispatch engine (this repo returns recommendations only).
- Any personal data (all data in this repo is synthetic).

---

## Repository layout

```text
configs/                 Runtime + policy + threshold configuration
  config.yaml            Thresholds & paths
  policies.yaml          Domain→primary sector + support sectors
  logging.yaml           Logging configuration

data/
  input_datasets/        Synthetic datasets (generated)
  output_plans/          Latest deployment plan + capacity map outputs

src/
  model/                 Model logic (features, scoring, optimizer, check layer)
  api/                   FastAPI surface (endpoints/controllers/validators)

contracts/
  openapi.yaml           API contract
  schemas/               Request/response JSON schemas

tests/                   Unit tests + API contract tests
scripts/                 Mock data gen + run scripts
artifacts/
  models/                Trained model artifacts (pkl)
  metrics/               Evaluation + scenario tests + check logs
```

---

## Input datasets (synthetic)

All files live under `data/input_datasets/`.

### Operational input (what the model needs at inference time)
At runtime (via API or direct function call), the model needs:
- **Incident input**: `latitude`, `longitude`, `incident_domain`, `severity` (+ optional `incident_id`, `timestamp`, `geo_area`)
- **Current state tables** (latest rows are used):
  - units master (`workforce_units.csv`)
  - skills (`workforce_skills.csv`)
  - availability (`on_shift_units.csv`)
  - locations (`unit_locations.csv`)
  - sector load (`sector_capacity.csv`)
  - policy (`policy_constraints.csv`, `configs/policies.yaml`)
  - cross-sector triggers (`cross_sector_rules.csv`)

### `incidents_history.csv` (training signal)
Used to learn historical patterns of what units were effective for which incidents.

Key columns:
- `incident_id`, `timestamp`, `geo_area`, `latitude`, `longitude`
- `incident_domain`, `incident_type`, `severity`
- `sectors_involved` (JSON list)
- `units_dispatched` (JSON list)
- `responders_count`, `resolution_time_minutes`
- `success_flag`, `escalation_flag`

### `workforce_units.csv` (unit master)
Each row = one operational unit.

Key columns:
- `unit_id`, `sector`, `ministry`
- `home_base_geo_area`, `home_latitude`, `home_longitude`
- `unit_type`, `max_capacity_people`

### `workforce_skills.csv` (skills repository)
Each row = a skill profile for a unit.

Key columns:
- `unit_id`, `skill_name`, `proficiency_level (1–5)`
- `years_experience`, `incidents_handled_count`, `success_rate`

### `on_shift_units.csv` (availability)
Key columns:
- `timestamp`, `unit_id`, `is_on_shift`
- `available_people_count`, `fatigue_score`, `last_dispatch_minutes_ago`

### `unit_locations.csv` (live-ish locations)
Key columns:
- `timestamp`, `unit_id`, `latitude`, `longitude`, `geo_area`
- `mobility_status` (stationary/patrolling/enroute)

### `sector_capacity.csv` (capacity/load)
Key columns:
- `timestamp`, `sector`, `geo_area`
- `total_units`, `total_people_on_shift`, `active_incidents`
- `utilization_ratio`, `overload_flag`

### `policy_constraints.csv` (policy constraints)
Key columns:
- `domain`, `primary_sector`, `allowed_support_sectors` (JSON list)
- `max_response_time_minutes`, `escalation_rules`, `mandatory_dual_sector_flag`

### `cross_sector_rules.csv` (cross-sector triggers)
Key columns:
- `incident_domain`, `severity_threshold`
- `trigger_sector`, `backup_sector`, `backup_activation_delay`, `priority_level`

### `shift_rosters.csv` (shift planning)
Key columns:
- `unit_id`, `date`, `shift_start`, `shift_end`, `planned_people`, `overtime_allowed`

---

## Outputs

All outputs are written to `data/output_plans/`.

### `deployment_plan_latest.json` / `deployment_plan_latest.csv`
For a given incident input, the model produces a ranked unit assignment with:

- `selected_units` (ranked)
- `sector_breakdown`
- `estimated_response_time`
- `confidence_score`
- `reasoning_tags` (e.g. `closest_unit`, `highest_skill_match`, `low_sector_load`, `policy_mandated`, `backup_required`)
- `check` (approval gate status)

Deployment output also includes:
- **per-unit fields**: `unit_id`, `sector`, `ministry`, `latitude`, `longitude`, `available_people_count`, `fatigue_score`
- **explainability**: `skills_matched`, `skill_match_score`, `reasoning_tags`
- **safety gate**: `check.approval_status`, `check.reasons`, `check.confidence_score`

### `capacity_risk_map.csv`
Capacity by area/sector with:

- `utilization_ratio`
- `capacity_status`: `green | amber | red`
- `recommendation`: `pre_position_units | add_extra_shift | activate_cross_sector_backup`

---

## Core logic (scoring + rule-constrained optimization)

### Feature engineering
Implemented in `src/model/features.py`.

Key features:
- **Distance** to incident (`Haversine`)
- **Skill match score** vs domain requirements
- **Availability score** (available people / capacity)
- **Fatigue penalty** and **recency penalty**
- **Sector load ratio**
- **Policy eligibility flag**
- **ETA estimate**
- **Cross-sector priority bonus/penalty**

### Scoring model
Implemented in `src/model/scorer.py`.

- Trains a lightweight **Logistic Regression** on synthetic historical incident-unit pairs.
- If artifacts aren’t present, it uses a deterministic **heuristic scoring fallback**.

Artifacts:
- `artifacts/models/resource_match_model.pkl`
- `artifacts/models/encoders.pkl`

### Optimizer
Implemented in `src/model/optimizer.py`.

Process:
1. Build candidate set from on-shift, availability, fatigue, distance.
2. Enforce policy eligibility (primary sector + allowed support sectors).
3. Score candidates (ML probability).
4. Select units while:
   - preferring fast ETA
   - maximizing skill fit
   - avoiding high-load sectors when possible
   - enforcing mandatory dual-sector responses when required

---

## Mandatory safety layer: Check Layer (hard gate)

This is enforced **before outputs are exposed to APIs**.

> “This check layer ensures that deployment decisions are safe, explainable, and operationally reliable before any unit is dispatched.”

Implemented in `src/model/check_layer.py`.

### Decisions
- `approved`: safe to use
- `warning`: output allowed but flagged
- `blocked`: output blocked → fallback plan returned

### Thresholds
Configured in `configs/config.yaml` under `thresholds:`.

Examples:
- `max_sector_utilization_allowed: 0.85`
- `max_fatigue_score_for_dispatch: 0.70`
- `min_confidence_score_for_auto_dispatch: 0.60`
- `max_policy_violation_rate: 0`

### API-facing gate output
Every deployment response includes:

```json
{
  "approval_status": "approved | warning | blocked",
  "reasons": ["low_confidence", "sector_overload_risk"],
  "confidence_score": 0.72
}
```

### Explainability artifacts
Written to `artifacts/metrics/`:
- `explanations.json` (global feature importance + future per-decision explanations)
- `latest_check.json` (latest gate decision)

---

## Evaluation & monitoring (required layer)

Artifacts live under `artifacts/metrics/`:

- `evaluation_report.json` (Phase A: offline training evaluation)
- `scenario_test_results.json` (Phase B: scenario stress testing)
- `capacity_stress_test.json` (capacity stress placeholder)
- `model_card.md` (model card)

Current offline evaluation includes:
- **Top-K Match Accuracy** (Top-5 and Top-10)
- **Precision / Recall / F1 / Accuracy** for unit-match classification (test set)
- **Best-F1 threshold** recommendation
- **ROC-AUC / PR-AUC** (probability quality)
- **Confidence distribution summary**

Important: training evaluation uses a **grouped split by `incident_id`** to prevent leakage.

---

## Setup

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Generate mock data (≥50k rows per dataset)

A generator script will create all required CSVs in `data/input_datasets/`.

```bash
python3 scripts/generate_mock_data.py --rows 50000
```

---

## Train the scoring model + write evaluation artifacts

```bash
python3 -m src.model.train
```

This writes:
- `artifacts/models/*.pkl`
- `artifacts/metrics/*.json` and `model_card.md`

---

## Run a deployment recommendation (local)

The simplest non-API entrypoint is:

```bash
python3 -c "from src.model.schemas import IncidentInput; from src.model.predict import create_deployment_plan; print(create_deployment_plan(IncidentInput(latitude=24.7136, longitude=46.6753, incident_domain='traffic', severity=3)).model_dump_json(indent=2))"
```

Outputs are also persisted to:
- `data/output_plans/deployment_plan_latest.json`
- `data/output_plans/deployment_plan_latest.csv`

---

## API

The API layer lives under `src/api/` and must expose:

- `POST /deploy`
- `GET /capacity/view`
- `GET /units/closest`
- `GET /sectors/{sector}/readiness`

Recommended monitoring endpoints:
- `GET /model/health`
- `GET /model/metrics`
- `GET /model/check/latest`

### Example: deploy

```bash
curl -X POST http://127.0.0.1:8000/deploy \
  -H 'Content-Type: application/json' \
  -d '{"incident_id":"INC-1","latitude":24.7136,"longitude":46.6753,"incident_domain":"traffic","severity":3}'
```

Response includes ranked units with:
- `unit_id`, `sector`, `ministry`, `lat/lon`
- `availability`, `skills_matched`
- `confidence` and `reasoning_tags`
- `check.approval_status` + reasons

### Direct (non-API) inference entrypoint
If you’re integrating this model into another backend service without HTTP, call:
- `src/model/predict.py:create_deployment_plan(IncidentInput(...))`

This returns a `DeploymentPlan` object and also persists `deployment_plan_latest.json/csv`.

---

## Contracts (ABI)
---

## Contracts (API)

- `contracts/openapi.yaml` defines endpoint contracts.
- `contracts/schemas/*.json` defines strict request/response schemas.

Tests in `tests/test_api_contracts.py` validate that API responses conform to these schemas.

---

## How this plugs into the larger system

Later, the platform orchestration engine will:

1. Receive a real incident.
2. Call the **incident prediction model** to infer domain/forecast context.
3. Call **this model** to compute:
   - recommended units
   - capacity risk + pre-positioning actions
4. Dispatch messages via a separate **live dispatch engine**.

This repo remains backend-only and model-only so it can be merged cleanly.

---

## Non-functional guarantees

- **No personal data**: datasets are fully synthetic.
- **Modular, merge-safe structure**: clean separation of model, API, contracts.
- **Safety gate enforced**: check layer can block outputs and trigger fallback.
