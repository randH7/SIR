# Model Card — Resource Readiness & Deployment Optimization

- **Model type**: Logistic Regression (unit matching / suitability scoring)
- **Training data**: Synthetic mock datasets in `data/input_datasets/`
- **Primary use**: Rank candidate response units for an incident, subject to policy constraints and safety thresholds.

## Intended Use
This model supports operational decisioning by suggesting ranked unit assignments. It does **not** predict incidents.

## Safety & Governance
A mandatory check layer gates outputs using thresholds (confidence, fatigue, utilization, policy compliance).

## Metrics
See `artifacts/metrics/evaluation_report.json` and scenario tests in `artifacts/metrics/scenario_test_results.json`.
