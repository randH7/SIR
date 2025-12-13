#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m venv "$REPO_ROOT/.venv" >/dev/null 2>&1 || true
source "$REPO_ROOT/.venv/bin/activate"

pip install -q -r "$REPO_ROOT/requirements.txt"

python3 "$REPO_ROOT/scripts/generate_mock_data.py" --repo-root "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" python3 -m src.model.train
PYTHONPATH="$REPO_ROOT" python3 -m src.model.predict
