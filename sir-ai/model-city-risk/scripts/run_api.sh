#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m venv "$REPO_ROOT/.venv" >/dev/null 2>&1 || true
source "$REPO_ROOT/.venv/bin/activate"

pip install -q -r "$REPO_ROOT/requirements.txt"

PYTHONPATH="$REPO_ROOT" uvicorn src.api.app:app --host 0.0.0.0 --port 8000
