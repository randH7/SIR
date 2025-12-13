#!/usr/bin/env bash
set -euo pipefail
 
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
 
# Simple cross-platform runner:
# - Windows: use `python scripts/run_train.py` or `py scripts\\run_train.py`
# - Linux/macOS/Git Bash: `bash scripts/run_train.sh`
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi
 
"$PY" "$REPO_ROOT/scripts/run_train.py"