#!/usr/bin/env bash
# start.sh — One-liner launcher for the MEV Bot Dashboard (Unix / macOS / Linux)
#
# Usage:
#   bash start.sh              # start on default port 5000
#   bash start.sh --port 8080  # custom port
#   bash start.sh --no-browser # don't auto-open browser
#
# For the full option list run: python run.py --help
set -euo pipefail

# Resolve the directory this script lives in (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find Python 3
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌  Python 3.10+ is required. Download from https://python.org"
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" run.py "$@"
