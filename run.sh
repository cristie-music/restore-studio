#!/usr/bin/env bash
# Restore Studio (Linux/macOS): starts the web studio on http://127.0.0.1:8790/ ; Ctrl+C to stop.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || bash ./install.sh
export PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python app/server.py "$@"
