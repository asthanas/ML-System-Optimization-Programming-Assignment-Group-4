#!/usr/bin/env bash
set -euo pipefail
echo "=== Running Test Suite ==="
source venv/bin/activate 2>/dev/null || true
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing -q
