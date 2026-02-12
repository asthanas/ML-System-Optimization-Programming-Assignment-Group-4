#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate 2>/dev/null || true
echo "=== Compression Benchmark ==="
python experiments/benchmark_compression.py
echo "=== Training Benchmark ==="
python experiments/benchmark_training.py
echo "=== Scalability Analysis ==="
python experiments/scalability_analysis.py
