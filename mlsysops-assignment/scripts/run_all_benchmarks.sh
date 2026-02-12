#!/usr/bin/env bash
# Run all benchmarks with Python 3.13 fixes

set -euo pipefail

echo "======================================================================"
echo "Running All Benchmarks (Python 3.13 Compatible)"
echo "======================================================================"
echo ""

echo "=== [1/2] Compression Benchmark ==="
python3 benchmark_compression_fixed.py

echo ""
echo "=== [2/2] Training Benchmark ==="
python3 benchmark_training_fixed.py

echo ""
echo "======================================================================"
echo "✅ All benchmarks complete!"
echo "======================================================================"
echo ""
echo "Results saved in: experiments/results/"
ls -lh experiments/results/*.csv 2>/dev/null || echo "(No CSV files found)"
