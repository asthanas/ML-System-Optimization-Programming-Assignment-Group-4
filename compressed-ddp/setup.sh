#!/usr/bin/env bash
# Compressed-DDP Setup Script (Linux/macOS)
# For Windows, use: python setup.py

set -euo pipefail

echo "======================================================================"
echo "Compressed-DDP Setup (Linux/macOS)"
echo "======================================================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python3 -c "import sys; assert sys.version_info>=(3,9),'Python 3.9+ required'" || {
    echo "❌ Error: Python 3.9+ required"
    exit 1
}
echo "✅ Python version OK"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d venv ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate and upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
source venv/bin/activate
pip install --upgrade pip -q
echo "✅ Pip upgraded"

# Install requirements
echo ""
echo "[4/5] Installing requirements..."
pip install -r requirements.txt -q
echo "✅ Requirements installed"

# Install package
echo ""
echo "[5/5] Installing package in editable mode..."
pip install -e . -q
echo "✅ Package installed"

# Check GPU
echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ GPU: Apple Metal (MPS)')
else:
    print('ℹ️  No GPU detected - CPU mode')
"

echo ""
echo "======================================================================"
echo "✅ Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run validation: python experiments/quick_validation.py"
echo "  3. Run tests: bash scripts/run_tests.sh"
echo "  4. Train model: python train.py --help"
echo ""
echo "======================================================================"
