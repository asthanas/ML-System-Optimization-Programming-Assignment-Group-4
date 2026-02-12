#!/usr/bin/env bash
# Compressed-DDP Setup & Verification Script (Linux/macOS)
# Skip editable install to avoid configuration conflicts

set -eo pipefail

echo "======================================================================"
echo "Compressed-DDP Setup & Verification"
echo "======================================================================"
echo ""

# ============================================================================
# PART 1: SETUP
# ============================================================================

# Check Python version
echo "[1/5] Checking Python version..."
if ! python3 -c "import sys; assert sys.version_info>=(3,9)" 2>/dev/null; then
    echo "❌ Error: Python 3.9+ required"
    python3 --version 2>/dev/null || echo "Python 3 not found"
    exit 1
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✅ Python $PY_VERSION"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ -d venv ]; then
    echo "⚠️  Removing existing venv..."
    rm -rf venv
fi
python3 -m venv venv
echo "✅ Virtual environment created"

# Define venv paths
VENV_PYTHON="venv/bin/python"
VENV_PIP="venv/bin/pip"

# Verify venv was created correctly
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment creation failed"
    exit 1
fi

# Upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
if $VENV_PYTHON -m pip install --upgrade pip -q 2>&1; then
    PIP_VERSION=$($VENV_PYTHON -m pip --version | awk '{print $2}')
    echo "✅ Pip upgraded to $PIP_VERSION"
else
    echo "⚠️  Could not upgrade pip (continuing with current version)"
    PIP_VERSION=$($VENV_PYTHON -m pip --version | awk '{print $2}')
fi

# Install requirements
echo ""
echo "[4/5] Installing requirements..."
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt not found, creating minimal version..."
    cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
pyyaml>=6.0
tensorboard>=2.10.0
pytest>=7.0.0
EOF
    echo "✅ Created requirements.txt"
fi

echo "   Installing packages (this may take 2-3 minutes)..."
if $VENV_PYTHON -m pip install -r requirements.txt 2>&1 | grep -q "Successfully installed"; then
    echo "✅ Requirements installed"
else
    echo "❌ Failed to install requirements"
    echo "   Try manually: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Skip editable install to avoid conflicts
echo ""
echo "[5/5] Package installation..."
echo "⚠️  Skipping editable install (not required for development)"
echo "   You can run code directly from src/ using PYTHONPATH"
EDITABLE_INSTALL="No (not required)"

# ============================================================================
# PART 2: VERIFICATION
# ============================================================================

echo ""
echo "[6/6] Verifying installation..."
echo ""

# Check PyTorch
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Core Packages:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TORCH_VERSION=$($VENV_PYTHON -c "import torch; print(torch.__version__)" 2>&1 || echo "NOT_INSTALLED")
if [[ "$TORCH_VERSION" == "NOT_INSTALLED" ]]; then
    echo "❌ PyTorch: NOT INSTALLED"
    exit 1
else
    echo "✅ PyTorch: $TORCH_VERSION"
fi

# Check GPU
GPU_INFO=$($VENV_PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'{torch.cuda.get_device_name(0)} (CUDA)')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple Metal (MPS)')
else:
    print('None (CPU mode)')
" 2>&1)
echo "✅ GPU: $GPU_INFO"

# Check torchvision
TORCHVISION_VERSION=$($VENV_PYTHON -c "import torchvision; print(torchvision.__version__)" 2>&1 || echo "NOT_INSTALLED")
if [[ "$TORCHVISION_VERSION" == "NOT_INSTALLED" ]]; then
    echo "❌ torchvision: NOT INSTALLED"
else
    echo "✅ torchvision: $TORCHVISION_VERSION"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dependencies:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for pkg in numpy yaml tensorboard pytest; do
    if $VENV_PYTHON -c "import $pkg" 2>/dev/null; then
        VERSION=$($VENV_PYTHON -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "installed")
        printf "✅ %-15s %s\n" "$pkg:" "$VERSION"
    else
        printf "❌ %-15s %s\n" "$pkg:" "NOT INSTALLED"
    fi
done

# Check project structure
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Project Structure:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "src" ]; then
    echo "✅ src/ directory"
    SRC_FILES=$(find src -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
    echo "   → $SRC_FILES Python files"
else
    echo "⚠️  src/ directory not found"
fi

if [ -d "tests" ]; then
    TEST_FILES=$(find tests -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ tests/ directory"
    echo "   → $TEST_FILES test files"
else
    echo "⚠️  tests/ directory not found"
fi

if [ -d "experiments" ]; then
    EXP_FILES=$(find experiments -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ experiments/ directory"
    echo "   → $EXP_FILES experiment files"
else
    echo "⚠️  experiments/ directory not found"
fi

[ -f "requirements.txt" ] && echo "✅ requirements.txt" || echo "⚠️  requirements.txt"
[ -f "train.py" ] && echo "✅ train.py" || echo "⚠️  train.py"

# Test imports (if src exists)
if [ -d "src" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Import Tests:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Set PYTHONPATH safely
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

    for module in compression error_feedback communication models data utils metrics; do
        if [ -d "src/$module" ]; then
            if $VENV_PYTHON -c "from src import $module" 2>/dev/null; then
                echo "✅ src.$module"
            else
                echo "⚠️  src.$module (import failed)"
            fi
        fi
    done
fi

# Check datasets
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Datasets:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "data/MNIST/raw" ]; then
    MNIST_FILES=$(ls data/MNIST/raw/*.gz 2>/dev/null | wc -l | tr -d ' ')
    if [ "$MNIST_FILES" -eq "4" ]; then
        echo "✅ MNIST (downloaded, $MNIST_FILES/4 files)"
    else
        echo "⚠️  MNIST (incomplete - $MNIST_FILES/4 files)"
    fi
else
    echo "⚠️  MNIST (not downloaded - will download on first use)"
fi

if [ -d "data/cifar-10-batches-py" ] || [ -d "data/cifar-10-batches" ]; then
    echo "✅ CIFAR-10 (downloaded)"
else
    echo "⚠️  CIFAR-10 (not downloaded - will download on first use)"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "======================================================================"
echo "✅ Setup & Verification Complete!"
echo "======================================================================"
echo ""
echo "Installation Summary:"
echo "  • Python:           $PY_VERSION"
echo "  • Pip:              $PIP_VERSION"
echo "  • PyTorch:          $TORCH_VERSION"
echo "  • GPU:              $GPU_INFO"
echo "  • Editable Install: $EDITABLE_INSTALL"
echo "  • Virtual Env:      $(pwd)/venv"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Next Steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set PYTHONPATH (required for imports):"
echo "   export PYTHONPATH=\$PWD:\$PYTHONPATH"
echo ""
echo "3. Then try:"
if [ -f "experiments/quick_validation.py" ]; then
    echo "   python experiments/quick_validation.py"
fi
if [ -d "tests" ]; then
    echo "   pytest tests/ -v"
fi
echo "   python -c 'import torch; print("PyTorch:", torch.__version__)'"
echo ""
echo "======================================================================"

# Create activation helper
cat > activate.sh << 'ACTIVATE_EOF'
#!/usr/bin/env bash
# Quick activation helper
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Environment activated!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "Ready to run:"
echo "  • python experiments/quick_validation.py"
echo "  • pytest tests/ -v"
echo "  • python train.py --help"
echo ""
ACTIVATE_EOF

chmod +x activate.sh

echo ""
echo "💡 Pro tip: Use 'source activate.sh' for quick activation!"
echo ""
