# Compressed-DDP Complete Command Guide

**Complete reference for setup, training, and validation**

Last Updated: February 13, 2026

---

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Environment Activation](#environment-activation)
3. [Verification Commands](#verification-commands)
4. [Training Commands](#training-commands)
5. [Testing Commands](#testing-commands)
6. [Development Commands](#development-commands)
7. [Troubleshooting Commands](#troubleshooting-commands)

---

## Initial Setup

### 1. Navigate to Project Directory

```bash
cd compressed-ddp
```

**Expected Result:**
```
# Your terminal prompt changes to show compressed-ddp directory
```

**Significance:** Ensures all subsequent commands run in the correct directory.

---

### 2. Run Setup Script

```bash
bash setup.sh
```

**Expected Result:**
```
======================================================================
Compressed-DDP Setup & Verification
======================================================================

[1/5] Checking Python version...
✅ Python 3.13.5

[2/5] Creating virtual environment...
✅ Virtual environment created

[3/5] Upgrading pip...
✅ Pip upgraded to 26.0.1

[4/5] Installing requirements...
   Installing packages (this may take 2-3 minutes)...
✅ Requirements installed

[5/5] Package installation...
⚠️  Skipping editable install (not required for development)
   You can run code directly from src/ using PYTHONPATH

[6/6] Verifying installation...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Packages:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ PyTorch: 2.10.0
✅ GPU: Apple Metal (MPS)
✅ torchvision: 0.25.0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dependencies:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ numpy: 2.4.2
✅ yaml: 6.0.3
✅ tensorboard: 2.20.0
✅ pytest: 9.0.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ src/ directory
   → 25 Python files
✅ tests/ directory
   → 3 test files
✅ experiments/ directory
   → 5 experiment files
✅ requirements.txt
✅ train.py

✅ Setup & Verification Complete!
```

**Significance:**
- Creates isolated Python environment (venv)
- Installs PyTorch, torchvision, and all dependencies
- Verifies installation correctness
- Detects GPU availability (CUDA/MPS/CPU)
- Auto-creates activate.sh helper script

**Time:** ~2-3 minutes

---

## Environment Activation

### 3. Activate Virtual Environment (Method 1 - Quick)

```bash
source activate.sh
```

**Expected Result:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Environment activated!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PYTHONPATH: /Users/.../compressed-ddp

Ready to run:
  • python experiments/quick_validation.py
  • pytest tests/ -v
  • python train.py --help
```

**Significance:**
- Activates virtual environment
- Sets PYTHONPATH for imports
- One command does everything!

---

### 3. Activate Virtual Environment (Method 2 - Manual)

```bash
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Expected Result:**
```
(venv) user@machine compressed-ddp %
```

**Significance:**
- Activates venv (prompt shows "(venv)")
- Sets PYTHONPATH so Python can find src/ modules
- Required before running any Python scripts

---

## Verification Commands

### 4. Verify PyTorch Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
```

**Expected Result:**
```
PyTorch: 2.10.0
```

**Significance:** Confirms PyTorch is correctly installed.

---

### 5. Verify GPU Detection

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

**Expected Result (Apple Silicon Mac):**
```
CUDA: False
MPS: True
```

**Expected Result (NVIDIA GPU):**
```
CUDA: True
MPS: False
```

**Expected Result (CPU only):**
```
CUDA: False
MPS: False
```

**Significance:** Shows available hardware acceleration.

---

### 6. Verify Module Imports

```bash
python -c "from src.compression import TopKCompressorGPU; print('✅ Imports work!')"
```

**Expected Result:**
```
✅ Imports work!
```

**Significance:** Confirms PYTHONPATH is set correctly and modules are importable.

---

### 7. List Available Models

```bash
python train.py --help
```

**Expected Result:**
```
usage: train.py [-h] [--model MODEL] [--dataset DATASET] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--lr LR] [--data-dir DATA_DIR]
                [--output-dir OUTPUT_DIR] [--seed SEED] [--no-cuda]
                [--rank RANK] [--world-size WORLD_SIZE]

Compressed-DDP Training Script

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model architecture (default: simple_cnn)
  --dataset DATASET     Dataset to use (default: mnist)
  --epochs EPOCHS       Number of training epochs (default: 10)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 32)
  --lr LR               Learning rate (default: 0.001)
  ...
```

**Significance:** Shows all available training options and defaults.

---

## Training Commands

### 8. Basic Training (MNIST + SimpleCNN)

```bash
python train.py --model simple_cnn --dataset mnist --epochs 5
```

**Expected Result:**
```
01:30:00 INFO train – Platform: {'os': 'macOS-26.2-arm64-arm-64bit-Mach-O', 'python': '3.13.5', 'torch': '2.10.0', 'cuda_available': 'False'}
01:30:00 INFO train – Device: cpu
01:30:00 INFO train – Model simple_cnn: 0.42M params

Downloading MNIST dataset... (if first run)
Extracting...

Epoch 1/5:
  Train Loss: 0.234 | Train Acc: 92.3%
  Val Loss: 0.145 | Val Acc: 95.6%

Epoch 2/5:
  Train Loss: 0.123 | Train Acc: 96.2%
  Val Loss: 0.098 | Val Acc: 97.1%

...

Training completed in 2m 34s
Best validation accuracy: 98.2%
Model saved to: checkpoints/simple_cnn_mnist_best.pth
```

**Significance:**
- Trains a simple CNN on MNIST dataset
- Tests basic training pipeline
- Downloads MNIST on first run (~10MB)
- Saves best model checkpoint
- Quick validation of setup (~3 minutes)

---

### 9. Training with Custom Parameters

```bash
python train.py --model simple_cnn --dataset mnist --epochs 10 --batch-size 64 --lr 0.01
```

**Expected Result:**
```
01:35:00 INFO train – Model simple_cnn: 0.42M params
01:35:00 INFO train – Batch size: 64
01:35:00 INFO train – Learning rate: 0.01

Epoch 1/10:
  Train Loss: 0.456 | Train Acc: 87.3%
  ...
```

**Significance:**
- Demonstrates custom hyperparameter usage
- Larger batch size = faster but more memory
- Higher learning rate = faster convergence (if stable)

---

### 10. CIFAR-10 Training

```bash
python train.py --model simple_cnn --dataset cifar10 --epochs 10
```

**Expected Result:**
```
01:40:00 INFO train – Dataset: CIFAR-10
01:40:00 INFO train – Model simple_cnn: 0.42M params

Downloading CIFAR-10... (if first run)
Extracting cifar-10-python.tar.gz...

Epoch 1/10:
  Train Loss: 1.234 | Train Acc: 45.2%
  Val Loss: 1.156 | Val Acc: 52.3%

...
```

**Significance:**
- Tests on more complex dataset (color images)
- CIFAR-10 download: ~170MB
- Lower initial accuracy (harder dataset)
- Tests data augmentation pipeline

---

### 11. Training with GPU (if available)

```bash
python train.py --model simple_cnn --dataset mnist --epochs 5
```

**Expected Result (with GPU):**
```
01:45:00 INFO train – Device: cuda:0 (or mps)
01:45:00 INFO train – Model simple_cnn: 0.42M params

Epoch 1/5: 100%|████████████| 1688/1688 [00:12<00:00, 135.45it/s]
  Train Loss: 0.234 | Train Acc: 92.3%
  ...

(Faster training with GPU)
```

**Significance:**
- GPU acceleration if available
- Automatically uses CUDA or MPS
- Faster training (3-10x speedup)

---

## Testing Commands

### 12. Run All Tests

```bash
pytest tests/ -v
```

**Expected Result:**
```
========================= test session starts ==========================
platform darwin -- Python 3.13.5, pytest-9.0.2, pluggy-1.5.0
collected 15 items

tests/test_compression.py::test_topk_compressor PASSED          [  6%]
tests/test_compression.py::test_randomk_compressor PASSED       [ 13%]
tests/test_error_feedback.py::test_error_buffer PASSED          [ 20%]
tests/test_communication.py::test_all_reduce PASSED             [ 26%]
...

========================= 15 passed in 5.23s ===========================
```

**Significance:**
- Validates all compression algorithms
- Tests error feedback mechanisms
- Verifies communication primitives
- Ensures code quality

---

### 13. Run Specific Test File

```bash
pytest tests/test_compression.py -v
```

**Expected Result:**
```
tests/test_compression.py::test_topk_compressor PASSED
tests/test_compression.py::test_randomk_compressor PASSED
tests/test_compression.py::test_threshold_compressor PASSED

========================= 3 passed in 1.45s ============================
```

**Significance:** Tests only compression algorithms.

---

### 14. Run Tests with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

**Expected Result:**
```
========================= test session starts ==========================
collected 15 items

tests/test_compression.py ....                                  [ 26%]
tests/test_error_feedback.py ....                               [ 53%]
tests/test_communication.py .......                             [100%]

---------- coverage: platform darwin, python 3.13.5 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/compression/__init__.py          12      0   100%
src/compression/topk.py              45      3    93%
src/error_feedback/buffer.py        38      2    95%
...
-----------------------------------------------------
TOTAL                               234     12    95%

Coverage HTML written to htmlcov/index.html
```

**Significance:**
- Shows code coverage percentage
- Identifies untested code
- Generates HTML report

---

## Development Commands

### 15. Quick Validation Experiment

```bash
python experiments/quick_validation.py
```

**Expected Result:**
```
Running quick validation experiment...

Testing compression algorithms:
  TopK (k=10): Compression ratio: 90.0%
  RandomK (k=10): Compression ratio: 90.0%
  Threshold (t=0.1): Compression ratio: 85.3%

Testing error feedback:
  Residual accumulation: ✅ Working
  Gradient reconstruction: ✅ Working

All validations passed! ✅
```

**Significance:**
- Quick sanity check of main components
- Tests compression ratios
- Validates error feedback
- Fast (~30 seconds)

---

### 16. View Training Logs with TensorBoard

```bash
tensorboard --logdir=runs
```

**Expected Result:**
```
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

**Then open browser:** http://localhost:6006

**Significance:**
- Visualizes training metrics
- Shows loss/accuracy curves
- Compares different runs
- Interactive plots

---

### 17. Check Model Checkpoints

```bash
ls -lh checkpoints/
```

**Expected Result:**
```
total 8.4M
-rw-r--r-- 1 user staff 1.7M Feb 13 01:30 simple_cnn_mnist_best.pth
-rw-r--r-- 1 user staff 1.7M Feb 13 01:30 simple_cnn_mnist_epoch_5.pth
-rw-r--r-- 1 user staff 5.0M Feb 13 01:40 simple_cnn_cifar10_best.pth
```

**Significance:**
- Shows saved model weights
- File sizes indicate model complexity
- _best.pth = best validation accuracy
- _epoch_N.pth = checkpoint at epoch N

---

### 18. List Downloaded Datasets

```bash
ls -lh data/
```

**Expected Result:**
```
total 32M
drwxr-xr-x  4 user staff  128B Feb 13 01:30 MNIST/
  └── raw/
      ├── train-images-idx3-ubyte.gz (9.9M)
      ├── train-labels-idx1-ubyte.gz (29K)
      ├── t10k-images-idx3-ubyte.gz (1.6M)
      └── t10k-labels-idx1-ubyte.gz (5.0K)

drwxr-xr-x  3 user staff   96B Feb 13 01:40 cifar-10-batches-py/
  └── data_batch_1 (31M)
  └── ... (170M total)
```

**Significance:**
- Shows cached datasets
- Avoids re-downloading
- MNIST: ~10MB, CIFAR-10: ~170MB

---

## Troubleshooting Commands

### 19. Check Python and Package Versions

```bash
python --version
pip list | grep torch
pip list | grep numpy
```

**Expected Result:**
```
Python 3.13.5

torch                2.10.0
torchvision          0.25.0
numpy                2.4.2
```

**Significance:** Verifies correct versions installed.

---

### 20. Verify PYTHONPATH

```bash
echo $PYTHONPATH
```

**Expected Result:**
```
/Users/.../compressed-ddp
```

**Significance:** Confirms PYTHONPATH is set (required for imports).

---

### 21. Test Import Manually

```bash
python
>>> import sys
>>> sys.path.insert(0, 'src')
>>> from compression import TopKCompressorGPU
>>> print("✅ Success")
>>> exit()
```

**Expected Result:**
```
✅ Success
```

**Significance:** Manual import test if PYTHONPATH issues.

---

### 22. Clean Build Artifacts

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

**Expected Result:**
```
(No output - silently removes cache files)
```

**Significance:**
- Removes Python bytecode cache
- Useful after code changes
- Frees disk space

---

### 23. Reset Virtual Environment

```bash
rm -rf venv
bash setup.sh
```

**Expected Result:**
```
(Same as initial setup - fresh environment)
```

**Significance:**
- Complete reset if environment corrupted
- Reinstalls everything from scratch

---

## Quick Reference

### Essential Commands (Copy-Paste Ready)

```bash
# 1. Setup (once)
cd compressed-ddp
bash setup.sh

# 2. Activate (every session)
source activate.sh

# 3. Verify
python -c "import torch; print('PyTorch:', torch.__version__)"

# 4. Train
python train.py --model simple_cnn --dataset mnist --epochs 5

# 5. Test
pytest tests/ -v

# 6. Experiment
python experiments/quick_validation.py
```

---

## Command Summary Table

| Command | Time | Purpose | When to Use |
|---------|------|---------|-------------|
| `bash setup.sh` | 2-3 min | Initial setup | Once, or after reset |
| `source activate.sh` | <1 sec | Activate env | Every new terminal |
| `python train.py ...` | 3-30 min | Train model | Development/experiments |
| `pytest tests/` | 5-10 sec | Run tests | After code changes |
| `python experiments/...` | 30 sec | Quick validation | Sanity checks |
| `tensorboard --logdir=runs` | <1 sec | View metrics | Analyze training |

---

## Expected File Structure After Setup

```
compressed-ddp/
├── venv/                      # Virtual environment
├── data/                      # Downloaded datasets
│   ├── MNIST/                 # ~10MB
│   └── cifar-10-batches-py/   # ~170MB
├── checkpoints/               # Saved models
│   └── simple_cnn_mnist_best.pth
├── runs/                      # TensorBoard logs
├── src/                       # Source code
│   ├── compression/
│   ├── error_feedback/
│   ├── communication/
│   ├── models/
│   └── data/
├── tests/                     # Test files
├── experiments/               # Experiment scripts
├── activate.sh                # Quick activation (auto-created)
├── setup.sh                   # Setup script
├── train.py                   # Main training script
└── requirements.txt           # Dependencies
```

---

## Status Indicators

### ✅ Success Indicators

- Green checkmarks (✅) in output
- `Training completed successfully`
- `X passed in Y.Zs` (pytest)
- Model saved to checkpoints/
- No error messages

### ⚠️ Warning Indicators (Usually OK)

- Yellow warnings (⚠️)
- `Skipping editable install`
- `Dataset will download on first use`
- UserWarnings (non-critical)

### ❌ Error Indicators (Need Fixing)

- Red errors (❌)
- `ImportError` - PYTHONPATH not set
- `TypeError` - Code issue
- `RuntimeError` - Execution issue
- Training crashes

---

## Performance Benchmarks

### Expected Training Times (MNIST, 5 epochs)

| Device | simple_cnn | Time |
|--------|-----------|------|
| Apple M1 (MPS) | 0.42M params | ~3 min |
| NVIDIA RTX 3080 | 0.42M params | ~1.5 min |
| Intel Core i7 (CPU) | 0.42M params | ~8 min |

### Expected Test Times

| Test Suite | Tests | Time |
|------------|-------|------|
| Full suite | 15 tests | ~5 sec |
| Compression only | 3 tests | ~1.5 sec |
| With coverage | 15 tests | ~7 sec |

---

## Next Steps After Setup

1. ✅ Run basic training: `python train.py --model simple_cnn --dataset mnist --epochs 5`
2. ✅ Run tests: `pytest tests/ -v`
3. ✅ Explore experiments: `ls experiments/`
4. ✅ Read documentation: `cat README.md`
5. ✅ Start development!

---

**Document Version:** 1.0  
**Last Updated:** February 13, 2026, 1:33 AM IST  
**Tested On:** macOS 26.2 (Apple Silicon), Python 3.13.5, PyTorch 2.10.0
