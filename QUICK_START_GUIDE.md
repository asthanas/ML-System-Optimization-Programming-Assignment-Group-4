# QUICK START GUIDE
## Setup, Usage, and Troubleshooting

**Date:** February 12, 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. INSTALLATION (2 MINUTES)

### Prerequisites

- Python 3.9+ (tested on 3.9, 3.10, 3.11, 3.13)
- pip package manager
- 4GB RAM minimum
- 1GB disk space for datasets

### Step-by-Step Setup

```bash
# 1. Extract the submission package
unzip compressed-ddp-final-submission.zip
cd compressed-ddp

# 2. Run automated setup
bash setup.sh

# 3. Activate virtual environment
source venv/bin/activate       # Linux/macOS
# OR
venv\Scripts\activate         # Windows
```

**What setup.sh does:**
- Creates Python virtual environment
- Installs PyTorch and dependencies
- Installs project in editable mode
- Verifies installation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 2. QUICK VALIDATION (30 SECONDS)

Verify everything works before diving deeper:

```bash
python experiments/quick_validation.py
```

**Expected Output:**
```
============================================
Quick Validation Suite
============================================

[PASS] Module imports  (120 ms)
[PASS] CPU Top-K compression  (45 ms)
[PASS] Error feedback buffer  (12 ms)
[PASS] SimpleCNN forward pass  (18 ms)
[PASS] Compressed training step  (230 ms)

============================================
✅ All checks passed!
============================================
```

If this passes, your environment is ready!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 3. BASIC USAGE

### 3.1 Baseline Training (No Compression)

```bash
python train.py --model simple_cnn --dataset mnist --epochs 5
```

**Expected:**
- Training time: ~60 seconds (CPU)
- Final accuracy: ~98.2%

### 3.2 Training with Compression

```bash
python train.py --model simple_cnn --dataset mnist \
    --epochs 5 --compress --ratio 0.01
```

**Expected:**
- Training time: ~62 seconds (minimal overhead)
- Final accuracy: ~97.9% (within 1%)
- Bandwidth saved: 97%

### 3.3 Run Test Suite

```bash
# Run all 22 tests
bash scripts/run_tests.sh

# Or use pytest directly
pytest tests/ -v
```

**Expected:**
```
tests/test_compression.py::test_topk_selects_largest PASSED
tests/test_compression.py::test_shape_preserved PASSED
...
tests/test_integration.py::test_accuracy_comparable PASSED

22 passed in 45.2s
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 4. ADVANCED USAGE

### 4.1 Multi-GPU Training

```bash
# 4 GPUs with NCCL backend
torchrun --nproc_per_node 4 train.py \
    --model resnet18 --dataset cifar10 \
    --epochs 50 --backend nccl \
    --compress --ratio 0.01 --batch-size 256
```

### 4.2 Benchmarks (macOS - Use Fixed Scripts)

```bash
# Compression throughput
python benchmark_compression_fixed.py

# Training speed & accuracy
python benchmark_training_fixed.py

# Or run all benchmarks
bash run_benchmarks_fixed.sh
```

### 4.3 Configuration Files

```bash
# Use YAML config instead of CLI args
python train.py --config configs/default.yaml
```

Edit `configs/default.yaml` to customize settings.

### 4.4 TensorBoard Monitoring

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir runs/

# Train with logging
python train.py --model simple_cnn --dataset mnist \
    --epochs 10 --compress --ratio 0.01

# View at http://localhost:6006
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 5. TROUBLESHOOTING

### 5.1 SSL Certificate Error (macOS)

**Error:**
```
RuntimeError: Error downloading train-images-idx3-ubyte.gz:
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Solutions:**

**Option 1: Manual Download (Recommended)**
```bash
bash download_mnist.sh
python train.py --model simple_cnn --dataset mnist --epochs 5
```

**Option 2: Use Fixed Training Script**
```bash
python train_fixed.py --model simple_cnn --dataset mnist --epochs 5
```

**Option 3: Fix Python Certificates**
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
# Then retry normal training
```

### 5.2 Multiprocessing Error (Python 3.13)

**Error:**
```
RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase.
```

**Solution:**
```bash
# Use fixed benchmark scripts
python benchmark_compression_fixed.py
python benchmark_training_fixed.py
bash run_benchmarks_fixed.sh
```

**Details:** See MULTIPROCESSING_FIX_GUIDE.md

### 5.3 CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Option 1: Reduce batch size
python train.py --batch-size 32 --device cuda

# Option 2: Use CPU
python train.py --device cpu

# Option 3: Use gradient accumulation (future feature)
```

### 5.4 Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Install in editable mode
pip install -e .

# Verify installation
python -c "import src; print('OK')"
```

### 5.5 Tests Failing

**Issue:** Some tests fail on first run

**Solution:**
```bash
# Ensure single-process mode (default)
pytest tests/ -v

# If still failing, check dependencies
pip install -r requirements.txt

# Run individual test files
pytest tests/test_compression.py -v
```

### 5.6 Slow Training on macOS

**Issue:** MPS (Metal) backend warnings

**Solution:**
```bash
# Use CPU explicitly (often faster for small models)
python train.py --device cpu

# Or ignore MPS warnings (they're harmless)
python train.py --device auto  # Auto-detects best device
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 6. COMMAND REFERENCE

### 6.1 Training Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--model` | simple_cnn | simple_cnn, resnet18, resnet50 | Model architecture |
| `--dataset` | mnist | mnist, cifar10 | Dataset to use |
| `--epochs` | 10 | int | Number of training epochs |
| `--batch-size` | 64 | int | Batch size per worker |
| `--lr` | 0.01 | float | Learning rate |
| `--compress` | False | flag | Enable compression |
| `--ratio` | 0.01 | 0.001-1.0 | Compression ratio ρ |
| `--no-error-feedback` | False | flag | Disable error feedback |
| `--backend` | gloo | gloo, nccl | Distributed backend |
| `--device` | auto | auto, cpu, cuda, mps | Device to use |
| `--seed` | 42 | int | Random seed |

### 6.2 Example Commands

**CPU Training:**
```bash
python train.py --model simple_cnn --dataset mnist \
    --epochs 10 --device cpu
```

**GPU Training with Compression:**
```bash
python train.py --model resnet18 --dataset cifar10 \
    --epochs 50 --device cuda --compress --ratio 0.01
```

**Aggressive Compression:**
```bash
python train.py --model simple_cnn --dataset mnist \
    --epochs 10 --compress --ratio 0.001  # 99.7% savings
```

**Without Error Feedback (comparison):**
```bash
python train.py --model simple_cnn --dataset mnist \
    --epochs 10 --compress --ratio 0.01 --no-error-feedback
# Expected: Poor convergence (demonstrates EF importance)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 7. PERFORMANCE EXPECTATIONS

### 7.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| GPU | None (CPU works) | CUDA-capable |
| Disk | 1GB | 5GB+ |

### 7.2 Training Time Estimates

**SimpleCNN on MNIST (10 epochs):**
- CPU (M1 Mac): ~2 minutes
- CPU (Intel i5): ~5 minutes
- GPU (RTX 3080): ~30 seconds

**ResNet-18 on CIFAR-10 (50 epochs):**
- CPU: ~2 hours
- GPU (RTX 3080): ~15 minutes

### 7.3 Expected Accuracies

| Model | Dataset | Epochs | Baseline | Compressed (ρ=0.01) |
|-------|---------|--------|----------|---------------------|
| SimpleCNN | MNIST | 10 | 98.2% | 97.9% (-0.3pp) |
| ResNet-18 | CIFAR-10 | 50 | 92.5% | 91.8% (-0.7pp) |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 8. GETTING HELP

### 8.1 Documentation

- **This guide:** Quick start and troubleshooting
- **COMPLETE_ASSIGNMENT_SOLUTION.md:** Comprehensive report
- **IMPLEMENTATION_GUIDE.md:** Technical deep-dive
- **CODE_MAPPING_GUIDE.md:** Theory → code mapping
- **compressed-ddp/docs/:** P0-P3 technical documentation

### 8.2 Common Questions

**Q: Do I need a GPU?**
A: No! CPU works fine for MNIST. GPU recommended for CIFAR-10.

**Q: How long does setup take?**
A: ~2-3 minutes (downloads PyTorch and dependencies).

**Q: Can I run on Windows?**
A: Yes, but use `venv\Scripts\activate` and avoid shell scripts.

**Q: What Python version?**
A: 3.9+ required. Tested on 3.9, 3.10, 3.11, 3.13.

**Q: Do tests require GPU?**
A: No, all tests run on CPU.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 9. NEXT STEPS

After completing quick start:

1. ✅ Read EXECUTIVE_SUMMARY.md (5 minutes)
2. ✅ Review COMPLETE_ASSIGNMENT_SOLUTION.md (30 minutes)
3. ✅ Explore compressed-ddp/docs/ (P0-P3 documentation)
4. ✅ Run benchmarks to verify performance
5. ✅ Experiment with different models/datasets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Status:** Ready to use ✅

For detailed technical information, see IMPLEMENTATION_GUIDE.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
