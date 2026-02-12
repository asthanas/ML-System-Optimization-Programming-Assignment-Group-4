# Quick Start Guide

**Getting up and running with Compressed-DDP**

---

## Before You Start

You'll need:
- Python 3.9 or newer (I'm using 3.13)
- pip for installing packages
- About 1GB of disk space (for datasets)
- 4GB of RAM (more is better)
- 5-10 minutes for setup

That's it! No GPU required (though it helps for bigger models).

---

## Installation

The automated setup makes this easy:

```bash
# 1. clone the github repo
git clone https://github.com/asthanas/ML-System-Optimization-Programming-Assignment-Group-4.git
cd mlsysops-assignment

# 2. Run setup (creates virtual env, installs dependencies)
bash setup.sh

# 3. Activate the environment
source venv/bin/activate
```

The setup script will:
- Check your Python version
- Create a virtual environment
- Install PyTorch and other dependencies
- Verify everything works

This takes about 2-3 minutes on a decent internet connection.

### Windows Users

If you're on Windows, use `setup.bat` instead of `setup.sh`:

```cmd
setup.bat
venv\Scripts\activate
```

Or use the universal Python script:

```bash
python setup.py
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS
```

---

## Quick Validation

Before doing anything else, make sure everything works:

```bash
python experiments/quick_validation.py
```

You should see:

```
[PASS] Module imports  (120 ms)
[PASS] CPU Top-K compression  (45 ms)
[PASS] Error feedback buffer  (12 ms)
[PASS] SimpleCNN forward pass  (18 ms)
[PASS] Compressed training step  (230 ms)

All checks passed ✅
```

If you see this, you're good to go! If not, check the troubleshooting section below.

---

## Basic Usage

### Training Without Compression (Baseline)

Let's start simple - train a model the normal way:

```bash
python train.py --model simple_cnn --dataset mnist --epochs 5
```

This will:
- Download MNIST (if needed)
- Train for 5 epochs
- Show training progress
- Reach about 98.2% accuracy

Takes about 5 minutes on CPU.

### Training With Compression

Now let's add compression:

```bash
python train.py --model simple_cnn --dataset mnist \
    --epochs 5 --compress --ratio 0.01
```

Same model, but now:
- Compresses gradients to 1% of original size
- Uses error feedback for convergence
- Saves 97% of bandwidth
- Reaches about 97.9% accuracy (only 0.3pp lower!)

The training time is similar because we're on a single machine. The benefits show up when you're actually doing distributed training across multiple GPUs.

---

## Running Tests

Check that everything works correctly:

```bash
# All tests
bash scripts/run_tests.sh

# Or use pytest directly
pytest tests/ -v

# Or run specific test files
pytest tests/test_compression.py -v
```

All 22 tests should pass. Takes about 2 minutes.

---

## Benchmarks

Want to see performance numbers?

### Compression Speed

```bash
python experiments/benchmark_compression.py
```

This tests how fast the compression algorithm is at different sizes and ratios.

### Training Comparison

```bash
python experiments/benchmark_training.py
```

Compares training time and accuracy with and without compression.

**macOS/Python 3.13 Users:** Use the fixed versions instead:

```bash
python benchmark_compression_fixed.py
python benchmark_training_fixed.py

# Or run both:
bash run_all_benchmarks.sh
```

---

## Common Use Cases

### Different Models

```bash
# SimpleCNN (small, fast)
python train.py --model simple_cnn --dataset mnist --epochs 5

# ResNet-18 (medium)
python train.py --model resnet18 --dataset cifar10 --epochs 50

# ResNet-50 (large)
python train.py --model resnet50 --dataset cifar10 --epochs 50
```

### Different Compression Ratios

```bash
# Conservative (10%)
python train.py --compress --ratio 0.1 --epochs 5

# Sweet spot (1%) - recommended
python train.py --compress --ratio 0.01 --epochs 5

# Aggressive (0.1%)
python train.py --compress --ratio 0.001 --epochs 5
```

Lower ratios save more bandwidth but may impact accuracy more.

### Multi-GPU Training

If you have multiple GPUs:

```bash
# 4 GPUs with NCCL backend
torchrun --nproc_per_node 4 train.py \
    --model resnet18 --dataset cifar10 \
    --epochs 50 --backend nccl \
    --compress --ratio 0.01
```

This is where compression really shines!

---

## Monitoring with TensorBoard

Want to see training progress visually?

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir runs/

# Train with logging
python train.py --model simple_cnn --dataset mnist --epochs 10

# Open http://localhost:6006 in your browser
```

You'll see:
- Training and validation accuracy
- Loss curves
- Compression statistics (if using compression)

---

## Troubleshooting

### SSL Certificate Error (macOS)

**Problem:** MNIST download fails with SSL errors

**Solution 1** (easiest):
```bash
bash download_mnist.sh
python train.py --model simple_cnn --dataset mnist --epochs 5
```

**Solution 2** (permanent fix):
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Multiprocessing Error (Python 3.13)

**Problem:** Benchmark scripts fail with multiprocessing errors

**Solution:**
```bash
# Use the fixed versions
python benchmark_compression_fixed.py
python benchmark_training_fixed.py
```

See `MULTIPROCESSING_FIX_GUIDE.md` for details.

### CUDA Out of Memory

**Problem:** GPU runs out of memory

**Solutions:**
```bash
# Reduce batch size
python train.py --batch-size 32

# Or use CPU
python train.py --device cpu
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the right directory
cd compressed-ddp

# Reinstall in editable mode
pip install -e .

# Activate virtual environment if you haven't
source venv/bin/activate
```

### MPS pin_memory Warning (macOS)

**Problem:** Warning about pin_memory not supported on MPS

**Solution:** This is harmless! You can ignore it or suppress it:
```bash
python -W ignore::UserWarning train.py --model simple_cnn --dataset mnist --epochs 5
```

---

## Command Reference

### Training Arguments

```bash
python train.py \
    --model simple_cnn        # Model architecture
    --dataset mnist           # Dataset to use
    --epochs 10               # Number of epochs
    --batch-size 64           # Batch size
    --lr 0.01                 # Learning rate
    --compress                # Enable compression
    --ratio 0.01              # Compression ratio (1%)
    --backend gloo            # Distributed backend
    --device auto             # Device (auto/cpu/cuda/mps)
    --seed 42                 # Random seed
```

### Common Combinations

**Quick test:**
```bash
python train.py --model simple_cnn --dataset mnist --epochs 3
```

**Full MNIST run:**
```bash
python train.py --model simple_cnn --dataset mnist --epochs 10 --compress --ratio 0.01
```

**CIFAR-10 baseline:**
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50
```

**CIFAR-10 with compression:**
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 50 --compress --ratio 0.01
```

---

## Expected Results

### MNIST (SimpleCNN, 10 epochs)

**Baseline (no compression):**
- Accuracy: ~98.2%
- Time: ~5 minutes (CPU)

**Compressed (1% ratio):**
- Accuracy: ~97.9% (0.3pp lower)
- Time: ~5 minutes (CPU)
- Bandwidth: 97% saved

### CIFAR-10 (ResNet-18, 50 epochs)

**Baseline:**
- Accuracy: ~92.5%
- Time: ~2 hours (CPU), ~15 min (GPU)

**Compressed (1% ratio):**
- Accuracy: ~91.8% (0.7pp lower)
- Time: Similar (compression overhead minimal)
- Bandwidth: 97% saved

---

## Tips and Tricks

**Use compression ratio 0.01 (1%)** - This is the sweet spot. 97% bandwidth savings with minimal accuracy impact.

**Start with quick_validation.py** - Always run this first to make sure your environment is set up correctly.

**Check TensorBoard** - It's really helpful to visualize what's happening during training.

**Use the fixed benchmarks** - If you're on macOS with Python 3.13, use the `*_fixed.py` versions to avoid multiprocessing headaches.

**Read the logs** - The training script outputs useful info about compression ratios and bandwidth saved.

---

## Next Steps

Once you've got things running:

1. Try different compression ratios and see how they affect accuracy
2. Compare baseline vs compressed training side-by-side
3. Run the benchmarks to measure performance
4. Check out the source code to see how it works
5. Read the full documentation in `docs/` for technical details

---

## Getting Help

**Documentation:**
- This guide - setup and basic usage
- COMPLETE_ASSIGNMENT_SOLUTION.md - full technical write-up
- IMPLEMENTATION_GUIDE.md - code architecture deep dive
- docs/ folder - detailed P0-P3 documentation

**Common Issues:**
- SSL certificates: Use download_mnist.sh
- Multiprocessing: Use *_fixed.py benchmark scripts  
- Import errors: Run `pip install -e .`
- CUDA OOM: Reduce batch size or use CPU

---

That's it! You should now be able to run everything. If you hit any issues not covered here, check the other documentation files or the troubleshooting section.

Happy training! 🚀
