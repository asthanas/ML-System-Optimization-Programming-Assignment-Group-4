# Compressed-DDP
### Communication-Efficient Distributed Deep Learning via Top-K Gradient Compression

---

## Quick Start

```bash
# 1. Unzip and enter
unzip compressed-ddp.zip && cd compressed-ddp

# 2. One-command setup
bash setup.sh && source venv/bin/activate

# 3. Smoke test (no downloads, ~30 s)
python experiments/quick_validation.py

# 4. Train baseline
python train.py --model simple_cnn --dataset mnist --epochs 5

# 5. Train with 1% gradient compression
python train.py --model simple_cnn --dataset mnist --epochs 5 --compress --ratio 0.01

# 6. Full test suite
bash scripts/run_tests.sh

# 7. Benchmarks
bash scripts/run_benchmarks.sh
```

---

## Phase Overview

| Phase | Document | Description |
|-------|----------|-------------|
| P0 | `docs/p0_problem.md` | Problem formulation, expectations, metrics |
| P1 | `docs/p1_design.md` | System design, architecture, algorithms |
| P1r | `docs/p1r_revised_design.md` | Revised design, platform choices, implementation plan |
| P2 | Source in `src/`, `train.py` | Full implementation |
| P3 | `docs/p3_analysis.md` | Test results, benchmarks, deviation analysis |

---

## Project Structure

```
compressed-ddp/
├── src/
│   ├── compression/        # Top-K compressor (GPU + CPU)
│   ├── error_feedback/     # Error residual buffer
│   ├── communication/      # DistributedBackend (NCCL/Gloo)
│   ├── models/             # SimpleCNN, ResNet-18/50
│   ├── data/               # MNIST, CIFAR-10 loaders
│   ├── metrics/            # MetricsTracker + TensorBoard
│   └── utils/              # Device detect, config, checkpoints
├── tests/                  # 22 unit + integration tests
├── experiments/            # Benchmarks, validation scripts
├── docs/                   # P0–P3 phase documents
├── configs/                # YAML config files
├── scripts/                # Shell helpers
├── train.py                # Main training entry-point
├── setup.sh                # One-command install
└── requirements.txt
```

---

## Key Command-Line Flags

| Flag | Default | Options | Description |
|------|---------|---------|-------------|
| `--model` | `simple_cnn` | `simple_cnn`, `resnet18`, `resnet50` | Model architecture |
| `--dataset` | `mnist` | `mnist`, `cifar10` | Dataset |
| `--epochs` | `10` | any int | Training epochs |
| `--batch-size` | `64` | any int | Per-process batch size |
| `--lr` | `0.01` | float | Learning rate |
| `--compress` | off | flag | Enable Top-K compression |
| `--ratio` | `0.01` | 0.001–1.0 | Fraction of gradients to transmit |
| `--no-error-feedback` | off | flag | Disable error feedback (not recommended) |
| `--backend` | `gloo` | `gloo`, `nccl` | Distributed backend |
| `--device` | `auto` | `auto`, `cpu`, `cuda` | Compute device |

---

## Multi-GPU Training

```bash
# 4 GPUs, NCCL backend
torchrun --nproc_per_node 4 train.py \
    --model resnet18 --dataset cifar10 --epochs 50 \
    --backend nccl --compress --ratio 0.01 --batch-size 256
```

---

## Requirements

- Python 3.9+
- PyTorch 2.1+
- torchvision 0.16+
- numpy, pyyaml, tqdm, matplotlib, tensorboard
- CUDA 11.8+ (optional, for GPU mode)

---

## References

1. Seide et al. (2014). *1-Bit SGD*. Microsoft Research.
2. Karimireddy et al. (2019). *Error Feedback Fixes SignSGD*. ICML.
3. Lin et al. (2018). *Deep Gradient Compression*. ICLR.
4. Li et al. (2020). *PyTorch Distributed*. VLDB.
