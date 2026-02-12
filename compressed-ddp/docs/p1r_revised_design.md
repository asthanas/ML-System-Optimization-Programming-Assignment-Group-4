# P1 (Revised) — Revised Design & Implementation Plan

## 1. Revisions from Initial Design

| Aspect | P1 (Initial) | P1r (Revised) |
|--------|-------------|---------------|
| Sparse AllReduce | Planned true sparse exchange | Simplified: decompress→allreduce→recompress (correct semantics, easier testing) |
| Compressor API | Single compress() | + stats tracking, reset_stats() |
| Error buffer | In-memory only | + save/load for checkpoint support |
| Config | Command-line only | + YAML config files |
| Logging | print() | Python logging + TensorBoard |
| Single-process support | Dist init required | world_size=1 skips dist entirely |

### Why True Sparse AllReduce Was Deferred

True sparse AllReduce (transmit k·12 bytes instead of N·4 bytes) requires
a custom MPI/NCCL primitive. PyTorch does not provide this out-of-the-box.
Instead, the current implementation demonstrates *correct gradient semantics*
(compression + error feedback work correctly) while leaving the low-level
transport optimisation as a production extension.

The accuracy, convergence, and correctness tests are unaffected by this
choice. Only the actual bytes-on-wire measurement differs from the
theoretical model.

---

## 2. Execution Platform

### 2.1 Minimum Requirements (CPU mode)
| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Linux, macOS, Windows/WSL | Ubuntu 22.04, macOS 13 |
| Python | 3.9 | 3.9, 3.11 |
| PyTorch | 2.1.0 | 2.1.0, 2.2.0 |
| RAM | 4 GB | 8 GB |

### 2.2 Recommended (GPU mode)
| Component | Recommended |
|-----------|-------------|
| GPU | NVIDIA V100 / A100 / RTX 3090+ |
| CUDA | 11.8+ |
| VRAM | 8 GB+ |
| Network | 10 Gbps Ethernet or NVLink |
| Distributed | torchrun (PyTorch ≥ 1.10) |

### 2.3 Development Environment
```
compressed-ddp/
  venv/            ← Python virtual environment (created by setup.sh)
  data/            ← Auto-downloaded datasets (MNIST, CIFAR-10)
  checkpoints/     ← Saved model checkpoints
  runs/            ← TensorBoard event files
  experiments/results/  ← Benchmark CSV + PNG outputs
```

---

## 3. Implementation Architecture

### 3.1 Module Dependency Graph

```
train.py
  ├── src.models.factory          (get_model)
  ├── src.data.loaders            (get_dataloaders)
  ├── src.compression.factory     (get_compressor)
  │     ├── src.compression.topk_gpu   (CUDA path)
  │     └── src.compression.topk_cpu   (CPU path)
  ├── src.error_feedback.buffer   (ErrorFeedbackBuffer)
  ├── src.communication.backend   (DistributedBackend)
  │     ├── src.compression.*
  │     └── src.error_feedback.*
  ├── src.metrics.tracker         (MetricsTracker)
  └── src.utils.*                 (device, config, checkpoint)
```

### 3.2 Data Flow

```
Raw gradient g_t (shape: same as parameter)
       │
       ▼
ErrorFeedbackBuffer.compensate()
       │  ẽ_t = g_t + e_{t-1}
       ▼
TopKCompressor.compress()
       │  (values[k], indices[k], original_shape)
       ▼
TopKCompressor.decompress()
       │  ĝ_t (dense, N elements, k non-zero)
       ▼
torch.distributed.all_reduce()
       │  ḡ_t = (1/P) Σ ĝ_t^w
       ▼
ErrorFeedbackBuffer.update()
       │  e_t = ẽ_t - ḡ_t
       ▼
param.grad.copy_(ḡ_t)
       │
       ▼
optimizer.step()
```

---

## 4. Algorithm Specifications

### 4.1 Top-K Compression (CPU)

```python
def compress_cpu(tensor, ratio):
    flat = tensor.reshape(-1)
    n    = len(flat)
    k    = max(1, int(n * ratio))
    arr  = flat.numpy()

    # O(n) average using argpartition (not full sort)
    part = numpy.argpartition(abs(arr), n - k)[n - k:]
    idx  = part[numpy.argsort(abs(arr[part]))[::-1]]

    return arr[idx], idx.astype(int64), tensor.shape
```

### 4.2 Top-K Compression (GPU)

```python
def compress_gpu(tensor, ratio):
    flat     = tensor.reshape(-1)
    k        = max(1, int(len(flat) * ratio))
    _, idx   = torch.topk(flat.abs(), k, largest=True)
    values   = flat[idx]          # preserves sign
    return values, idx, tensor.shape
```

### 4.3 Error Feedback Update

```python
def allreduce_compressed(name, grad, compressor, error_buf, world_size):
    # 1. Add residual
    compensated = grad + error_buf[name]

    # 2. Compress
    values, indices, shape = compressor.compress(compensated)

    # 3. Decompress to dense (for allreduce)
    approx = compressor.decompress(values, indices, shape)

    # 4. AllReduce
    dist.all_reduce(approx)
    approx /= world_size

    # 5. Update residual
    error_buf[name] = compensated - approx

    # 6. Return averaged compressed gradient
    return approx
```

---

## 5. Testing Strategy

### 5.1 Unit Tests
- **Compression:** correctness of Top-K selection, shape preservation,
  sign preservation, edge cases (zeros, single spike, full ratio)
- **Error Feedback:** accumulation, convergence (normalised metric),
  checkpoint roundtrip, buffer isolation

### 5.2 Integration Tests
- Full training loop convergence (loss decreases over 50 steps)
- Compressed vs baseline accuracy within 15 pp after 100 steps

### 5.3 Performance Benchmarks
- `experiments/benchmark_compression.py` — throughput vs tensor size
- `experiments/benchmark_training.py`    — speed vs accuracy table
- `experiments/scalability_analysis.py`  — theoretical E(P) curves

---

## 6. Configuration Management

YAML files in `configs/`:
```yaml
# configs/default.yaml
training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.01

compression:
  enabled: true
  method: topk
  ratio: 0.01
  error_feedback: true

distributed:
  backend: gloo
  world_size: 1
```

Overridable at runtime via command-line flags (`--ratio`, `--epochs`, etc.).

---

## 7. Deployment Considerations

### Single Machine, Multiple GPUs
```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py \
    --backend nccl --compress --ratio 0.01
```

### Multi-Node Cluster
```bash
# Node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 train.py \
    --backend nccl --compress --ratio 0.01

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 train.py \
    --backend nccl --compress --ratio 0.01
```

### Checkpoint Resume
```bash
python train.py --resume checkpoints/simple_cnn_mnist/best_model.pt
```
