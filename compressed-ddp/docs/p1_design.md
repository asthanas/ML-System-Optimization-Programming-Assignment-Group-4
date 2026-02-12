# P1 — System Design

## 1. Design Goals

1. **Correctness**: Compressed training must converge to the same accuracy
   as uncompressed training (within 1%) on standard benchmarks.
2. **Modularity**: Compression, communication, and training components
   are independently replaceable.
3. **Platform-agnostic**: Runs on CPU (for testing) and GPU (for performance).
4. **Reproducibility**: Deterministic seeds, checkpoint/resume support.
5. **Extensibility**: Easy to add new compressors (e.g., random-K, 1-bit).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Loop                              │
│                                                                  │
│  ┌──────────┐   forward   ┌──────────┐   backward  ┌─────────┐  │
│  │  Model   │────────────▶│  Loss   │────────────▶│  Grad   │  │
│  └──────────┘             └──────────┘             └────┬────┘  │
│                                                         │       │
│                    ┌────────────────────────────────────▼────┐  │
│                    │         DistributedBackend               │  │
│                    │                                          │  │
│                    │  ┌──────────────┐   ┌────────────────┐  │  │
│                    │  │ ErrorFeedback│   │  Compressor    │  │  │
│                    │  │   Buffer     │──▶│  (Top-K)       │  │  │
│                    │  └──────────────┘   └───────┬────────┘  │  │
│                    │                             │           │  │
│                    │                    ┌────────▼────────┐  │  │
│                    │                    │   AllReduce     │  │  │
│                    │                    │  (NCCL/Gloo)   │  │  │
│                    │                    └────────┬────────┘  │  │
│                    └─────────────────────────────┼───────────┘  │
│                                                  │              │
│  ┌──────────┐                          ┌─────────▼─────────┐   │
│  │Optimizer │◀─────────────────────────│  Averaged Grad ḡ  │   │
│  └──────────┘                          └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Compression Module (`src/compression/`)

**Interface** (Abstract Base Class):
```python
class BaseCompressor:
    def compress(tensor) -> (values, indices, shape)
    def decompress(values, indices, shape) -> tensor
```

**Implementations:**
- `TopKCompressorGPU` — uses `torch.topk` on CUDA tensors (O(n) average)
- `TopKCompressorCPU` — uses `numpy.argpartition` (O(n) average, no CUDA)
- `get_compressor(ratio, device)` — factory; auto-selects based on device

**Algorithm complexity:**
| Operation | Time | Space |
|-----------|------|-------|
| compress  | O(n) avg | O(k) |
| decompress| O(k)     | O(n) |

**Compression ratio:**
```
Bytes_out / Bytes_in = k·(4+8) / (n·4) = 3ρ
```
(float32 value + int64 index per selected element)

### 3.2 Error Feedback Module (`src/error_feedback/`)

Maintains per-parameter residual buffer `e` such that:
```
Step t:
    compensated = gradient + e_{t-1}
    compressed  = Compress(compensated)
    e_t         = compensated - Decompress(compressed)
    send        = compressed
```

**Memory cost:** One float32 tensor per parameter → same as the model itself.

**Checkpoint support:** `save()` / `load()` persist residuals across restarts.

### 3.3 Communication Module (`src/communication/`)

`DistributedBackend` wraps `torch.distributed`:
- **Uncompressed path:** `dist.all_reduce(grad)` + divide by world_size
- **Compressed path:** error-compensate → compress → allreduce → update error
- **Broadcast:** sync model parameters from rank 0 at training start

Backends:
- `gloo` — CPU and cross-platform (used for correctness testing)
- `nccl` — GPU-optimised (used for performance)

### 3.4 Training Pipeline (`train.py`)

```
parse_args()
  ↓
setup_distributed()
  ↓
build model, compressor, error_buffer, backend, dataloaders
  ↓
for epoch in range(epochs):
    for batch in train_loader:
        forward → loss → backward
        backend.allreduce_gradients()   ← compression here
        optimizer.step()
    validate()
    save_checkpoint()
```

---

## 4. Algorithm: Top-K with Error Feedback (Pseudocode)

```
Initialise: θ₀, e₀ = 0, η (learning rate)

for t = 1, 2, ..., T:
    [each worker w in parallel]
    g_w = stochastic_gradient(θ_{t-1}, batch_w)

    for each parameter p:
        # Error compensation
        ẽ_p = g_p + e_{p,t-1}

        # Top-K selection
        k    = ceil(ρ · |ẽ_p|)
        vals = top-k-values(|ẽ_p|)
        idx  = top-k-indices(|ẽ_p|)

        # Allreduce (ring, O(N·k/P) bytes per worker)
        ḡ_p = AllReduce(Decompress(vals, idx)) / P

        # Error update
        e_{p,t} = ẽ_p - ḡ_p

    # SGD update
    θ_t = θ_{t-1} - η · [ḡ_p for all p]
```

---

## 5. Performance Model

### Compute time per step
```
T_compute = T_forward + T_backward = C  (model-dependent constant)
```

### Communication time per step (ring-allreduce)
```
T_comm = 2(P-1)/P · k·12 / B
       ≈ 2 · N·ρ·12 / B        (for large P)
```

### Total time and efficiency
```
T_total = T_compute / P + T_comm
E(P)    = (T_compute/P) / T_total
        = 1 / (1 + P · T_comm / T_compute)
```

### Projected efficiency (ResNet-18, V100, P=8)

| Network | ρ=1.0 (baseline) | ρ=0.01 (compressed) |
|---------|-----------------|---------------------|
| 1 Gbps  | E=0.13 (13%)    | E=0.82 (82%)        |
| 10 Gbps | E=0.59 (59%)    | E=0.98 (98%)        |
| NVLink  | E=0.99 (99%)    | E=1.00 (100%)       |

---

## 6. Expected Accuracy vs Compression Trade-off

Based on published results (Lin et al. 2018, Aji & Heafield 2017):

| ρ     | Expected accuracy (MNIST/ResNet) | Epochs to convergence |
|-------|----------------------------------|-----------------------|
| 1.0   | 99.2% (baseline)                 | 10                    |
| 0.1   | 99.0% (−0.2%)                    | 11                    |
| 0.01  | 98.8% (−0.4%)                    | 13                    |
| 0.001 | 97.5% (−1.7%)                    | 20+                   |

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Divergence at low ρ | Low | High | Error feedback guarantees convergence |
| GPU OOM for error buffer | Medium | Medium | Buffer shares dtype, same size as model |
| CPU too slow for benchmark | High | Low | GPU target only; CPU for correctness |
| NCCL version mismatch | Medium | Medium | Gloo fallback always available |
