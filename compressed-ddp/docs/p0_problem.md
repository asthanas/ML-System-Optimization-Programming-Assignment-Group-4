# P0 — Problem Formulation

## 1. Background: Data Parallelization in Distributed Training

Modern deep learning training is embarrassingly parallelisable across data:
split a mini-batch across P workers, each computes gradients on its shard,
then aggregate via AllReduce before the optimizer step.

```
Worker 0: ∇L(θ; x₀)  ─┐
Worker 1: ∇L(θ; x₁)  ─┤──▶ AllReduce (avg) ──▶ θ ← θ − η·ḡ
Worker 2: ∇L(θ; x₂)  ─┤
Worker P: ∇L(θ; x_P) ─┘
```

This is **synchronous SGD (S-SGD)**, which converges identically to
single-machine SGD when the effective batch size is B·P.

---

## 2. The Communication Bottleneck

### 2.1 Communication Volume

Each AllReduce transmits 2(P−1)/P × N × 4 bytes ≈ 2N bytes per step
(ring-allreduce), where N = number of model parameters.

| Model | Params (N) | Gradient size | @ 1 Gbps | @ 10 Gbps |
|-------|-----------|---------------|----------|-----------|
| SimpleCNN | 200 K | 0.8 MB | 6.4 ms | 0.64 ms |
| ResNet-18 | 11 M | 44 MB | 352 ms | 35.2 ms |
| ResNet-50 | 23 M | 92 MB | 736 ms | 73.6 ms |
| GPT-2 | 117 M | 468 MB | 3744 ms | 374 ms |

**A single V100 forward+backward on ResNet-50 takes ~50 ms.**
At 1 Gbps, communication (736 ms) is **14× the compute time** → 93% idle.

### 2.2 Scaling Efficiency

Define efficiency as:
```
E(P) = Ideal_speedup / Actual_speedup
     = (T_compute / P) / (T_compute / P + T_comm)
     = 1 / (1 + P · T_comm / T_compute)
```

For ResNet-50 at 1 Gbps, T_compute = 50 ms, T_comm = 736 ms:
- E(2) = 0.06  (6% efficiency)  — 94% time spent waiting for network
- E(8) = 0.02  (2% efficiency)  — essentially no benefit from 8 GPUs

---

## 3. Problem Statement

**Given:** A synchronous data-parallel training system with P workers
connected by a network of bandwidth B Gbps, training a model with N
parameters (N ≫ 10⁶).

**Problem:** AllReduce communication dominates training time, making
multi-GPU efficiency E(P) unacceptably low on commodity networks.

**Goal:** Reduce gradient communication volume by 10–100× while preserving
model convergence guarantees and final accuracy within 1% of full-precision
baseline.

---

## 4. Proposed Solution: Top-K Gradient Compression

Transmit only the k = ⌈ρ·N⌉ gradient components with the largest absolute
values (Top-K sparsification), where ρ ∈ (0, 1] is the *compression ratio*.

```
Compress(g) = { (g_i, i) : i ∈ argmax_k |g| }   [sparse, k entries]
```

Combined with **error feedback** to correct the compression bias:
```
ẽ_t    = g_t + e_{t−1}          [compensated gradient]
g̃_t    = Compress(ẽ_t)          [compressed]
e_t    = ẽ_t − Decompress(g̃_t) [residual saved for next step]
```

---

## 5. Quantitative Expectations

### 5.1 Communication Reduction

```
Bytes_compressed   k × (4 + 8)    ρ × 12
────────────────── = ──────────── = ────────
Bytes_baseline     N × 4          4

Reduction ratio = 4 / (12ρ) = 1 / (3ρ)
```

| ρ (ratio) | k/N | Reduction |
|-----------|-----|-----------|
| 0.1       | 10% | 3.3×      |
| 0.01      | 1%  | 33×       |
| 0.001     | 0.1%| 333×      |

### 5.2 Expected Speedup (ResNet-18, 10 Gbps, P=8)

```
T_comm_baseline    = 2 × 44 MB / (10 Gbps / 8) = 352 ms
T_comm_compressed  = T_comm_baseline / 33       = 10.7 ms
T_compute          = 500 ms (ResNet-18 on V100)

E_baseline   (P=8) = (500/8) / (500/8 + 352)  = 0.15 = 15%
E_compressed (P=8) = (500/8) / (500/8 + 10.7) = 0.85 = 85%  ← target
```

### 5.3 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Communication reduction | ≥ 33× at ρ=0.01 | Formula: 1/(3ρ) |
| Multi-GPU efficiency E(8) @ 10 Gbps | ≥ 70% | Practical DGX cluster |
| Accuracy degradation at ρ=0.01 | < 1% vs baseline | Acceptable for production |
| Compression overhead (GPU) | < 5 ms for 25M params | Negligible vs compute |
| Convergence | Same order as SGD | Error feedback guarantee |

---

## 6. Theoretical Guarantee

**Theorem** (Karimireddy et al. 2019, simplified):
For any δ-contractive compressor C (Top-K satisfies this with δ = 1−ρ)
and error feedback, the sequence of iterates satisfies:

```
(1/T) Σ E[‖∇L(θ_t)‖²] ≤ O(1/√T) + O(δ·σ²/T)
```

where σ² is the gradient variance. As T→∞ and ρ→1, this matches the SGD
convergence rate O(1/√T) exactly. For small ρ, convergence is slowed by
the O(δ·σ²/T) term, which vanishes as T grows.

**Practical implication:** With error feedback, Top-K compression does NOT
diverge and achieves the same final accuracy as SGD given sufficient epochs.
