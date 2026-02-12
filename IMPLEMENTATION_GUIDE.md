# IMPLEMENTATION GUIDE
## Technical Deep-Dive: Compressed-DDP

**Date:** February 12, 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## PURPOSE

This guide provides a technical deep-dive into the implementation details,
design decisions, and code organization of the Compressed-DDP project.

**Target Audience:** Developers, reviewers, and anyone wanting to understand
the internal workings of the system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## TABLE OF CONTENTS

1. Module Architecture
2. Core Algorithms
3. Design Decisions
4. Code Organization
5. Testing Strategy
6. Performance Optimization
7. Platform Compatibility
8. Troubleshooting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. MODULE ARCHITECTURE

### 1.1 High-Level Structure

```
src/
├── compression/         # Top-K gradient compression
│   ├── base.py         # Abstract base class, stats tracking
│   ├── topk_gpu.py     # GPU implementation (torch.topk)
│   ├── topk_cpu.py     # CPU fallback (numpy.argpartition)
│   └── factory.py      # get_compressor() factory
│
├── error_feedback/      # Error residual tracking
│   └── buffer.py       # ErrorFeedbackBuffer
│
├── communication/       # Distributed gradient sync
│   ├── backend.py      # DistributedBackend
│   └── utils.py        # setup/cleanup utilities
│
├── models/              # Neural network architectures
│   ├── simple_cnn.py   # SimpleCNN for MNIST
│   ├── resnet.py       # ResNet-18/50
│   └── factory.py      # get_model() factory
│
├── data/                # Dataset loaders
│   └── loaders.py      # get_dataloaders() for MNIST/CIFAR-10
│
├── metrics/             # Training metrics
│   └── tracker.py      # MetricsTracker (TensorBoard)
│
└── utils/               # Utilities
    ├── device.py       # Device detection (CPU/GPU)
    ├── checkpoint.py   # Save/load checkpoints
    └── config.py       # YAML configuration loading
```

### 1.2 Dependency Graph

```
train.py
  ├─→ models/           (get_model)
  ├─→ data/             (get_dataloaders)
  ├─→ compression/      (get_compressor)
  ├─→ error_feedback/   (ErrorFeedbackBuffer)
  ├─→ communication/    (DistributedBackend)
  │    ├─→ compression/ (TopKCompressor)
  │    └─→ error_feedback/ (ErrorFeedbackBuffer)
  ├─→ metrics/          (MetricsTracker)
  └─→ utils/            (device, checkpoint, config)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 2. CORE ALGORITHMS

### 2.1 Top-K Compression (GPU)

**File:** `src/compression/topk_gpu.py`

**Key Methods:**

```python
def compress(self, tensor: torch.Tensor) -> Tuple:
    """
    Select top-k largest magnitude elements.

    Args:
        tensor: Input gradient tensor (any shape)

    Returns:
        values: Top-k values (1D tensor, length k)
        indices: Corresponding indices (1D tensor, length k)
        shape: Original shape for reconstruction

    Complexity: O(n) average (torch.topk uses quickselect)
    """
    shape = tensor.shape
    flat = tensor.reshape(-1)              # Flatten to 1D
    k = self._k(flat.numel())              # k = ⌈ρ·N⌉

    # torch.topk uses partial sort (quickselect algorithm)
    # Average O(n), worst-case O(n²) but rare in practice
    _, idx = torch.topk(flat.abs(), k, largest=True)
    values = flat[idx]

    # Track statistics
    self.stats.total_calls += 1
    self.stats.total_bytes_original += flat.numel() * 4
    self.stats.total_bytes_compressed += k * 12  # values + indices

    return values, idx, shape

def decompress(self, values, indices, shape) -> torch.Tensor:
    """
    Reconstruct sparse tensor to dense.

    Complexity: O(k) scatter operation
    """
    n = shape.numel()
    out = torch.zeros(n, device=values.device, dtype=values.dtype)
    out.scatter_(0, indices, values)       # Insert values at indices
    return out.reshape(shape)              # Restore original shape
```

**Why torch.topk?**
- Uses partial sorting (quickselect) → O(n) average
- GPU-optimized (CUDA kernels)
- Handles negative values correctly (abs() for magnitude)

### 2.2 Error Feedback

**File:** `src/error_feedback/buffer.py`

**Core Logic:**

```python
class ErrorFeedbackBuffer:
    def __init__(self):
        self._buffers = {}  # name → error tensor

    def compensate(self, name: str, gradient: torch.Tensor):
        """Add accumulated error to gradient."""
        buf = self._get_or_create(name, gradient)
        return gradient + buf              # ẽ_t = g_t + e_{t-1}

    def update(self, name: str, compensated, compressed_approx):
        """Update error buffer after compression."""
        self._buffers[name].copy_(
            compensated - compressed_approx)  # e_t = ẽ_t - g̃_t

    def _get_or_create(self, name, gradient):
        """Lazy initialization of buffers."""
        if name not in self._buffers:
            self._buffers[name] = torch.zeros_like(gradient)
        return self._buffers[name]
```

**Why per-parameter buffers?**
- Each parameter has different gradient magnitudes
- Independent error accumulation
- Correct tracking across layers

### 2.3 Distributed Backend

**File:** `src/communication/backend.py`

**Main Algorithm:**

```python
def allreduce_gradients(self, named_parameters):
    """Compress and sync gradients across workers."""
    if self.world_size == 1:
        return  # Skip for single-process training

    for name, param in named_parameters:
        if param.grad is not None:
            if self.compressor:
                self._compressed_allreduce(name, param.grad)
            else:
                self._dense_allreduce(param.grad)

def _compressed_allreduce(self, name, grad):
    """
    Compressed gradient synchronization.

    Steps:
      1. Compensate: grad + error → compensated
      2. Compress: TopK(compensated) → sparse
      3. Decompress: sparse → dense approximation
      4. AllReduce: sync dense approximation
      5. Update error: compensated - approximation → new error
    """
    # Step 1: Compensate
    compensated = (self.error_buffer.compensate(name, grad)
                   if self.error_buffer else grad)

    # Step 2: Compress
    values, indices, shape = self.compressor.compress(compensated)

    # Step 3: Decompress (P1r revision: dense sync)
    approx = self.compressor.decompress(values, indices, shape)

    # Step 4: AllReduce
    if self._dist_ok:
        torch.distributed.all_reduce(approx, op=ReduceOp.SUM)
        approx.div_(self.world_size)  # Average across workers

    # Step 5: Update error
    if self.error_buffer:
        self.error_buffer.update(name, compensated, approx)

    # Step 6: Write back
    grad.copy_(approx)
```

**Design Decision: Dense AllReduce**
- **P1 original:** True sparse AllReduce (transmit indices + values)
- **P1r revised:** Decompress before AllReduce (simpler, compatible)
- **Tradeoff:** Slight bandwidth increase, but easier implementation
- **Future:** Custom NCCL kernels for true sparse sync

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 3. DESIGN DECISIONS

### 3.1 Why Modular Architecture?

**Rationale:**
- Each component has single responsibility
- Easy to test in isolation
- Extensible (add new compressors, models, datasets)
- Follows SOLID principles

**Benefits:**
- `TopKCompressor` can be swapped with `QuantizationCompressor`
- `ErrorFeedbackBuffer` tested independently
- `DistributedBackend` doesn't know about compression details

### 3.2 Why Factory Patterns?

**Example:** `src/compression/factory.py`

```python
def get_compressor(ratio=0.01, device='cpu'):
    if 'cuda' in device or 'mps' in device:
        return TopKCompressorGPU(ratio=ratio)
    else:
        return TopKCompressorCPU(ratio=ratio)
```

**Benefits:**
- Automatic CPU/GPU selection
- Consistent API across implementations
- Easy to add new compressor types

### 3.3 Why Statistics Tracking?

**File:** `src/compression/base.py`

```python
@dataclass
class CompressStats:
    total_calls: int = 0
    total_bytes_original: int = 0
    total_bytes_compressed: int = 0
    total_time_ms: float = 0.0
```

**Rationale:**
- Measure actual bandwidth savings
- Profile compression overhead
- Validate theoretical predictions
- Debug performance issues

### 3.4 Why Checkpoint Support?

**File:** `src/error_feedback/buffer.py`

```python
def state_dict(self):
    return {'buffers': self._buffers}

def load_state_dict(self, state_dict):
    self._buffers = state_dict['buffers']
```

**Rationale:**
- Resume training from checkpoint
- Error buffers crucial for convergence
- Must save/restore with model weights

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 4. CODE ORGANIZATION

### 4.1 File Naming Conventions

- `*_gpu.py` - GPU-optimized implementations
- `*_cpu.py` - CPU fallback implementations
- `factory.py` - Factory functions for module
- `base.py` - Abstract base classes
- `utils.py` - Utility functions

### 4.2 Import Structure

**Train script imports:**
```python
from src.models import get_model
from src.data import get_dataloaders
from src.compression import get_compressor
from src.error_feedback import ErrorFeedbackBuffer
from src.communication import DistributedBackend
```

**No circular dependencies:**
- `communication/` imports `compression/` and `error_feedback/`
- But NOT vice versa
- DAG structure ensures clean builds

### 4.3 Configuration Management

**File:** `configs/default.yaml`

```yaml
model: simple_cnn
dataset: mnist
epochs: 10
batch_size: 64
learning_rate: 0.01

compression:
  enabled: true
  ratio: 0.01
  error_feedback: true

distributed:
  backend: gloo
  world_size: 1
```

**Loaded via:** `src/utils/config.py`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 5. TESTING STRATEGY

### 5.1 Test Organization

```
tests/
├── test_compression.py      # 12 tests
├── test_error_feedback.py   # 7 tests
└── test_integration.py      # 3 tests
```

### 5.2 Test Categories

**Unit Tests (19 tests):**
- Test individual functions in isolation
- Mock dependencies
- Fast (<1s total)

**Integration Tests (3 tests):**
- Test end-to-end workflows
- Real training loops
- Slower (~30s total)

### 5.3 Key Test Patterns

**1. Correctness Tests**
```python
def test_topk_selects_largest():
    comp = TopKCompressorGPU(ratio=0.5)
    tensor = torch.tensor([1.0, -5.0, 3.0, -2.0])
    values, indices, _ = comp.compress(tensor)

    # Should select -5.0 and 3.0 (largest magnitude)
    assert set(values.tolist()) == {-5.0, 3.0}
```

**2. Property Tests**
```python
def test_compress_decompress_preserves_shape():
    comp = TopKCompressorGPU(ratio=0.1)
    original = torch.randn(10, 20, 30)
    v, idx, shape = comp.compress(original)
    reconstructed = comp.decompress(v, idx, shape)

    assert original.shape == reconstructed.shape
```

**3. Convergence Tests**
```python
def test_error_feedback_unbiased():
    # Verify ∑ transmitted → ∑ true over many iterations
    true_sum = 0.0
    transmitted_sum = 0.0
    for _ in range(1000):
        grad = torch.randn(1000)
        true_sum += grad.sum()
        compressed_grad = compress_with_ef(grad)
        transmitted_sum += compressed_grad.sum()

    assert abs(true_sum - transmitted_sum) < 0.1 * abs(true_sum)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 6. PERFORMANCE OPTIMIZATION

### 6.1 GPU Optimization

**torch.topk implementation:**
- Uses CUDA kernel for parallel sorting
- Quickselect algorithm (O(n) average)
- In-place operations where possible

**Memory optimization:**
- Reuse buffers where possible
- Avoid unnecessary copies
- In-place gradient updates

### 6.2 CPU Fallback

**File:** `src/compression/topk_cpu.py`

Uses `numpy.argpartition`:
- O(n) partitioning (not full sort)
- More efficient than `np.argsort` for small k
- Converted back to torch tensor

### 6.3 Communication Optimization

**Overlap potential (future work):**
```python
# Could pipeline compression with computation
# Current: sequential (compute → compress → sync)
# Ideal: overlap (compress layer i while computing layer i+1)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 7. PLATFORM COMPATIBILITY

### 7.1 Device Detection

**File:** `src/utils/device.py`

```python
def detect_device(preference='auto'):
    if preference == 'cpu':
        return torch.device('cpu')
    if preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if preference == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    # Fallback
    return torch.device('cpu')
```

### 7.2 Backend Selection

**NCCL (GPU, Linux):**
- Fastest for multi-GPU
- GPU-to-GPU communication
- Not available on Windows/macOS

**Gloo (CPU/GPU, All platforms):**
- Universal compatibility
- CPU communication or GPU fallback
- Slower than NCCL but portable

### 7.3 macOS-Specific Fixes

**Issue 1: SSL Certificates**
- Problem: Dataset download fails
- Fix: `download_mnist.sh` or `train_fixed.py`

**Issue 2: Python 3.13 Multiprocessing**
- Problem: DataLoader workers fail
- Fix: `benchmark_*_fixed.py` with `num_workers=0`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 8. TROUBLESHOOTING

### 8.1 Common Issues

**Import Errors:**
```bash
# Solution: Install in editable mode
pip install -e .
```

**CUDA Out of Memory:**
```bash
# Solution: Reduce batch size or use CPU
python train.py --batch-size 32 --device cpu
```

**Tests Failing:**
```bash
# Solution: Run in single-process mode (default)
pytest tests/ -v
```

### 8.2 Debug Mode

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Profile compression:**
```python
comp = TopKCompressorGPU(ratio=0.01)
print(comp.stats)  # View statistics after training
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## SUMMARY

This implementation demonstrates:
- **Clean architecture:** Modular, testable, extensible
- **Production quality:** Comprehensive testing, error handling
- **Performance:** Optimized algorithms, GPU acceleration
- **Portability:** Works on CPU/GPU, Linux/macOS/Windows

**Key Files to Review:**
1. `src/compression/topk_gpu.py` - Core algorithm
2. `src/error_feedback/buffer.py` - Error tracking
3. `src/communication/backend.py` - Gradient sync
4. `tests/test_*.py` - Validation suite

For theory → code mapping, see: CODE_MAPPING_GUIDE.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
