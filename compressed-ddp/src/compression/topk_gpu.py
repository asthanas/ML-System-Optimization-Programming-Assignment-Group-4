import time
from typing import Tuple
import torch
from .base import BaseCompressor

class TopKCompressorGPU(BaseCompressor):
    """GPU Top-K compressor using torch.topk (CUDA accelerated)."""

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        t0 = time.perf_counter()
        shape = tensor.shape
        flat  = tensor.reshape(-1)
        k     = self._k(flat.numel())
        _, idx = torch.topk(flat.abs(), k, largest=True, sorted=False)
        values = flat[idx]
        ms = (time.perf_counter() - t0) * 1000
        self.stats.total_compress_calls   += 1
        self.stats.total_compress_ms      += ms
        self.stats.total_bytes_original   += flat.numel() * 4
        self.stats.total_bytes_compressed += k * (4 + 8)
        return values, idx, shape

    def decompress(self, values: torch.Tensor, indices: torch.Tensor,
                   original_shape: tuple) -> torch.Tensor:
        t0 = time.perf_counter()
        n = 1
        for d in original_shape:
            n *= d
        out = torch.zeros(n, dtype=values.dtype, device=values.device)
        out.scatter_(0, indices, values)
        self.stats.total_decompress_calls += 1
        self.stats.total_decompress_ms    += (time.perf_counter() - t0) * 1000
        return out.reshape(original_shape)
