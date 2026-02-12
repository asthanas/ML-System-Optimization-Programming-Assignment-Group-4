import time
from typing import Tuple
import numpy as np
import torch
from .base import BaseCompressor

class TopKCompressorCPU(BaseCompressor):
    """CPU Top-K compressor using numpy.argpartition (O(n) average)."""

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        t0 = time.perf_counter()
        shape = tensor.shape
        flat = tensor.reshape(-1)
        n = flat.numel()
        k = self._k(n)
        arr = flat.detach().cpu().numpy()
        abs_arr = np.abs(arr)
        # argpartition gives the k largest indices in O(n) average
        part_idx = np.argpartition(abs_arr, n - k)[n - k:]
        # sort so results are deterministic
        part_idx = part_idx[np.argsort(abs_arr[part_idx])[::-1]]
        values  = torch.from_numpy(arr[part_idx].copy())
        indices = torch.from_numpy(part_idx.astype(np.int64).copy())
        ms = (time.perf_counter() - t0) * 1000
        self.stats.total_compress_calls += 1
        self.stats.total_compress_ms    += ms
        self.stats.total_bytes_original   += n * 4
        self.stats.total_bytes_compressed += k * (4 + 8)
        return values, indices, shape

    def decompress(self, values: torch.Tensor, indices: torch.Tensor,
                   original_shape: tuple) -> torch.Tensor:
        t0 = time.perf_counter()
        n = 1
        for d in original_shape:
            n *= d
        out = torch.zeros(n, dtype=values.dtype)
        out.scatter_(0, indices, values)
        self.stats.total_decompress_calls += 1
        self.stats.total_decompress_ms    += (time.perf_counter() - t0) * 1000
        return out.reshape(original_shape)
