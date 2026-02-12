import abc, time
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class CompressStats:
    total_compress_calls: int = 0
    total_compress_ms: float = 0.0
    total_decompress_calls: int = 0
    total_decompress_ms: float = 0.0
    total_bytes_original: int = 0
    total_bytes_compressed: int = 0

    @property
    def mean_compress_ms(self):
        return self.total_compress_ms / max(self.total_compress_calls, 1)

    @property
    def compression_ratio(self):
        return self.total_bytes_original / max(self.total_bytes_compressed, 1)

    def __str__(self):
        return (f"CompressStats(calls={self.total_compress_calls}, "
                f"mean={self.mean_compress_ms:.2f}ms, "
                f"ratio={self.compression_ratio:.1f}x)")


class BaseCompressor(abc.ABC):
    def __init__(self, ratio: float = 0.01):
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0,1], got {ratio}")
        self.ratio = ratio
        self.stats = CompressStats()

    @abc.abstractmethod
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        ...

    @abc.abstractmethod
    def decompress(self, values: torch.Tensor, indices: torch.Tensor,
                   original_shape: tuple) -> torch.Tensor:
        ...

    def _k(self, numel: int) -> int:
        return max(1, int(numel * self.ratio))

    def reset_stats(self):
        self.stats = CompressStats()

    def __repr__(self):
        return f"{self.__class__.__name__}(ratio={self.ratio})"
