import torch
from .base import BaseCompressor
from .topk_gpu import TopKCompressorGPU
from .topk_cpu import TopKCompressorCPU

def get_compressor(ratio: float = 0.01, device: str | None = None) -> BaseCompressor:
    """Return GPU compressor if CUDA available, else CPU compressor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        return TopKCompressorGPU(ratio=ratio)
    return TopKCompressorCPU(ratio=ratio)
