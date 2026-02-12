import logging
from typing import Optional
import torch
import torch.distributed as dist
from src.compression.base import BaseCompressor
from src.error_feedback.buffer import ErrorFeedbackBuffer

log = logging.getLogger(__name__)

class DistributedBackend:
    """Gradient allreduce with optional Top-K compression + error feedback."""

    def __init__(self, compressor: Optional[BaseCompressor] = None,
                 error_buffer: Optional[ErrorFeedbackBuffer] = None,
                 world_size: int = 1, rank: int = 0):
        self.compressor   = compressor
        self.error_buffer = error_buffer
        self.world_size   = world_size
        self.rank         = rank
        self._dist_ok     = dist.is_available() and dist.is_initialized()

    def allreduce_gradients(self, named_params) -> None:
        if self.world_size == 1 and self.compressor is None:
            return
        for name, param in named_params:
            if param.grad is None:
                continue
            if self.compressor is not None:
                self._compressed_allreduce(name, param.grad)
            elif self._dist_ok:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    def broadcast_parameters(self, model: torch.nn.Module, src: int = 0) -> None:
        if not self._dist_ok or self.world_size == 1:
            return
        for p in model.parameters():
            dist.broadcast(p.data, src=src)

    def _compressed_allreduce(self, name: str, grad: torch.Tensor) -> None:
        compensated = (self.error_buffer.compensate(name, grad)
                       if self.error_buffer else grad)
        values, indices, shape = self.compressor.compress(compensated)
        approx = self.compressor.decompress(values, indices, shape)
        if self._dist_ok:
            dist.all_reduce(approx, op=dist.ReduceOp.SUM)
            approx.div_(self.world_size)
        if self.error_buffer:
            self.error_buffer.update(name, compensated, approx)
        grad.copy_(approx)
