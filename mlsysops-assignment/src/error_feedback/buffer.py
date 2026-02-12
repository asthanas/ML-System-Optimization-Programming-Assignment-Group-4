import logging
from pathlib import Path
from typing import Dict, Optional
import torch

log = logging.getLogger(__name__)

class ErrorFeedbackBuffer:
    """
    Per-parameter error residual buffers for biased gradient compressors.

    Maintains e such that:  g_comp = g + e,  e_new = g_comp - Compress(g_comp)
    This ensures the compressed gradients are unbiased in expectation.
    """
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype  = dtype
        self._buffers: Dict[str, torch.Tensor] = {}

    def compensate(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """Return gradient + accumulated error residual."""
        buf = self._get_or_create(name, gradient)
        return gradient + buf

    def update(self, name: str, compensated: torch.Tensor,
               compressed_approx: torch.Tensor) -> None:
        """Update buffer: e <- compensated - compressed_approx."""
        self._buffers[name].copy_(compensated - compressed_approx)

    def reset(self, name: Optional[str] = None) -> None:
        if name:
            if name in self._buffers:
                self._buffers[name].zero_()
        else:
            for b in self._buffers.values():
                b.zero_()

    def error_norm(self, name: str) -> float:
        if name not in self._buffers:
            return 0.0
        return float(self._buffers[name].norm().item())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._buffers.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            self._buffers[k] = v.to(device=self.device, dtype=self.dtype)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)

    def _get_or_create(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        if name not in self._buffers:
            self._buffers[name] = torch.zeros_like(ref, device=self.device, dtype=self.dtype)
        return self._buffers[name]

    def __len__(self):
        return len(self._buffers)
