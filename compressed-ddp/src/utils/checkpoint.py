import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

log = logging.getLogger(__name__)

def save_checkpoint(state: Dict[str,Any], path, is_best=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        torch.save(state, path.parent / "best_model.pt")

def load_checkpoint(path, map_location="cpu") -> Optional[Dict[str,Any]]:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location)
