import platform, sys
from typing import Dict
import torch

def detect_device(preferred="auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)

def get_platform_info() -> Dict[str, str]:
    info = {"os": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": str(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
