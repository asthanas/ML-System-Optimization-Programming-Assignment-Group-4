import logging, os
import torch.distributed as dist

log = logging.getLogger(__name__)

def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()

def setup_distributed(backend="gloo", rank=0, world_size=1,
                      init_method="env://") -> None:
    if world_size <= 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, init_method=init_method,
                            world_size=world_size, rank=rank)

def cleanup_distributed() -> None:
    if is_dist_available():
        dist.destroy_process_group()
