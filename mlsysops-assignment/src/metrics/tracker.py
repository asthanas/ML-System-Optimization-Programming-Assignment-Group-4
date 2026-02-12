import csv, logging, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

class MetricsTracker:
    def __init__(self, log_dir=None, use_tensorboard=True):
        self._data: Dict[str, List[tuple]] = defaultdict(list)
        self._writer = None
        self._t0 = time.perf_counter()
        if use_tensorboard and _TB and log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

    def update(self, key: str, value: float, step: int) -> None:
        self._data[key].append((step, value))
        if self._writer:
            self._writer.add_scalar(key, value, step)

    def latest(self, key: str) -> Optional[float]:
        h = self._data.get(key)
        return h[-1][1] if h else None

    def save_csv(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"metric":k,"step":s,"value":v} for k,e in self._data.items() for s,v in e]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric","step","value"])
            w.writeheader(); w.writerows(rows)

    def close(self):
        if self._writer:
            self._writer.close()
