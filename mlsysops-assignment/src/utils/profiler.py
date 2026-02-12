import time
from collections import defaultdict

class SimpleProfiler:
    def __init__(self):
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._start  = {}

    class _T:
        def __init__(self, p, tag):
            self._p = p; self._tag = tag
        def __enter__(self):
            self._p._start[self._tag] = time.perf_counter(); return self
        def __exit__(self, *_):
            ms = (time.perf_counter() - self._p._start[self._tag]) * 1000
            self._p._totals[self._tag] += ms
            self._p._counts[self._tag] += 1

    def record(self, tag): return self._T(self, tag)

    def summary(self):
        lines = ["Profiler (ms):"]
        for tag in sorted(self._totals):
            t = self._totals[tag]; n = self._counts[tag]
            lines.append(f"  {tag:30s} total={t:.1f} calls={n} mean={t/n:.2f}")
        return "\n".join(lines)
