from pathlib import Path
import yaml

class _D(dict):
    def __getattr__(self, k):
        try: v = self[k]
        except KeyError: raise AttributeError(k)
        return _D(v) if isinstance(v, dict) else v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config(path):
    with open(path) as f:
        return _D(yaml.safe_load(f) or {})
