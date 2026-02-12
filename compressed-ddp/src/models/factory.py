from .simple_cnn import SimpleCNN
from .resnet import resnet18, resnet50

_REG = {"simple_cnn": SimpleCNN, "resnet18": resnet18, "resnet50": resnet50}

def get_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name not in _REG:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(_REG)}")
    return _REG[name](num_classes) if name == "simple_cnn" else _REG[name](num_classes=num_classes)
