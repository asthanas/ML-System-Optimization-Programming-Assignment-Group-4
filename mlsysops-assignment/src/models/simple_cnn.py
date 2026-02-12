import torch, torch.nn as nn, torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Lightweight 3-layer CNN for MNIST (~200K params)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(64*7*7, 128)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
