from typing import List, Type, Union
import torch, torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.sc    = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.sc = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
    def forward(self, x):
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.sc(x))

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_planes, planes, 1, bias=False); self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False); self.b2 = nn.BatchNorm2d(planes)
        self.c3 = nn.Conv2d(planes, planes*4, 1, bias=False); self.b3 = nn.BatchNorm2d(planes*4)
        self.sc = nn.Sequential()
        if stride != 1 or in_planes != planes*4:
            self.sc = nn.Sequential(
                nn.Conv2d(in_planes, planes*4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*4))
    def forward(self, x):
        out = torch.relu(self.b1(self.c1(x)))
        out = torch.relu(self.b2(self.c2(out)))
        return torch.relu(self.b3(self.c3(out)) + self.sc(x))

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.ip = 64
        self.stem = nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.l1 = self._layer(block, 64,  num_blocks[0], 1)
        self.l2 = self._layer(block, 128, num_blocks[1], 2)
        self.l3 = self._layer(block, 256, num_blocks[2], 2)
        self.l4 = self._layer(block, 512, num_blocks[3], 2)
        self.avg= nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _layer(self, block, planes, nb, stride):
        layers = [block(self.ip, planes, stride)]
        self.ip = planes * block.expansion
        for _ in range(1, nb):
            layers.append(block(self.ip, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)
        return self.fc(self.avg(x).flatten(1))

def resnet18(num_classes=10): return ResNet(BasicBlock, [2,2,2,2], num_classes)
def resnet50(num_classes=10): return ResNet(Bottleneck, [3,4,6,3], num_classes)
