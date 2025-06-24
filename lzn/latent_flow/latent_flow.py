from abc import ABC, abstractmethod
from torch import nn


class LatentFlow(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass
