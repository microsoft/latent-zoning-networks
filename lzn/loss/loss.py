from abc import ABC, abstractmethod
from torch import nn


class Loss(ABC, nn.Module):
    def initialize(self):
        pass

    @abstractmethod
    def forward(self, x1, x1_hat, x1_m_x0, x1_m_x0_hat):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
