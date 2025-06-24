# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from torch import nn


class Model(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @property
    def num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        return num_params


class Encoder(Model):
    pass


class Decoder(Model):
    pass
