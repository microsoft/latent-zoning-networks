# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class Sampler(ABC):
    def initialize(self, trainer):
        self._trainer = trainer

    @abstractmethod
    def sample(self, latent, label, return_all):
        pass

    @abstractmethod
    def predict_x1_and_x1_m_x0_from_xt_and_x0(
        self, latent, label, t, xt, x0, clip_x1=None
    ):
        pass

    @abstractmethod
    def generate_x0(self, batch_size, device):
        pass

    @abstractmethod
    def generate_xt(self, t, x1):
        pass
