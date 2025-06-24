# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class TScheduler(ABC):

    @abstractmethod
    def sample(self, batch_size, device):
        pass
