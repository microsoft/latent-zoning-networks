# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class Prior(ABC):

    @abstractmethod
    def get_data_loader(self, batch_size):
        pass

    @abstractmethod
    def get_batch_unnormalized_pdf(self, x):
        pass
