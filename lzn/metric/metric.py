# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class Metric(ABC):
    def initialize(self, trainer):
        pass

    @abstractmethod
    def evaluate(self):
        pass
