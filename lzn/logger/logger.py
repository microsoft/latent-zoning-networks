# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class Logger(ABC):
    def initialize(self, trainer):
        pass

    @abstractmethod
    def log(self, epoch, iteration, metric_items):
        pass

    def clean_up(self):
        pass
