from abc import ABC, abstractmethod
from collections import namedtuple


LossInfo = namedtuple("LossInfo", ["loss", "weight"])
MetricInfo = namedtuple(
    "MetricInfo",
    [
        "metric",
        "iteration_freq",
        "epoch_freq",
        "apply_on_ema",
        "apply_on_samplers",
    ],
)
SamplerInfo = namedtuple("SamplerInfo", ["sampler", "name"])


class Trainer(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def train(self):
        pass
