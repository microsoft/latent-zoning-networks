from .logger import Logger
from ..metric.metric_item import FloatMetricItem, FloatListMetricItem
from ..logging import execution_logger


class LogPrint(Logger):
    def __init__(self, log_iteration_freq=100):
        self._log_iteration_freq = log_iteration_freq

    def log(self, epoch, iteration, metric_items):
        if iteration % self._log_iteration_freq != 0:
            return
        metric_items = [
            item
            for item in metric_items
            if isinstance(item, (FloatMetricItem, FloatListMetricItem))
        ]
        if len(metric_items) == 0:
            return
        execution_logger.info(f"Epoch: {epoch}, Iteration: {iteration}")
        for item in metric_items:
            if isinstance(item, FloatMetricItem):
                value = [item.value]
            else:
                value = item.value
            value = ",".join([f"{v:.8f}" for v in value])
            execution_logger.info(f"\t{item.name}: {value}")
