from .metric import Metric


class TrainingMetrics(Metric):
    def initialize(self, trainer):
        self._trainer = trainer

    def evaluate(self):
        metric_items = self._trainer.training_metric_items

        return metric_items
