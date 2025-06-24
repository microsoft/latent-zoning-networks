import torch
import math

from .metric import Metric
from .metric_item import FloatMetricItem

from lzn.logging import execution_logger
from lzn.pytorch_utils.class_weights import get_class_weights
from lzn.pytorch_utils.distributed import distributed_reduce_sum


class TunableStdClassAccuracy(Metric):
    def __init__(
        self,
        num_samples,
        batch_size,
        std_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        num_workers=0,
    ):
        super().__init__()
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._std_list = std_list
        self._num_workers = num_workers

    def initialize(self, trainer):
        self._trainer = trainer
        if self._num_samples > 0:
            self._num_samples_per_device = math.ceil(
                self._num_samples / self._trainer.config.distributed.world_size
            )
        else:
            self._num_samples_per_device = -1

    def _evaluate(
        self, data_loader, prior_data_loader, std, with_class_weights
    ):
        device = self._trainer.config.distributed.device
        total = 0
        correct = 0
        for (data, label), prior_data in zip(data_loader, prior_data_loader):
            data = data.to(device)
            label = label.to(device)
            prior_data = prior_data.to(device)
            prior_data = prior_data[: data.shape[0]]
            prior_data = prior_data * std
            latent = self._trainer.get_latent_from_data(
                data=data, prior_data=prior_data
            )
            if with_class_weights:
                class_weights = get_class_weights(
                    label, self._trainer.data.num_classes
                )
            else:
                class_weights = None
            predicted_label = (
                self._trainer.get_discrete_class_label_from_latent(
                    latent=latent, class_weights=class_weights
                )
            )
            if self._num_samples_per_device > 0:
                label = label[: self._num_samples_per_device - total]
                predicted_label = predicted_label[
                    : self._num_samples_per_device - total
                ]
            correct += (predicted_label == label).sum().item()
            total += data.shape[0]
            if (
                self._num_samples_per_device > 0
                and total >= self._num_samples_per_device
            ):
                break
        correct = distributed_reduce_sum(value=correct, device=device)
        total = distributed_reduce_sum(value=total, device=device)
        execution_logger.info(
            f"Correct: {correct}, Total: {total}, Accuracy: {correct / total}"
        )
        return correct / total

    @torch.no_grad()
    def evaluate(self):
        metric_items = []
        for std in self._std_list:
            execution_logger.info(f"std: {std}")
            data_loader, sampler = (
                self._trainer.data.get_data_loader_and_sampler(
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    seed=0,
                    train=False,
                    evaluation=True,
                    rank=self._trainer.config.distributed.global_rank,
                    world_size=self._trainer.config.distributed.world_size,
                    drop_last=True,
                )
            )
            prior_data_loader = self._trainer.prior.get_data_loader(
                batch_size=sampler.per_device_batch_size
            )
            execution_logger.info("Evaluating test class accuracy.")
            test_accuracy = self._evaluate(
                data_loader=data_loader,
                prior_data_loader=prior_data_loader,
                std=std,
                with_class_weights=False,
            )
            test_metric_item = FloatMetricItem(
                name=f"test_accuracy_{std}", value=test_accuracy
            )
            metric_items.append(test_metric_item)
            execution_logger.info(
                "Evaluating test class accuracy with class_weights."
            )
            test_accuracy = self._evaluate(
                data_loader=data_loader,
                prior_data_loader=prior_data_loader,
                std=std,
                with_class_weights=True,
            )
            test_metric_item = FloatMetricItem(
                name=f"test_accuracy_{std}_with_class_weights",
                value=test_accuracy,
            )
            metric_items.append(test_metric_item)

            data_loader, _ = self._trainer.data.get_data_loader_and_sampler(
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                seed=0,
                train=True,
                evaluation=True,
                rank=self._trainer.config.distributed.global_rank,
                world_size=self._trainer.config.distributed.world_size,
                drop_last=True,
            )
            execution_logger.info("Evaluating train class accuracy.")
            train_accuracy = self._evaluate(
                data_loader=data_loader,
                prior_data_loader=prior_data_loader,
                std=std,
                with_class_weights=False,
            )
            train_metric_item = FloatMetricItem(
                name=f"train_accuracy_{std}", value=train_accuracy
            )
            metric_items.append(train_metric_item)
            execution_logger.info(
                "Evaluating train class accuracy with class weights."
            )
            train_accuracy = self._evaluate(
                data_loader=data_loader,
                prior_data_loader=prior_data_loader,
                std=std,
                with_class_weights=True,
            )
            train_metric_item = FloatMetricItem(
                name=f"train_accuracy_{std}_with_class_weights",
                value=train_accuracy,
            )
            metric_items.append(train_metric_item)

        return metric_items
