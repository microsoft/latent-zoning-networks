# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import math

from .metric import Metric
from .metric_item import ImageListMetricItem
from .metric_item import ArrayMetricItem

from lzn.logging import execution_logger
from lzn.pytorch_utils.distributed import distributed_reduce_max


class RandomSamples(Metric):
    def __init__(self, num_samples_per_class, batch_size, num_classes=-1):
        super().__init__()
        self._num_samples_per_class = num_samples_per_class
        self._batch_size = batch_size
        self._num_classes = num_classes

    def initialize(self, trainer):
        self._trainer = trainer
        if self._num_classes == -1:
            self._num_classes = trainer.data.num_classes
        else:
            self._num_classes = self._num_classes

        self._num_samples_per_class_per_device = math.ceil(
            self._num_samples_per_class
            / self._trainer.config.distributed.world_size
        )

        if hasattr(trainer, "prior"):
            prior_data_loader = trainer.prior.get_data_loader(
                batch_size=self._num_samples_per_class_per_device
                * self._num_classes
            )
            self._z = next(iter(prior_data_loader)).cpu().detach().numpy()
        else:
            self._z = None
        self._c = (
            torch.arange(self._num_classes)
            .repeat_interleave(self._num_samples_per_class_per_device)
            .cpu()
            .detach()
            .numpy()
        )

    @torch.no_grad()
    def evaluate(self):
        device = self._trainer.config.distributed.device
        label = torch.from_numpy(self._c).to(device)
        if self._z is not None:
            prior_data = torch.from_numpy(self._z).to(device)
            latent = self._trainer.get_latent_from_label(
                label=label,
                prior_data=prior_data,
            )
        else:
            latent = None
        y = self._trainer.get_data_from_latent(
            latent=latent,
            discrete_class_label=label,
            uint8=True,
            return_all=True,
            num_samples=label.size(0),
            device=device,
        )
        final_y = y[-1]
        y = list(y)
        num_steps = len(y)
        max_num_steps = distributed_reduce_max(value=num_steps, device=device)

        execution_logger.info(
            f"RandomSamples num_steps: {num_steps},"
            f" max_num_steps: {max_num_steps}"
        )

        if max_num_steps > num_steps:
            y += [np.zeros_like(y[-1])] * (max_num_steps - num_steps)

        y = np.stack(y, axis=1)
        y = np.reshape(y, (-1, *y.shape[2:]))

        y = torch.from_numpy(y).to(device)
        y = torch.distributed.nn.functional.all_gather(y)
        y = [torch.split(a, max_num_steps) for a in y]
        y = torch.cat([a for b in zip(*y) for a in b])
        y = y.cpu().detach().numpy()

        image_metric_item = ImageListMetricItem(
            name="random_samples",
            value=y,
            num_images_per_row=max_num_steps,
        )

        final_y = torch.from_numpy(final_y).to(device)
        final_y = torch.distributed.nn.functional.all_gather(final_y)
        final_y = [torch.split(a, 1) for a in final_y]
        final_y = torch.cat([a for b in zip(*final_y) for a in b])
        final_y = final_y.cpu().detach().numpy()

        array_metric_item = ArrayMetricItem(
            name="random_samples_array",
            value=final_y,
        )

        return [image_metric_item, array_metric_item]
