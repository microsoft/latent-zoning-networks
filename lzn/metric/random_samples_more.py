# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import math
from tqdm import tqdm

from .metric import Metric
from .metric_item import ArrayMetricItem


class RandomSamplesMore(Metric):
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
            self._num_samples_per_class / self._trainer.config.distributed.world_size
        )

        if hasattr(trainer, "prior"):
            prior_data_loader = trainer.prior.get_data_loader(
                batch_size=self._num_samples_per_class_per_device * self._num_classes
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
        self._per_device_batch_size = self._batch_size // self._trainer.config.distributed.world_size

    @torch.no_grad()
    def evaluate(self):
        device = self._trainer.config.distributed.device
        all_ys = []
        for i in tqdm(range(0, len(self._c), self._per_device_batch_size)):
            batch_c = self._c[i : i + self._per_device_batch_size]
            batch_label = torch.from_numpy(batch_c).to(device)
            if self._z is not None:
                batch_z = self._z[i : i + self._per_device_batch_size]
                batch_prior_data = torch.from_numpy(batch_z).to(device)
                batch_latent = self._trainer.get_latent_from_label(
                    label=batch_label,
                    prior_data=batch_prior_data,
                )
            else:
                batch_latent = None
            y = self._trainer.get_data_from_latent(
                latent=batch_latent,
                discrete_class_label=batch_label,
                uint8=True,
                return_all=False,
                num_samples=batch_label.size(0),
                device=device,
            )
            final_y = y
            final_y = torch.from_numpy(final_y).to(device)
            final_y = torch.distributed.nn.functional.all_gather(final_y)
            final_y = torch.cat(final_y, dim=0)
            final_y = final_y.cpu().detach().numpy()
            all_ys.append(final_y)
        all_ys = np.concatenate(all_ys, axis=0)

        array_metric_item = ArrayMetricItem(
            name="random_samples_more_array",
            value=all_ys,
        )

        return [array_metric_item]
