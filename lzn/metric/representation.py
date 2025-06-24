# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import math
from tqdm import tqdm

from .metric import Metric
from .metric_item import ArrayMetricItem
from lzn.pytorch_utils.distributed import all_gather_nd


from lzn.logging import execution_logger


class Representation(Metric):
    def __init__(
        self,
        batch_size,
        mode,
        num_workers=0,
        num_classes=-1,
        num_samples_per_class=50,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._mode = mode
        if self._mode not in ["no_head", "head", "latent", "latent_center"]:
            raise ValueError(f"Invalid mode: {self._mode}")
        self._num_workers = num_workers
        self._num_classes = num_classes
        self._num_samples_per_class = num_samples_per_class
        self._name = f"representation_{self._mode}"

    def collect_samples_per_class(self, target_num_samples_per_class):
        if self._num_classes == -1:
            num_classes = self._trainer.data.num_classes
        else:
            num_classes = self._num_classes
        data_loader, _ = self._trainer.data.get_data_loader_and_sampler(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            train=False,
            evaluation=True,
            rank=self._trainer.config.distributed.global_rank,
            world_size=self._trainer.config.distributed.world_size,
        )
        xs = [[] for _ in range(num_classes)]
        cs = [[] for _ in range(num_classes)]
        num_samples_completed = [0 for _ in range(num_classes)]
        for x, c in tqdm(data_loader):
            x = x.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            for i in range(num_classes):
                mask = c == i
                xs[i].append(x[mask])
                cs[i].append(c[mask])
                num_samples_completed[i] += mask.sum()
            if all([num_samples_completed[i] >= target_num_samples_per_class for i in range(num_classes)]):
                break
        xs = [np.concatenate(x, axis=0) for x in xs]
        xs = [x[:target_num_samples_per_class] for x in xs]
        cs = [np.concatenate(c, axis=0) for c in cs]
        cs = [c[:target_num_samples_per_class] for c in cs]
        return np.concatenate(xs, axis=0), np.concatenate(cs, axis=0)

    def initialize(self, trainer):
        execution_logger.info(f"Initializing {self._name} metric")
        self._trainer = trainer
        self._num_samples_per_class_per_device = math.ceil(
            self._num_samples_per_class / self._trainer.config.distributed.world_size
        )

        x, c = self.collect_samples_per_class(self._num_samples_per_class_per_device)
        self._x = x
        self._c = c

    def _get_feature(self, data, prior_data):
        if self._mode == "latent":
            return self._trainer.get_latent_from_data(
                data=data,
                prior_data=prior_data,
                return_full=False,
            )
        elif self._mode == "latent_center":
            return self._trainer.get_latent_from_data(
                data=data,
                prior_data=prior_data * 0.0,
                return_full=False,
            )
        elif self._mode == "head":
            return self._trainer._encoder(data, apply_head=True)
        elif self._mode == "no_head":
            return self._trainer._encoder(data, apply_head=False)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def evaluate(self):
        execution_logger.info(f"Evaluating {self._name} metric")
        per_device_batch_size = self._batch_size // self._trainer.config.distributed.world_size
        prior_data_loader = self._trainer.prior.get_data_loader(batch_size=per_device_batch_size)
        prior_data_iter = iter(prior_data_loader)

        representations = []
        for i in range(0, self._x.shape[0], per_device_batch_size):
            x = self._x[i : i + per_device_batch_size]
            x = torch.from_numpy(x).to(self._trainer.config.distributed.device)
            prior_data = next(prior_data_iter)[: x.shape[0]]
            prior_data = prior_data.to(self._trainer.config.distributed.device)
            representation = self._get_feature(data=x, prior_data=prior_data)
            representations.append(representation.cpu().detach().numpy())
        representations = np.concatenate(representations, axis=0)

        representations = torch.from_numpy(representations).to(self._trainer.config.distributed.device)
        representations = all_gather_nd(representations)
        representations = torch.cat(representations, dim=0)
        representations = representations.cpu().detach().numpy()

        labels = torch.from_numpy(self._c).to(self._trainer.config.distributed.device)
        labels = all_gather_nd(labels)
        labels = torch.cat(labels, dim=0)
        labels = labels.cpu().detach().numpy()

        metric_items = []
        metric_items.append(
            ArrayMetricItem(
                name=self._name + "_representations",
                value=representations,
            )
        )
        metric_items.append(
            ArrayMetricItem(
                name=self._name + "_labels",
                value=labels,
            )
        )

        return metric_items
