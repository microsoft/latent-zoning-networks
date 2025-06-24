# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data import Sampler
import torch
import math
from lzn.logging import execution_logger


class StatefulSampler(Sampler):
    def __init__(
        self, dataset, batch_size, rank, world_size, drop_last, seed=0
    ):
        super().__init__(dataset)
        self._num_samples = len(dataset)
        self._start_iter = 0
        self._epoch = 0
        self._seed = seed
        self._per_device_batch_size = batch_size // world_size
        self._batch_size = self._per_device_batch_size * world_size
        self._num_iterations_per_epoch = (
            math.ceil(math.floor(self._num_samples / world_size) / self._per_device_batch_size)
            if not drop_last
            else math.floor(self._num_samples / self._batch_size)
        )
        self._num_samples_per_epoch = (
            (self._num_iterations_per_epoch * self._batch_size)
            if drop_last
            else math.floor(self._num_samples / world_size) * world_size
        )

        self._rank = rank
        self._world_size = world_size
        execution_logger.info(
            f"Sampler: num_samples={self._num_samples}, "
            f"num_iterations_per_epoch={self._num_iterations_per_epoch}, "
            f"num_samples_per_epoch={self._num_samples_per_epoch}, "
            f"batch_size={self._batch_size}, "
            f"per_device_batch_size={self._per_device_batch_size}, "
            f"rank={self._rank}, "
            f"world_size={self._world_size}"
        )

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)
        shuffling = torch.randperm(
            self._num_samples, generator=generator
        ).tolist()
        start_index = self._start_iter * self._batch_size
        indices = shuffling[: self._num_samples_per_epoch][
            start_index + self._rank:: self._world_size
        ]
        return iter(indices)

    @property
    def num_iterations_per_epoch(self):
        return self._num_iterations_per_epoch

    @property
    def per_device_batch_size(self):
        return self._per_device_batch_size

    def set_start_iter(self, start_iter):
        self._start_iter = start_iter

    def set_start_iter_by_past_iters(self, past_iters):
        self._start_iter = past_iters % self._num_iterations_per_epoch

    def set_epoch(self, epoch):
        self._epoch = epoch
