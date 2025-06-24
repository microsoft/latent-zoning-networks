# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torchvision.datasets as datasets
import torch
import torchvision.transforms as T

from .data import Data
from .sampler import StatefulSampler


def normalization(x):
    return x * 2 - 1


class CIFAR10(Data):
    def __init__(self, root, conditional, horizontal_flip=True):
        super().__init__()
        transforms = []
        if horizontal_flip:
            transforms.append(T.RandomHorizontalFlip())
        self._train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=T.Compose(
                transforms + [T.ToTensor(), T.Lambda(normalization)]
            ),
        )
        self._train_evaluation_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=T.Compose([T.ToTensor(), T.Lambda(normalization)]),
        )
        self._test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=T.Compose([T.ToTensor(), T.Lambda(normalization)]),
        )

        if conditional:
            self._num_classes = 10
        else:
            self._num_classes = 1
            self._train_dataset.targets = [0] * len(
                self._train_dataset.targets
            )
            self._train_evaluation_dataset.targets = [0] * len(
                self._train_evaluation_dataset.targets
            )
            self._test_dataset.targets = [0] * len(self._test_dataset.targets)

    def num_samples(self, train):
        return len(self._train_dataset) if train else len(self._test_dataset)

    def get_data_loader_and_sampler(
        self,
        batch_size,
        num_workers,
        rank,
        world_size,
        seed=0,
        train=True,
        evaluation=False,
        drop_last=True,
    ):
        if not train:
            dataset = self._test_dataset
        else:
            if evaluation:
                dataset = self._train_evaluation_dataset
            else:
                dataset = self._train_dataset
        sampler = StatefulSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            rank=rank,
            world_size=world_size,
            drop_last=drop_last,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=sampler.per_device_batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=True if num_workers > 0 else False,
        )
        return data_loader, sampler
