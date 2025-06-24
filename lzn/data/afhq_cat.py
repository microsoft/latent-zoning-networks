# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os
import torchvision.transforms as T

from .data import Data
from .sampler import StatefulSampler
from .image_folder_dataset import ImageFolderDataset


def normalization(x):
    return x * 2 - 1


class AFHQCat(Data):
    def __init__(self, afhq_root, image_size, horizontal_flip=True):
        super().__init__()
        transforms = []
        if horizontal_flip:
            transforms.append(T.RandomHorizontalFlip())
        default_transforms = [T.Resize(image_size), T.ToTensor(), T.Lambda(normalization)]
        self._train_root = os.path.join(afhq_root, "train", "cat")
        self._test_root = os.path.join(afhq_root, "val", "cat")
        self._train_dataset = ImageFolderDataset(
            folder=self._train_root, transform=T.Compose(transforms + default_transforms), conditional=False
        )
        self._train_evaluation_dataset = ImageFolderDataset(
            folder=self._train_root, transform=T.Compose(default_transforms), conditional=False
        )
        self._test_dataset = ImageFolderDataset(
            folder=self._test_root, transform=T.Compose(default_transforms), conditional=False
        )
        self._num_classes = 1

    def num_samples(self, train):
        return len(self._dataset)

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
