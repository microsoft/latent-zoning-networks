# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torchvision.transforms as T

from .data import Data
from .sampler import StatefulSampler
from .image_folder_dataset import ImageFolderDataset


def normalization(x):
    return x * 2 - 1


class CelebAHQ(Data):
    def __init__(self, root, image_size, horizontal_flip=True):
        super().__init__()
        transforms = []
        if horizontal_flip:
            transforms.append(T.RandomHorizontalFlip())
        default_transforms = [T.Resize(image_size), T.ToTensor(), T.Lambda(normalization)]
        self._dataset = ImageFolderDataset(
            folder=root, transform=T.Compose(transforms + default_transforms), conditional=False
        )
        self._evaluation_dataset = ImageFolderDataset(
            folder=root, transform=T.Compose(default_transforms), conditional=False
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
        if evaluation:
            dataset = self._evaluation_dataset
        else:
            dataset = self._dataset
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
