import torch
import torchvision.transforms as T
from torchvision.datasets import ImageNet

from .data import Data
from .sampler import StatefulSampler


def normalization(x):
    return x * 2 - 1


class DoubleImageNet(ImageNet):
    def __getitem__(self, index):
        sample1, target1 = super().__getitem__(index)
        sample2, target2 = super().__getitem__(index)
        assert target1 == target2
        samples = torch.stack([sample1, sample2], dim=0)
        return samples, target1


class ImageNetAug(Data):
    def __init__(self, root):
        super().__init__()
        self._train_transform = T.Compose(
            [
                T.RandomResizedCrop(size=224),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=23),  # 23 == int(0.1 * 224) + 1
                T.ToTensor(),
                T.Lambda(normalization),
            ]
        )
        self._test_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Lambda(normalization),
            ]
        )
        self._train_dataset = DoubleImageNet(
            root=root,
            split="train",
            transform=self._train_transform,
        )
        self._train_evaluation_dataset = ImageNet(
            root=root,
            split="train",
            transform=self._test_transform,
        )
        self._test_dataset = ImageNet(
            root=root,
            split="val",
            transform=self._test_transform,
        )

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
