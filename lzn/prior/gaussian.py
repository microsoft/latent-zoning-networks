# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .prior import Prior


class GaussianDataset(torch.utils.data.IterableDataset):
    def __init__(self, latent_dim):
        super().__init__()
        self._latent_dim = latent_dim

    def __iter__(self):
        return self._generate_sample()

    def _generate_sample(self):
        while True:
            sample = torch.normal(mean=0, std=1, size=(self._latent_dim,))
            yield sample.float()
        return sample


class Gaussian(Prior):
    def __init__(self, latent_dim):
        super().__init__()
        self._latent_dim = latent_dim
        self._dataset = GaussianDataset(latent_dim)

    def get_data_loader(self, batch_size):
        dataloader = torch.utils.data.DataLoader(
            self._dataset, batch_size=batch_size, num_workers=0
        )
        return dataloader

    def get_batch_unnormalized_pdf(self, x):
        x = -0.5 * x.square().sum(dim=(2,))
        x_shift = x - torch.max(x, dim=1).values[:, None]
        return torch.exp(x_shift)
