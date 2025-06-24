# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .sampler import Sampler


class EulerSampler(Sampler):
    def __init__(self, image_size, image_channels, num_steps, clip_x1):
        self._image_size = image_size
        self._image_channels = image_channels
        self._num_steps = num_steps
        self._clip_x1 = clip_x1

    def inverse(self, latent, label, return_all, x1, eps=1e-3):
        delta_t = (1 - eps) / self._num_steps
        t = 1
        x = x1
        if return_all:
            xs = []
        for _ in range(self._num_steps):
            _, grad = self.predict_x0_and_x1_m_x0_from_xt_and_x1(
                latent=latent, label=label, t=t, xt=x, x1=x1
            )
            x = x - grad * delta_t
            t -= delta_t
            if return_all:
                x0 = (x - x1 * t) / (1 - t)
                xs.append(x0)
        return tuple(xs) if return_all else x

    def sample(
        self,
        latent,
        label,
        return_all,
        clip_x1=None,
        eps=1e-3,
        x0=None,
        num_samples=None,
        device=None,
    ):
        num_samples = (
            num_samples
            or (latent.size(0) if latent is not None else None)
            or (label.size(0) if label is not None else None)
        )
        device = (
            device
            or (latent.device if latent is not None else None)
            or (label.device if label is not None else None)
        )
        delta_t = (1 - eps) / self._num_steps
        t = eps
        if x0 is None:
            x0 = self.generate_x0(batch_size=num_samples, device=device)
        x = x0
        if return_all:
            xs = []
        for _ in range(self._num_steps):
            _, grad = self.predict_x1_and_x1_m_x0_from_xt_and_x0(
                latent=latent, label=label, t=t, xt=x, x0=x0, clip_x1=clip_x1
            )
            x = x + grad * delta_t
            t += delta_t
            if return_all:
                x1 = (x - x0 * (1 - t)) / t
                xs.append(x1)
        return tuple(xs) if return_all else x

    def predict_x1_and_x1_m_x0_from_xt_and_x0(
        self, latent, label, t, xt, x0, clip_x1=None
    ):
        if isinstance(t, (int, float)):
            t = torch.tensor(t).to(xt.device).type(xt.dtype).repeat(xt.size(0))
        x1_m_x0 = self._trainer.decoder(latent=latent, label=label, xt=xt, t=t)
        x1 = x0 + x1_m_x0
        if clip_x1 is None:
            clip_x1 = self._clip_x1
        if clip_x1:
            x1 = torch.clamp(x1, 0, 1)
            x1_m_x0 = x1 - x0
        return x1, x1_m_x0

    def predict_x0_and_x1_m_x0_from_xt_and_x1(self, latent, label, t, xt, x1):
        if isinstance(t, (int, float)):
            t = torch.tensor(t).to(xt.device).type(xt.dtype).repeat(xt.size(0))
        x1_m_x0 = self._trainer.decoder(latent=latent, label=label, xt=xt, t=t)
        x0 = x1 - x1_m_x0
        return x0, x1_m_x0

    def generate_x0(self, batch_size, device):
        x0 = torch.randn(
            batch_size,
            self._image_channels,
            self._image_size,
            self._image_size,
        ).to(device)
        return x0

    def generate_xt(self, t, x1):
        x0 = self.generate_x0(batch_size=x1.size(0), device=x1.device)
        x0 = x0.type(x1.dtype).to(x1.device)
        if isinstance(t, (int, float)):
            t = torch.tensor(t).to(x1.device).type(x1.dtype).unsqueeze(0)
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        xt = x1 * t + x0 * (1 - t)
        return xt, x0
