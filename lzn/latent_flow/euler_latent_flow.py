# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .latent_flow import LatentFlow


class EulerLatentFlow(LatentFlow):
    def __init__(self, trainer, eps=1e-4, use_gradient_checkpoint=True):
        super().__init__()
        self._trainer = trainer
        self._eps = eps
        self._use_gradient_checkpoint = use_gradient_checkpoint

    def _velocity(self, full_encoded, x, t, weights):
        if weights is None:
            weights = torch.ones(full_encoded.shape[0]).to(x.device)
        noise = (x.unsqueeze(1) - full_encoded * (1 - t)) / t
        curr_pdf = self._trainer.prior.get_batch_unnormalized_pdf(noise)

        weights = weights.view(1, -1)
        curr_pdf = curr_pdf * weights

        # calculate prob-weighted sum
        e = curr_pdf[:, :, None] * (-full_encoded + noise)
        e = e.sum(dim=1)

        # calculate probability sum
        p = curr_pdf.sum(dim=1)
        result = e / p[:, None]
        return result, curr_pdf

    def _velocity_with_gradient_checkpoint(self, *args):
        if self._use_gradient_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._velocity, *args, use_reentrant=False)
        else:
            return self._velocity(*args)

    def forward(self, reverse=False, **kwargs):
        if reverse:
            return self._reverse(**kwargs)
        else:
            return self._forward(**kwargs)

    def _forward(self, full_encoded, encoded, prior_data, num_steps):
        full_batch_size = full_encoded.shape[0]
        device = encoded.device

        step_size = (1 - self._eps) / num_steps

        t = torch.ones(full_batch_size, 1).to(device) * self._eps
        initial_point = self._eps * prior_data + (1 - self._eps) * encoded

        current_point = initial_point

        for _ in range(num_steps):
            gradient, _ = self._velocity_with_gradient_checkpoint(full_encoded, current_point, t, None)
            current_point = current_point + step_size * gradient
            t = t + step_size
        final_point = current_point
        return final_point

    def _reverse(self, full_encoded, initial_point, num_steps, weights=None):
        full_batch_size = full_encoded.shape[0]
        device = initial_point.device
        t = torch.ones(full_batch_size, 1).to(device)
        current_point = initial_point
        step_size = 1.0 / num_steps
        pdfs = []
        for _ in range(num_steps):
            gradient, pdf = self._velocity_with_gradient_checkpoint(full_encoded, current_point, t, weights)
            current_point = current_point - step_size * gradient
            t = t - step_size
            pdfs.append(pdf)
        final_point = current_point
        return final_point, pdfs
