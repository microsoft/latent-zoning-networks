# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from .latent_flow import LatentFlow
from .dpm_solver_LZN import DPM_Solver, NoiseScheduleRectFlow


class DPMSolverLatentFlow(LatentFlow):
    def __init__(
        self,
        trainer,
        denoise=False,
        order=3,
        skip_type="time_uniform",
        method="multistep",
        algorithm_type="dpmsolver++",
        thresholding=False,
        rtol=0.05,
        atol=0.0078,
        lower_order_final=True,
        eps_start=1e-3,
        eps_end=1e-3,
    ):
        super().__init__()
        self._trainer = trainer
        self._denoise = denoise
        self._order = order
        self._skip_type = skip_type
        self._method = method
        self._algorithm_type = algorithm_type
        self._thresholding = thresholding
        self._rtol = rtol
        self._atol = atol
        self._lower_order_final = lower_order_final
        self._eps_start = eps_start
        self._eps_end = eps_end

    def _velocity(self, full_encoded, x, t, weights):
        if weights is None:
            weights = torch.ones(full_encoded.shape[0]).to(x.device)
        noise = (x.unsqueeze(1) - full_encoded * (1 - t[0])) / t[0]
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

    def forward(self, reverse=False, **kwargs):
        if reverse:
            return self._reverse(**kwargs)
        else:
            return self._forward(**kwargs)

    def _forward(self, full_encoded, encoded, prior_data, num_steps):
        def noise_fn(x, t):
            alpha_t, sigma_t, d_alpha_d_t, d_sigma_d_t = 1 - t[0], t[0], -1, 1
            v, pdf = self._velocity(full_encoded, x, t, weights=None)
            noise = (v - (d_alpha_d_t / alpha_t) * x) / (d_sigma_d_t - sigma_t * d_alpha_d_t / alpha_t)
            return noise, pdf

        initial_point = self._eps_end * prior_data + (1 - self._eps_end) * encoded
        x = initial_point
        ns = NoiseScheduleRectFlow()
        noise_pred_fn = noise_fn
        dpm_solver = DPM_Solver(
            noise_pred_fn,
            ns,
            algorithm_type=self._algorithm_type,
            correcting_x0_fn="dynamic_thresholding" if self._thresholding else None,
        )
        x = dpm_solver.sample(
            x,
            steps=num_steps - 1 if self._denoise else num_steps,
            t_start=self._eps_end,
            t_end=1 - self._eps_start,
            order=self._order,
            skip_type=self._skip_type,
            method=self._method,
            denoise_to_zero=self._denoise,
            atol=self._atol,
            rtol=self._rtol,
            lower_order_final=self._lower_order_final,
        )
        return x

    def _reverse(self, full_encoded, initial_point, num_steps, weights=None):
        def noise_fn(x, t):
            alpha_t, sigma_t, d_alpha_d_t, d_sigma_d_t = 1 - t[0], t[0], -1, 1
            v, pdf = self._velocity(full_encoded, x, t, weights)
            noise = (v - (d_alpha_d_t / alpha_t) * x) / (d_sigma_d_t - sigma_t * d_alpha_d_t / alpha_t)
            return noise, pdf

        x = initial_point
        ns = NoiseScheduleRectFlow()
        noise_pred_fn = noise_fn
        dpm_solver = DPM_Solver(
            noise_pred_fn,
            ns,
            algorithm_type=self._algorithm_type,
            correcting_x0_fn="dynamic_thresholding" if self._thresholding else None
        )
        x, _, intermediates_pdf = dpm_solver.sample(
            x,
            steps=num_steps - 1 if self._denoise else num_steps,
            t_start=1 - self._eps_start,
            t_end=self._eps_end,
            order=self._order,
            skip_type=self._skip_type,
            method=self._method,
            denoise_to_zero=self._denoise,
            atol=self._atol,
            rtol=self._rtol,
            lower_order_final=self._lower_order_final,
            return_intermediate=True,
        )
        return x, intermediates_pdf
