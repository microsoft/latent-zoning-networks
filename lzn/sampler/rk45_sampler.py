import torch
import numpy as np
from lzn.logging import execution_logger
from scipy import integrate

from .sampler import Sampler


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape, device):
    return torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)


class RK45Sampler(Sampler):
    def __init__(self, image_size, image_channels, clip_x1):
        self._image_size = image_size
        self._image_channels = image_channels
        self._clip_x1 = clip_x1

    def inverse(
        self,
        latent,
        label,
        return_all,
        x1,
        eps=1e-3,
        rtol=1e-5,
        atol=1e-5,
    ):
        shape = x1.shape
        device = x1.device

        def ode_func(t, xt):
            xt = from_flattened_numpy(xt, shape, device)
            _, x1_m_x0 = self.predict_x0_and_x1_m_x0_from_xt_and_x1(
                latent=latent, label=label, t=t, xt=xt, x1=x1
            )
            x1_m_x0 = to_flattened_numpy(x1_m_x0)
            return x1_m_x0

        solution = integrate.solve_ivp(
            ode_func,
            (1, eps),
            to_flattened_numpy(x1),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        t = solution.t
        execution_logger.info(f"NFE: {nfe}.")
        execution_logger.info(f"Ts: {list(t)}.")
        xt = from_flattened_numpy(solution.y.T, [-1] + list(shape), device)
        t_reshape = np.reshape(t, (-1, 1, 1, 1, 1))
        t_reshape = torch.tensor(t_reshape).to(xt.device).type(xt.dtype)
        x0 = (xt - x1.unsqueeze(0) * t_reshape) / (1 - t_reshape)
        if return_all:
            return tuple(x0)
        else:
            return x0[-1]

    def sample(
        self,
        latent,
        label,
        return_all,
        clip_x1=None,
        eps=1e-3,
        rtol=1e-5,
        atol=1e-5,
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
        if x0 is None:
            x0 = self.generate_x0(batch_size=num_samples, device=device)
        shape = x0.shape
        device = x0.device
        cnt = 0

        def ode_func(t, xt):
            nonlocal cnt
            execution_logger.info(f"ODE: {cnt}.")
            cnt += 1
            xt = from_flattened_numpy(xt, shape, device)
            _, x1_m_x0 = self.predict_x1_and_x1_m_x0_from_xt_and_x0(
                latent=latent, label=label, t=t, xt=xt, x0=x0, clip_x1=clip_x1
            )
            x1_m_x0 = to_flattened_numpy(x1_m_x0)
            return x1_m_x0

        solution = integrate.solve_ivp(
            ode_func,
            (eps, 1),
            to_flattened_numpy(x0),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        t = solution.t
        execution_logger.info(f"NFE: {nfe}.")
        execution_logger.info(f"Ts: {list(t)}.")
        xt = from_flattened_numpy(solution.y.T, [-1] + list(shape), device)
        t_reshape = np.reshape(t, (-1, 1, 1, 1, 1))
        t_reshape = torch.tensor(t_reshape).to(xt.device).type(xt.dtype)
        x1 = (xt - x0.unsqueeze(0) * (1 - t_reshape)) / t_reshape
        if return_all:
            return tuple(x1)
        else:
            return x1[-1]

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
