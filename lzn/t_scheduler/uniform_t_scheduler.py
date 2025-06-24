# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .t_scheduler import TScheduler


class UniformTScheduler(TScheduler):

    def sample(self, batch_size, device, eps=1e-3):
        t = torch.rand(batch_size).to(device) * (1 - eps) + eps
        return t
