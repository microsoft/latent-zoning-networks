# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn


class CustomDataParallel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = nn.DataParallel(model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


class CustomDistributedDataParallel(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = nn.parallel.DistributedDataParallel(
            model, *args, **kwargs
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
