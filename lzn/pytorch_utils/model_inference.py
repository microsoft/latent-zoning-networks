# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch


def batch_forward(model, batch_size, **kwargs):
    num_samples = len(next(iter(kwargs.values())))
    num_batches = int(np.ceil(num_samples / batch_size))
    y = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        parameters = {
            k: v[start:end] if isinstance(v, (torch.Tensor, np.ndarray)) else v
            for k, v in kwargs.items()
        }
        y.append(model(**parameters))
    if not isinstance(y[0], tuple):
        y = [(a,) for a in y]
        is_tuple = False
    else:
        is_tuple = True
    new_y = []
    for i in range(len(y[0])):
        if isinstance(y[0][i], np.ndarray):
            new_y.append(np.concatenate([z[i] for z in y], axis=0))
        elif isinstance(y[0][i], torch.Tensor):
            new_y.append(torch.cat([z[i] for z in y], dim=0))
        else:
            raise ValueError(f"Unsupported type: {type(y[0][i])}")
    return tuple(new_y) if is_tuple else new_y[0]


def to_uint8(x, min, max):
    x = (x - min) / (max - min)
    x = np.around(np.clip(x * 255, a_min=0, a_max=255)).astype(np.uint8)
    return x
