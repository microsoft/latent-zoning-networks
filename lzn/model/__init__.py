# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from .ema import ExponentialMovingAverage


def from_config(config, **additional_params):
    class_name = config.name
    params = config.params
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(**params, **additional_params)


def ema_from_config(parameters, config):
    if config.enabled:
        ema = ExponentialMovingAverage(parameters, **config.params)
    else:
        ema = None
    return ema
