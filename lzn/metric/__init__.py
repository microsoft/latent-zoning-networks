# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib

from .metric_item import metric_scope
from .metric_item import FloatMetricItem
from .metric_item import FloatListMetricItem
from .metric_item import ImageMetricItem
from .metric_item import ImageListMetricItem


def from_config(config):
    class_name = config.name
    params = config.params
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(**params)


__all__ = [
    metric_scope,
    from_config,
    FloatMetricItem,
    FloatListMetricItem,
    ImageMetricItem,
    ImageListMetricItem,
]
