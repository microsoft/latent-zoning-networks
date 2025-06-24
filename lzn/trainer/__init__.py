# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib

from .generation_trainer import GenerationTrainer
from .contrastive_trainer import ContrastiveTrainer


def from_config(config):
    class_name = config.trainer
    module_name, class_name = class_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_(config=config)
