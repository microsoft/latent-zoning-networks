# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os

execution_logger = logging.getLogger()


def setup_from_config(config, name="logger"):
    execution_logger.name = name

    os.makedirs(os.path.dirname(config.log_file), exist_ok=True)

    execution_logger.handlers.clear()
    execution_logger.setLevel(config.level)

    log_formatter = logging.Formatter(
        fmt=config.fmt,
        datefmt=config.datefmt,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    execution_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(log_formatter)
    execution_logger.addHandler(file_handler)
