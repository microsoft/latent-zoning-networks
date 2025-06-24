# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np

from .logger import Logger
from ..metric.metric_item import ArrayMetricItem
from ..pytorch_utils.distributed import master_only


class NpyFile(Logger):
    def __init__(
        self,
        output_folder,
        path_separator="-",
        epoch_format="09d",
        iteration_format="09d",
        preview_size=None,
    ):
        self._output_folder = output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._path_separator = path_separator
        self._epoch_format = epoch_format
        self._iteration_format = iteration_format
        self._preview_size = preview_size

    @master_only
    def log(self, epoch, iteration, metric_items):
        for item in metric_items:
            if not isinstance(item, ArrayMetricItem):
                continue
            npy_path = self._get_npy_path(epoch, iteration, item)
            np.save(npy_path, item.value)

    def _get_npy_path(self, epoch, iteration, item):
        npy_name = item.name
        npy_name = npy_name.replace("/", self._path_separator)
        npy_name = npy_name.replace("\\", self._path_separator)
        npy_folder = os.path.join(self._output_folder, npy_name)
        os.makedirs(npy_folder, exist_ok=True)
        epoch_string = format(epoch, self._epoch_format)
        iteration_string = format(iteration, self._iteration_format)
        npy_file_name = f"{epoch_string}_{iteration_string}.npy"
        npy_path = os.path.join(
            npy_folder,
            npy_file_name,
        )
        return npy_path
