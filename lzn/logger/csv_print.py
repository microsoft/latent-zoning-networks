# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import torch
import numpy as np
from collections import defaultdict

from .logger import Logger
from ..metric.metric_item import FloatMetricItem
from ..metric.metric_item import FloatListMetricItem
from ..pytorch_utils.distributed import get_rank


class CSVPrint(Logger):
    def __init__(
        self,
        output_folder,
        path_separator="-",
        float_format=".8f",
        flush_iteration_freq=500,
    ):
        self._output_folder = output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._path_separator = path_separator
        self._float_format = float_format
        self._flush_iteration_freq = flush_iteration_freq
        self._clear_logs()

    def _clear_logs(self):
        self._logs = defaultdict(list)

    def _get_log_path(self, epoch, iteration, item):
        log_path = item.name
        log_path = log_path.replace("/", self._path_separator)
        log_path = log_path.replace("\\", self._path_separator)
        log_path = f"[rank{get_rank()}]" + log_path
        log_path = os.path.join(self._output_folder, log_path + ".csv")
        return log_path

    def _flush(self):
        for path in self._logs:
            with open(path, "a") as f:
                writer = csv.writer(f)
                writer.writerows(self._logs[path])

    def _log_float(self, log_path, epoch, iteration, item):
        str_epoch = str(epoch)
        str_iteration = str(iteration)
        str_value = item.value
        if isinstance(item.value, torch.Tensor):
            str_value = item.value.cpu().detach().numpy()
        if isinstance(str_value, np.ndarray):
            str_value = str_value.tolist()
        if isinstance(str_value, list):
            str_value = ",".join(
                [format(v, self._float_format) for v in str_value]
            )
        else:
            str_value = format(str_value, self._float_format)
        self._logs[log_path].append([str_epoch, str_iteration, str_value])

    def log(self, epoch, iteration, metric_items):
        for item in metric_items:
            if not isinstance(item, (FloatMetricItem, FloatListMetricItem)):
                continue
            log_path = self._get_log_path(epoch, iteration, item)
            self._log_float(log_path, epoch, iteration, item)
        if iteration % self._flush_iteration_freq == 0:
            self._flush()
            self._clear_logs()

    def clean_up(self):
        self._flush()
        self._clear_logs()
