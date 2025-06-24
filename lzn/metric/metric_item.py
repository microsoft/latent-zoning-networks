# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import matplotlib.pyplot as plt

scopes = []


class metric_scope(object):
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        scopes.append(self._name)

    def __exit__(self, type, value, traceback):
        scopes.pop()


class MetricItem(object):
    def __init__(self, name, value):
        self._name = "/".join(scopes + [name])
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def clean_up(self):
        pass


class MatplotlibMetricItem(MetricItem):
    def clean_up(self):
        plt.close(self._value)


class FloatMetricItem(MetricItem):
    pass


class FloatListMetricItem(MetricItem):
    pass


class ImageMetricItem(MetricItem):
    pass


class ArrayMetricItem(MetricItem):
    pass


class ImageListMetricItem(MetricItem):
    def __init__(self, num_images_per_row=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_images_per_row = num_images_per_row

    @property
    def num_images_per_row(self):
        return self._num_images_per_row
