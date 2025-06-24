# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import imageio
import math
import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

from .logger import Logger
from ..metric.metric_item import ImageMetricItem, ImageListMetricItem
from ..pytorch_utils.distributed import master_only


class ImageFile(Logger):
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
            if not isinstance(item, (ImageMetricItem, ImageListMetricItem)):
                continue
            image_path, preview_image_path = self._get_image_path(epoch, iteration, item)
            if isinstance(item, ImageMetricItem):
                self._log_image(image_path=image_path, preview_image_path=preview_image_path, item=item)
            elif isinstance(item, ImageListMetricItem):
                self._log_image_list(image_path=image_path, preview_image_path=preview_image_path, item=item)

    def _get_image_path(self, epoch, iteration, item):
        image_name = item.name
        image_name = image_name.replace("/", self._path_separator)
        image_name = image_name.replace("\\", self._path_separator)
        image_folder = os.path.join(self._output_folder, image_name)
        os.makedirs(image_folder, exist_ok=True)
        epoch_string = format(epoch, self._epoch_format)
        iteration_string = format(iteration, self._iteration_format)
        image_file_name = f"{epoch_string}_{iteration_string}.png"
        image_path = os.path.join(
            image_folder,
            image_file_name,
        )
        preview_image_file_name = f"{epoch_string}_{iteration_string}_preview.png"
        preview_image_path = os.path.join(
            image_folder,
            preview_image_file_name,
        )
        return image_path, preview_image_path

    def _save_preview_image(self, image, image_path):
        if self._preview_size is not None:
            preview_image = image
            if preview_image.shape[0] > self._preview_size or preview_image.shape[1] > self._preview_size:
                scale = self._preview_size / max(preview_image.shape[0], preview_image.shape[1])
                new_size = (
                    int(preview_image.shape[1] * scale),
                    int(preview_image.shape[0] * scale),
                )
                preview_image = np.array(Image.fromarray(preview_image).resize(new_size))
            imageio.imwrite(image_path, preview_image)

    def _log_image(self, image_path, preview_image_path, item):
        image = item.value
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        imageio.imwrite(image_path, image)
        self._save_preview_image(image=image, image_path=preview_image_path)

    def _log_image_list(self, image_path, preview_image_path, item):
        images = item.value
        num_images_per_row = item.num_images_per_row
        if num_images_per_row is None:
            num_images_per_row = int(math.sqrt(len(images)))

        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(image) for image in images]

        image = make_grid(images, nrow=num_images_per_row).cpu().detach().numpy()
        image = image.transpose((1, 2, 0))
        imageio.imwrite(image_path, image)
        self._save_preview_image(image=image, image_path=preview_image_path)
