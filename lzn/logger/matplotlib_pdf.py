import os

from .logger import Logger
from ..metric.metric_item import MatplotlibMetricItem
from ..pytorch_utils.distributed import master_only


class MatplotlibPDF(Logger):
    def __init__(
        self,
        output_folder,
        path_separator="-",
        epoch_format="09d",
        iteration_format="09d",
    ):
        self._output_folder = output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._path_separator = path_separator
        self._epoch_format = epoch_format
        self._iteration_format = iteration_format

    @master_only
    def log(self, epoch, iteration, metric_items):
        for item in metric_items:
            if not isinstance(item, (MatplotlibMetricItem,)):
                continue
            pdf_path = self._get_pdf_path(epoch, iteration, item)
            item.value.savefig(pdf_path)

    def _get_pdf_path(self, epoch, iteration, item):
        image_name = item.name
        image_name = image_name.replace("/", self._path_separator)
        image_name = image_name.replace("\\", self._path_separator)
        image_folder = os.path.join(self._output_folder, image_name)
        os.makedirs(image_folder, exist_ok=True)
        epoch_string = format(epoch, self._epoch_format)
        iteration_string = format(iteration, self._iteration_format)
        image_file_name = f"{epoch_string}_{iteration_string}.pdf"
        image_path = os.path.join(
            image_folder,
            image_file_name,
        )
        return image_path
