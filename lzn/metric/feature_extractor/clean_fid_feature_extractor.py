import torch.nn as nn
import tempfile
import numpy as np
import torch

from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer

from lzn.pytorch_utils.model_inference import to_uint8


class CleanFIDFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._temp_folder = tempfile.TemporaryDirectory()
        self._inception = InceptionV3W(
            path=self._temp_folder.name,
            download=True,
            resize_inside=False)
        self._resizer = build_resizer("clean")

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        x = x.cpu().detach().numpy()
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        x = to_uint8(x=x, min=-1, max=1)
        transformed_x = []
        for i in range(x.shape[0]):
            image = x[i]
            image = image.transpose((1, 2, 0))
            image = self._resizer(image)
            transformed_x.append(image)
        transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
        transformed_x = torch.from_numpy(transformed_x).to(device)
        return self._inception(transformed_x)
