import numpy as np
import torch
import gc

from .rf_evaluation import get_inception_model
from .rf_evaluation import run_inception_local

from lzn.pytorch_utils.model_inference import to_uint8


class RFFIDFeatureExtractor:
    def __init__(self):
        super().__init__()
        self._inceptionv3 = False
        self._inception_model = get_inception_model(
            inceptionv3=self._inceptionv3
        )

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        type_ = x.dtype
        x = x.cpu().detach().numpy()
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        x = x.transpose((0, 2, 3, 1))
        x = to_uint8(x=x, min=-1, max=1)
        gc.collect()
        latents = run_inception_local(
            x, self._inception_model, inceptionv3=self._inceptionv3
        )
        features = latents["pool_3"].numpy()
        features = torch.from_numpy(features).to(device).to(type_)
        gc.collect()

        return features
