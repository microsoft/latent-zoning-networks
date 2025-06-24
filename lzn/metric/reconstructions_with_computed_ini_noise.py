import torch
import numpy as np

from .metric import Metric
from .metric_item import ImageListMetricItem, FloatMetricItem
from lzn.pytorch_utils.model_inference import to_uint8
from lzn.pytorch_utils.distributed import distributed_reduce_mean
from lzn.logging import execution_logger


class Reconstructions(Metric):
    def __init__(self, num_samples, batch_size, num_recontruction_per_sample, num_workers=0):
        super().__init__()
        self._num_samples = num_samples
        self._num_recontruction_per_sample = num_recontruction_per_sample
        self._batch_size = batch_size
        self._num_workers = num_workers

    def initialize(self, trainer):
        self._trainer = trainer
        data_loader, _ = trainer.data.get_data_loader_and_sampler(
            batch_size=(
                self._num_samples
                + self._trainer.config.distributed.world_size
                - 1
            ),
            num_workers=self._num_workers,
            seed=0,
            train=True,
            evaluation=True,
            rank=self._trainer.config.distributed.global_rank,
            world_size=self._trainer.config.distributed.world_size,
        )
        self._x, self._c = next(iter(data_loader))
        self._x = self._x.cpu().detach().numpy()
        self._c = self._c.cpu().detach().numpy()

        if hasattr(self._trainer, "prior"):
            prior_data_loader = trainer.prior.get_data_loader(
                batch_size=self._x.shape[0]
            )
            self._zs = []
            prior_data_loader_iter = iter(prior_data_loader)
            for i in range(self._num_recontruction_per_sample):
                self._zs.append(
                    next(prior_data_loader_iter).cpu().detach().numpy()
                )
        else:
            self._zs = None

    @torch.no_grad()
    def evaluate(self):
        device = self._trainer.config.distributed.device
        x = torch.from_numpy(self._x).to(device)
        c = torch.from_numpy(self._c).to(device)
        decodeds = []
        l2s = []
        for i in range(self._num_recontruction_per_sample):
            if self._zs is not None:
                prior_data = torch.from_numpy(self._zs[i]).to(device)
            else:
                prior_data = None
            encoded = self._trainer.get_latent_from_data(
                batch_size=self._batch_size, data=x, prior_data=prior_data
            )
            ini_noise = self._trainer.get_ini_noise_from_latent_and_data(
                latent=encoded,
                discrete_class_label=c,
                return_all=False,
                batch_size=self._batch_size,
                x1=x,
            )
            decoded = self._trainer.get_data_from_latent(
                latent=encoded,
                discrete_class_label=c,
                batch_size=self._batch_size,
                uint8=False,
                return_all=False,
                x0=ini_noise,
            ).cpu().detach().numpy()
            decoded_image = to_uint8(decoded, min=-1, max=1)
            decodeds.append(decoded_image)
            l2 = np.sqrt(np.sum(
                (decoded - self._x) ** 2, axis=(1, 2, 3)
            )).mean()
            l2s.append(l2)
        x = to_uint8(x=x.cpu().detach().numpy(), min=-1, max=1)
        image = np.concatenate([x] + decodeds, axis=3)
        image = torch.from_numpy(image).to(device)
        image = torch.distributed.nn.functional.all_gather(image)
        image = torch.cat(image, dim=0)
        image = image.cpu().detach().numpy()

        l2 = np.mean(l2s)
        execution_logger.info(f"Reconstructions with computed ini noise local L2: {l2}")
        l2 = distributed_reduce_mean(value=l2, device=device)
        execution_logger.info(f"Reconstructions with computed ini noise global L2: {l2}")

        metric_item = ImageListMetricItem(
            name="reconstructions_with_computed_ini_noise",
            value=image,
            num_images_per_row=1,
        )

        l2_metric_item = FloatMetricItem(
            name="reconstructions_with_computed_ini_noise_l2",
            value=l2,
        )

        return [metric_item, l2_metric_item]
