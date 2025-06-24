# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import cleanfid.fid
from tqdm import tqdm
import math
import itertools

import tensorflow as tf

from lzn.pytorch_utils.distributed import get_world_size
from lzn.pytorch_utils.distributed import all_gather_nd
from lzn.pytorch_utils.distributed import distributed_reduce_sum

from .metric import Metric
from .metric_item import FloatMetricItem
from .feature_extractor.rf_fid_feature_extractor import RFFIDFeatureExtractor

from lzn.logging import execution_logger


class RFFID(Metric):
    def __init__(
        self,
        batch_size,
        num_gen_samples=-1,
        num_real_samples=-1,
        num_workers=0,
        seed=0,
        gpu_memory_limit=8192,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_gen_samples = num_gen_samples
        self._num_real_samples = num_real_samples
        self._num_workers = num_workers
        self._seed = seed
        self._gpu_memory_limit = gpu_memory_limit

        self._set_num_gen_samples_per_device()
        self._set_num_real_samples_per_device()

    def _set_num_gen_samples_per_device(self):
        if self._num_gen_samples > 0:
            self._num_gen_samples_per_device = math.ceil(self._num_gen_samples / get_world_size())
        else:
            self._num_gen_samples_per_device = -1

    def _set_num_real_samples_per_device(self):
        if self._num_real_samples > 0:
            self._num_real_samples_per_device = math.ceil(self._num_real_samples / get_world_size())
        else:
            self._num_real_samples_per_device = -1

    def initialize(self, trainer):
        execution_logger.info("Initializing RFFID metric")
        tf_devices = tf.config.list_physical_devices("GPU")
        tf_devices.sort(key=lambda gpu: int(gpu.name.split(':')[-1]))
        local_device = tf_devices[trainer.config.distributed.local_rank]
        tf.config.experimental.set_memory_growth(local_device, True)
        tf.config.set_visible_devices(local_device, "GPU")
        if self._gpu_memory_limit > 0:
            tf.config.set_logical_device_configuration(
                local_device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=self._gpu_memory_limit)],
            )

        self._trainer = trainer

        self._feature_extractor = RFFIDFeatureExtractor()
        features = []
        labels = []
        data_loader, sampler = trainer.data.get_data_loader_and_sampler(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            seed=self._seed,
            train=True,
            evaluation=True,
            rank=self._trainer.config.distributed.global_rank,
            world_size=self._trainer.config.distributed.world_size,
            drop_last=False,
        )
        self._per_device_batch_size = sampler.per_device_batch_size
        num_samples = 0
        for data, label in tqdm(iter(data_loader)):
            data = data.to(self._trainer.config.distributed.device)
            feature = self._feature_extractor.forward(data)
            features.append(feature)
            labels.append(label)
            num_samples += data.shape[0]
            if (
                self._num_real_samples_per_device != -1
                and num_samples >= self._num_real_samples_per_device
                and (self._num_gen_samples_per_device == -1 or num_samples >= self._num_gen_samples_per_device)
            ):
                break
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        execution_logger.info(f"RFFID labels shape: {labels.shape}")

        num_samples = distributed_reduce_sum(value=num_samples, device=self._trainer.config.distributed.device)
        execution_logger.info(f"RFFID num_samples: {num_samples}")

        features = all_gather_nd(features)
        features = torch.cat(features, dim=0)

        if self._num_real_samples == -1:
            self._num_real_samples = num_samples
            self._set_num_real_samples_per_device()
        else:
            features = features[: self._num_real_samples]
        execution_logger.info(f"RFFID real features shape: {features.shape}")
        features = features.cpu().detach().numpy()
        self._real_mu = np.mean(features, axis=0)
        self._real_sigma = np.cov(features, rowvar=False)

        if self._num_gen_samples == -1:
            self._num_gen_samples = self._num_real_samples
            self._set_num_gen_samples_per_device()
        labels = labels[: self._num_gen_samples_per_device]
        self._labels = labels.cpu().detach().numpy()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self):
        execution_logger.info("Evaluating RFFID metric")
        torch.cuda.empty_cache()
        device = self._trainer.config.distributed.device
        if hasattr(self._trainer, "prior"):
            prior_data_loader = self._trainer.prior.get_data_loader(batch_size=self._per_device_batch_size)
            prior_data_iter = iter(prior_data_loader)
        else:
            prior_data_iter = itertools.repeat(None)
        features = []
        num_samples = 0
        num_target_samples = self._labels.shape[0]
        for i, prior_data in tqdm(enumerate(prior_data_iter)):
            label = torch.from_numpy(
                self._labels[i * self._per_device_batch_size : (i + 1) * self._per_device_batch_size]
            ).to(device)
            if prior_data is not None:
                prior_data = prior_data.to(device)
                prior_data = prior_data[: label.shape[0]]
                latent = self._trainer.get_latent_from_label(
                    label=label,
                    prior_data=prior_data,
                )
            else:
                latent = None
            gen = self._trainer.get_data_from_latent(
                latent=latent, discrete_class_label=label, return_all=False, num_samples=label.size(0), device=device
            )
            feature = self._feature_extractor.forward(gen)
            features.append(feature)
            num_samples += gen.shape[0]
            if num_samples >= num_target_samples:
                break
        features = torch.cat(features, dim=0)
        execution_logger.info(f"RFFID gen features shape local: {features.shape}")
        features = features[:num_target_samples]

        features = all_gather_nd(features)
        features = torch.cat(features, dim=0)
        features = features[: self._num_gen_samples]

        execution_logger.info(f"RFFID gen features shape: {features.shape}")
        features = features.cpu().detach().numpy()
        gen_mu = np.mean(features, axis=0)
        gen_sigma = np.cov(features, rowvar=False)
        fid = cleanfid.fid.frechet_distance(
            mu1=self._real_mu,
            sigma1=self._real_sigma,
            mu2=gen_mu,
            sigma2=gen_sigma,
        )
        metric_item = FloatMetricItem(name="rf_fid", value=fid)
        torch.cuda.empty_cache()
        return [metric_item]
