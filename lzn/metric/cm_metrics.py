# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Adapted from https://github.com/openai/consistency_models/blob/main/evaluations/evaluator.py"""

import torch
import numpy as np
import cleanfid.fid
from tqdm import tqdm
import math
import itertools
from typing import Tuple
from multiprocessing.pool import ThreadPool
from functools import partial
from multiprocessing import cpu_count

import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from lzn.pytorch_utils.distributed import get_world_size
from lzn.pytorch_utils.distributed import all_gather_nd
from lzn.pytorch_utils.distributed import distributed_reduce_sum

from .metric import Metric
from .metric_item import FloatMetricItem
from .feature_extractor.cm_feature_extractor import CMFeatureExtractor

from lzn.logging import execution_logger


class CMMetrics(Metric):
    def __init__(
        self,
        batch_size,
        num_gen_samples=-1,
        num_real_samples=-1,
        num_workers=0,
        seed=0,
        gpu_memory_limit=8192,
        is_split_size=5000,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_gen_samples = num_gen_samples
        self._num_real_samples = num_real_samples
        self._num_workers = num_workers
        self._seed = seed
        self._gpu_memory_limit = gpu_memory_limit
        self._is_split_size = is_split_size

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
        execution_logger.info("Initializing CMMetrics metric")
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

        config = tf_v1.ConfigProto(allow_soft_placement=True)  # allows DecodeJpeg to run on CPU in Inception graph
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(trainer.config.distributed.local_rank)
        self._session = tf_v1.Session(config=config)
        self._feature_extractor = CMFeatureExtractor(sess=self._session)
        self._manifold_estimator = ManifoldEstimator(self._session)

        preds = []
        spatial_preds = []
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
            preds.append(feature.pred)
            spatial_preds.append(feature.spatial_pred)
            labels.append(label)
            num_samples += data.shape[0]
            if (
                self._num_real_samples_per_device != -1
                and num_samples >= self._num_real_samples_per_device
                and (self._num_gen_samples_per_device == -1 or num_samples >= self._num_gen_samples_per_device)
            ):
                break
        preds = torch.cat(preds, dim=0)
        spatial_preds = torch.cat(spatial_preds, dim=0)
        labels = torch.cat(labels, dim=0)

        execution_logger.info(f"CMMetrics labels shape: {labels.shape}")

        num_samples = distributed_reduce_sum(value=num_samples, device=self._trainer.config.distributed.device)
        execution_logger.info(f"CMMetrics num_samples: {num_samples}")

        preds = all_gather_nd(preds)
        preds = torch.cat(preds, dim=0)

        spatial_preds = all_gather_nd(spatial_preds)
        spatial_preds = torch.cat(spatial_preds, dim=0)

        if self._num_real_samples == -1:
            self._num_real_samples = num_samples
            self._set_num_real_samples_per_device()
        else:
            preds = preds[: self._num_real_samples]
            spatial_preds = spatial_preds[: self._num_real_samples]
        execution_logger.info(f"CMMetrics real features shape: {preds.shape}, {spatial_preds.shape}")
        preds = preds.cpu().detach().numpy()
        self._real_pred_mu = np.mean(preds, axis=0)
        self._real_pred_sigma = np.cov(preds, rowvar=False)
        spatial_preds = spatial_preds.cpu().detach().numpy()
        self._real_spatial_pred_mu = np.mean(spatial_preds, axis=0)
        self._real_spatial_pred_sigma = np.cov(spatial_preds, rowvar=False)

        self._real_pred = preds  # Save it for computing precision and recall
        self._real_radii = self._manifold_estimator.manifold_radii(preds)  # Save it for computing precision and recall

        if self._num_gen_samples == -1:
            self._num_gen_samples = self._num_real_samples
            self._set_num_gen_samples_per_device()
        labels = labels[: self._num_gen_samples_per_device]
        self._labels = labels.cpu().detach().numpy()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self):
        execution_logger.info("Evaluating CMMetrics metric")
        torch.cuda.empty_cache()
        device = self._trainer.config.distributed.device
        if hasattr(self._trainer, "prior"):
            prior_data_loader = self._trainer.prior.get_data_loader(batch_size=self._per_device_batch_size)
            prior_data_iter = iter(prior_data_loader)
        else:
            prior_data_iter = itertools.repeat(None)
        preds = []
        spatial_preds = []
        softmaxs = []
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
            preds.append(feature.pred)
            spatial_preds.append(feature.spatial_pred)
            softmaxs.append(feature.softmax)
            num_samples += gen.shape[0]
            if num_samples >= num_target_samples:
                break

        preds = torch.cat(preds, dim=0)
        spatial_preds = torch.cat(spatial_preds, dim=0)
        softmaxs = torch.cat(softmaxs, dim=0)
        execution_logger.info(
            f"CMMetrics gen features shape local: {preds.shape}, {spatial_preds.shape}, {softmaxs.shape}"
        )

        preds = preds[:num_target_samples]
        preds = all_gather_nd(preds)
        preds = torch.cat(preds, dim=0)
        preds = preds[: self._num_real_samples]

        spatial_preds = spatial_preds[:num_target_samples]
        spatial_preds = all_gather_nd(spatial_preds)
        spatial_preds = torch.cat(spatial_preds, dim=0)
        spatial_preds = spatial_preds[: self._num_real_samples]

        softmaxs = softmaxs[:num_target_samples]
        softmaxs = all_gather_nd(softmaxs)
        softmaxs = torch.cat(softmaxs, dim=0)
        softmaxs = softmaxs[: self._num_real_samples]

        execution_logger.info(f"CMMetrics gen features shape: {preds.shape}, {spatial_preds.shape}, {softmaxs.shape}")

        metric_items = []

        # FID
        preds = preds.cpu().detach().numpy()
        gen_pred_mu = np.mean(preds, axis=0)
        gen_pred_sigma = np.cov(preds, rowvar=False)
        fid = cleanfid.fid.frechet_distance(
            mu1=self._real_pred_mu,
            sigma1=self._real_pred_sigma,
            mu2=gen_pred_mu,
            sigma2=gen_pred_sigma,
        )
        metric_item = FloatMetricItem(name="cm_fid", value=fid)
        metric_items.append(metric_item)

        # sFID
        spatial_preds = spatial_preds.cpu().detach().numpy()
        gen_spatial_pred_mu = np.mean(spatial_preds, axis=0)
        gen_spatial_pred_sigma = np.cov(spatial_preds, rowvar=False)
        sfid = cleanfid.fid.frechet_distance(
            mu1=self._real_spatial_pred_mu,
            sigma1=self._real_spatial_pred_sigma,
            mu2=gen_spatial_pred_mu,
            sigma2=gen_spatial_pred_sigma,
        )
        metric_item = FloatMetricItem(name="cm_sfid", value=sfid)
        metric_items.append(metric_item)

        # IS
        softmaxs = softmaxs.cpu().detach().numpy()
        scores = []
        for i in range(0, len(softmaxs), self._is_split_size):
            part = softmaxs[i : i + self._is_split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        is_ = np.mean(scores)
        metric_item = FloatMetricItem(name="cm_IS", value=is_)
        metric_items.append(metric_item)

        # Precision and Recall
        gen_radii = self._manifold_estimator.manifold_radii(preds)
        pr = self._manifold_estimator.evaluate_pr(
            features_1=self._real_pred, radii_1=self._real_radii, features_2=preds, radii_2=gen_radii
        )
        metric_item = FloatMetricItem(name="cm_precision", value=float(pr[0]))
        metric_items.append(metric_item)
        metric_item = FloatMetricItem(name="cm_recall", value=float(pr[1]))
        metric_items.append(metric_item)

        torch.cuda.empty_cache()
        return metric_items


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from
    https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        session,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats, radii = (
            np.zeros([1, 2048], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0 : end1 - begin1, begin2:end2] = self.distance_block.pairwise_distances(
                    row_batch, col_batch
                )

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [x[:, self.nhood_sizes] for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[0 : end1 - begin1, begin2:end2] = self.distance_block.pairwise_distances(
                    feature_batch, ref_batch
                )

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=np.bool_)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=np.bool_)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    """
    Calculate pairwise distances between vectors.

    Adapted from
    https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self, session):
        self.session = session

        # Initialize TF graph to calculate pairwise distances.
        with session.graph.as_default():
            self._features_batch1 = tf_v1.placeholder(tf_v1.float32, shape=[None, None])
            self._features_batch2 = tf_v1.placeholder(tf_v1.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf_v1.cast(self._features_batch1, tf_v1.float16),
                tf_v1.cast(self._features_batch2, tf_v1.float16),
            )
            self.distance_block = tf_v1.cond(
                tf_v1.reduce_all(tf_v1.math.is_finite(distance_block_16)),
                lambda: tf_v1.cast(distance_block_16, tf_v1.float32),
                lambda: _batch_pairwise_distances(self._features_batch1, self._features_batch2),
            )

            # Extra logic for less thans.
            self._radii1 = tf_v1.placeholder(tf_v1.float32, shape=[None, None])
            self._radii2 = tf_v1.placeholder(tf_v1.float32, shape=[None, None])
            dist32 = tf_v1.cast(self.distance_block, tf_v1.float32)[..., None]
            self._batch_1_in = tf_v1.math.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf_v1.math.reduce_any(dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )


def _batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors.
    """
    with tf_v1.variable_scope("pairwise_dist_block"):
        # Squared norms of each row in U and V.
        norm_u = tf_v1.reduce_sum(tf_v1.square(U), 1)
        norm_v = tf_v1.reduce_sum(tf_v1.square(V), 1)

        # norm_u as a column and norm_v as a row vectors.
        norm_u = tf_v1.reshape(norm_u, [-1, 1])
        norm_v = tf_v1.reshape(norm_v, [1, -1])

        # Pairwise squared Euclidean distances.
        D = tf_v1.maximum(norm_u - 2 * tf_v1.matmul(U, V, False, True) + norm_v, 0.0)

    return D


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))
