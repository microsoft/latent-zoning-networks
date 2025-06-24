# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed.nn.functional
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import glob
import os
import datetime
import numpy as np

from .trainer import Trainer
from .trainer import MetricInfo

import lzn.latent_flow
import lzn.data
import lzn.model
import lzn.optimizer
import lzn.prior
import lzn.logging
import lzn.loss
import lzn.metric
import lzn.logger
import lzn.label_anchor
import lzn.sampler
import lzn.t_scheduler
from lzn.logging import execution_logger
from lzn.metric import FloatMetricItem
from lzn.metric import FloatListMetricItem
from lzn.metric import metric_scope
from lzn.pytorch_utils.data_parallel import CustomDistributedDataParallel
from lzn.pytorch_utils.model_inference import batch_forward
from lzn.pytorch_utils.distributed import master_only


class ContrastiveTrainer(Trainer):
    def __init__(self, config):
        self._config = config

        self._dump_config()

        # Let the master thread download the data first to avoid race
        # condition on data file writing
        if self._config.distributed.local_rank == 0:
            self._data = lzn.data.from_config(config.data)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            self._data = lzn.data.from_config(config.data)

        self._prior = lzn.prior.from_config(config.prior)
        self._initialize_encoder()
        self._initialize_metrics()
        self._initialize_loggers()
        self._initialize_latent_flow()
        self._initialize_grad_scalar()

        self._encoder_ema = lzn.model.ema_from_config(self._encoder.parameters(), config.encoder_ema)

        self._optimizer = lzn.optimizer.from_config(
            [
                (self._encoder, config.optimizer.additional_params.encoder_lr),
            ],
            config.optimizer,
        )
        self._lr_scheduler = lzn.optimizer.lr_scheduler_from_config(self._optimizer, config.lr_scheduler)
        self._parameters = list(self._encoder.parameters())

        self._epoch = 0
        self._iteration = 0

    @property
    def config(self):
        return self._config

    @property
    def data(self):
        return self._data

    @property
    def encoder(self):
        return self._encoder

    @property
    def prior(self):
        return self._prior

    @master_only
    def _dump_config(self):
        os.makedirs(self._config.result_folder, exist_ok=True)
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        path = os.path.join(self._config.result_folder, f"config_{formatted_datetime}.yaml")
        with open(path, "w") as f:
            f.write(self._config.to_yaml())
        execution_logger.info(f"Config dumped to {path}.")

    def _initialize_metrics(self):
        self._metrics = []
        for metric_config in self._config.metrics:
            self._metrics.append(
                MetricInfo(
                    metric=lzn.metric.from_config(metric_config),
                    iteration_freq=metric_config.iteration_freq,
                    epoch_freq=metric_config.epoch_freq,
                    apply_on_ema=metric_config.apply_on_ema,
                    apply_on_samplers=None,
                )
            )

        for metric_info in self._metrics:
            metric_info.metric.initialize(self)

    def _initialize_loggers(self):
        self._loggers = []
        for logger_config in self._config.loggers:
            self._loggers.append(lzn.logger.from_config(logger_config))

        for logger in self._loggers:
            logger.initialize(self)

    def _clean_up_loggers(self):
        for logger in self._loggers:
            logger.clean_up()

    @master_only
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "encoder": self._encoder.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "epoch": self._epoch,
            "iteration": self._iteration,
        }
        if self._encoder_ema is not None:
            state["encoder_ema"] = self._encoder_ema.state_dict()
        state["grad_scalar"] = self._grad_scalar.state_dict()
        torch.save(state, path)

    @master_only
    def _save_checkpoint_if_needed(self, iter_end=False, epoch_end=False):
        epoch = self._epoch
        iteration = self._iteration

        case1 = (
            (self._config.checkpoint.epoch_freq > 0)
            and (epoch % self._config.checkpoint.epoch_freq == 0)
            and epoch_end
        )
        case2 = (
            (self._config.checkpoint.iteration_freq > 0)
            and (iteration % self._config.checkpoint.iteration_freq == 0)
            and iter_end
        )
        case3 = (iteration == self._num_iterations) and iter_end

        if case1 or case2 or case3:
            self.save_checkpoint(
                os.path.join(
                    self._config.checkpoint.folder,
                    self._config.checkpoint.format.format(epoch=epoch, iteration=iteration),
                )
            )

    def load_checkpoint(self, path):
        map_location = {
            "cuda:0": self._config.distributed.device,
        }
        state = torch.load(path, map_location=map_location)
        self._encoder.load_state_dict(state["encoder"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._lr_scheduler.load_state_dict(state["lr_scheduler"])
        if self._encoder_ema is not None:
            self._encoder_ema.load_state_dict(state["encoder_ema"], device=self._config.distributed.device)
        self._epoch = state["epoch"]
        self._iteration = state["iteration"]
        self._grad_scalar.load_state_dict(state["grad_scalar"])

    def _load_checkpoint_if_needed(self):
        if self._config.checkpoint.load_checkpoint == "auto":
            checkpoint_path = self.find_checkpoint()
            if checkpoint_path is not None:
                self.load_checkpoint(checkpoint_path)
                execution_logger.info(f"Checkpoint loaded from {checkpoint_path}.")
            else:
                execution_logger.warning("No checkpoint found.")
        elif self._config.checkpoint.load_checkpoint == "manual":
            checkpoint_path = self._config.checkpoint.path
            self.load_checkpoint(checkpoint_path)
            execution_logger.info(f"Checkpoint loaded from {checkpoint_path}.")

    def _log_ema_metrics_if_needed(self, iter_end=False, epoch_end=False):
        if self._encoder_ema is None:
            return
        self._encoder_ema.store(self._encoder.parameters())
        self._encoder_ema.copy_to(self._encoder.parameters())
        with metric_scope("ema"):
            self._log_metrics_if_needed(iter_end=iter_end, epoch_end=epoch_end, ema=True)
        self._encoder_ema.restore(self._encoder.parameters())

    def _log_metrics_if_needed(self, iter_end=False, epoch_end=False, ema=False):
        self.eval_mode()
        metric_items = []
        for metric_info in self._metrics:
            case1 = (
                (metric_info.iteration_freq > 0) and (self._iteration % metric_info.iteration_freq == 0) and iter_end
            )
            case2 = (metric_info.epoch_freq > 0) and (self._epoch % metric_info.epoch_freq == 0) and epoch_end
            case3 = (self._iteration == self._num_iterations) and iter_end
            case4 = (not ema) or metric_info.apply_on_ema
            if case4 and (case1 or case2 or case3):
                metric_items += metric_info.metric.evaluate()

        if len(metric_items) > 0:
            for logger in self._loggers:
                logger.log(
                    epoch=self._epoch,
                    iteration=self._iteration,
                    metric_items=metric_items,
                )
            for metric_item in metric_items:
                metric_item.clean_up()

    def find_checkpoint(self):
        candidates = glob.glob(
            os.path.join(
                self._config.checkpoint.folder,
                "*." + self._config.checkpoint.format.split(".")[-1],
            )
        )
        if len(candidates) == 0:
            return None
        else:
            return sorted(candidates)[-1]

    def _initialize_latent_flow(self):
        self._encoder_latent_flow = lzn.latent_flow.from_config(self._config.encoder_latent_flow, trainer=self).to(
            self._config.distributed.device
        )

    def _initialize_grad_scalar(self):
        self._grad_scalar = GradScaler(enabled=self._config.training.enable_mixed_precision_training)

    def _initialize_encoder(self):
        self._encoder = lzn.model.from_config(self._config.encoder).to(self._config.distributed.device)
        execution_logger.info(f"#params in encoder: {self._encoder.num_parameters}")

        self._encoder = CustomDistributedDataParallel(self._encoder, device_ids=[self._config.distributed.local_rank])

    def _get_latent_from_data(self, data, prior_data=None, return_full=False):
        assert prior_data is not None
        """
        if prior_data is None:
            prior_data = torch.zeros(
                data.shape[0],
                self._config.latent_dim,
                dtype=data.dtype,
                device=data.device,
            )
        """
        anchor = self._encoder(data)
        # This all_gather ensures that gradients can go through
        all_anchor = torch.distributed.nn.functional.all_gather(anchor)
        all_anchor = torch.cat(all_anchor, dim=0)
        num_steps = self._config.training.ode_num_steps
        latent = self._encoder_latent_flow(
            full_encoded=all_anchor,
            encoded=anchor,
            prior_data=prior_data,
            num_steps=num_steps,
        )
        if return_full:
            return latent, anchor
        else:
            return latent

    def get_latent_from_data(self, batch_size=None, **kwargs):
        if batch_size is None:
            return self._get_latent_from_data(**kwargs)
        else:
            return batch_forward(
                model=self._get_latent_from_data,
                batch_size=batch_size,
                **kwargs,
            )

    def get_data_assignment_from_latent(self, latent, data):
        anchor = self._encoder(data)
        # This all_gather ensures that gradients can go through
        all_anchor = torch.distributed.nn.functional.all_gather(anchor)
        all_anchor = torch.cat(all_anchor, dim=0)

        final_point, pdfs = self._encoder_latent_flow(
            full_encoded=all_anchor,
            initial_point=latent,
            num_steps=self._config.training.ode_num_steps,
            reverse=True,
        )
        return final_point, pdfs, all_anchor

    def train_mode(self):
        self._encoder.train()

    def eval_mode(self):
        self._encoder.eval()

    def _train_iteration(self, data, prior_data):
        self._optimizer.zero_grad()
        self.train_mode()

        data = data.to(self._config.distributed.device)
        prior_data = prior_data.to(self._config.distributed.device)

        with autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self._config.training.enable_mixed_precision_training,
        ):
            encoded = self.get_latent_from_data(
                data=data[:, 0],
                prior_data=prior_data * self._config.training.prior_data_scale,
                return_full=False,
            )
            final_point, pdfs, all_anchor = self.get_data_assignment_from_latent(
                latent=encoded,
                data=data[:, 1],
            )
            with torch.no_grad():
                ode_error = all_anchor[pdfs[-1].argmax(dim=1)] - final_point
                ode_error = ode_error.norm(dim=1) / all_anchor[pdfs[-1].argmax(dim=1)].norm(dim=1)
                ode_error = ode_error.mean().item()
            pdfs = torch.stack(pdfs, dim=0)
            pdfs = pdfs / pdfs.sum(dim=2, keepdim=True)
            shift = self._config.distributed.global_rank * data.shape[0]
            pdfs = pdfs[:, np.arange(pdfs.shape[1]), np.arange(pdfs.shape[1]) + shift]
            cut_off_step = int(pdfs.shape[0] * self._config.training.assignment_loss_cut_off_step)
            max_prob = torch.max(pdfs[cut_off_step:, :], dim=0)[0]
            prob_loss = -max_prob.mean()
            cross_entropy_loss = -torch.log(max_prob).mean()

            loss = (
                prob_loss * self._config.training.loss_weight.prob
                + cross_entropy_loss * self._config.training.loss_weight.cross_entropy
            )

            assignment_acc = pdfs[-1].mean()

            metric_items = []
            with metric_scope("loss"):
                metric_items.append(
                    FloatMetricItem(
                        name="total",
                        value=loss.item(),
                    )
                )
                metric_items.append(
                    FloatMetricItem(
                        name="prob",
                        value=prob_loss.item(),
                    )
                )
                metric_items.append(
                    FloatMetricItem(
                        name="cross_entropy",
                        value=cross_entropy_loss.item(),
                    )
                )
            metric_items.append(
                FloatMetricItem(
                    name="assignment_acc",
                    value=assignment_acc.item(),
                )
            )
            metric_items.append(
                FloatMetricItem(
                    name="ode_error",
                    value=ode_error,
                )
            )
        is_finite = torch.isfinite(loss)
        if torch.all(torch.stack(torch.distributed.nn.functional.all_gather(is_finite))).item():
            metric_items.append(
                FloatMetricItem(
                    name="grad_scale",
                    value=self._grad_scalar.get_scale(),
                )
            )
            self._grad_scalar.scale(loss).backward()
            self._grad_scalar.unscale_(self._optimizer)

            gradient_norm = torch.nn.utils.clip_grad_norm_(self._parameters, self._config.training.gradient_clipping)
            metric_items.append(
                FloatMetricItem(
                    name="gradient_norm",
                    value=gradient_norm,
                )
            )
            self._grad_scalar.step(self._optimizer)
            self._grad_scalar.update()

            metric_items.append(
                FloatListMetricItem(
                    name="lr",
                    value=self._lr_scheduler.get_last_lr(),
                )
            )
            if torch.isfinite(gradient_norm):
                # ==> Otherwise, grad scalar did not perform the step
                self._lr_scheduler.step()
                if self._encoder_ema is not None:
                    self._encoder_ema.update(self._encoder.parameters())
        else:
            execution_logger.warning("Infinite loss detected. Skip the iteration.")
        metric_items.append(
            FloatMetricItem(
                name="is_finite",
                value=is_finite,
            )
        )
        self._training_metric_items = metric_items

    @property
    def training_metric_items(self):
        return self._training_metric_items

    def evaluate(self):
        self._load_checkpoint_if_needed()

        self._num_iterations = self._config.training.num_encoder_iterations

        self._log_ema_metrics_if_needed(epoch_end=True, iter_end=True)
        self._log_metrics_if_needed(epoch_end=True, iter_end=True)

    def train(self):
        self._load_checkpoint_if_needed()

        data_loader, sampler = self._data.get_data_loader_and_sampler(
            batch_size=self._config.training.batch_size,
            num_workers=self._config.training.data_num_workers,
            seed=self._config.training.data_seed,
            train=True,
            rank=self._config.distributed.global_rank,
            world_size=self._config.distributed.world_size,
        )
        prior_data_loader = self._prior.get_data_loader(batch_size=sampler.per_device_batch_size)
        prior_data_iter = iter(prior_data_loader)

        stop = False

        self._num_iterations = self._config.training.num_encoder_iterations

        try:
            while self._iteration < self._num_iterations:
                sampler.set_epoch(self._epoch)
                sampler.set_start_iter_by_past_iters(self._iteration)
                for data, _ in iter(data_loader):
                    prior_data = next(prior_data_iter)
                    self._train_iteration(data=data, prior_data=prior_data)
                    self._iteration += 1
                    if self._iteration % sampler.num_iterations_per_epoch != 0:
                        self._log_metrics_if_needed(iter_end=True)
                        self._log_ema_metrics_if_needed(iter_end=True)
                        self._save_checkpoint_if_needed(iter_end=True)

                        if self._iteration == self._num_iterations:
                            stop = True
                            break
                if stop:
                    break
                self._epoch += 1
                self._log_metrics_if_needed(epoch_end=True, iter_end=True)
                self._log_ema_metrics_if_needed(epoch_end=True, iter_end=True)
                self._save_checkpoint_if_needed(epoch_end=True, iter_end=True)
        finally:
            self._clean_up_loggers()
