# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed.nn.functional
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import glob
import os
import numpy as np
from contextlib import contextmanager
import datetime

from .trainer import Trainer
from .trainer import LossInfo, MetricInfo, SamplerInfo

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
from lzn.pytorch_utils.model_inference import to_uint8
from lzn.pytorch_utils.distributed import master_only
from lzn.pytorch_utils.class_weights import get_class_weights


class GenerationTrainer(Trainer):
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
        self._initialize_decoder()
        self._initialize_label_anchor()
        self._initialize_training_sampler()
        self._initialize_evaluation_samplers()
        self._initialize_t_scheduler()
        self._initialize_metrics()
        self._initialize_loggers()
        self._initialize_losses()
        self._initialize_latent_flow()
        self._initialize_grad_scalar()

        self._decoder_ema = lzn.model.ema_from_config(self._decoder.parameters(), config.decoder_ema)
        self._encoder_ema = lzn.model.ema_from_config(self._encoder.parameters(), config.encoder_ema)
        if self._label_anchor is not None:
            self._label_anchor_ema = lzn.model.ema_from_config(
                self._label_anchor.parameters(), config.label_anchor_ema
            )
        else:
            self._label_anchor_ema = None

        self._optimizer = lzn.optimizer.from_config(
            [
                (self._encoder, config.optimizer.additional_params.encoder_lr),
                (self._decoder, config.optimizer.additional_params.decoder_lr),
                (
                    self._label_anchor,
                    None if self._label_anchor is None else config.optimizer.additional_params.label_anchor_lr,
                ),
            ],
            config.optimizer,
        )
        self._lr_scheduler = lzn.optimizer.lr_scheduler_from_config(self._optimizer, config.lr_scheduler)
        self._parameters = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + (list(self._label_anchor.parameters()) if self._label_anchor is not None else [])
        )

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
    def decoder(self):
        return self._decoder

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
                    apply_on_samplers=metric_config.apply_on_samplers,
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

    def _initialize_losses(self):
        self._losses = []
        for loss_config in self._config.losses:
            self._losses.append(
                LossInfo(
                    loss=lzn.loss.from_config(loss_config),
                    weight=loss_config.weight,
                )
            )

    @master_only
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "encoder": self._encoder.state_dict(),
            "decoder": self._decoder.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "epoch": self._epoch,
            "iteration": self._iteration,
        }
        if self._label_anchor is not None:
            state["label_anchor"] = self._label_anchor.state_dict()
        if self._decoder_ema is not None:
            state["decoder_ema"] = self._decoder_ema.state_dict()
        if self._encoder_ema is not None:
            state["encoder_ema"] = self._encoder_ema.state_dict()
        if self._label_anchor_ema is not None:
            state["label_anchor_ema"] = self._label_anchor_ema.state_dict()
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
        case3 = (iteration in self._num_iterations) and iter_end

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
        self._decoder.load_state_dict(state["decoder"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._lr_scheduler.load_state_dict(state["lr_scheduler"])
        if self._label_anchor is not None:
            self._label_anchor.load_state_dict(state["label_anchor"])
        if self._decoder_ema is not None:
            self._decoder_ema.load_state_dict(state["decoder_ema"], device=self._config.distributed.device)
        if self._encoder_ema is not None:
            self._encoder_ema.load_state_dict(state["encoder_ema"], device=self._config.distributed.device)
        if self._label_anchor_ema is not None:
            self._label_anchor_ema.load_state_dict(
                state["label_anchor_ema"],
                device=self._config.distributed.device,
            )
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
        if self._decoder_ema is None and self._encoder_ema is None and self._label_anchor_ema is None:
            return
        if self._decoder_ema is not None:
            self._decoder_ema.store(self._decoder.parameters())
            self._decoder_ema.copy_to(self._decoder.parameters())
        if self._encoder_ema is not None:
            self._encoder_ema.store(self._encoder.parameters())
            self._encoder_ema.copy_to(self._encoder.parameters())
        if self._label_anchor_ema is not None:
            self._label_anchor_ema.store(self._label_anchor.parameters())
            self._label_anchor_ema.copy_to(self._label_anchor.parameters())
        with metric_scope("ema"):
            self._log_metrics_if_needed(iter_end=iter_end, epoch_end=epoch_end, ema=True)
        if self._decoder_ema is not None:
            self._decoder_ema.restore(self._decoder.parameters())
        if self._encoder_ema is not None:
            self._encoder_ema.restore(self._encoder.parameters())
        if self._label_anchor_ema is not None:
            self._label_anchor_ema.restore(self._label_anchor.parameters())

    def _log_metrics_if_needed(self, iter_end=False, epoch_end=False, ema=False):
        self.eval_mode()
        metric_items = []
        for metric_info in self._metrics:
            case1 = (
                (metric_info.iteration_freq > 0) and (self._iteration % metric_info.iteration_freq == 0) and iter_end
            )
            case2 = (metric_info.epoch_freq > 0) and (self._epoch % metric_info.epoch_freq == 0) and epoch_end
            case3 = (self._iteration in self._num_iterations) and iter_end
            case4 = (not ema) or metric_info.apply_on_ema
            if case4 and (case1 or case2 or case3):
                if metric_info.apply_on_samplers:
                    for evaluation_sampler_info in self._evaluation_samplers:
                        sampler_name = evaluation_sampler_info.name
                        with self.using_sampler(evaluation_sampler_info.sampler):
                            with metric_scope(sampler_name):
                                metric_items += metric_info.metric.evaluate()
                else:
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
        if self._config.training.conditional:
            self._label_latent_flow = lzn.latent_flow.from_config(self._config.label_latent_flow, trainer=self).to(
                self._config.distributed.device
            )
        else:
            self._label_latent_flow = None

    def _initialize_grad_scalar(self):
        self._grad_scalar = GradScaler(enabled=self._config.training.enable_mixed_precision_training)

    def _initialize_training_sampler(self):
        self._training_sampler = lzn.sampler.from_config(self._config.training_sampler)

        self._training_sampler.initialize(self)

    def _initialize_evaluation_samplers(self):
        self._evaluation_samplers = []
        for evaluation_sampler_config in self._config.evaluation_samplers:
            self._evaluation_samplers.append(
                SamplerInfo(
                    sampler=lzn.sampler.from_config(evaluation_sampler_config),
                    name=evaluation_sampler_config.sampler_name,
                )
            )

        for evaluation_sampler in self._evaluation_samplers:
            evaluation_sampler.sampler.initialize(self)

    def _initialize_t_scheduler(self):
        self.t_scheduler = lzn.t_scheduler.from_config(self._config.t_scheduler)

    def _initialize_encoder(self):
        self._encoder = lzn.model.from_config(self._config.encoder).to(self._config.distributed.device)
        execution_logger.info(f"#params in encoder: {self._encoder.num_parameters}")

        self._encoder = CustomDistributedDataParallel(self._encoder, device_ids=[self._config.distributed.local_rank])

        self._encoder_enabled = True

    def _initialize_decoder(self):
        self._decoder = lzn.model.from_config(self._config.decoder).to(self._config.distributed.device)
        execution_logger.info(f"#params in decoder: {self._decoder.num_parameters}")

        self._decoder = CustomDistributedDataParallel(self._decoder, device_ids=[self._config.distributed.local_rank])

        self._decoder_enabled = True

    def _initialize_label_anchor(self):
        if self._config.training.conditional:
            self._label_anchor = lzn.label_anchor.from_config(self._config.label_anchor).to(
                self._config.distributed.device
            )
            execution_logger.info(f"#params in label anchor: {self._label_anchor.num_parameters}")

            self._label_anchor = CustomDistributedDataParallel(
                self._label_anchor,
                device_ids=[self._config.distributed.local_rank],
            )
        else:
            self._label_anchor = None

        self._label_anchor_enabled = True

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

    def get_class_anchor_from_latent(self, latent, class_weights=None):
        if self._config.training.conditional:
            final_point, pdfs = self._label_latent_flow(
                full_encoded=self._label_anchor(None),
                initial_point=latent,
                num_steps=self._config.training.label_ode_num_steps,
                reverse=True,
                weights=class_weights,
            )
            return final_point, pdfs
        else:
            return None, None

    def get_discrete_class_label_from_latent(self, latent, class_weights=None):
        if self._config.training.conditional:
            with torch.no_grad():
                final_point, _ = self.get_class_anchor_from_latent(latent, class_weights=class_weights)
                return self._label_anchor.get_label(final_point)
        else:
            return torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device)

    def _get_ini_noise_from_latent_and_data(
        self,
        latent,
        discrete_class_label,
        sampler,
        return_all,
        x1,
    ):
        if self._config.training.conditional:
            if discrete_class_label is None:
                discrete_class_label = self.get_discrete_class_label_from_latent(latent)
            label = self._get_label_from_discrete_class_label(discrete_class_label)
        else:
            label = None
        ini_noise = sampler.inverse(
            latent=latent,
            label=label,
            return_all=return_all,
            x1=x1,
        )
        return ini_noise

    def get_ini_noise_from_latent_and_data(
        self,
        latent,
        return_all,
        sampler=None,
        discrete_class_label=None,
        batch_size=None,
        x1=None,
    ):
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self._config.distributed.device)
        if batch_size is None:
            ini_noise = self._get_ini_noise_from_latent_and_data(
                latent=latent,
                discrete_class_label=discrete_class_label,
                sampler=sampler or self._sampler,
                return_all=return_all,
                x1=x1,
            )
        else:
            ini_noise = batch_forward(
                model=self._get_ini_noise_from_latent_and_data,
                batch_size=self._config.training.batch_size,
                latent=latent,
                discrete_class_label=discrete_class_label,
                sampler=sampler or self._sampler,
                return_all=return_all,
                x1=x1,
            )
        return ini_noise

    def _get_label_from_discrete_class_label(self, discrete_class_label):
        if self._config.training.conditional:
            return torch.nn.functional.one_hot(discrete_class_label, num_classes=self._data.num_classes).float()
        else:
            return None

    def _get_data_from_latent(self, latent, discrete_class_label, sampler, return_all, x0, num_samples, device):
        if self._config.training.conditional:
            if discrete_class_label is None:
                discrete_class_label = self.get_discrete_class_label_from_latent(latent)
            label = self._get_label_from_discrete_class_label(discrete_class_label=discrete_class_label)
        else:
            label = None
        data = sampler.sample(
            latent=latent, label=label, return_all=return_all, x0=x0, num_samples=num_samples, device=device
        )
        return data

    def get_data_from_latent(
        self,
        latent,
        return_all,
        sampler=None,
        discrete_class_label=None,
        batch_size=None,
        uint8=False,
        x0=None,
        num_samples=None,
        device=None,
    ):
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self._config.distributed.device)
        if batch_size is None:
            data = self._get_data_from_latent(
                latent=latent,
                discrete_class_label=discrete_class_label,
                sampler=sampler or self._sampler,
                return_all=return_all,
                x0=x0,
                num_samples=num_samples,
                device=device,
            )
        else:
            data = batch_forward(
                model=self._get_data_from_latent,
                batch_size=batch_size,
                latent=latent,
                discrete_class_label=discrete_class_label,
                sampler=sampler or self._sampler,
                return_all=return_all,
                x0=x0,
                num_samples=num_samples,
                device=device,
            )
        if uint8:
            if not isinstance(data, tuple):
                data = (data,)
                is_tuple = False
            else:
                is_tuple = True
            new_data = []
            for d in data:
                d = d.cpu().detach().numpy()
                d = to_uint8(
                    x=d,
                    min=-1,
                    max=1,
                )
                new_data.append(d)
            data = tuple(new_data) if is_tuple else new_data[0]
        return data

    def get_latent_from_label(self, label, prior_data):
        if not self._config.training.conditional:
            return prior_data
        num_steps = self._config.training.label_ode_num_steps
        anchor = self._label_anchor(None)
        encoded = anchor[label]
        latent = self._label_latent_flow(
            full_encoded=anchor,
            encoded=encoded,
            prior_data=prior_data,
            num_steps=num_steps,
        )
        return latent

    @contextmanager
    def using_sampler(self, sampler):
        self._sampler = sampler
        try:
            yield
        finally:
            self._sampler = None

    def train_mode(self):
        self._encoder.train()
        self._decoder.train()
        if self._label_anchor is not None:
            self._label_anchor.train()

    def eval_mode(self):
        self._encoder.eval()
        self._decoder.eval()
        if self._label_anchor is not None:
            self._label_anchor.eval()

    def _train_iteration(self, data, label, prior_data):
        self._optimizer.zero_grad()
        self.train_mode()

        data = data.to(self._config.distributed.device)
        label = label.to(self._config.distributed.device)
        prior_data = prior_data.to(self._config.distributed.device)

        with autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self._config.training.enable_mixed_precision_training,
        ):
            with torch.set_grad_enabled(self._encoder_enabled):
                encoded = self.get_latent_from_data(
                    data=data,
                    prior_data=prior_data,
                    return_full=False,
                )
            if self._decoder_enabled or self._encoder_enabled:
                t = self.t_scheduler.sample(
                    batch_size=data.shape[0],
                    device=self._config.distributed.device,
                )
                xt, x0 = self._training_sampler.generate_xt(t=t, x1=data)
                (
                    x1_hat,
                    x1_m_x0_hat,
                ) = self._training_sampler.predict_x1_and_x1_m_x0_from_xt_and_x0(
                    latent=encoded,
                    label=self._get_label_from_discrete_class_label(label),
                    t=t,
                    xt=xt,
                    x0=x0,
                )
            else:
                x1_hat = x1_m_x0_hat = xt = x0 = None

            if self._config.training.conditional and (self._encoder_enabled or self._label_anchor_enabled):
                class_weights = get_class_weights(label, self._data.num_classes)
                _, pdfs = self.get_class_anchor_from_latent(encoded, class_weights=class_weights)
                pdfs = torch.stack(pdfs, dim=0)
                pdfs = pdfs / pdfs.sum(dim=2, keepdim=True)
                pdfs = pdfs[:, np.arange(pdfs.shape[1]), label]
                cut_off_step = int(pdfs.shape[0] * self._config.training.label_loss_cut_off_step)
                label_loss = -torch.max(pdfs[cut_off_step:, :], dim=0)[0].mean()
            else:
                label_loss = 0.0

            total_loss = 0.0
            metric_items = []
            with metric_scope("loss"):
                if self._config.training.conditional:
                    with metric_scope("label"):
                        total_loss += self._config.training.label_loss_coe * label_loss
                        metric_items.append(
                            FloatMetricItem(
                                name="label",
                                value=label_loss,
                            )
                        )
                for loss_info_i, loss_info in enumerate(self._losses):
                    with metric_scope(loss_info.loss.name):
                        if x1_hat is None:
                            loss = 0.0
                        else:
                            loss = loss_info.loss(
                                x1=data,
                                x1_hat=x1_hat,
                                x1_m_x0=data - x0,
                                x1_m_x0_hat=x1_m_x0_hat,
                            ).mean()
                        total_loss += loss_info.weight * self._config.training.recon_loss_coe * loss
                        metric_items.append(
                            FloatMetricItem(
                                name="recon",
                                value=loss,
                            )
                        )
                metric_items.append(
                    FloatMetricItem(
                        name="total",
                        value=total_loss,
                    )
                )
        is_finite = torch.isfinite(total_loss)
        if torch.all(torch.stack(torch.distributed.nn.functional.all_gather(is_finite))).item():
            metric_items.append(
                FloatMetricItem(
                    name="grad_scale",
                    value=self._grad_scalar.get_scale(),
                )
            )
            self._grad_scalar.scale(total_loss).backward()
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
                if self._decoder_ema is not None and self._decoder_enabled:
                    self._decoder_ema.update(self._decoder.parameters())
                if self._encoder_ema is not None and self._encoder_enabled:
                    self._encoder_ema.update(self._encoder.parameters())
                if self._label_anchor_ema is not None and self._label_anchor_enabled:
                    self._label_anchor_ema.update(self._label_anchor.parameters())
        else:
            execution_logger.warning("Infinite loss detected. Skip the iteration.")
        metric_items.append(
            FloatMetricItem(
                name="is_finite",
                value=is_finite,
            )
        )
        metric_items.append(
            FloatMetricItem(
                name="decoder_enabled",
                value=self._decoder_enabled,
            )
        )
        metric_items.append(
            FloatMetricItem(
                name="encoder_enabled",
                value=self._encoder_enabled,
            )
        )
        metric_items.append(
            FloatMetricItem(
                name="label_anchor_enabled",
                value=self._label_anchor_enabled,
            )
        )

        self._training_metric_items = metric_items

    def _disable(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _disable_models_if_needed(self):
        if self._iteration >= self._config.training.num_decoder_iterations and self._decoder_enabled:
            self._disable(self._decoder)
            self._decoder_enabled = False
        if self._iteration >= self._config.training.num_encoder_iterations and self._encoder_enabled:
            self._disable(self._encoder)
            self._encoder_enabled = False
        if self._label_anchor is not None and (
            self._iteration >= self._config.training.num_label_anchor_iterations and self._label_anchor_enabled
        ):
            self._disable(self._label_anchor)
            self._label_anchor_enabled = False

    @property
    def training_metric_items(self):
        return self._training_metric_items

    def evaluate(self):
        self._load_checkpoint_if_needed()

        self._num_iterations = [
            self._config.training.num_decoder_iterations,
            self._config.training.num_encoder_iterations,
        ]
        if self._label_anchor is not None:
            self._num_iterations.append(self._config.training.num_label_anchor_iterations)

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

        self._num_iterations = [
            self._config.training.num_decoder_iterations,
            self._config.training.num_encoder_iterations,
        ]
        if self._label_anchor is not None:
            self._num_iterations.append(self._config.training.num_label_anchor_iterations)
        max_num_iterations = max(self._num_iterations)

        try:
            while self._iteration < max_num_iterations:
                sampler.set_epoch(self._epoch)
                sampler.set_start_iter_by_past_iters(self._iteration)
                for data, label in iter(data_loader):
                    self._disable_models_if_needed()
                    prior_data = next(prior_data_iter)
                    self._train_iteration(data=data, label=label, prior_data=prior_data)
                    self._iteration += 1
                    if self._iteration % sampler.num_iterations_per_epoch != 0:
                        self._log_metrics_if_needed(iter_end=True)
                        self._log_ema_metrics_if_needed(iter_end=True)
                        self._save_checkpoint_if_needed(iter_end=True)

                        if self._iteration == max_num_iterations:
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
