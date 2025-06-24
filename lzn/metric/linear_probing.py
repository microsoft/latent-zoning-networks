# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from collections import namedtuple
from tqdm import tqdm

from .metric import Metric
from .metric_item import FloatListMetricItem

from lzn.logging import execution_logger
from lzn.pytorch_utils.data_parallel import CustomDistributedDataParallel
from lzn.pytorch_utils.distributed import distributed_reduce_mean
from lzn.pytorch_utils.distributed import distributed_reduce_sum

TopkResult = namedtuple("TopkResult", ["accuracy", "correct", "total"])
IterationResult = namedtuple("IterationResult", ["loss", "topk_result"])
TrainTestResult = namedtuple(
    "TrainTestResult", ["train_loss", "train_topk_accuracy", "test_loss", "test_topk_accuracy"]
)


class LinearHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class LinearProbing(Metric):
    def __init__(
        self,
        batch_size,
        lr,
        weight_decay,
        num_epochs,
        mode,
        num_workers=16,
        seed=100,
        num_classes=1000,
        topk=[1, 5, 10],
    ):
        super().__init__()
        self._batch_size = batch_size
        self._lr = lr
        self._weight_decay = weight_decay
        self._num_epochs = num_epochs
        self._mode = mode
        if self._mode not in ["no_head", "head", "latent", "latent_center"]:
            raise ValueError(f"Invalid mode: {self._mode}")
        self._num_workers = num_workers
        self._seed = seed
        self._num_classes = num_classes
        self._topk = topk
        self._name = f"linear_probing_{self._mode}"

    def initialize(self, trainer):
        execution_logger.info(f"Initializing {self._name} metric")
        self._trainer = trainer
        if self._mode in ["head", "latent", "latent_center"]:
            self._input_dim = self._trainer.encoder.head_out_dim
        elif self._mode == "no_head":
            self._input_dim = self._trainer.encoder.channels_out
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def _initialize_model(self):
        self._linear_head = LinearHead(in_dim=self._input_dim, out_dim=self._num_classes)
        self._linear_head.to(self._trainer.config.distributed.device)
        self._linear_head = CustomDistributedDataParallel(
            self._linear_head, device_ids=[self._trainer.config.distributed.local_rank]
        )

    def _initialize_optimizer(self):
        self._optimizer = torch.optim.AdamW(
            self._linear_head.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

    def topk_result(self, logit, label, topk):
        size = label.size(0)
        _, pred = logit.topk(k=max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        accuracy_result = {}
        correct_result = {}
        total_result = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum().item()
            accuracy_result[k] = correct_k / size
            correct_result[k] = correct_k
            total_result[k] = size
        return TopkResult(accuracy=accuracy_result, correct=correct_result, total=total_result)

    def _get_feature(self, data, prior_data):
        if self._mode == "latent":
            return self._trainer.get_latent_from_data(
                data=data,
                prior_data=prior_data,
                return_full=False,
            )
        elif self._mode == "latent_center":
            return self._trainer.get_latent_from_data(
                data=data,
                prior_data=prior_data * 0.0,
                return_full=False,
            )
        elif self._mode == "head":
            return self._trainer._encoder(data, apply_head=True)
        elif self._mode == "no_head":
            return self._trainer._encoder(data, apply_head=False)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def _run_iteration(self, data, label, prior_data, train=True):
        if train:
            self._linear_head.train()
        else:
            self._linear_head.eval()
        data = data.to(self._trainer.config.distributed.device)
        label = label.to(self._trainer.config.distributed.device)
        prior_data = prior_data.to(self._trainer.config.distributed.device)
        if train:
            self._optimizer.zero_grad()
        with torch.no_grad():
            feature = self._get_feature(data=data, prior_data=prior_data).detach()
        logit = self._linear_head(feature)
        loss = torch.nn.functional.cross_entropy(input=logit, target=label)
        if train:
            loss.backward()
            self._optimizer.step()
        topk_result = self.topk_result(logit=logit, label=label, topk=self._topk)
        return IterationResult(loss=loss.item(), topk_result=topk_result)

    def _train(self, train_data_loader, train_sampler, test_data_loader, prior_data_iter):
        iteration = 0
        train_losses = []
        train_topk_accuracies = {k: [] for k in self._topk}
        test_losses = []
        test_topk_accuracies = {k: [] for k in self._topk}
        for epoch in range(self._num_epochs):
            train_sampler.set_epoch(epoch)
            train_sampler.set_start_iter_by_past_iters(iteration)
            epoch_losses = []
            epoch_topk_accuracies = {k: [] for k in self._topk}
            for data, label in tqdm(train_data_loader):
                prior_data = next(prior_data_iter)
                prior_data = prior_data[: data.shape[0]]
                result = self._run_iteration(data=data, label=label, prior_data=prior_data)
                iteration += 1
                epoch_losses.append(result.loss)
                for k in self._topk:
                    epoch_topk_accuracies[k].append(result.topk_result.accuracy[k])
            epoch_losses = np.mean(epoch_losses)
            for k in self._topk:
                epoch_topk_accuracies[k] = np.mean(epoch_topk_accuracies[k])
            train_losses.append(epoch_losses)
            for k in self._topk:
                train_topk_accuracies[k].append(epoch_topk_accuracies[k])
            test_result = self._test(data_loader=test_data_loader, prior_data_iter=prior_data_iter)
            test_losses.append(test_result.loss)
            for k in self._topk:
                test_topk_accuracies[k].append(test_result.topk_result.accuracy[k])
            execution_logger.info(
                f"Epoch {epoch}: train_loss={epoch_losses}, train_topk_accuracy={epoch_topk_accuracies}, "
                f"test_loss={test_result.loss}, test_topk_accuracy={test_result.topk_result.accuracy}"
            )
        return TrainTestResult(
            train_loss=train_losses,
            train_topk_accuracy=train_topk_accuracies,
            test_loss=test_losses,
            test_topk_accuracy=test_topk_accuracies,
        )

    @torch.no_grad()
    def _test(self, data_loader, prior_data_iter):
        test_losses = []
        test_topk_correct = {k: [] for k in self._topk}
        test_topk_total = {k: [] for k in self._topk}
        for data, label in tqdm(data_loader):
            prior_data = next(prior_data_iter)
            prior_data = prior_data[: data.shape[0]]
            result = self._run_iteration(data=data, label=label, prior_data=prior_data, train=False)
            test_losses.append(result.loss)
            for k in self._topk:
                test_topk_correct[k].append(result.topk_result.correct[k])
                test_topk_total[k].append(result.topk_result.total[k])
        test_losses = np.mean(test_losses)
        test_losses = distributed_reduce_mean(value=test_losses, device=self._trainer.config.distributed.device)
        test_topk_accuracy = {}
        for k in self._topk:
            test_topk_correct[k] = np.sum(test_topk_correct[k])
            test_topk_correct[k] = distributed_reduce_sum(
                value=test_topk_correct[k], device=self._trainer.config.distributed.device
            )
            test_topk_total[k] = np.sum(test_topk_total[k])
            test_topk_total[k] = distributed_reduce_sum(
                value=test_topk_total[k], device=self._trainer.config.distributed.device
            )
            test_topk_accuracy[k] = test_topk_correct[k] / test_topk_total[k]
        topk_result = TopkResult(
            accuracy=test_topk_accuracy,
            correct=test_topk_correct,
            total=test_topk_total,
        )
        return IterationResult(loss=test_losses, topk_result=topk_result)

    def _training_cleanup(self):
        del self._linear_head
        del self._optimizer
        torch.cuda.empty_cache()

    def evaluate(self):
        execution_logger.info(f"Evaluating {self._name} metric")
        torch.cuda.empty_cache()
        train_data_loader, train_sampler = self._trainer.data.get_data_loader_and_sampler(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            seed=self._seed,
            train=True,
            evaluation=False,
            rank=self._trainer.config.distributed.global_rank,
            world_size=self._trainer.config.distributed.world_size,
            drop_last=True,
        )
        test_data_loader, test_sampler = self._trainer.data.get_data_loader_and_sampler(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            seed=0,
            train=False,
            evaluation=True,
            rank=self._trainer.config.distributed.global_rank,
            world_size=self._trainer.config.distributed.world_size,
            drop_last=False,
        )
        prior_data_loader = self._trainer.prior.get_data_loader(batch_size=train_sampler.per_device_batch_size)
        prior_data_iter = iter(prior_data_loader)

        self._initialize_model()
        self._initialize_optimizer()
        training_result = self._train(
            train_data_loader=train_data_loader,
            train_sampler=train_sampler,
            test_data_loader=test_data_loader,
            prior_data_iter=prior_data_iter,
        )

        metric_items = []
        metric_items.append(FloatListMetricItem(name=f"{self._name}_train_loss", value=training_result.train_loss))
        for k in self._topk:
            metric_items.append(
                FloatListMetricItem(
                    name=f"{self._name}_train_top{k}_accuracy", value=training_result.train_topk_accuracy[k]
                )
            )
        metric_items.append(FloatListMetricItem(name=f"{self._name}_test_loss", value=training_result.test_loss))
        for k in self._topk:
            metric_items.append(
                FloatListMetricItem(
                    name=f"{self._name}_test_top{k}_accuracy", value=training_result.test_topk_accuracy[k]
                )
            )
        self._training_cleanup()

        return metric_items
