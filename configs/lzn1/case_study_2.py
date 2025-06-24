# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ml_collections
import logging
import os


def get_config():
    config = ml_collections.ConfigDict()

    config.trainer = "lzn.trainer.ContrastiveTrainer"

    config.latent_dim = 256
    config.eval_batch_size = 8192
    config.result_folder = "./results/cast_study_2"

    config.distributed = ml_collections.ConfigDict()
    config.distributed.num_gpus_per_node = 8

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 8192
    config.training.data_num_workers = 16
    config.training.data_seed = 0
    config.training.num_encoder_iterations = 5000000
    config.training.ode_num_steps = 20
    config.training.assignment_loss_cut_off_step = 0.2
    config.training.loss_weight = ml_collections.ConfigDict()
    config.training.loss_weight.prob = 0.0
    config.training.loss_weight.cross_entropy = 1.0
    config.training.prior_data_scale = 0.45

    config.training.enable_mixed_precision_training = False

    config.training.gradient_clipping = 1

    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.folder = config.get_ref("result_folder") + os.sep + "ckpts"
    config.checkpoint.format = "{epoch:09d}-{iteration:09d}.pt"
    config.checkpoint.iteration_freq = 2000
    config.checkpoint.epoch_freq = -1
    config.checkpoint.load_checkpoint = "auto"
    config.checkpoint.path = ""

    config.encoder_ema = ml_collections.ConfigDict()
    config.encoder_ema.enabled = True
    config.encoder_ema.params = ml_collections.ConfigDict()
    config.encoder_ema.params.decay = 0.9999
    config.encoder_ema.params.use_num_updates = True

    config.encoder_latent_flow = ml_collections.ConfigDict()
    config.encoder_latent_flow.name = "lzn.latent_flow.euler_latent_flow.EulerLatentFlow"
    config.encoder_latent_flow.params = ml_collections.ConfigDict()

    config.logging = ml_collections.ConfigDict()
    config.logging.level = logging.DEBUG
    config.logging.log_file = config.get_ref("result_folder") + os.sep + "log.log"
    config.logging.datefmt = "%m/%d/%Y %H:%M:%S %p"
    config.logging.fmt = "%(asctime)s [%(name)s] [%(levelname)-5.5s]" "  %(message)s"

    config.data = ml_collections.ConfigDict()
    config.data.name = "lzn.data.imagenet_aug.ImageNetAug"
    config.data.params = ml_collections.ConfigDict()
    config.data.params.root = "/tmp/data/ImageNet"

    config.encoder = ml_collections.ConfigDict()
    config.encoder.name = "lzn.model.resnet.get_resnet"
    config.encoder.params = ml_collections.ConfigDict()
    config.encoder.params.head_out_dim = config.get_ref("latent_dim")
    config.encoder.params.head_num_layers = 2
    config.encoder.params.depth = 50
    config.encoder.params.width_multiplier = 1

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "torch.optim.Adam"
    config.optimizer.params = ml_collections.ConfigDict()
    config.optimizer.additional_params = ml_collections.ConfigDict()
    config.optimizer.additional_params.encoder_lr = 8e-4

    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "torch.optim.lr_scheduler.LinearLR"
    config.lr_scheduler.params = ml_collections.ConfigDict()
    config.lr_scheduler.params.start_factor = 1.0 / 5000
    config.lr_scheduler.params.end_factor = 1.0
    config.lr_scheduler.params.total_iters = 5000

    config.prior = ml_collections.ConfigDict()
    config.prior.name = "lzn.prior.gaussian.Gaussian"
    config.prior.params = ml_collections.ConfigDict()
    config.prior.params.latent_dim = config.get_ref("latent_dim")

    training_metrics_metric_config = ml_collections.ConfigDict()
    training_metrics_metric_config.name = "lzn.metric.training_metrics.TrainingMetrics"
    training_metrics_metric_config.params = ml_collections.ConfigDict()
    training_metrics_metric_config.iteration_freq = 1
    training_metrics_metric_config.epoch_freq = -1
    training_metrics_metric_config.apply_on_ema = False

    linear_probing_no_head_metric_config = ml_collections.ConfigDict()
    linear_probing_no_head_metric_config.name = "lzn.metric.linear_probing.LinearProbing"
    linear_probing_no_head_metric_config.params = ml_collections.ConfigDict()
    linear_probing_no_head_metric_config.params.mode = "no_head"
    linear_probing_no_head_metric_config.params.batch_size = config.get_ref("eval_batch_size")
    linear_probing_no_head_metric_config.params.lr = 3e-4
    linear_probing_no_head_metric_config.params.weight_decay = 0.0
    linear_probing_no_head_metric_config.params.num_epochs = 300
    linear_probing_no_head_metric_config.iteration_freq = 1000
    linear_probing_no_head_metric_config.epoch_freq = -1
    linear_probing_no_head_metric_config.apply_on_ema = True

    representation_metric_config = ml_collections.ConfigDict()
    representation_metric_config.name = "lzn.metric.representation.Representation"
    representation_metric_config.params = ml_collections.ConfigDict()
    representation_metric_config.params.mode = "no_head"
    representation_metric_config.params.batch_size = config.get_ref("eval_batch_size")
    representation_metric_config.iteration_freq = 1000000
    representation_metric_config.epoch_freq = -1
    representation_metric_config.apply_on_ema = True

    config.metrics = [
        training_metrics_metric_config,
        linear_probing_no_head_metric_config,
        representation_metric_config
    ]

    image_file_logger_config = ml_collections.ConfigDict()
    image_file_logger_config.name = "lzn.logger.image_file.ImageFile"
    image_file_logger_config.params = ml_collections.ConfigDict()
    image_file_logger_config.params.output_folder = config.get_ref("result_folder")
    image_file_logger_config.params.preview_size = 4096

    log_print_logger_config = ml_collections.ConfigDict()
    log_print_logger_config.name = "lzn.logger.log_print.LogPrint"
    log_print_logger_config.params = ml_collections.ConfigDict()

    csv_print_logger_config = ml_collections.ConfigDict()
    csv_print_logger_config.name = "lzn.logger.csv_print.CSVPrint"
    csv_print_logger_config.params = ml_collections.ConfigDict()
    csv_print_logger_config.params.output_folder = config.get_ref("result_folder")

    matplotlib_pdf_logger_config = ml_collections.ConfigDict()
    matplotlib_pdf_logger_config.name = "lzn.logger.matplotlib_pdf.MatplotlibPDF"
    matplotlib_pdf_logger_config.params = ml_collections.ConfigDict()
    matplotlib_pdf_logger_config.params.output_folder = config.get_ref("result_folder")

    npy_file_logger_config = ml_collections.ConfigDict()
    npy_file_logger_config.name = "lzn.logger.npy_file.NpyFile"
    npy_file_logger_config.params = ml_collections.ConfigDict()
    npy_file_logger_config.params.output_folder = config.get_ref(
        "result_folder"
    )

    config.loggers = [
        image_file_logger_config,
        log_print_logger_config,
        csv_print_logger_config,
        matplotlib_pdf_logger_config,
        npy_file_logger_config,
    ]

    return config
