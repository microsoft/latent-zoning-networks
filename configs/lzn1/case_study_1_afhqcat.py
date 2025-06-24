# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ml_collections
import logging
import os


def get_config():
    config = ml_collections.ConfigDict()

    config.trainer = "lzn.trainer.GenerationTrainer"

    config.latent_dim = 200
    config.eval_batch_size = 256
    config.result_folder = "./results/case_study_1_afhqcat"
    config.conditional = False
    config.image_size = 256
    config.image_channels = 3

    config.distributed = ml_collections.ConfigDict()
    config.distributed.num_gpus_per_node = 8

    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 256
    config.training.data_num_workers = 2
    config.training.data_seed = 0
    config.training.num_decoder_iterations = 60000
    config.training.num_encoder_iterations = 60000
    config.training.ode_num_steps = 20

    config.training.enable_mixed_precision_training = False

    config.training.recon_loss_coe = 1.0
    config.training.gradient_clipping = 1

    config.training.conditional = config.get_ref("conditional")

    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.folder = (
        config.get_ref("result_folder") + os.sep + "ckpts"
    )
    config.checkpoint.format = "{epoch:09d}-{iteration:09d}.pt"
    config.checkpoint.iteration_freq = 2000
    config.checkpoint.epoch_freq = -1
    config.checkpoint.load_checkpoint = "auto"
    config.checkpoint.path = ""

    config.decoder_ema = ml_collections.ConfigDict()
    config.decoder_ema.enabled = True
    config.decoder_ema.params = ml_collections.ConfigDict()
    config.decoder_ema.params.decay = 0.999999
    config.decoder_ema.params.use_num_updates = True

    config.encoder_ema = ml_collections.ConfigDict()
    config.encoder_ema.enabled = True
    config.encoder_ema.params = ml_collections.ConfigDict()
    config.encoder_ema.params.decay = 0.999999
    config.encoder_ema.params.use_num_updates = True

    config.encoder_latent_flow = ml_collections.ConfigDict()
    config.encoder_latent_flow.name = "lzn.latent_flow.dpm_solver_latent_flow.DPMSolverLatentFlow"
    config.encoder_latent_flow.params = ml_collections.ConfigDict()

    config.logging = ml_collections.ConfigDict()
    config.logging.level = logging.DEBUG
    config.logging.log_file = (
        config.get_ref("result_folder") + os.sep + "log.log"
    )
    config.logging.datefmt = "%m/%d/%Y %H:%M:%S %p"
    config.logging.fmt = (
        "%(asctime)s [%(name)s] [%(levelname)-5.5s]"
        "  %(message)s"
    )

    config.data = ml_collections.ConfigDict()
    config.data.name = "lzn.data.afhq_cat.AFHQCat"
    config.data.params = ml_collections.ConfigDict()
    config.data.params.afhq_root = "/tmp/data/AFHQ"
    config.data.params.image_size = config.get_ref("image_size")

    config.t_scheduler = ml_collections.ConfigDict()
    config.t_scheduler.name = (
        "lzn.t_scheduler.uniform_t_scheduler.UniformTScheduler"
    )
    config.t_scheduler.params = ml_collections.ConfigDict()

    config.training_sampler = ml_collections.ConfigDict()
    config.training_sampler.name = "lzn.sampler.rk45_sampler.RK45Sampler"
    config.training_sampler.params = ml_collections.ConfigDict()
    config.training_sampler.params.image_size = config.get_ref("image_size")
    config.training_sampler.params.image_channels = config.get_ref(
        "image_channels"
    )
    config.training_sampler.params.clip_x1 = False

    evaluation_sampler1 = ml_collections.ConfigDict()
    evaluation_sampler1.name = "lzn.sampler.rk45_sampler.RK45Sampler"
    evaluation_sampler1.sampler_name = "rk45"
    evaluation_sampler1.params = ml_collections.ConfigDict()
    evaluation_sampler1.params.image_size = config.get_ref("image_size")
    evaluation_sampler1.params.image_channels = config.get_ref(
        "image_channels"
    )
    evaluation_sampler1.params.clip_x1 = False

    evaluation_sampler2 = ml_collections.ConfigDict()
    evaluation_sampler2.name = "lzn.sampler.euler_sampler.EulerSampler"
    evaluation_sampler2.sampler_name = "euler10"
    evaluation_sampler2.params = ml_collections.ConfigDict()
    evaluation_sampler2.params.image_size = config.get_ref("image_size")
    evaluation_sampler2.params.image_channels = config.get_ref(
        "image_channels"
    )
    evaluation_sampler2.params.num_steps = 10
    evaluation_sampler2.params.clip_x1 = False

    config.evaluation_samplers = [evaluation_sampler1, evaluation_sampler2]

    config.encoder = ml_collections.ConfigDict()
    config.encoder.name = "lzn.model.ncsnpp.NCSNPPEncoder"
    config.encoder.params = ml_collections.ConfigDict()
    config.encoder.params.latent_dim = config.get_ref("latent_dim")
    config.encoder.params.num_base_channels = 20
    config.encoder.params.num_base_units = 200
    config.encoder.params.num_latent_transformation_layers = 2
    config.encoder.params.latent_transformation_resolutions = [64, 32, 16, 8, 4]
    config.encoder.params.config = ml_collections.ConfigDict()
    config.encoder.params.config.model = ml_collections.ConfigDict()
    config.encoder.params.config.model.scale_by_sigma = True
    config.encoder.params.config.model.ema_rate = 0.9999
    config.encoder.params.config.model.normalization = "GroupNorm"
    config.encoder.params.config.model.nonlinearity = "swish"
    config.encoder.params.config.model.nf = 128
    config.encoder.params.config.model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    config.encoder.params.config.model.num_res_blocks = 2
    config.encoder.params.config.model.attn_resolutions = (16,)
    config.encoder.params.config.model.resamp_with_conv = True
    config.encoder.params.config.model.conditional = True
    config.encoder.params.config.model.fir = True
    config.encoder.params.config.model.fir_kernel = [1, 3, 3, 1]
    config.encoder.params.config.model.skip_rescale = True
    config.encoder.params.config.model.resblock_type = "biggan"
    config.encoder.params.config.model.progressive = "output_skip"
    config.encoder.params.config.model.progressive_input = "input_skip"
    config.encoder.params.config.model.progressive_combine = "sum"
    config.encoder.params.config.model.attention_type = "ddpm"
    config.encoder.params.config.model.init_scale = 0.0
    config.encoder.params.config.model.embedding_type = "fourier"
    config.encoder.params.config.model.fourier_scale = 16
    config.encoder.params.config.model.conv_size = 3
    config.encoder.params.config.model.sigma_min = 0.01
    config.encoder.params.config.model.sigma_max = 378
    config.encoder.params.config.model.num_scales = 2000
    config.encoder.params.config.model.beta_min = 0.1
    config.encoder.params.config.model.beta_max = 20.0
    config.encoder.params.config.model.dropout = 0.
    config.encoder.params.config.data = ml_collections.ConfigDict()
    config.encoder.params.config.data.image_size = config.get_ref("image_size")
    config.encoder.params.config.data.num_channels = config.get_ref(
        "image_channels"
    )
    config.encoder.params.config.training = ml_collections.ConfigDict()
    config.encoder.params.config.training.continuous = True

    config.decoder = ml_collections.ConfigDict()
    config.decoder.name = "lzn.model.ncsnpp.NCSNPPDecoder"
    config.decoder.params = ml_collections.ConfigDict()
    config.decoder.params.latent_dim = config.get_ref("latent_dim")
    config.decoder.params.num_classes = config.get_ref("conditional") * 10
    config.decoder.params.config = config.encoder.params.get_ref("config")

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "torch.optim.Adam"
    config.optimizer.params = ml_collections.ConfigDict()
    config.optimizer.additional_params = ml_collections.ConfigDict()
    config.optimizer.additional_params.encoder_lr = 2e-6
    config.optimizer.additional_params.decoder_lr = 2e-4

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

    l2loss = ml_collections.ConfigDict()
    l2loss.name = "lzn.loss.l2.L2"
    l2loss.params = ml_collections.ConfigDict()
    l2loss.weight = 1.0

    config.losses = [l2loss]

    random_samples_more_metric_config = ml_collections.ConfigDict()
    random_samples_more_metric_config.name = (
        "lzn.metric.random_samples_more.RandomSamplesMore"
    )
    random_samples_more_metric_config.params = ml_collections.ConfigDict()
    random_samples_more_metric_config.params.num_samples_per_class = 5120
    random_samples_more_metric_config.params.batch_size = config.get_ref(
        "eval_batch_size"
    )
    random_samples_more_metric_config.iteration_freq = -1
    random_samples_more_metric_config.epoch_freq = -1
    random_samples_more_metric_config.apply_on_ema = True
    random_samples_more_metric_config.apply_on_samplers = True

    random_samples_metric_config = ml_collections.ConfigDict()
    random_samples_metric_config.name = (
        "lzn.metric.random_samples.RandomSamples"
    )
    random_samples_metric_config.params = ml_collections.ConfigDict()
    random_samples_metric_config.params.num_samples_per_class = 256
    random_samples_metric_config.params.batch_size = config.get_ref(
        "eval_batch_size"
    )
    random_samples_metric_config.iteration_freq = 10000
    random_samples_metric_config.epoch_freq = -1
    random_samples_metric_config.apply_on_ema = True
    random_samples_metric_config.apply_on_samplers = True

    reconstructions_with_computed_ini_noise_metric_config = ml_collections.ConfigDict()
    reconstructions_with_computed_ini_noise_metric_config.name = (
        "lzn.metric.reconstructions_with_computed_ini_noise.Reconstructions"
    )
    reconstructions_with_computed_ini_noise_metric_config.params = ml_collections.ConfigDict()
    reconstructions_with_computed_ini_noise_metric_config.params.num_samples = 256
    reconstructions_with_computed_ini_noise_metric_config.params.batch_size = config.get_ref(
        "eval_batch_size"
    )
    reconstructions_with_computed_ini_noise_metric_config.params.num_recontruction_per_sample = 20
    reconstructions_with_computed_ini_noise_metric_config.iteration_freq = 10000
    reconstructions_with_computed_ini_noise_metric_config.epoch_freq = -1
    reconstructions_with_computed_ini_noise_metric_config.apply_on_ema = True
    reconstructions_with_computed_ini_noise_metric_config.apply_on_samplers = True

    training_metrics_metric_config = ml_collections.ConfigDict()
    training_metrics_metric_config.name = (
        "lzn.metric.training_metrics.TrainingMetrics"
    )
    training_metrics_metric_config.params = ml_collections.ConfigDict()
    training_metrics_metric_config.iteration_freq = 1
    training_metrics_metric_config.epoch_freq = -1
    training_metrics_metric_config.apply_on_ema = False
    training_metrics_metric_config.apply_on_samplers = False

    clean_fid_metric_config = ml_collections.ConfigDict()
    clean_fid_metric_config.name = "lzn.metric.clean_fid.CleanFID"
    clean_fid_metric_config.params = ml_collections.ConfigDict()
    clean_fid_metric_config.params.batch_size = config.get_ref(
        "eval_batch_size"
    )
    clean_fid_metric_config.params.num_workers = config.training.get_ref(
        "data_num_workers"
    )
    clean_fid_metric_config.params.num_real_samples = 1024
    clean_fid_metric_config.iteration_freq = 10000
    clean_fid_metric_config.epoch_freq = -1
    clean_fid_metric_config.apply_on_ema = True
    clean_fid_metric_config.apply_on_samplers = True

    rf_fid_metric_config = ml_collections.ConfigDict()
    rf_fid_metric_config.name = "lzn.metric.rf_fid.RFFID"
    rf_fid_metric_config.params = ml_collections.ConfigDict()
    rf_fid_metric_config.params.batch_size = config.get_ref(
        "eval_batch_size"
    )
    rf_fid_metric_config.params.num_workers = config.training.get_ref(
        "data_num_workers"
    )
    rf_fid_metric_config.params.num_real_samples = 1024
    rf_fid_metric_config.iteration_freq = 10000
    rf_fid_metric_config.epoch_freq = -1
    rf_fid_metric_config.apply_on_ema = True
    rf_fid_metric_config.apply_on_samplers = True

    config.metrics = [
        random_samples_more_metric_config,
        random_samples_metric_config,
        reconstructions_with_computed_ini_noise_metric_config,
        training_metrics_metric_config,
        clean_fid_metric_config,
        rf_fid_metric_config,
    ]

    image_file_logger_config = ml_collections.ConfigDict()
    image_file_logger_config.name = "lzn.logger.image_file.ImageFile"
    image_file_logger_config.params = ml_collections.ConfigDict()
    image_file_logger_config.params.output_folder = config.get_ref(
        "result_folder"
    )
    image_file_logger_config.params.preview_size = 4096

    npy_file_logger_config = ml_collections.ConfigDict()
    npy_file_logger_config.name = "lzn.logger.npy_file.NpyFile"
    npy_file_logger_config.params = ml_collections.ConfigDict()
    npy_file_logger_config.params.output_folder = config.get_ref(
        "result_folder"
    )

    log_print_logger_config = ml_collections.ConfigDict()
    log_print_logger_config.name = "lzn.logger.log_print.LogPrint"
    log_print_logger_config.params = ml_collections.ConfigDict()

    csv_print_logger_config = ml_collections.ConfigDict()
    csv_print_logger_config.name = "lzn.logger.csv_print.CSVPrint"
    csv_print_logger_config.params = ml_collections.ConfigDict()
    csv_print_logger_config.params.output_folder = config.get_ref(
        "result_folder"
    )

    matplotlib_pdf_logger_config = ml_collections.ConfigDict()
    matplotlib_pdf_logger_config.name = (
        "lzn.logger.matplotlib_pdf.MatplotlibPDF"
    )
    matplotlib_pdf_logger_config.params = ml_collections.ConfigDict()
    matplotlib_pdf_logger_config.params.output_folder = config.get_ref(
        "result_folder"
    )

    config.loggers = [
        image_file_logger_config,
        npy_file_logger_config,
        log_print_logger_config,
        csv_print_logger_config,
        matplotlib_pdf_logger_config,
    ]

    return config
