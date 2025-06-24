# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.multiprocessing as mp

from ml_collections.config_flags import config_flags
from absl import app
from absl import flags
import lzn.trainer
import lzn.logging
import lzn.pytorch_utils.distributed


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)


def run(config):
    lzn.logging.setup_from_config(
        config.logging, name=f"rank_{config.distributed.global_rank}"
    )

    lzn.pytorch_utils.distributed.setup_distributed(config)

    trainer = lzn.trainer.from_config(config=config)
    trainer.train()
    lzn.pytorch_utils.distributed.cleanup_distributed()


def main(_):
    lzn.logging.setup_from_config(FLAGS.config.logging, name="starter")
    configs = lzn.pytorch_utils.distributed.get_distributed_configs(
        FLAGS.config
    )

    processes = []
    mp.set_start_method("spawn")
    for config in configs:
        p = mp.Process(target=run, args=(config,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    app.run(main)
