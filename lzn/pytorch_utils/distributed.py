# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import pdb
import sys
import functools
import torch
from datetime import timedelta
import torch.distributed as dist
from ml_collections import config_dict
from lzn.logging import execution_logger


def setup_distributed(config):
    os.environ["MASTER_ADDR"] = config.distributed.master_addr
    os.environ["MASTER_PORT"] = config.distributed.master_port
    dist.init_process_group(
        "nccl",
        rank=config.distributed.global_rank,
        world_size=config.distributed.world_size,
        timeout=timedelta(seconds=3*60*60),
    )

    execution_logger.info(f"Running on rank {config.distributed.global_rank}.")
    assert dist.is_initialized()
    assert dist.get_rank() == config.distributed.global_rank
    torch.cuda.set_device(config.distributed.local_rank)


def get_distributed_configs(config):
    with config.unlocked():
        if not hasattr(config, "distributed"):
            config.distributed = config_dict.ConfigDict()

        if "MASTER_ADDR" in os.environ:
            config.distributed.master_addr = os.environ["MASTER_ADDR"]
        elif not hasattr(config.distributed, "master_addr"):
            config.distributed.master_addr = "localhost"

        if "MASTER_PORT" in os.environ:
            config.distributed.master_port = os.environ["MASTER_PORT"]
        elif not hasattr(config.distributed, "master_port"):
            config.distributed.master_port = "29500"

        # In Amulet, WORLD_SIZE means the number of nodes
        if "WORLD_SIZE" in os.environ:
            config.distributed.num_nodes = int(os.environ["WORLD_SIZE"])
        elif not hasattr(config.distributed, "num_nodes"):
            config.distributed.num_nodes = 1

        if "NODE_RANK" in os.environ:
            config.distributed.node_rank = int(os.environ["NODE_RANK"])
        elif not hasattr(config.distributed, "node_rank"):
            config.distributed.node_rank = 0

        if not hasattr(config.distributed, "num_gpus_per_node"):
            config.distributed.num_gpus_per_node = 1

        config.distributed.world_size = config.distributed.num_nodes * config.distributed.num_gpus_per_node

    all_configs = []
    for i in range(config.distributed.num_gpus_per_node):
        node_config = copy.deepcopy(config)
        with node_config.unlocked():
            node_config.distributed.local_rank = i
            node_config.distributed.global_rank = (
                config.distributed.node_rank * config.distributed.num_gpus_per_node + i
            )
            node_config.distributed.device = f"cuda:{i}"
        all_configs.append(node_config)

    execution_logger.info("Distributed configurations.")
    for sub_config_i, sub_config in enumerate(all_configs):
        execution_logger.info(f"Config {sub_config_i}:")
        execution_logger.info(f"  - master_addr: {sub_config.distributed.master_addr}")
        execution_logger.info(f"  - master_port: {sub_config.distributed.master_port}")
        execution_logger.info(f"  - num_nodes: {sub_config.distributed.num_nodes}")
        execution_logger.info(f"  - node_rank: {sub_config.distributed.node_rank}")
        execution_logger.info(f"  - world_size: {sub_config.distributed.world_size}")
        execution_logger.info(f"  - local_rank: {sub_config.distributed.local_rank}")
        execution_logger.info(f"  - global_rank: {sub_config.distributed.global_rank}")
    return all_configs


class _DistributedPdb(pdb.Pdb):
    """
    From:
    https://github.com/pytorch/pytorch/blob/e235db98c92638c7544a5768df012a626196391e/torch/distributed/__init__.py#L60
    Supports using PDB from inside a multiprocessing child process.

    Usage:
    _DistributedPdb().set_trace()
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def breakpoint(rank=0):
    """
    From:
    https://github.com/pytorch/pytorch/blob/e235db98c92638c7544a5768df012a626196391e/torch/distributed/__init__.py#L76
    Set a breakpoint, but only on a single rank.  All other ranks will wait for
    you to be done with the breakpoint before continuing.

    Args:
        rank (int): Which rank to break on.  Default: ``0``
    """
    if get_rank() == rank:
        pdb = _DistributedPdb()
        pdb.message(
            "\n!!! ATTENTION !!!\n\n" "Type 'up' to get to the frame that called " f"dist.breakpoint(rank={rank})\n"
        )
        pdb.set_trace()
    torch.distributed.barrier()


def cleanup_distributed():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def distributed_reduce_max(value, device):
    torch_value = torch.tensor(value, device=device)
    torch.distributed.all_reduce(torch_value, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.barrier()
    value = torch_value.item()
    return value


def distributed_reduce_sum(value, device):
    torch_value = torch.tensor(value, device=device)
    torch.distributed.all_reduce(torch_value)
    torch.distributed.barrier()
    value = torch_value.item()
    return value


def distributed_reduce_mean(value, device):
    torch_value = torch.tensor(value, device=device)
    torch.distributed.all_reduce(torch_value, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.barrier()
    value = torch_value.item()
    return value


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
        dist.barrier()

    return wrapper


def all_gather_nd(tensor):
    """
    From:
    https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in
    the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[: size[0]])
    return all_tensors
