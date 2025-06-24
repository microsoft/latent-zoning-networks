# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Modified from
# https://github.com/Separius/SimCLRv2-Pytorch/blob/main/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Encoder


class GroupNormRelu(nn.Sequential):
    def __init__(self, num_channels, num_groups=32, relu=True):
        super().__init__(nn.GroupNorm(num_groups, num_channels), nn.ReLU() if relu else nn.Identity())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias
    )


class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(
            conv(in_channels, 2 * out_channels, stride=stride), GroupNormRelu(2 * out_channels)
        )
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(
            conv(out_channels, mid_dim, kernel_size=1),
            GroupNormRelu(mid_dim),
            conv(mid_dim, 2 * out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                # kernel_size = 2 => padding = 1
                nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                conv(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = GroupNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [conv(in_channels, out_channels, kernel_size=1), GroupNormRelu(out_channels)]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(GroupNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(GroupNormRelu(out_channels * 4, relu=False))
        nn.init.constant_(ops[-1][0].weight, 0)
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList([BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(GroupNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(GroupNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(GroupNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


class ResNet(Encoder):
    def __init__(
        self, layers, width_multiplier, sk_ratio, head_out_dim, head_num_layers, num_gradient_checkpoint_segments=5
    ):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.head_out_dim = head_out_dim
        self.net = nn.Sequential(*ops)
        # self.fc = nn.Linear(channels_in, 1000)
        self.head = ContrastiveHead(channels_in=channels_in, out_dim=head_out_dim, num_layers=head_num_layers)
        self._num_gradient_checkpoint_segments = num_gradient_checkpoint_segments

    def forward(self, x, apply_head=True):
        x = (x + 1.0) / 2.0
        if self._num_gradient_checkpoint_segments > 0:
            x = torch.utils.checkpoint.checkpoint_sequential(
                self.net, self._num_gradient_checkpoint_segments, x, use_reentrant=False
            )
        else:
            x = self.net(x)
        h = x.mean(dim=[2, 3])
        if apply_head:
            h = self.head(h)
        return h


class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, num_groups=32, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.GroupNorm(num_groups, dim)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


def get_resnet(depth=50, width_multiplier=1, sk_ratio=0, **kwargs):  # sk_ratio=0.0625 is recommended
    layers = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}[depth]
    resnet = ResNet(layers, width_multiplier, sk_ratio, **kwargs)
    return resnet


def name_to_params(checkpoint):
    sk_ratio = 0.0625 if "_sk1" in checkpoint else 0
    if "r50_" in checkpoint:
        depth = 50
    elif "r101_" in checkpoint:
        depth = 101
    elif "r152_" in checkpoint:
        depth = 152
    else:
        raise NotImplementedError

    if "_1x_" in checkpoint:
        width = 1
    elif "_2x_" in checkpoint:
        width = 2
    elif "_3x_" in checkpoint:
        width = 3
    else:
        raise NotImplementedError

    return depth, width, sk_ratio
