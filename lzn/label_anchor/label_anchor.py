import torch
from torch import nn


class LabelAnchor(nn.Module):
    def __init__(self, z_dim, num_classes):
        super().__init__()
        self.anchor = nn.Parameter(
            torch.randn(num_classes, z_dim), requires_grad=True
        )

    def get_label(self, latent):
        diff = latent.unsqueeze(1) - self(None)
        diff = diff.pow(2).sum(dim=2)
        return diff.argmin(dim=1)

    @property
    def num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        return num_params

    def forward(self, _):
        # Dummy input to cheat Pytorch
        return self.anchor
