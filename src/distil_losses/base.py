from typing import Union, Dict
import torch
from torch import nn
from lightning.pytorch import LightningModule


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        validation: bool = False,
        **kwargs,
    ) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError
