import torch
from lightning.pytorch import LightningModule
from torch import nn


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        hyp_projected_features: torch.Tensor = None,
        hyp_teacher_features: torch.Tensor = None,
        candidates_per_anchor: int = 1,
        validation: bool = False,
        **kwargs,
    ) -> dict | torch.Tensor:
        raise NotImplementedError
