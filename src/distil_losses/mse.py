"""
This implementation is based on (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss)
distilの実装のもう一個の方もこれかも？
TODO：実装
"""

import torch
from torch.nn import functional as F
from .base import DistilLoss
from lightning import LightningModule

class MSE(DistilLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(projected_features, teacher_features)
        return {"loss": loss}