"""
This implementation is based on (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss)
distilの実装のもう一個の方もこれかも？
TODO：実装
"""

import torch
from torch.nn import functional as F
from .base import DistilLoss
from lightning import LightningModule

def forward_mse(
    projected_features: torch.Tensor,
    teacher_features: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    MSE損失を計算する関数
    """
    return F.mse_loss(projected_features, teacher_features)

def make_mse_features(
    projected_features: torch.Tensor,
    teacher_features: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return projected_features, teacher_features

class MSE(DistilLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        
        projected_features, teacher_features = make_mse_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
        )
        # MSE損失を計算
        loss = forward_mse(projected_features, teacher_features)
        return {"loss": loss}