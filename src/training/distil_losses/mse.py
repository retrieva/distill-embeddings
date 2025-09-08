"""
This implementation is based on (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss)
distilの実装のもう一個の方もこれかも？
TODO：実装
"""

import torch
from lightning import LightningModule
from torch.nn import functional as F

from .base import DistilLoss


class MSE(DistilLoss):
    def __init__(self, args=None):
        super().__init__()

    def make_features(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return projected_features, teacher_features

    def compute_loss(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        candidates_per_anchor: int = 1,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        MSE損失を計算する関数
        """
        loss = F.mse_loss(projected_features, teacher_features)
        return loss, {"loss": loss}

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        candidates_per_anchor: int = 1,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        projected_features, teacher_features = self.make_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
        )
        # MSE損失を計算
        loss, loss_dict = self.compute_loss(projected_features, teacher_features, validation=validation)
        return loss_dict
