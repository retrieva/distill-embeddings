"""
This implementation is based on [DistilCSE's](https://github.com/caskcsg/sentemb/blob/main/DistilCSE/ckd_contrastive.py)
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss
from lightning import LightningModule
from typing import Dict, Optional

class CKD(DistilLoss):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the CKD loss using contrastive knowledge distillation.
        生徒と教師の文埋め込みの類似度を測り、クロスエントロピー損失を計算する
        同じ文を埋め込んだ時の類似度が高くなるように学習する
        公式の実装だと、queueに過去のバッチについての教師モデルの埋め込みを保持して、in batch negを増やしている
        """
        labels = torch.arange(projected_features.size(0), device=projected_features.device)
        query = F.normalize(projected_features, dim=-1)
        key = torch.cat([projected_features, teacher_features], dim=0)
        scores = torch.einsum('ab,cb->ac', query, key) / self.temp
        loss = F.cross_entropy(scores, labels.to(scores.device))
        return loss