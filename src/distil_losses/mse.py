"""
This implementation is based on (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss)
distilの実装のもう一個の方もこれかも？
TODO：実装
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss

class MSE(DistilLoss):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        pass