"""
This implementation is based on (https://github.com/UKPLab/sentence-transformers/blob/240cf3053c5d2dc9a04108412e63b57e05f17ff6/sentence_transformers/losses/DistillKLDivLoss.py)
TODO：実装
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss

class KLD(DistilLoss):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        pass