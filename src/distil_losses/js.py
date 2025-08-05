"""
This implementation is based on Jasper(https://github.com/NLPJCL/RAG-Retrieval/blob/master/rag_retrieval/train/embedding/model_distill.py)
TODO：実装
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss

class JasperStella(DistilLoss):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        pass