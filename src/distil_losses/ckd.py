"""
This implementation is based on [DistilCSE's](https://github.com/caskcsg/sentemb/blob/main/DistilCSE/ckd_contrastive.py)
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss
from lightning import LightningModule
from typing import Dict, Optional
from einops import einsum

class CKD(DistilLoss):
    """
    Contrastive Knowledge Distillation for sentence embeddings.
    
    対照学習ベースの知識蒸留。生徒モデルの埋め込みが、同じサンプルの教師埋め込みに最も類似するように学習する。
    バッチ内で生徒埋め込み（クエリ）と、生徒・教師埋め込み（キー）の類似度を計算し、
    正解ラベル（自分自身のインデックス）でクロスエントロピー損失を適用する。
    """
    def __init__(self, temp: float = 0.02):
        super().__init__()
        self.temp = temp

    def _compute_similarity_matrix(self, query_emb: torch.Tensor, key_emb: torch.Tensor) -> torch.Tensor:
        """共通の類似度行列計算"""
        return einsum(query_emb, key_emb, 'b d, k d -> b k') / self.temp

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:  
        # 各サンプルのインデックスをラベルとする（対角要素が正解）
        labels = torch.arange(projected_features.size(0), device=projected_features.device)
        
        # 生徒埋め込みをクエリとして使用
        student_features = F.normalize(projected_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        
        # 生徒と教師の埋め込みを結合してキーとする
        key = torch.cat([student_features, teacher_features], dim=0)
        
        # クエリとキー間の類似度スコアを計算
        scores = self._compute_similarity_matrix(student_features, key)

        # 対照学習損失：生徒埋め込みが対応する教師埋め込みに最も類似するように学習
        loss = F.cross_entropy(scores, labels)
        return {"loss": loss}