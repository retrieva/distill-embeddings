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
    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        self.temp = args.ckd_temp if hasattr(args, 'ckd_temp') else 0.02
        self.max_queue_len = args.ckd_max_queue_len if hasattr(args, 'ckd_max_queue_len') else 65536
        self.teacher_queue = torch.tensor([])  # 教師埋め込みのキュー

    def make_features(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生徒と教師の埋め込みを正規化し、類似度行列を計算する関数
        """
        # 類似度よりは埋め込みの重みつきわの方が良さそうだから、一旦こうしてみる
        # ここでやっとかないとTAIDで混ぜる時にNormの違いが影響しちゃいそう
        student_features = F.normalize(projected_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        return student_features, teacher_features

    def compute_loss(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        temp: float = 0.02,
        validation: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        対照学習ベースの知識蒸留損失を計算する関数
        生徒埋め込みが教師埋め込みに最も類似するように学習する。
        """
        # 各サンプルのインデックスをラベルとする（対角要素が正解）
        labels = torch.arange(projected_features.size(0), device=projected_features.device)
        # 生徒と教師の埋め込みを正規化
        # ２回やっちゃっても結果は一緒のはず
        student_features = F.normalize(projected_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        key = torch.cat([teacher_features, self.teacher_queue.to(student_features.device)], dim=0)

        # クエリとキー間の類似度スコアを計算 ab,cb->ac
        scores = einsum(student_features, key, 'b d, k d -> b k') / temp

        # 対照学習損失：生徒埋め込みが対応する教師埋め込みに最も類似するように学習
        loss = F.cross_entropy(scores, labels)
        if not validation:
            self.teacher_queue = key[:key.shape[0] - max(key.shape[0] - self.max_queue_len, 0)]
            self.teacher_queue = self.teacher_queue.detach().cpu()  # 勾配を伝播しないようにする
        return loss, {"loss": loss, "teacher_queue_length": self.teacher_queue.shape[0]}
    
    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        
        projected_features, teacher_features = self.make_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
        )
        loss, loss_dict = self.compute_loss(
            projected_features,
            teacher_features,
            temp=self.temp,
            validation=validation,
        )
        return loss_dict