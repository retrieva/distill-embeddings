"""
This implementation is based on [DistillCSE's](https://github.com/Jiahao004/DistillCSE/blob/main/distillcse/models_distill_calibrate.py)
"""

import torch
import torch.nn.functional as F
from .base import DistilLoss
from lightning import LightningModule
from typing import Dict, Optional
from einops import einsum


class DistillLoss(DistilLoss):
    """ """

    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        self.temp_s = 0.02
        self.temp_t = 0.01
        self.temp_cse = 0.05
        self.use_pos = args.use_pos if hasattr(args, "use_pos") else False

    def make_features(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_projected_features is not None and pos_teacher_features is not None and self.use_pos:
            # 対照学習のために、正例の埋め込みも必要な場合
            return projected_features, pos_teacher_features
        else:
            return projected_features, teacher_features

    def compute_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        temp: float = 0.05,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """ """
        loss_dict = {}
        distill_loss = F.cross_entropy(
            student_features / self.temp_s, F.softmax(teacher_features / self.temp_t, dim=-1)
        )
        # 各サンプルのインデックスをラベルとする（対角要素が正解）
        labels = torch.arange(student_features.size(0), device=student_features.device)
        # 生徒と教師の埋め込みを正規化
        # ２回やっちゃっても結果は一緒のはず
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        # クエリとキー間の類似度スコアを計算 ab,cb->ac
        scores = einsum(student_features, teacher_features, "b d, k d -> b k") / temp

        # 対照学習損失：生徒埋め込みが対応する教師埋め込みに最も類似するように学習
        cse_loss = F.cross_entropy(scores, labels)
        loss_dict["distill_loss"] = distill_loss
        loss_dict["cse_loss"] = cse_loss
        loss_dict["loss"] = cse_loss + distill_loss
        return loss_dict["loss"], loss_dict

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        projected_features, teacher_features = self.make_features(
            projected_features=projected_features,
            pos_projected_features=pos_projected_features,
            teacher_features=teacher_features,
            pos_teacher_features=pos_teacher_features,
        )

        loss, loss_dict = self.compute_loss(
            projected_features,
            teacher_features,
            temp=self.temp_cse,
            validation=validation,
        )
        return loss_dict
