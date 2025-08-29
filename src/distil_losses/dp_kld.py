"""
This implementation is based on (https://github.com/UKPLab/sentence-transformers/blob/240cf3053c5d2dc9a04108412e63b57e05f17ff6/sentence_transformers/losses/DistillKLDivLoss.py)
simとしてcosineではなく内積を使う
"""

import torch
import torch.nn.functional as F
from einops import einsum
from .base import DistilLoss
from lightning import LightningModule
from typing import Dict, Optional


class DP_KLD(DistilLoss):
    """
    Knowledge Distillation using KL Divergence Loss for sentence embeddings.
    
    バッチ内の文埋め込み同士の類似度分布を教師モデルから生徒モデルに蒸留する。
    教師と生徒それぞれのバッチ内類似度行列を計算し、その確率分布間のKLダイバージェンスを損失とする。
    これにより生徒モデルが教師モデルの埋め込み空間構造を学習する。
    """
    def __init__(self, args: Optional[Dict] = None, ):
        super().__init__()
        self.temp = args.kld_temp if hasattr(args, 'kld_temp') else 2.0
        self.use_pos = args.use_pos if hasattr(args, 'use_pos') else False

    def compute_loss(
        self,
        sim_s: torch.Tensor,
        sim_t: torch.Tensor,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        KL Divergence Lossを計算する関数
        """
        # KLダイバージェンス計算
        teacher_probs = F.softmax(sim_t, dim=1, dtype=torch.float32)
        student_log_probs = F.log_softmax(sim_s, dim=1, dtype=torch.float32)
        # KL(teacher || student) を計算
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (max(1.0, self.temp ** 2))  # 温度パラメータで調整

        return kl_loss, {"loss": kl_loss, "temp": self.temp}

    def make_features(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        student_features = projected_features
        teacher_features = teacher_features
        if pos_projected_features is not None and pos_teacher_features is not None and self.use_pos:
            pos_student_features = pos_projected_features
            pos_teacher_features = pos_teacher_features
            # 類似度行列を計算（バッチ内の各文同士の類似度）
            sim_s = einsum(student_features, pos_student_features, 'b d, k d -> b k') / self.temp
            sim_t = einsum(teacher_features, pos_teacher_features, 'b d, k d -> b k') / self.temp
        else: 
            # 類似度行列を計算（バッチ内の各文同士の類似度）
            sim_s = einsum(student_features, student_features, 'b d, k d -> b k') / self.temp
            sim_t = einsum(teacher_features, teacher_features, 'b d, k d -> b k') / self.temp

        return sim_s, sim_t

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
        # studentとteacherそれぞれで類似度行列を計算
        # Tempで割った後に、それぞれlog_softmaxもしくはsoftmaxを適用する
        # その後、KLダイバージェンス適用
        # 文埋め込みを正規化
        sim_s, sim_t = self.make_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
            pos_projected_features=pos_projected_features,
            pos_teacher_features=pos_teacher_features,
        )
        kl_loss,loss_dict = self.compute_loss(
            sim_s=sim_s,
            sim_t=sim_t,
            validation=validation,
        )

        return loss_dict