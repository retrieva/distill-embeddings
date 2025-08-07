"""
This implementation is based on (https://github.com/UKPLab/sentence-transformers/blob/240cf3053c5d2dc9a04108412e63b57e05f17ff6/sentence_transformers/losses/DistillKLDivLoss.py)
"""

import torch
import torch.nn.functional as F
from einops import einsum
from .base import DistilLoss
from lightning import LightningModule

def forward_kld(
    sim_s: torch.Tensor,
    sim_t: torch.Tensor,
    temp: float = 2.0,
) -> torch.Tensor:
    """
    KL Divergence Lossを計算する関数
    """
    # KLダイバージェンス計算
    teacher_probs = F.softmax(sim_t, dim=1, dtype=torch.float32)
    student_log_probs = F.log_softmax(sim_s, dim=1, dtype=torch.float32)

    # KL(teacher || student) を計算
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    kl_loss = kl_loss * (temp ** 2)  # 温度パラメータで調整

    return kl_loss

def make_kld_features(
    projected_features: torch.Tensor,
    teacher_features: torch.Tensor,
    temp: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_features = F.normalize(projected_features, dim=-1)
    teacher_features = F.normalize(teacher_features, dim=-1)

    # 類似度行列を計算（バッチ内の各文同士の類似度）
    sim_s = einsum(student_features, student_features, 'b d, k d -> b k') / temp
    sim_t = einsum(teacher_features, teacher_features, 'b d, k d -> b k') / temp

    return sim_s, sim_t


class KLD(DistilLoss):
    """
    Knowledge Distillation using KL Divergence Loss for sentence embeddings.
    
    バッチ内の文埋め込み同士の類似度分布を教師モデルから生徒モデルに蒸留する。
    教師と生徒それぞれのバッチ内類似度行列を計算し、その確率分布間のKLダイバージェンスを損失とする。
    これにより生徒モデルが教師モデルの埋め込み空間構造を学習する。
    """
    def __init__(self, temp: float = 2.0):
        super().__init__()
        self.temp = temp

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        # studentとteacherそれぞれで類似度行列を計算
        # Tempで割った後に、それぞれlog_softmaxもしくはsoftmaxを適用する
        # その後、KLダイバージェンス適用
        # 文埋め込みを正規化
        sim_s, sim_t = make_kld_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
            temp=self.temp,
        )
        kl_loss = forward_kld(
            sim_s=sim_s,
            sim_t=sim_t,
            temp=self.temp,
        )

        return {"loss": kl_loss}