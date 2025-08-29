"""
This implementation is based on (https://github.com/UKPLab/sentence-transformers/blob/240cf3053c5d2dc9a04108412e63b57e05f17ff6/sentence_transformers/losses/DistillKLDivLoss.py)
"""

import torch
import torch.nn.functional as F
from .kld import KLD


class RKL(KLD):
    """
    Knowledge Distillation using Reverse KL Divergence Loss for sentence embeddings.

    バッチ内の文埋め込み同士の類似度分布を教師モデルから生徒モデルに蒸留する。
    先生の分布を生徒の分布に近づける、"逆"KL
    """

    def compute_loss(
        self,
        sim_s: torch.Tensor,
        sim_t: torch.Tensor,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Reverce KL Divergence Lossを計算する関数
        """
        # KLダイバージェンス計算
        student_probs = F.softmax(sim_s, dim=1, dtype=torch.float32)
        teacher_log_probs = F.log_softmax(sim_t, dim=1, dtype=torch.float32)

        # KL(student || teacher) を計算
        kl_loss = F.kl_div(teacher_log_probs, student_probs, reduction="batchmean")
        kl_loss = kl_loss * (self.temp**2)  # 温度パラメータで調整

        return kl_loss, {"loss": kl_loss, "temp": self.temp}
