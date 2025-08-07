"""
TAID (Temperature Adaptive Interpolated Distillation) for sentence embeddings.
"""

import torch
import torch.nn.functional as F
from einops import einsum
from lightning import LightningModule
from .base import DistilLoss
from typing import Optional, Dict

def forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_probs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KLダイバージェンス損失の計算
    teacher_probs: 教師の確率分布 (softmax済み)
    student_logprobs: 生徒の対数確率分布 (log_softmax済み)
    """
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    
    # inf値をマスク
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.mean(x)
    
    return distil_loss

class TAID(DistilLoss):
    """
    Temperature Adaptive Interpolated Distillation for sentence embeddings.
    
    適応的温度調整機能付きのKL蒸留。学習の進行に応じて教師と生徒の重みを動的に調整し、
    損失の変化率に基づいて温度パラメータを適応的に更新する。
    バッチ内の文埋め込み同士の類似度分布で蒸留を行う。
    """
    def __init__(
        self,
        temp: float = 0.05,
        t_start: float = 0.4,
        t_end: float = 1.0,
        alpha: float = 5e-4,
        beta: float = 0.99,
        disable_adaptive: bool = False,
    ):
        super().__init__()
        # validation
        assert 0.0 <= t_start < 1.0
        assert 0.0 < t_end <= 1.0
        assert 0.0 <= alpha <= 1.0

        self.temp = temp
        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        
        self.register_buffer("t", torch.tensor(t_start, dtype=torch.float32))
        self.register_buffer("prev_loss", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("momentum", torch.zeros([], dtype=torch.float32))

    def _compute_similarity_matrix(self, query_emb: torch.Tensor, key_emb: torch.Tensor) -> torch.Tensor:
        """共通の類似度行列計算"""
        return einsum(query_emb, key_emb, 'b d, k d -> b k') / self.temp

    def update_t(
        self, loss: torch.Tensor, global_step: int, num_train_steps: int
    ) -> torch.Tensor:
        """適応的温度パラメータの更新"""
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return torch.tensor(0.0, device=loss.device)
            
        # 相対変化率を計算
        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        # モメンタムを更新
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        # 適応的デルタを計算
        adaptive_delta = torch.sigmoid(self.momentum)
        # tを更新（単調増加を保証）
        progress = global_step / num_train_steps
        t_target = self.t_start + (self.t_end - self.t_start) * progress
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        
        t = (
            min(self.t_end, max(t_target, self.t + delta_t))
            if not self.disable_adaptive
            else t_target
        )
        
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        TAID損失の計算
        生徒と教師のlogitsを補間して新しい教師分布を作成し、KL蒸留を行う
        """
        # 教師と生徒のlogitsを補間 (TAIDの核心部分)
        p_t = (1 - self.t) * student_logits.detach() + self.t * teacher_logits
        p_t = F.softmax(p_t, dim=-1, dtype=torch.float32)
        
        # KL蒸留損失を計算
        distil_loss = forward_kl(
            logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_probs=p_t,
        )
        return distil_loss

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 文埋め込みを正規化
        student_emb = F.normalize(projected_features, dim=-1)
        teacher_emb = F.normalize(teacher_features, dim=-1)
        
        # 類似度行列を計算（これをlogitsとして使用）
        student_logits = self._compute_similarity_matrix(student_emb, student_emb)
        teacher_logits = self._compute_similarity_matrix(teacher_emb, teacher_emb)
        
        # TAID損失を計算
        loss = self.compute_loss(student_logits, teacher_logits)

        # 温度パラメータを更新
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )

        loss_dict = {
            "loss": loss,
            "taid_t": self.t,
            "delta_t": delta_t,
        }
        return loss_dict
