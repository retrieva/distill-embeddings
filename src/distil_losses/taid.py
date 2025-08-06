"""
TODO: まだ実装合わせてない

"""


import torch
import torch.nn.functional as F
from lightning import LightningModule
from .base import DistilLoss
from typing import Optional, Dict

def forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    teacher_probs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

class TAID(DistilLoss):
    def __init__(
        self,
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

        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        self.register_buffer(
            "t", torch.tensor(t_start, device="cuda", dtype=torch.float32)
        )
        self.register_buffer(
            "prev_loss", torch.tensor(float("inf"), device="cuda", dtype=torch.float32)
        )
        self.register_buffer(
            "momentum", torch.zeros([], device="cuda", dtype=torch.float32)
        )

    def update_t(
        self, loss: torch.Tensor, global_step: int, num_train_steps: int
    ) -> torch.Tensor:
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return
        # Calculate relative change rate
        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        # Update momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        # Calculate adaptive delta
        adaptive_delta = torch.sigmoid(self.momentum)
        # Update t (ensure monotonic increase)
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
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ):
        p_t = (1 - self.t) * logits.detach() + self.t * teacher_logits
        p_t = F.softmax(p_t, dim=-1, dtype=torch.float32)
        distil_loss = forward_kl(
            logits=logits,
            teacher_logits=teacher_logits,
            mask=mask,
            teacher_probs=p_t,
        )
        return distil_loss

    def forward(
        self,
        lightning_module: LightningModule,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # compute kd loss
        loss = self.compute_loss(logits, teacher_logits, mask)

        # update t
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )

        loss_dict = {
            "distil_loss": loss,
            "tiki_t": self.t,
            "delta_t": delta_t,
        }
        return loss_dict
