import torch
import torch.nn.functional as F
from lightning import LightningModule
from .base import DistilLoss
from typing import Dict, Optional


class TAID(DistilLoss):
    def __init__(
        self,
        forward_fn: DistilLoss,
        t_start: float = 0.4,
        t_end: float = 1.0,
        alpha: float = 5e-4,
        beta: float = 0.99,
        disable_adaptive: bool = False,
        args: Optional[Dict] = None,
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
        self.forward_fn = forward_fn
    def update_t(
        self, loss: torch.Tensor, global_step: int, num_train_steps: int
    ) -> torch.Tensor:
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return
        # Calculate relative change rate
        # 前回のlossとの差分
        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        # Update momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        # Calculate adaptive delta
        # momentumを使った適応的な変化量を計算
        adaptive_delta = torch.sigmoid(self.momentum)
        # Update t (ensure monotonic increase)
        # 今全体のどんぐらいか
        progress = global_step / num_train_steps
        # 線形な場合のt
        t_target = self.t_start + (self.t_end - self.t_start) * progress
        # alphaは学習率っぽい？ adaptive_deltaを更新量にして、実際の更新幅を決めてそう
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (
            # 更新幅を加えた時に、線形増加よりは多くなるようにする
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
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
        validation: bool = False,
    ):
        student_features, teacher_features = self.forward_fn.make_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
            pos_projected_features=pos_projected_features,
            pos_teacher_features=pos_teacher_features,
        )
        p_t = (1 - self.t) * student_features.detach() + self.t * teacher_features
        distil_loss, loss_dict = self.forward_fn.compute_loss(student_features,p_t,validation=validation)
        return distil_loss, loss_dict

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
        # compute kd loss
        loss, loss_dict = self.compute_loss(
            projected_features=projected_features,
            teacher_features=teacher_features,
            pos_projected_features=pos_projected_features,
            pos_teacher_features=pos_teacher_features,
            validation=validation,
        )

        # update t
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )
        loss_dict["tiki_t"] = self.t
        loss_dict["delta_t"] = delta_t

        return loss_dict
