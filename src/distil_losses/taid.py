import torch
import torch.nn.functional as F
from lightning import LightningModule
from .base import DistilLoss
from .mse import forward_mse, make_mse_features
from .kld import forward_kld, make_kld_features
# from .ckd import forward_ckd, make_ckd_features


class TAID(DistilLoss):
    def __init__(
        self,
        forward_fn:str,
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
        if forward_fn == "ckd":
            # self.forward_fn = forward_ckd
            # self.make_features_fn = make_ckd_features
            self.forward_fn = forward_mse
            self.make_features_fn = make_mse_features
        elif forward_fn == "kld":
            self.forward_fn = forward_kld
            self.make_features_fn = make_kld_features
        elif forward_fn == "mse":
            self.forward_fn = forward_mse
            self.make_features_fn = make_mse_features
        else:
            raise ValueError(
                f"Invalid forward_fn: {forward_fn}. Must be one of ['ckd', 'kld', 'mse']"
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
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ):
        student_features, teacher_features = self.make_features_fn(
            projected_features=projected_features,
            teacher_features=teacher_features,
            temp=self.t,
        )
        p_t = (1 - self.t) * student_features.detach() + self.t * teacher_features
        distil_loss = self.forward_fn(student_features,p_t)
        return distil_loss

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        # compute kd loss
        loss = self.compute_loss(
            projected_features=projected_features,
            teacher_features=teacher_features,
        )

        # update t
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )

        loss_dict = {
            "loss": loss,
            "tiki_t": self.t,
            "delta_t": delta_t,
        }
        return loss_dict
