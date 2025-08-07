from typing import Dict, Optional
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from lightning import LightningModule
from src.distil_losses import *
from src.data import Batch


@dataclass
class LossOutput:
    loss: Tensor
    loss_dict: Dict[str, Tensor]


class KDLoss(nn.Module):
    def __init__(
        self,
        distil_loss_fn: Optional[DistilLoss] = None,
    ):
        super().__init__()
        self.distil_loss_fn = distil_loss_fn

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
    ) -> LossOutput:
        loss_dict = {}
        student_features = lightning_module.student_model(batch)['sentence_embedding']
        # Project student features to teacher's embedding space
        projected_features = lightning_module.linear(student_features)

        # TODO: 複数GPUの場合、この辺りでGatherの処理が必要かもしれない

        teacher_features = batch["teacher_features"]
        if isinstance(teacher_features, list):
            teacher_features = torch.stack(teacher_features, dim=0)
            
        loss = self.distil_loss_fn(
            lightning_module=lightning_module,
            projected_features=projected_features,
            teacher_features=teacher_features,
        )
        loss_dict = loss
        return LossOutput(
            loss=loss["loss"],
            loss_dict=loss_dict,
        )


def get_loss_fn(args):
    if args.loss_type == "taid":
        distil_loss_fn = TAID(
            forward_fn=args.taid_forward_fn,
            t_start=args.taid_t_start,
            t_end=args.taid_t_end,
            alpha=args.taid_alpha,
            beta=args.taid_beta,
            disable_adaptive=args.taid_disable_adaptive,
        )
    elif args.loss_type == "ckd":
        distil_loss_fn = CKD(args.temp)
    elif args.loss_type == "mse":
        distil_loss_fn = MSE()
    elif args.loss_type == "kld":
        distil_loss_fn = KLD(args.temp)
    else:
        raise NotImplementedError(args.loss_type)

    return KDLoss(distil_loss_fn=distil_loss_fn)
