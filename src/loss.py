from typing import Dict, Optional
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from lightning import LightningModule
from src.distil_losses import *
from src.data import Batch
from einops import einsum

taid_forward_fn_map = {
    "ckd": CKD,
    "kld": KLD,
    "mse": MSE,
    "dp_kld": DP_KLD,
    "js":JasperStella
}

@dataclass
class LossOutput:
    loss: Tensor
    loss_dict: Dict[str, Tensor]

class InfoCSE(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.use_pos = args.use_pos if hasattr(args, 'use_pos') else False
        self.temp = args.cse_temp if hasattr(args, 'cse_temp') else 0.05

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
        validation: bool = False,
        **kwargs,
    ) -> LossOutput:
        local_features = lightning_module.student_model(batch)['sentence_embedding']
        local_features = F.normalize(local_features, dim=-1)
        # (gpu_num, bs, dim)
        gathered_features = lightning_module.all_gather(local_features,sync_grads=True)
        # (gpu_num x bs, dim)
        global_features = gathered_features.view(-1, gathered_features.shape[-1])
        if "pos" in batch.keys() and "pos_features" in batch.keys() and self.use_pos:
            local_pos_features = lightning_module.student_model(batch["pos"])['sentence_embedding']
        else:
            # unsupと同じように同じ文2回かける（dropoutでちょっと違う埋め込みになるはず）
            local_pos_features = lightning_module.student_model(batch)['sentence_embedding']
        local_pos_features = F.normalize(local_pos_features, dim=-1)
        # 3. 全てのGPUから 'pos_features' を収集して結合
        gathered_pos_features_list = lightning_module.all_gather(local_pos_features,sync_grads=True)
        global_pos_features = gathered_pos_features_list.view(-1, gathered_pos_features_list.shape[-1])

        loss = self.loss_fn(
            features=global_features,
            pos_features=global_pos_features
        )
        return loss
    
    def loss_fn(
        self,
        features: torch.Tensor,
        pos_features: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        labels = torch.arange(features.size(0), device=features.device)
        # クエリとキー間の類似度スコアを計算 ab,cb->ac
        scores = einsum(features, pos_features, 'b d, k d -> b k') / self.temp

        # 対照学習損失：生徒埋め込みが対応する教師埋め込みに最も類似するように学習
        loss = F.cross_entropy(scores, labels)

        return LossOutput(
            loss=loss,
            loss_dict={"loss": loss},
        )

class KDLoss(nn.Module):
    def __init__(
        self,
        use_pos,
        distil_loss_fn: Optional[DistilLoss] = None,
    ):
        super().__init__()
        self.use_pos = use_pos
        self.distil_loss_fn = distil_loss_fn

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
        validation: bool = False,
        **kwargs,
    ) -> LossOutput:
        loss_dict = {}
        local_student_features = lightning_module.student_model(batch)['sentence_embedding']

        local_teacher_features = batch["teacher_features"]
        if isinstance(local_teacher_features, list):
            local_teacher_features = torch.stack(local_teacher_features, dim=0)
        local_teacher_features = local_teacher_features[:,:local_student_features.shape[1]]
        # 3. 全GPUから学生・教師の特徴量を集約
        # (gpu_num, bs, dim)
        gathered_student_features = lightning_module.all_gather(local_student_features,sync_grads=True)
        # (gpu_num x bs, dim)
        global_student_features = gathered_student_features.view(-1, gathered_student_features.shape[-1])

        gathered_teacher_features = lightning_module.all_gather(local_teacher_features,sync_grads=True)
        global_teacher_features = gathered_teacher_features.view(-1, gathered_teacher_features.shape[-1])
        if self.use_pos:
            local_pos_student_features = lightning_module.student_model(batch["pos"])['sentence_embedding']
            local_pos_teacher_features = batch["pos_features"]
            if isinstance(local_pos_teacher_features, list):
                local_pos_teacher_features = torch.stack(local_pos_teacher_features, dim=0)
            local_pos_teacher_features = local_pos_teacher_features[:,:local_pos_student_features.shape[1]]

            # Positiveペアの特徴量も全GPUから集約
            gathered_pos_student_features = lightning_module.all_gather(local_pos_student_features,sync_grads=True)
            global_pos_student_features = gathered_pos_student_features.view(-1, gathered_pos_student_features.shape[-1])
            gathered_pos_teacher_features = lightning_module.all_gather(local_pos_teacher_features,sync_grads=True)
            global_pos_teacher_features = gathered_pos_teacher_features.view(-1, gathered_pos_teacher_features.shape[-1])
        else:
            global_pos_student_features=None
            global_pos_teacher_features=None
            
        loss = self.distil_loss_fn(
           lightning_module=lightning_module,
            projected_features=global_student_features,
            teacher_features=global_teacher_features,
            pos_projected_features=global_pos_student_features,
            pos_teacher_features=global_pos_teacher_features,
            validation=validation,
            **kwargs,
        )
        loss_dict = loss
        return LossOutput(
            loss=loss["loss"],
            loss_dict=loss_dict,
        )


def get_loss_fn(args):
    if args.loss_type == "infocse":
        return InfoCSE(args)
    if "taid-" in args.loss_type:
        taid_forward_fn = args.loss_type.split("-")[1]
        distil_loss_fn = TAID(
            forward_fn=taid_forward_fn_map[taid_forward_fn](args),
            t_start=args.taid_t_start,
            t_end=args.taid_t_end,
            alpha=args.taid_alpha,
            beta=args.taid_beta,
            disable_adaptive=args.taid_disable_adaptive,
        )
    elif args.loss_type == "ckd":
        distil_loss_fn = CKD(args)
    elif args.loss_type == "mse":
        distil_loss_fn = MSE(args)
    elif args.loss_type == "kld":
        distil_loss_fn = KLD(args)
    elif args.loss_type == "dp_kld":
        distil_loss_fn = DP_KLD(args)
    elif args.loss_type == "js":
        distil_loss_fn = JasperStella(args)
    elif args.loss_type == "distill":
        distil_loss_fn = DistillLoss(args)
    elif args.loss_type == "infocse":
        distil_loss_fn = InfoCSE(args)
    else:
        raise NotImplementedError(args.loss_type)
    return KDLoss(use_pos = args.use_pos, distil_loss_fn=distil_loss_fn)
