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
        features = lightning_module.student_model(batch)['sentence_embedding']
        features = F.normalize(features, dim=-1)
        # 2. 全てのGPUから 'features' を収集して結合
        # all_gather はテンソルのリストを返します (例: [tensor_gpu0, tensor_gpu1, ...])
        gathered_features_list = lightning_module.all_gather(features)
        # リストを次元0で結合し、(world_size * batch_size_per_gpu, embedding_dim) のテンソルを作成
        gathered_features = torch.cat(gathered_features_list, dim=0)
        print(gathered_features.shape)

        if "pos" in batch.keys() and "pos_features" in batch.keys() and self.use_pos:
            pos_features = lightning_module.student_model(batch["pos"])['sentence_embedding']
        else:
            # unsupと同じように同じ文2回かける（dropoutでちょっと違う埋め込みになるはず）
            pos_features = lightning_module.student_model(batch)['sentence_embedding']
        pos_features = F.normalize(pos_features, dim=-1)
        # 3. 全てのGPUから 'pos_features' を収集して結合
        gathered_pos_features_list = lightning_module.all_gather(pos_features)
        gathered_pos_features = torch.cat(gathered_pos_features_list, dim=0)
        print(gathered_pos_features.shape)

        loss = self.loss_fn(
            features=gathered_features,
            pos_features=gathered_pos_features
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
        student_features = lightning_module.student_model(batch)['sentence_embedding']
        # Project student features to teacher's embedding space
        projected_features = lightning_module.linear(student_features)

        teacher_features = batch["teacher_features"]
        if isinstance(teacher_features, list):
            teacher_features = torch.stack(teacher_features, dim=0)
        # 3. 全GPUから学生・教師の特徴量を集約
        gathered_projected_features = torch.cat(lightning_module.all_gather(projected_features), dim=0)
        gathered_teacher_features = torch.cat(lightning_module.all_gather(teacher_features), dim=0)
        if self.use_pos:
            pos_student_features = lightning_module.student_model(batch["pos"])['sentence_embedding']
            pos_projected_features = lightning_module.linear(pos_student_features)
            pos_teacher_features = batch["pos_features"]
            if isinstance(pos_teacher_features, list):
                pos_teacher_features = torch.stack(pos_teacher_features, dim=0)

            # Positiveペアの特徴量も全GPUから集約
            gathered_pos_projected_features = torch.cat(lightning_module.all_gather(pos_projected_features), dim=0)
            gathered_pos_teacher_features = torch.cat(lightning_module.all_gather(pos_teacher_features), dim=0)
        else:
            pos_projected_features=None
            pos_teacher_features=None
            
        loss = self.distil_loss_fn(
           lightning_module=lightning_module,
            projected_features=gathered_projected_features,
            teacher_features=gathered_teacher_features,
            pos_projected_features=gathered_pos_projected_features,
            pos_teacher_features=gathered_pos_teacher_features,
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
    elif args.loss_type == "js":
        distil_loss_fn = JasperStella(args)
    elif args.loss_type == "infocse":
        distil_loss_fn = InfoCSE(args)
    else:
        raise NotImplementedError(args.loss_type)
    return KDLoss(use_pos = args.use_pos, distil_loss_fn=distil_loss_fn)
