from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import einsum
from lightning import LightningModule
from torch import Tensor, nn

from src.training.data import Batch
from src.training.distil_losses import CKD, DP_KLD, KLD, MSE, TAID, DistillLoss, DistilLoss, JasperStella

taid_forward_fn_map = {"ckd": CKD, "kld": KLD, "mse": MSE, "dp_kld": DP_KLD, "js": JasperStella}


@dataclass
class LossOutput:
    loss: Tensor
    loss_dict: dict[str, Tensor]


def _is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather(lightning_module: LightningModule, tensor: torch.Tensor, sync_grads: bool = True) -> torch.Tensor:
    if not _is_dist():
        return tensor
    # world_size , bs, dim
    batch_size: int = tensor.size(0)
    pid: int = lightning_module.global_rank
    gathered_stacked = lightning_module.all_gather(tensor, sync_grads=sync_grads)
    # world_size x bs, dim
    gathered_concatenated = gathered_stacked.reshape(-1, *gathered_stacked.shape[2:])
    gathered_concatenated[batch_size * pid : batch_size * (pid + 1)] = tensor
    return gathered_concatenated


class InfoCSE(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.use_pos = args.use_pos if hasattr(args, "use_pos") else False
        self.temp = args.cse_temp if hasattr(args, "cse_temp") else 0.05

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
        validation: bool = False,
        **kwargs,
    ) -> LossOutput:
        # あとで全部これにする
        local_features = torch.cat(
            [lightning_module.student_model(anc)["sentence_embedding"] for anc in batch.anc], dim=0
        )
        local_features = F.normalize(local_features, dim=-1)
        global_features = gather(lightning_module, local_features)
        if self.use_pos:
            local_pos_features = torch.cat(
                [lightning_module.student_model(pos)["sentence_embedding"] for pos in batch.pos], dim=0
            )
        else:
            # unsupと同じように同じ文2回かける（dropoutでちょっと違う埋め込みになるはず）
            local_pos_features = torch.cat(
                [lightning_module.student_model(anc)["sentence_embedding"] for anc in batch.anc], dim=0
            )
        local_pos_features = F.normalize(local_pos_features, dim=-1)
        # 3. 全てのGPUから 'pos_features' を収集して結合
        global_pos_features = gather(lightning_module, local_pos_features)
        loss = self.loss_fn(features=global_features, pos_features=global_pos_features)
        return loss

    def loss_fn(
        self,
        features: torch.Tensor,
        pos_features: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        labels = torch.arange(features.size(0), device=features.device)
        # クエリとキー間の類似度スコアを計算 ab,cb->ac
        scores = einsum(features, pos_features, "b d, k d -> b k") / self.temp

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
        distil_loss_fn: DistilLoss | None = None,
        distill_weight: float = 1.0,
        cse_temp: float = 0.05,
        use_neg: bool = False,
    ):
        super().__init__()
        self.use_pos = use_pos
        self.use_neg = use_neg
        self.distil_loss_fn = distil_loss_fn
        self.distill_weight = distill_weight
        self.cse_temp = cse_temp

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
        validation: bool = False,
        **kwargs,
    ) -> LossOutput:
        loss_dict = {}

        local_student_features = torch.cat(
            [lightning_module.student_model(anc)["sentence_embedding"] for anc in batch.anc], dim=0
        )
        local_teacher_features = batch.teacher_features
        if isinstance(local_teacher_features, list):
            local_teacher_features = torch.stack(local_teacher_features, dim=0)
        # studentと次元数を揃える（マトリョーシカ埋め込みを想定）
        local_teacher_features = local_teacher_features[:, : local_student_features.shape[1]]

        # 全GPUから学生・教師の特徴量を集約
        global_student_features = gather(lightning_module, local_student_features)
        global_teacher_features = gather(lightning_module, local_teacher_features)
        if self.use_pos:
            local_pos_student_features = torch.cat(
                [lightning_module.student_model(pos)["sentence_embedding"] for pos in batch.pos], dim=0
            )
            local_pos_teacher_features = batch.pos_features
            if isinstance(local_pos_teacher_features, list):
                local_pos_teacher_features = torch.stack(local_pos_teacher_features, dim=0)
            local_pos_teacher_features = local_pos_teacher_features[:, : local_pos_student_features.shape[1]]

            # Positiveペアの特徴量も全GPUから集約
            global_pos_student_features = gather(lightning_module, local_pos_student_features)
            global_pos_teacher_features = gather(lightning_module, local_pos_teacher_features)
        else:
            global_pos_student_features = None
            global_pos_teacher_features = None

        distill_loss_dict = self.distil_loss_fn(
            lightning_module=lightning_module,
            projected_features=global_student_features,
            teacher_features=global_teacher_features,
            pos_projected_features=global_pos_student_features,
            pos_teacher_features=global_pos_teacher_features,
            validation=validation,
            **kwargs,
        )
        loss_dict = distill_loss_dict
        if self.distill_weight != 1.0:
            labels = torch.arange(global_student_features.size(0), device=global_student_features.device)
            # クエリとキー間の類似度スコアを計算 ab,cb->ac
            if self.use_pos:
                scores = (
                    einsum(
                        F.normalize(global_student_features, dim=-1),
                        F.normalize(global_pos_student_features, dim=-1),
                        "b d, k d -> b k",
                    )
                    / self.cse_temp
                )
            else:
                local_pos_features = torch.cat(
                    [lightning_module.student_model(anc)["sentence_embedding"] for anc in batch.anc], dim=0
                )
                global_pos_features = gather(lightning_module, local_pos_features)
                scores = (
                    einsum(
                        F.normalize(global_student_features, dim=-1),
                        F.normalize(global_pos_features, dim=-1),
                        "b d, k d -> b k",
                    )
                    / self.cse_temp
                )
            # 対照学習損失：生徒埋め込みが対応する教師埋め込みに最も類似するように学習
            cse_loss = F.cross_entropy(scores, labels)
            loss = distill_loss_dict["loss"] * self.distill_weight + cse_loss * (1 - self.distill_weight)
            loss_dict["distill_loss"] = distill_loss_dict["loss"]
            loss_dict["loss"] = loss
            loss_dict["cse_loss"] = cse_loss
        return LossOutput(
            loss=loss_dict["loss"],
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
    return KDLoss(use_pos=args.use_pos, distil_loss_fn=distil_loss_fn, distill_weight=args.distill_weight)
