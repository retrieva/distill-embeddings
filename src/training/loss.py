from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import einsum
from lightning import LightningModule
from torch import Tensor, nn

from src.training.data import Batch
from src.training.distil_losses import CKD, KLD, MSE, TAID, DistilLoss

taid_forward_fn_map = {"ckd": CKD, "kld": KLD, "mse": MSE}


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


def encode_chunks(model, chunks):
    # chunks: list[BatchEncoding]（チャンク列）
    return torch.cat([model(c)["sentence_embedding"] for c in chunks], dim=0)  # [B_local, D]


def build_candidates_flat(lm: LightningModule, batch: Batch, use_pos: bool, use_neg: bool):
    """
    returns:
      q_s: [B_all, D]         （anchor student）
      q_t: [B_all, D]         （anchor teacher）
      pos_s: [B_all, D] or None   （pos only → CSE用）
      cand_s_flat: [B_all*C, D] or None   （pos(+neg) を flat）
      cand_t_flat: [B_all*C, D] or None
      C: int（1+K or 0）
    """
    # anchors
    q_s_local = encode_chunks(lm.student_model, batch.anc)
    q_t_local = (
        torch.stack(batch.teacher_features, dim=0)
        if isinstance(batch.teacher_features, list)
        else batch.teacher_features
    )
    q_t_local = q_t_local[:, : q_s_local.shape[1]]
    q_s = gather(lm, q_s_local, sync_grads=True)
    q_t = gather(lm, q_t_local, sync_grads=False)

    if not use_pos:
        return q_s, q_t, None, None, None, 0

    # pos (student)
    pos_s_local = encode_chunks(lm.student_model, batch.pos)  # [B_l, D]
    pos_s = gather(lm, pos_s_local, sync_grads=True)  # [B_all, D]

    # candidates: start with pos
    C = 1
    cand_s_local = pos_s_local.unsqueeze(1)  # [B_l, 1, D]

    # teacher candidates
    pos_t_local = (
        torch.stack(batch.pos_features, dim=0) if isinstance(batch.pos_features, list) else batch.pos_features
    )
    pos_t_local = pos_t_local[:, : q_s_local.shape[1]]
    cand_t_local = pos_t_local.unsqueeze(1)  # [B_l, 1, D]

    K = getattr(batch, "num_neg", 0) if (use_neg and len(getattr(batch, "neg", [])) > 0) else 0
    if K > 0 and use_neg:
        neg_locals = [encode_chunks(lm.student_model, batch.neg[j]) for j in range(K)]  # list of [B_l, D]
        cand_s_local = torch.cat([cand_s_local] + [n.unsqueeze(1) for n in neg_locals], dim=1)  # [B_l, 1+K, D]

        neg_t_local = torch.stack([torch.stack(nlist, dim=0) for nlist in batch.neg_features], dim=0)  # [B_l, K, D]
        neg_t_local = neg_t_local[:, :, : q_s_local.shape[1]]
        cand_t_local = torch.cat([cand_t_local, neg_t_local], dim=1)  # [B_l, 1+K, D]
        C = 1 + K

    # flat にしてから gather（効率＆実装簡潔）
    B_l = cand_s_local.shape[0]
    cand_s_flat = gather(lm, cand_s_local.reshape(B_l * C, -1), sync_grads=True)  # [B_all*C, D]
    cand_t_flat = gather(lm, cand_t_local.reshape(B_l * C, -1), sync_grads=False)  # [B_all*C, D]
    return q_s, q_t, pos_s, cand_s_flat, cand_t_flat, C


def build_student_cand_flat(lm: LightningModule, batch: Batch, use_pos: bool, use_neg: bool):
    """
    returns:
      q_s: [B_all, D]                 # anchor student
      cand_s_flat: [B_all*C, D] or None
      C: int  # 1+K or 0
    """
    # anchors
    anc_s_local = encode_chunks(lm.student_model, batch.anc)  # [B_l, D]
    anc_s = gather(lm, anc_s_local, sync_grads=True)  # [B_all, D]

    if not use_pos:
        return anc_s, None, 0

    # pos
    pos_s_local = encode_chunks(lm.student_model, batch.pos)  # [B_l, D]
    cand_s_local = pos_s_local.unsqueeze(1)  # [B_l, 1, D]
    C = 1

    # hard neg
    K = getattr(batch, "num_neg", 0) if (use_neg and len(getattr(batch, "neg", [])) > 0) else 0
    if K > 0 and use_neg:
        neg_locals = [encode_chunks(lm.student_model, batch.neg[j]) for j in range(K)]  # list of [B_l, D]
        cand_s_local = torch.cat([cand_s_local] + [n.unsqueeze(1) for n in neg_locals], dim=1)  # [B_l,1+K,D]
        C = 1 + K

    B_l = cand_s_local.shape[0]
    cand_s_flat = gather(lm, cand_s_local.reshape(B_l * C, -1), sync_grads=True)  # [B_all*C, D]
    return anc_s, cand_s_flat, C


class InfoCSE(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.use_pos = args.use_pos if hasattr(args, "use_pos") else False
        self.use_neg = args.use_neg if hasattr(args, "use_neg") else False
        self.temp = args.cse_temp if hasattr(args, "cse_temp") else 0.05

    def forward(
        self,
        lightning_module: LightningModule,
        batch: Batch,
        validation: bool = False,
        **kwargs,
    ) -> LossOutput:
        anc_s, cand_s_flat, C = build_student_cand_flat(
            lightning_module,
            batch,
            self.use_pos,
            self.use_neg,
        )
        if not self.use_pos:
            # unsupと同じように同じ文2回かける（dropoutでちょっと違う埋め込みになるはず）
            local_anc_s_2 = encode_chunks(lightning_module.student_model, batch.anc)
            global_anc_s_2 = gather(lightning_module, local_anc_s_2)
            cand_s_flat = global_anc_s_2
        loss = self.loss_fn(F.normalize(anc_s, dim=-1), F.normalize(cand_s_flat, dim=-1), C)
        return loss

    def loss_fn(
        self,
        features: torch.Tensor,
        pos_features: torch.Tensor,
        candidates_per_anchor: int = 1,
        **kwargs,
    ) -> LossOutput:
        labels = torch.arange(features.size(0), device=features.device) * max(candidates_per_anchor, 1)
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
        anc_s, anc_t, pos_s, cand_s_flat, cand_t_flat, C = build_candidates_flat(
            lightning_module,
            batch,
            self.use_pos,
            self.use_neg,
        )
        distill_loss_dict = self.distil_loss_fn(
            lightning_module=lightning_module,
            projected_features=anc_s,
            teacher_features=anc_t,
            hyp_projected_features=cand_s_flat,
            hyp_teacher_features=cand_t_flat,
            candidates_per_anchor=C,
            validation=validation,
            **kwargs,
        )
        loss_dict = distill_loss_dict
        if self.distill_weight != 1.0:
            labels = torch.arange(anc_s.size(0), device=anc_s.device) * max(C, 1)
            # クエリとキー間の類似度スコアを計算 ab,cb->ac
            if not self.use_pos:
                local_anc_2 = encode_chunks(lightning_module.student_model, batch.anc)
                global_anc_2 = gather(lightning_module, local_anc_2, sync_grads=True)
                cand_s_flat = global_anc_2
            scores = (
                einsum(
                    F.normalize(anc_s, dim=-1),
                    F.normalize(cand_s_flat, dim=-1),
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
    else:
        raise NotImplementedError(args.loss_type)
    return KDLoss(
        use_pos=args.use_pos, use_neg=args.use_neg, distil_loss_fn=distil_loss_fn, distill_weight=args.distill_weight
    )
