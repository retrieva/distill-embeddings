"""
This implementation is based on Jasper(https://github.com/NLPJCL/RAG-Retrieval/blob/master/rag_retrieval/train/embedding/model_distill.py)
TODO：実装
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule

from .base import DistilLoss


class JasperStella(DistilLoss):
    def __init__(self, args: dict | None = None):
        super().__init__()
        self.lambda_cos = 10
        self.lambda_sim = 200
        self.lambda_tri = 20
        self.triplet_margin = 0.015
        self.use_pos = args.use_pos if hasattr(args, "use_pos") else False

    def compute_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        validation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        loss_dict = {}
        cosine_loss = self.cosine_embedding_loss(student_features, teacher_features)
        similarity_loss = self.pair_inbatch_similarity_loss(student_features, teacher_features)
        triplet_loss = self.pair_inbatch_triplet_loss(student_features, teacher_features)

        loss_dict["loss"] = (
            cosine_loss * self.lambda_cos + similarity_loss * self.lambda_sim + triplet_loss * self.lambda_tri
        )
        loss_dict["cosine_loss"] = cosine_loss * self.lambda_cos
        loss_dict["similarity_loss"] = similarity_loss * self.lambda_sim
        loss_dict["triplet_loss"] = triplet_loss * self.lambda_tri
        return loss_dict["loss"], loss_dict

    def make_features(
        self,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        hyp_projected_features: torch.Tensor = None,
        hyp_teacher_features: torch.Tensor = None,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        入力埋め込みを正規化し、比較対象となる教師側の埋め込みを選ぶ。

        - KDLoss/TAID 経由では `hyp_*` が渡される想定（pos が先頭、neg が続く）。
        - 互換性のため、従来の `pos_*` も受け付ける。
        - C(= candidates_per_anchor) > 1 の場合は形状が合わないため、anc 側の `teacher_features` を用いる。
        """
        projected_features = F.normalize(projected_features, dim=-1)

        if self.use_pos:
            # 優先順位: hyp_teacher_features(形状がBと一致) > pos_teacher_features > teacher_features
            if hyp_teacher_features is not None and hyp_teacher_features.shape[0] == projected_features.shape[0]:
                return projected_features, F.normalize(hyp_teacher_features, dim=-1)
            if pos_teacher_features is not None and pos_teacher_features.shape[0] == projected_features.shape[0]:
                return projected_features, F.normalize(pos_teacher_features, dim=-1)

        # それ以外は通常の teacher_features を使用
        return projected_features, F.normalize(teacher_features, dim=-1)

    def forward(
        self,
        lightning_module: LightningModule,
        projected_features: torch.Tensor,
        teacher_features: torch.Tensor,
        hyp_projected_features: torch.Tensor = None,
        hyp_teacher_features: torch.Tensor = None,
        candidates_per_anchor: int = 1,
        validation: bool = False,
        pos_projected_features: torch.Tensor = None,
        pos_teacher_features: torch.Tensor = None,
        **kwargs,
    ) -> dict | torch.Tensor:
        if candidates_per_anchor > 1:
            raise ValueError(
                f"JasperStella does not support negatives: candidates_per_anchor={candidates_per_anchor} (>1)."
            )
        # hyp_*（pos/neg 候補）も受け取りつつ、形が合う場合だけ pos を採用する
        student_features, teacher_features = self.make_features(
            projected_features=projected_features,
            teacher_features=teacher_features,
            hyp_projected_features=hyp_projected_features,
            hyp_teacher_features=hyp_teacher_features,
            pos_projected_features=pos_projected_features,
            pos_teacher_features=pos_teacher_features,
        )
        loss, loss_dict = self.compute_loss(
            student_features=student_features,
            teacher_features=teacher_features,
            validation=validation,
        )
        return loss_dict

    def get_score_diff(self, embedding):
        scores = torch.matmul(embedding, embedding.T)
        scores = scores[torch.triu(torch.ones_like(scores), diagonal=1).bool()]
        score_diff = scores.reshape((1, -1)) - scores.reshape((-1, 1))
        score_diff = score_diff[torch.triu(torch.ones_like(score_diff), diagonal=1).bool()]
        return score_diff

    def cosine_embedding_loss(
        self,
        student_features,  # [batch_size,dim]
        teacher_features,  # [batch_size,dim]
    ):
        # get cosine loss
        # positive pairs only
        labels = torch.ones(student_features.size(0), device=student_features.device, dtype=student_features.dtype)
        loss = F.cosine_embedding_loss(student_features, teacher_features, labels)
        return loss

    def pair_inbatch_similarity_loss(
        self,
        student_features,  # [batch_size,dim]
        teacher_features,  # [batch_size,dim]
    ):
        # get mse loss
        # [batch_size,batch_size]<- [batch_size,dim],[dim,batch_size]
        student_similarity = student_features @ student_features.transpose(-1, -2)
        teacher_similarity = teacher_features @ teacher_features.transpose(-1, -2)
        loss = F.mse_loss(student_similarity, teacher_similarity)
        return loss

    def pair_inbatch_triplet_loss(
        self,
        student_features,  # [batch_size,dim]
        teacher_features,  # [batch_size,dim]
    ):
        triplet_label = torch.where(self.get_score_diff(teacher_features) < 0, 1, -1)
        # get triplets loss
        loss = F.relu(self.get_score_diff(student_features) * triplet_label + self.triplet_margin).mean()

        return loss
