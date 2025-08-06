from typing import Dict, Any
import torch
from torch import Tensor
import lightning as L
from transformers import get_scheduler, AutoConfig
from src.loss import get_loss_fn, LossOutput
from sentence_transformers import SentenceTransformer
from src.data import Batch

class KDForSentEmb(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.student_model = None
        self.teacher_model_config = None
        self.linear = None
        self.loss_fn = get_loss_fn(args)
        self.args = args
        self.validation_step_outputs = {}

    def configure_model(self):
        self.student_model = SentenceTransformer(
            self.args.student_model,
        ).bfloat16()
        self.teacher_model_config = AutoConfig.from_pretrained(
            self.args.teacher_model,
            trust_remote_code=True,
        )
        # up projection layer
        self.linear = torch.nn.Linear(
            self.student_model.get_sentence_embedding_dimension(),
            self.teacher_model_config.hidden_size
        )

    def forward(self, batch: Batch, **kwargs) -> LossOutput:
        outputs: LossOutput = self.loss_fn(lightning_module=self, batch=batch, **kwargs)
        return outputs

    def get_batch_size(self, batch: Batch) -> int:
        return batch["input_ids"].size(0)

    def training_step(self, batch: Batch, batch_idx) -> Tensor:
        batch_size = self.get_batch_size(batch)
        # compute loss
        outputs = self(batch)
        loss_dict = outputs.loss_dict
        loss_dict = {
            f"train/{k}": v for k, v in loss_dict.items() if v is not None
        }
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return outputs.loss

    def encode(self, inputs):
        """
        Encode the input sentences using the student model.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = self.student_model.encode(inputs, convert_to_tensor=True)
        return embeddings

    # def _compute_metric(self, outputs, step="val"):
    #     preds = flatten_list([item["preds"] for item in outputs])
    #     target = flatten_list([item["target"] for item in outputs])
    #     metric_res = compute_metrics(preds, target)
    #     metric_res = {f"{step}/{k}": v for k, v in metric_res.items()}
    #     return metric_res

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        res = {}
        batch_size = self.get_batch_size(batch)
        outputs = self(batch)
        loss_dict = outputs.loss_dict
        loss_dict = {
            f"val_{dataloader_idx}/{k}": v for k, v in loss_dict.items() if v is not None
        }

        self.log_dict(
            loss_dict,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        res["loss"] = outputs.loss
        # validation, STSとかで適当にやる？
        # JSICKかJSTS　MTEBで簡単にできるはず
        # if "model_inputs_gen" in batch:
        #     model_inputs_gen = batch.pop("model_inputs_gen")
        #     preds, target = self.generate(model_inputs_gen)
        #     res["preds"] = preds
        #     res["target"] = target
        # if dataloader_idx not in self.validation_step_outputs:
        #     self.validation_step_outputs[dataloader_idx] = []
        self.validation_step_outputs[dataloader_idx].append(res)
        return outputs.loss

    # def on_validation_epoch_end(self):
    #     res = {}
    #     for (
    #         idx,
    #         step_outputs,
    #     ) in self.validation_step_outputs.items():
    #         losses = torch.stack([item["loss"] for item in step_outputs])
    #         eval_loss = losses.mean()
    #         if "preds" in step_outputs[0]:
    #             metric_res = self._compute_metric(step_outputs, step=f"val_{idx}")
    #             res.update(metric_res)
    #         if self.sampler is not None and idx == 0:
    #             self.sampler.update(loss=eval_loss)
    #             if self.sampler.sampling_type == "adaptive":
    #                 res["adaptive_threshold"] = self.sampler.adaptive_threshold
    #     self.log_dict(res, logger=True, sync_dist=True)
    #     self.validation_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        for key in list(sd.keys()):
            if "teacher_model." in key:
                del sd[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        sd = self.state_dict()
        for key in list(sd.keys()):
            if "teacher_model." in key:
                if "state_dict" in checkpoint:
                    checkpoint["state_dict"][key] = sd[key]
                else:
                    checkpoint[key] = sd[key]

    def configure_optimizers(self):
        # distilcseからbetaが少し変わっているので注意　https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        # original: eps=1e-8, betas=(0.9, 0.98）
        optim = torch.optim.AdamW(self.student_model.parameters(),lr=self.args.lr)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optim,
            num_training_steps=num_training_steps,
            num_warmup_steps=self.args.warmup_ratio * num_training_steps,
        )
        self.print(
            f"Setting up scheduler (estimated_stepping_batches: {num_training_steps})..."
        )
        scheduler = [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optim], scheduler
