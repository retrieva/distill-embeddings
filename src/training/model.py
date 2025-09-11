from typing import Any

import lightning as L
import mteb
import skdim
import torch
import yaml
from datasets import load_from_disk
from IsoScore.IsoScore import IsoScore
from mteb.encoder_interface import PromptType
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoConfig

from src.training.data import Batch
from src.training.loss import LossOutput, get_loss_fn
from src.training.scheduler import get_scheduler

PROMPT_MAP = {
    "none": "",
    "query": "query: ",
    "passage": "passage: ",
    "retrieval": "Given a question, retrieve passages that answer the question",
    "sts": "Retrieve semantically similar text",
    "classification": "Given a text, classify its topic",
}


class SentEmb(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.student_model = None
        self.loss_fn = get_loss_fn(args)
        self.args = args
        self.validation_step_outputs = {}
        self.mteb_dict = {}
        with open("tasks.yaml") as file:
            self.on_train_tasks = yaml.safe_load(file)[args.language]["on_train_tasks"]
        with open("tasks.yaml") as file:
            self.on_eval_tasks = yaml.safe_load(file)[args.language]["on_train_end_tasks"]
        self.save_hyperparameters(vars(args))

    def add_lora_adapter(self, model: SentenceTransformer):
        peft_config = LoraConfig(
            target_modules="all-linear",
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
        )
        model.add_adapter(peft_config)
        print(model.active_adapters())
        return model

    def configure_model(self):
        self.student_model = SentenceTransformer(
            self.args.student_model,
        ).bfloat16()
        if self.args.use_lora:
            self.student_model = self.add_lora_adapter(self.student_model)
        if self.args.gradient_checkpointing:
            self.student_model[0].auto_model.config.use_cache = False
            self.student_model[0].auto_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )

    def forward(self, batch: Batch, validation: bool = False, **kwargs) -> LossOutput:
        outputs: LossOutput = self.loss_fn(lightning_module=self, batch=batch, validation=validation, **kwargs)
        return outputs

    def get_batch_size(self, batch: Batch) -> int:
        return len(batch)

    def training_step(self, batch: Batch, batch_idx) -> Tensor:
        batch_size = self.get_batch_size(batch)
        # compute loss
        outputs = self(batch)
        loss_dict = outputs.loss_dict
        loss_dict = {f"train/{k}": v for k, v in loss_dict.items() if v is not None}
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return outputs.loss

    @torch.no_grad()
    def encode(self, inputs, **kwargs):
        """
        Encode the input sentences using the student model.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = self.student_model.encode(inputs, convert_to_tensor=True, **kwargs)
        return embeddings

    def on_train_epoch_start(self):
        self.trainer.datamodule.rebuild_train_batches_for_epoch(self.current_epoch)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        res = {}
        batch_size = self.get_batch_size(batch)
        outputs = self(batch, validation=True)
        loss_dict = outputs.loss_dict
        loss_dict = {f"val_{dataloader_idx}/{k}": v for k, v in loss_dict.items() if v is not None}

        self.log_dict(
            loss_dict,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        res["loss"] = outputs.loss
        if dataloader_idx not in self.validation_step_outputs:
            self.validation_step_outputs[dataloader_idx] = []
        self.validation_step_outputs[dataloader_idx].append(res)
        return outputs.loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        if not self.args.mteb_eval:
            return
        try:
            # MTEB evaluation
            if self.args.add_prefix:
                model_prompts = {
                    PromptType.query.value: "query: ",
                    PromptType.passage.value: "document: ",
                }
                self.student_model.prompts = model_prompts
            output_folder = self.args.output_dir / "mteb_eval"
            evaluation = mteb.MTEB(
                tasks=self.on_eval_tasks,
                task_langs=[self.args.language],
            )
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = evaluation.run(
                    self.student_model,
                    output_folder=output_folder,
                    num_workers=self.args.num_workers,
                    overwrite_results=True,
                    verbosity=1,
                    encode_kwargs={"batch_size": self.args.batch_size},
                )
            mteb_dict = {score.task_name: score.get_score() for score in scores}
            if self._save_mteb_flag(mteb_dict):
                self.mteb_dict = mteb_dict

            self.log_dict(mteb_dict, logger=True, sync_dist=False)
            self.print(f"MTEB evaluation results: {mteb_dict}")
        except Exception as e:
            self.print(f"Error during MTEB evaluation: {e}")
            self.print("Skipping MTEB evaluation due to an error.")

    def _save_mteb_flag(self, mteb_dict: dict) -> bool:
        return True

    def _on_train_end_mteb(self):
        self.trainer.save_checkpoint(self.args.output_dir / "last.ckpt")
        if not self.args.mteb_eval:
            return
        try:
            if self.args.add_prefix:
                model_prompts = {
                    PromptType.query.value: "query: ",
                    PromptType.passage.value: "document: ",
                }
                self.student_model.prompts = model_prompts
            # MTEB evaluation
            output_folder = self.args.output_dir / "mteb_eval"
            tasks = mteb.get_benchmark("MTEB(eng, v2)").tasks
            evaluation = mteb.MTEB(
                tasks=tasks,
                # tasks=self.on_eval_tasks,
                task_langs=[self.args.language],
            )
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = evaluation.run(
                    self.student_model,
                    output_folder=output_folder,
                    num_workers=self.args.num_workers,
                    overwrite_results=True,
                    verbosity=1,
                    encode_kwargs={"batch_size": self.args.batch_size},
                )
            # MTEBの最終結果をSummaryに保存
            final_mteb_dict = {score.task_name: score.get_score() for score in scores}
            final_summary_dict = {f"mteb_final/{k}": v for k, v in final_mteb_dict.items()}

            self.logger.experiment.summary.update(final_summary_dict)
        except Exception as e:
            self.print(f"Error during MTEB evaluation: {e}")
            self.print("Skipping MTEB evaluation due to an error.")

    def get_id_iso_score(self):
        if not self.args.get_id_iso:
            return
        score_dict = {}
        wiki = load_from_disk("data/anly-wiki/en")
        texts = wiki["text"]
        for prompt_name, prompt in PROMPT_MAP.items():
            embeddings = self.student_model.encode(texts, convert_to_tensor=True, prompt=prompt).to("cpu")
            twonn = skdim.id.TwoNN()
            # .to(torch.float32)で型を変換し、.cpu()でCPUに転送してから.numpy()を呼ぶ
            embeddings_np = embeddings.to(torch.float32).cpu().numpy()
            twonn.fit(embeddings_np)
            intrinsic_dimension_twonn = twonn.dimension_
            iso_score = IsoScore(embeddings_np)
            score_dict[f"{prompt_name}/iso_score"] = iso_score
            score_dict[f"{prompt_name}/id"] = intrinsic_dimension_twonn
        self.logger.experiment.summary.update(score_dict)

    def on_train_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self.get_id_iso_score()
        self._on_train_end_mteb()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["student_model_name"] = self.args.student_model
        checkpoint["code_name"] = self.args.output_dir.name

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
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
        optim = torch.optim.AdamW(self.student_model.parameters(), lr=self.args.lr)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name=self.args.scheduler,
            optimizer=optim,
            num_training_steps=num_training_steps,
            num_warmup_steps=self.args.warmup_ratio * num_training_steps,
        )
        self.print(f"Setting up scheduler (estimated_stepping_batches: {num_training_steps})...")
        scheduler = [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]
        return [optim], scheduler


class KDForSentEmb(SentEmb):
    def __init__(self, args):
        super().__init__(args)
        self.teacher_model_config = None

    def configure_model(self):
        super().configure_model()
        self.teacher_model_config = AutoConfig.from_pretrained(
            self.args.teacher_model,
            trust_remote_code=True,
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["teacher_model_name"] = self.args.teacher_model
