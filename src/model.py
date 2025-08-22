from typing import Dict, Any
import torch
from torch import Tensor
import lightning as L
from transformers import AutoConfig
from src.loss import get_loss_fn, LossOutput
from sentence_transformers import SentenceTransformer
from src.data import Batch
from src.scheduler import get_scheduler
import mteb
import yaml
from IsoScore.IsoScore import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import skdim

PROMPT_MAP = {
    "none": "",
    "retrieval": "Given a question, retrieve passages that answer the question",
    "sts": "Retrieve semantically similar text",
    "classification": "Given a text, classify its topic"
}

class SentEmb(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.student_model = None
        self.loss_fn = get_loss_fn(args)
        self.args = args
        self.validation_step_outputs = {}
        self.mteb_dict = {}
        with open("tasks.yaml", 'r') as file:
            self.on_train_tasks = yaml.safe_load(file)[args.language]["on_train_tasks"]
        with open("tasks.yaml", 'r') as file:
            self.on_eval_tasks = yaml.safe_load(file)[args.language]["on_train_end_tasks"]
        # argsをsave_hyperparametersで保存（wandbにも自動的に送信される）
        self.save_hyperparameters(vars(args))
    def configure_model(self):
        self.student_model = SentenceTransformer(
            self.args.student_model,
        ).bfloat16()

    def forward(self, batch: Batch, validation:bool = False, **kwargs) -> LossOutput:
        outputs: LossOutput = self.loss_fn(lightning_module=self, batch=batch, validation=validation, **kwargs)
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
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        res = {}
        batch_size = self.get_batch_size(batch)
        outputs = self(batch,validation=True)
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
        if dataloader_idx not in self.validation_step_outputs:
            self.validation_step_outputs[dataloader_idx] = []
        self.validation_step_outputs[dataloader_idx].append(res)
        return outputs.loss
    
    def on_train_epoch_end(self):
        if not self.args.mteb_eval:
            return
            # 評価処理全体をメインプロセス (rank 0) でのみ実行するようにガードする
        if not self.trainer.is_global_zero:
            return
        try:
            self.print("🚀 Starting MTEB evaluation on global_rank 0...")
            # MTEB evaluation
            output_folder = self.args.output_dir / "mteb_eval"
            evaluation = mteb.MTEB(tasks=self.on_train_tasks, task_langs=[self.args.language],)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = evaluation.run(self.student_model, output_folder=output_folder,
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

    def _save_mteb_flag(self,mteb_dict:dict) -> bool:
        return True
    
    def _on_train_end_mteb(self):
        if not self.args.mteb_eval:
            return
        try:
            # MTEB evaluation
            output_folder = self.args.output_dir / "mteb_eval"
            evaluation = mteb.MTEB(tasks=self.on_eval_tasks, task_langs=[self.args.language],)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = evaluation.run(self.student_model, output_folder=output_folder,
                                        num_workers=self.args.num_workers,
                                        overwrite_results=True,
                                        verbosity=1,
                                        encode_kwargs={"batch_size": self.args.batch_size},
                                        )
            # MTEBの最終結果をSummaryに保存
            final_mteb_dict = {score.task_name: score.get_score() for score in scores}
            final_summary_dict = {f"mteb_final/{k}": v for k, v in final_mteb_dict.items()}
            
            # 平均スコアも計算して保存
            avg_score = sum(final_mteb_dict.values()) / len(final_mteb_dict)
            final_summary_dict["mteb_final/average"] = avg_score
            
            self.logger.experiment.summary.update(final_summary_dict)
        except Exception as e:
            self.print(f"Error during MTEB evaluation: {e}")
            self.print("Skipping MTEB evaluation due to an error.")

    def get_id_iso_score(self):
        if not self.args.get_id_iso:
            return
        score_dict = {}
        simple_wiki = pd.read_json("data/triplet-eng/SimpleWiki.jsonl",lines=True,orient="records")
        texts = simple_wiki.sample(10000,random_state=42)["anc"].unique().tolist()
        for prompt_name, prompt in PROMPT_MAP.items():
            print(f"Calculating IsoScore for model: {self.args.student_model}, prompt: {prompt_name}")
            model = SentenceTransformer(self.args.student_model)
            embeddings = model.encode(texts, convert_to_tensor=True, prompt=prompt).to("cpu")
            twonn = skdim.id.TwoNN()
            twonn.fit(embeddings)
            intrinsic_dimension_twonn = twonn.dimension_
            iso_score = IsoScore(embeddings)
            score_dict[f"{prompt_name}/iso_score"] = iso_score
            score_dict[f"{prompt_name}/id"] = intrinsic_dimension_twonn
        self.logger.experiment.summary.update(score_dict)
    
    def on_train_end(self) -> None:
        self.on_train_end_mteb()
        self.get_id_iso_score()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["student_model_name"] = self.args.student_model
        checkpoint["code_name"] = self.args.output_dir.name

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
            name=self.args.scheduler,
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
    
class KDForSentEmb(SentEmb):
    def __init__(self, args):
        super().__init__(args)
        self.teacher_model_config = None
        # self.linear = None

    def configure_model(self):
        super().configure_model()
        self.teacher_model_config = AutoConfig.from_pretrained(
            self.args.teacher_model,
            trust_remote_code=True,
        )
        # # up projection layer
        # self.linear = torch.nn.Linear(
        #     self.student_model.get_sentence_embedding_dimension(),
        #     self.teacher_model_config.hidden_size
        # )
    # def on_save_checkpoint(self, trainer: L.Trainer, lightning_module: L.LightningModule, checkpoint: Dict[str, Any]):
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["teacher_model_name"] = self.args.teacher_model