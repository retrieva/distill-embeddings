from typing import Dict, Any
import torch
from torch import Tensor
import lightning as L
from transformers import get_scheduler, AutoConfig
from src.loss import get_loss_fn, LossOutput
from sentence_transformers import SentenceTransformer
from src.data import Batch
import mteb
import yaml

class KDForSentEmb(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.student_model = None
        self.teacher_model_config = None
        self.linear = None
        self.loss_fn = get_loss_fn(args)
        self.args = args
        self.validation_step_outputs = {}
        self.mteb_dict = {}
        with open("tasks.yaml", 'r') as file:
            self.tasks = yaml.safe_load(file)[args.language]["tasks"]

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
        try:
            # MTEB evaluation
            output_folder = self.args.output_dir / "mteb_eval"
            evaluation = mteb.MTEB(tasks=self.tasks, task_langs=[self.args.language],)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = evaluation.run(self.student_model, output_folder=output_folder,
                                        num_workers=self.args.num_workers,
                                        overwrite_results=True,
                                        verbosity=0,
                                        encode_kwargs={"batch_size": self.args.batch_size},
                                        )
            mteb_dict = {score.task_name: score.get_score() for score in scores}
            if self._save_mteb_flag(mteb_dict):
                self.mteb_dict = mteb_dict
            self.log_dict(mteb_dict,logger=True,sync_dist=True)
            self.print(f"MTEB evaluation results: {mteb_dict}")
        except Exception as e:
            self.print(f"Error during MTEB evaluation: {e}")
            self.print("Skipping MTEB evaluation due to an error.")

    def _save_mteb_flag(self,mteb_dict:dict) -> bool:
        return True

    def on_train_end(self) -> None:
        for task_name, score in self.mteb_dict.items():
            print(f"{task_name}: {score}")
        with open(self.args.output_dir / "mteb_eval" / "scores.txt", "w") as f:
            f.write(f"|")
            for task_name in self.mteb_dict.keys():
                f.write(f"{task_name}|")
            f.write(f"\n|")
            for score in self.mteb_dict.values():
                f.write(f"{score}|")
        print(f"Scores saved to {self.args.output_dir / "mteb_eval" / 'scores.txt'}")

    # def on_save_checkpoint(self, trainer: L.Trainer, lightning_module: L.LightningModule, checkpoint: Dict[str, Any]):
    def on_save_checkpoint(self, checkpoint):
        checkpoint["teacher_model_name"] = self.args.teacher_model
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
