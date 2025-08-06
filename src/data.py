import os
from typing import Optional

import torch
import lightning as L
from transformers import PreTrainedTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    teacher_features: torch.Tensor

class DataCollatorForDistill:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, samples):
        texts = [s["text"] for s in samples]
        teacher_features = [torch.Tensor(s["teacher_features"]) for s in samples]
        inputs = self.preprocess(texts)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "teacher_features": teacher_features}


class DataModuleForDistill(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        student_tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
        eval_batch_size: Optional[int] = None,
        max_length: int = 4096
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = student_tokenizer

        self.collate_fn = DataCollatorForDistill(
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        self.datasets = {}

    def setup(self, stage: str):
        datasets = load_from_disk(self.data_path)
        self.datasets = datasets.train_test_split(test_size=1000, seed=42, shuffle=True)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
