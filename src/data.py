import os
from typing import Optional

import torch
import lightning as L
from transformers import PreTrainedTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pos: dict[str, torch.Tensor]
    teacher_features: torch.Tensor
    pos_features: torch.Tensor


class DistilDataset(Dataset):
    def __init__(self, dataset:Dataset, embedding:np.ndarray):
        super().__init__()
        self.dataset = dataset
        self.embedding = embedding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # mmap_mode='r'で読むと書き込み不可になるため、警告も出るし一応コピーしとく
        return {
            "anc": self.dataset[idx]["anc"],
            "anc_features": self.embedding[self.dataset[idx]["anc_emb_idx"]].copy(),
            "pos": self.dataset[idx]["pos"],
            "pos_features": self.embedding[self.dataset[idx]["pos_emb_idx"]].copy()
        }


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
        texts = [s["anc"] for s in samples]
        teacher_features = [torch.Tensor(s["anc_features"]) for s in samples]
        inputs = self.preprocess(texts)

        return {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "teacher_features": teacher_features,
            }
class DataCollatorForContrastiveDistill(DataCollatorForDistill):
    def __call__(self, samples):
        anc_text = [s["anc"] for s in samples]
        pos_text = [s["pos"] for s in samples]
        anc_features = [torch.Tensor(s["anc_features"]) for s in samples]
        pos_features = [torch.Tensor(s["pos_features"]) for s in samples]
        anc_inputs = self.preprocess(anc_text)
        pos_inputs = self.preprocess(pos_text)
        return {"input_ids": anc_inputs["input_ids"],
                "attention_mask": anc_inputs["attention_mask"],
                "pos": pos_inputs,
                "teacher_features": anc_features,
                "pos_features": pos_features
            }
    
class DataModuleForDistill(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_num: str,
        student_tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
        eval_batch_size: Optional[int] = None,
        max_length: int = 4096
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_num = data_num
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = student_tokenizer

        if "triplet" in str(self.data_dir):
            self.collate_fn = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
            )
        else:
            self.collate_fn = DataCollatorForDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
            )
        self.datasets = {}

    def load_data_and_emb(self,data_path):
        datasets = load_from_disk(data_path)
        embeddings = np.load(os.path.join(data_path ,"emb.npy"),  mmap_mode='r')
        return datasets, embeddings

    def setup(self, stage: str):
        data_path = os.path.join(self.data_dir, self.data_num)
        if os.path.exists(data_path):
            datasets, embeddings = self.load_data_and_emb(data_path)
        else:
            exist_data_dir_list = os.listdir(self.data_dir)
            exist_data_num_list = []
            for num in exist_data_dir_list:
                try:
                    exist_data_num_list.append(int(num))
                except ValueError:
                    continue
            max_data_num = max(exist_data_num_list) if exist_data_num_list else 0
            if int(self.data_num) < max_data_num:
                datasets, embeddings = self.load_data_and_emb(os.path.join(self.data_dir, str(max_data_num)))
                datasets = datasets.shuffle(seed=42).select(range(int(self.data_num)))
            else:
                raise ValueError(f"Data path {data_path} does not exist.")
        datasets = datasets.train_test_split(test_size=int(min(len(datasets)*0.1, 1000)), seed=42, shuffle=True)
        logger.info(f"Total samples: {len(datasets)}, embeddings: {embeddings.shape}")
        self.datasets["train"] = DistilDataset(datasets["train"], embeddings)
        self.datasets["test"] = DistilDataset(datasets["test"], embeddings)

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
