import os
from typing import Optional

import torch
import lightning as L
from transformers import PreTrainedTokenizer, AutoTokenizer
from src.utils import load_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StreamingDataCollatorForLM:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_input_len, max_output_len):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_length = max_input_len + max_output_len

    def pad(self, inputs, max_length: int, padding_side: str = "right"):
        data = {}
        keys = inputs[0].keys()
        for k in keys:
            if k == "input_ids":
                pad_id = self.tokenizer.pad_token_id
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k == "attention_mask":
                pad_id = 0
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k == "labels":
                pad_id = -100
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            else:
                data[k] = [d[k] for d in inputs]
        if "attention_mask" not in data:
            data["attention_mask"] = (
                data["input_ids"].ne(self.tokenizer.pad_token_id).long()
            )
        return data

    def __call__(self, samples):
        result = {}
        model_inputs_columns = [
            col for col in list(samples[0].keys()) if col.startswith("model_inputs")
        ]
        for col in model_inputs_columns:
            # list of dict
            inputs = [sample[col] for sample in samples]
            if col == "model_inputs_gen":
                result[col] = self.pad(inputs, self.max_input_len, padding_side="left")
            else:
                result[col] = self.pad(inputs, self.max_length, padding_side="right")
        if "model_inputs_gen" in result:
            result["model_inputs_gen"]["response"] = [s["response"] for s in samples]
        return result


class StreamingSFTDataModule(L.LightningDataModule):
    def __init__(
        self,
        student_tokenizer: str,
        teacher_model: str,
        data_path: str,
        batch_size: int,
        num_workers: int,
        eval_batch_size: Optional[int] = None,
        max_input_len: int = 1536
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = load_tokenizer(tokenizer_path)

        self.collate_fn = StreamingDataCollatorForLM(
            tokenizer=self.tokenizer,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
        )
        self.loader_cls = StreamingDataLoader
        self.datasets = {}

    def setup(self):
        self.datasets["train"] = load_dataset("hotchpotch/fineweb-2-edu-japanese","sample_10BT",split="train", streaming=True)
        self.datasets["val"] = load_dataset("hotchpotch/fineweb-2-edu-japanese","sample_10BT",split="train", streaming=True)

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
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
