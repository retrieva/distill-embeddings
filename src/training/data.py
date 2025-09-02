import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PREFIX_MAP = {
    "HotpotQA": ["query:", "document:"],
    "NQ": ["query:", "document:"],
    "SQuAD": ["query:", "document:"],
    "Trivia": ["query:", "document:"],
    "en_NLI_data": ["query:", "query:"],
    "fever": ["query:", "document:"],
    "ms-marco": ["query:", "document:"],
}


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pos: dict[str, torch.Tensor]
    teacher_features: torch.Tensor
    pos_features: torch.Tensor
    subset: list[str]


class ShuffledTaskBatchSampler(Sampler[list[int]]):
    """
    バッチ内のタスクを統一しつつ、バッチの順序をタスク間でシャッフルするサンプラー
    """

    def __init__(self, task_indices: dict[str, list[int]], batch_size: int, shuffle: bool = True):
        """
        Args:
            task_indices (dict): タスク名をキー、インデックスのリストを値とする辞書
            batch_size (int): 各バッチのサイズ
            shuffle (bool): エポックごとにデータをシャッフルするかどうか
        """
        self.task_indices = task_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.all_batches = self._generate_all_batches()

    def _generate_all_batches(self) -> list[list[int]]:
        """全タスクの全バッチを生成する"""
        all_batches = []
        for task_name in self.task_indices:
            indices = self.task_indices[task_name][:]  # コピーを作成
            if self.shuffle:
                # タスク内のデータ順をシャッフル
                random.shuffle(indices)

            # バッチサイズごとに区切っていく
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i : i + self.batch_size])
        return all_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_batches)
        yield from self.all_batches

    def __len__(self) -> int:
        return len(self.all_batches)


class DistilDataset(Dataset):
    def __init__(self, dataset: Dataset, embedding: np.ndarray):
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
            "pos_features": self.embedding[self.dataset[idx]["pos_emb_idx"]].copy(),
            "subset": self.dataset[idx]["subset"],
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

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "teacher_features": teacher_features,
        }


class DataCollatorForContrastiveDistill(DataCollatorForDistill):
    def __init__(self, tokenizer, max_length=4096, disable_instruction: bool = False, add_prefix: bool = False):
        super().__init__(tokenizer, max_length)
        self.disable_instruction = disable_instruction
        self.add_prefix = add_prefix

    def __call__(self, samples):
        anc_text = [s["anc"] for s in samples]
        pos_text = [s["pos"] for s in samples]
        if self.disable_instruction:
            anc_text = [text.split("Query:")[1].strip() for text in anc_text]
        if self.add_prefix:
            for s in samples:
                subset = s["subset"]
                anc_text = [f"{PREFIX_MAP[subset][0]} {text}" for text in anc_text]
                pos_text = [f"{PREFIX_MAP[subset][1]} {text}" for text in pos_text]
        anc_features = [torch.Tensor(s["anc_features"]) for s in samples]
        pos_features = [torch.Tensor(s["pos_features"]) for s in samples]
        anc_inputs = self.preprocess(anc_text)
        pos_inputs = self.preprocess(pos_text)
        return {
            "input_ids": anc_inputs["input_ids"],
            "attention_mask": anc_inputs["attention_mask"],
            "pos": pos_inputs,
            "teacher_features": anc_features,
            "pos_features": pos_features,
        }


class DataModuleForDistill(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_num: str,
        student_tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
        eval_batch_size: int | None = None,
        max_length: int = 4096,
        add_prefix: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_num = data_num
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = student_tokenizer

        # TODO: ここのハードコード気持ち悪いかも！！
        if "triplet" or "gte" in str(self.data_dir):
            self.collate_fn = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
                disable_instruction=True if "gte" in str(self.data_dir) else False,
                add_prefix=add_prefix,
            )
        else:
            self.collate_fn = DataCollatorForDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
            )
        self.datasets = {}

    def load_data_and_emb(self, data_path):
        datasets = load_from_disk(data_path)
        embeddings = np.load(os.path.join(data_path, "emb.npy"), mmap_mode="r")
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
        datasets = datasets.train_test_split(test_size=int(min(len(datasets) * 0.1, 1000)), seed=42, shuffle=True)
        logger.info(f"Total samples: {len(datasets)}, embeddings: {embeddings.shape}")
        self.datasets["train"] = DistilDataset(datasets["train"], embeddings)
        self.datasets["test"] = DistilDataset(datasets["test"], embeddings)

    def _get_task_indices(self, dataset: Dataset) -> dict[str, list[int]]:
        """
        データセットをスキャンし、タスク名ごとのインデックスリストを作成する。

        Args:
            dataset (Dataset): `subset`属性にタスク名のリストを持つデータセット。

        Returns:
            dict[str, list[int]]: タスク名をキー、インデックスのリストを値とする辞書。
                                    (例: {'task_a': [0, 2, 5], 'task_b': [1, 3, 4]})
        """
        task_indices = defaultdict(list)
        for idx, task_name in enumerate(dataset.dataset["subset"]):
            task_indices[task_name].append(idx)

        return dict(task_indices)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            num_workers=self.num_workers,
            batch_sampler=ShuffledTaskBatchSampler(
                task_indices=self._get_task_indices(self.datasets["train"]),
                batch_size=self.batch_size,
                shuffle=True,
            ),
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            num_workers=self.num_workers,
            batch_sampler=ShuffledTaskBatchSampler(
                task_indices=self._get_task_indices(self.datasets["test"]),
                batch_size=self.eval_batch_size,
                shuffle=False,
            ),
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
