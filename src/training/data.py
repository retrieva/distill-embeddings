import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from more_itertools import divide
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, PreTrainedTokenizer

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
    "S2ORC_title_abstract": ["query:", "document:"],
    "SimpleWiki": ["query:", "document:"],
    "agnews": ["query:", "document:"],
    "amazon-qa": ["query:", "document:"],
    "amazon_review_2018": ["query:", "document:"],
    "ccnews_title_text": ["query:", "document:"],
    "cnn_dailymail": ["query:", "document:"],
    "coco_captions": ["query:", "query:"],
    "codesearchnet": ["query:", "query:"],
    "eli5_question_answer": ["query:", "document:"],
    "gooaq_pairs": ["query:", "document:"],
    "npr": ["query:", "document:"],
    "searchQA_top5_snippets": ["query:", "query:"],
    "sentence-compression": ["query:", "query:"],
    "stackexchange_duplicate_questions_body_body": ["query:", "query:"],
    "stackexchange_duplicate_questions_title-body_title-body": ["query:", "query:"],
    "stackexchange_duplicate_questions_title_title": ["query:", "query:"],
    "wikihow": ["query:", "document:"],
    "xsum": ["query:", "document:"],
    "yahoo_answers_title_answer": ["query:", "document:"],
}


@dataclass
class Batch:

    anc: list[BatchEncoding]
    pos: list[BatchEncoding]
    teacher_features: list[torch.Tensor]
    pos_features: list[torch.Tensor]
    subset: list[str]

    neg: list[list[BatchEncoding]]  # 長さK。neg[j] は「j番目negの全サンプル分」を分割したチャンク列
    neg_features: list[list[torch.Tensor]]  # [B][K] でも [K][B] でもOKだがここは [B][K] にします
    num_neg: int = 0  # K

    def __len__(self):
        return sum(len(b["input_ids"]) for b in self.anc)


def get_rank_safe():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size_safe() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class TaskBatchDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        embeddings: np.ndarray,
        per_rank_batch_size: int,  # 変更: per-rank のBS
        world_size: int,  # 追加
        drop_last=False,
        shuffle_within_task=True,
        shuffle_task_batches=True,
        seed=42,
    ):
        self.hf_dataset = hf_dataset
        self.embeddings = embeddings
        self.per_rank_batch_size = per_rank_batch_size
        self.world_size = world_size
        self.global_batch_size = per_rank_batch_size * world_size
        self.drop_last = drop_last
        self.shuffle_within_task = shuffle_within_task
        self.shuffle_task_batches = shuffle_task_batches
        self.seed = seed
        self._batches = []
        self.rebuild_batches(0)

    def _group_indices_by_task(self):
        task_indices = defaultdict(list)
        for i in range(len(self.hf_dataset)):
            task_indices[self.hf_dataset[i]["subset"]].append(i)
        return task_indices

    def rebuild_batches(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        grouped = self._group_indices_by_task()
        batches = []
        for _, idxs in grouped.items():
            if self.shuffle_within_task:
                rng.shuffle(idxs)
            # ここが「グローバルBS」刻み
            for j in range(0, len(idxs), self.global_batch_size):
                chunk = idxs[j : j + self.global_batch_size]
                if len(chunk) < self.global_batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle_task_batches:
            rng.shuffle(batches)
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, bidx):
        idxs = self._batches[bidx]
        items = []
        for i in idxs:
            rec = self.hf_dataset[i]
            item = {
                "anc": rec["anc"],
                "anc_features": self.embeddings[rec["anc_emb_idx"]],
                "pos": rec["pos"],
                "pos_features": self.embeddings[rec["pos_emb_idx"]],
                "subset": rec["subset"],
            }

            # neg は存在すれば使う。無ければ空。
            neg_list = rec.get("neg") or []
            neg_idx = rec.get("neg_emb_idx") or []
            m = min(len(neg_list), len(neg_idx))
            if m > 0:
                neg_list = neg_list[:m]
                neg_idx = neg_idx[:m]
                item["neg"] = neg_list
                item["neg_features"] = [self.embeddings[j] for j in neg_idx]
            else:
                item["neg"] = []
                item["neg_features"] = []

            items.append(item)

        if not items:
            return items
        K = min(len(it["neg"]) for it in items)

        # そろえる（K=0 ならそのまま）
        if K >= 0:
            for it in items:
                it["neg"] = it["neg"][:K]
                it["neg_features"] = it["neg_features"][:K]

        return items
@dataclass
class DataCollatorForContrastiveDistill:
    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    disable_instruction: bool = False
    add_prefix: bool = False
    num_chunk: int = 4
    per_rank_batch_size: int = 32  # ★追加

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]


        # rank スライス
        rank = get_rank_safe()
        start = rank * self.per_rank_batch_size
        end = start + self.per_rank_batch_size
        samples = samples[start:end]

        # anc/pos（既存どおり）
        anc_text = [s["anc"] for s in samples]
        pos_text = [s["pos"] for s in samples]
        if self.disable_instruction:
            anc_text = [t.split("Query:", 1)[1].strip() if "Query:" in t else t for t in anc_text]

        if self.add_prefix:
            new_anc, new_pos = [], []
            for s in samples:
                subset = s["subset"]
                p_query, p_doc = PREFIX_MAP.get(subset, ("query:", "document:"))
                new_anc.append(f"{p_query} {s['anc']}")
                new_pos.append(f"{p_doc} {s['pos']}")
            anc_text, pos_text = new_anc, new_pos
        anc_feats = [torch.as_tensor(s["anc_features"], dtype=torch.float32) for s in samples]
        pos_feats = [torch.as_tensor(s["pos_features"], dtype=torch.float32) for s in samples]

        # --- negatives ---
        K = 0
        neg_tokenized_per_slot = []
        neg_features = []

        if len(samples) > 0:
            # __getitem__ 側で既に整形済み（全サンプル同じ長さ）
            K = len(samples[0].get("neg", []))

        if K > 0:
            # スロットごとに束ねる
            texts_per_slot = [[s["neg"][j] for s in samples] for j in range(K)]
            for j in range(K):
                neg_tokenized_per_slot.append([
                    self.preprocess(list(chunk)) for chunk in divide(self.num_chunk, texts_per_slot[j])
                ])
            neg_features = [[torch.as_tensor(f, dtype=torch.float32) for f in s["neg_features"]] for s in samples]
        else:
            neg_tokenized_per_slot = []  # 空でOK
            neg_features = []  # 空でOK
            

        return Batch(
            anc=[self.preprocess(list(a)) for a in divide(self.num_chunk, anc_text)],
            pos=[self.preprocess(list(p)) for p in divide(self.num_chunk, pos_text)],
            teacher_features=anc_feats,
            pos_features=pos_feats,
            subset=[s["subset"] for s in samples],
            neg=neg_tokenized_per_slot,  # 空でもよい
            neg_features=neg_features,  # 空でもよい
            num_neg=K,
        )


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
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_num = data_num
        self.per_rank_batch_size = batch_size  # 意味を明確化
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = student_tokenizer
        self.seed = seed

        if ("triplet" in str(self.data_dir)) or ("gte" in str(self.data_dir)):
            self.collate_fn_train = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
                disable_instruction=("gte" in str(self.data_dir)),
                add_prefix=add_prefix,
                num_chunk=4,

                per_rank_batch_size=self.per_rank_batch_size,
            )
            self.collate_fn_val = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
                disable_instruction=("gte" in str(self.data_dir)),
                add_prefix=add_prefix,
                num_chunk=4,
                per_rank_batch_size=self.eval_batch_size,  # ← val 用
            )
        else:
            raise ValueError(f"Unknown data type in {self.data_dir}")
        self.train_batches_ds: TaskBatchDataset | None = None
        self.val_batches_ds: TaskBatchDataset | None = None

    def load_data_and_emb(self, data_path):
        datasets = load_from_disk(data_path)
        embeddings = np.load(os.path.join(data_path, "emb.npy"), mmap_mode="r")
        return datasets, embeddings

    def prepare_data(self):
        data_path = os.path.join(self.data_dir, self.data_num)
        if not os.path.exists(data_path):
            exist_data_dir_list = os.listdir(self.data_dir)
            exist_nums = [int(d) for d in exist_data_dir_list if d.isdigit()]
            if not exist_nums:
                raise ValueError(f"No fallback data dirs in {self.data_dir}")

    def setup(self, stage: str):
        data_path = os.path.join(self.data_dir, self.data_num)
        if os.path.exists(data_path):
            datasets, embeddings = self.load_data_and_emb(data_path)
        else:
            exist_nums = [int(d) for d in os.listdir(self.data_dir) if d.isdigit()]
            max_data_num = max(exist_nums) if exist_nums else 0
            if int(self.data_num) < max_data_num:
                datasets, embeddings = self.load_data_and_emb(os.path.join(self.data_dir, str(max_data_num)))
                datasets = datasets.shuffle(seed=42).select(range(int(self.data_num)))
            else:
                raise ValueError(f"Data path {data_path} does not exist.")

        datasets = datasets.train_test_split(test_size=int(min(len(datasets) * 0.1, 1000)), seed=42, shuffle=True)
        logger.info(f"Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, embeddings: {embeddings.shape}")

        world_size = get_world_size_safe()  # ← ここで取得
        logger.info(f"World size: {world_size}")

        self.train_batches_ds = TaskBatchDataset(
            datasets["train"],
            embeddings,

            per_rank_batch_size=self.per_rank_batch_size,
            world_size=world_size,

            drop_last=True,
            shuffle_within_task=True,
            shuffle_task_batches=True,
            seed=self.seed,
        )
        self.val_batches_ds = TaskBatchDataset(
            datasets["test"],
            embeddings,

            per_rank_batch_size=self.eval_batch_size,
            world_size=world_size,

            drop_last=True,
            shuffle_within_task=False,
            shuffle_task_batches=False,
            seed=self.seed,
        )

    def rebuild_train_batches_for_epoch(self, epoch: int):
        if self.train_batches_ds is not None:
            self.train_batches_ds.rebuild_batches(epoch)

    def train_dataloader(self):
        # batch_size=1: 1 “要素” が既に 1 ミニバッチ
        return DataLoader(
            self.train_batches_ds,
            batch_size=1,
            shuffle=False,  # ← Sampler を入れ替えさせない
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_train,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_batches_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_val,
            pin_memory=True,
        )
