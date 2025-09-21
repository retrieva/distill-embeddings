import logging
import os
import random
import time
from collections import defaultdict
from collections.abc import Collection
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
        max_effective_pairs_per_rank: int | None = None,  # 追加: 動的BS用の上限（B*C）
        max_neg_per_sample: int | None = None,  # 追加: K の上限
        drop_last=False,
        shuffle_within_task=True,
        shuffle_task_batches=True,
        seed=42,
        label_group_subsets: Collection[str] | None = None,
        label_column: str = "label",
    ):
        self.hf_dataset = hf_dataset
        self.embeddings = embeddings
        self.per_rank_batch_size = per_rank_batch_size
        self.world_size = world_size
        self.global_batch_size = per_rank_batch_size * world_size
        self.max_effective_pairs_per_rank = (
            int(max_effective_pairs_per_rank)
            if max_effective_pairs_per_rank and max_effective_pairs_per_rank > 0
            else None
        )
        self.max_neg_per_sample = (
            int(max_neg_per_sample)
            if max_neg_per_sample is not None and int(max_neg_per_sample) > 0
            else None
        )
        self.drop_last = drop_last
        self.shuffle_within_task = shuffle_within_task
        self.shuffle_task_batches = shuffle_task_batches
        self.seed = seed
        self._batches = []
        self._neg_len = None  # lazily prepared cache of negatives length per sample
        self.label_group_subsets = set(label_group_subsets or [])
        self.label_column = label_column
        self.rebuild_batches(0)

    def _group_indices_by_task(self):
        # Faster: avoid per-row __getitem__; read the column once
        t0 = time.time()
        subsets_col = self.hf_dataset["subset"]  # list[str]
        task_indices = defaultdict(list)
        labels = None
        if self.label_group_subsets:
            if self.label_column in self.hf_dataset.column_names:
                raw_labels = self.hf_dataset[self.label_column]
                labels = ["" if v is None else str(v) for v in raw_labels]
            else:
                logger.warning(
                    "[TaskBatchDataset] label_column='%s' not found; falling back to subset-only grouping",
                    self.label_column,
                )
        use_label = labels is not None
        for i, s in enumerate(subsets_col):
            key = s
            if use_label and s in self.label_group_subsets:
                key = (s, labels[i])
            task_indices[key].append(i)
        logger.info(f"Grouped {len(subsets_col)} rows into {len(task_indices)} subsets in {time.time() - t0:.1f}s")
        return task_indices

    def rebuild_batches(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        # Cache grouping to avoid recomputing every epoch
        if not hasattr(self, "_grouped_cache") or self._grouped_cache is None:
            self._grouped_cache = self._group_indices_by_task()
        grouped = self._grouped_cache
        # Prepare neg length cache once to avoid per-row access below
        if self._neg_len is None:
            t0 = time.time()
            cols = set(self.hf_dataset.column_names)
            if "neg_num" in cols:
                vals = self.hf_dataset["neg_num"]
                neg_len = [int(v) if isinstance(v, (int, np.integer)) else 0 for v in vals]
            elif "neg_emb_idx" in cols:
                vals = self.hf_dataset["neg_emb_idx"]
                neg_len = [len(v) if isinstance(v, (list, tuple)) else 0 for v in vals]
            elif "neg" in cols:
                vals = self.hf_dataset["neg"]
                neg_len = [len(v) if isinstance(v, (list, tuple)) else 0 for v in vals]
            else:
                neg_len = [0] * len(self.hf_dataset)
            self._neg_len = np.asarray(neg_len, dtype=np.int32)
            logger.info(f"Prepared neg_len cache in {time.time() - t0:.1f}s")
        batches = []
        for _, idxs in grouped.items():
            if self.shuffle_within_task:
                rng.shuffle(idxs)
            # 動的なグローバルBSで刻む
            j = 0
            N = len(idxs)
            while j < N:
                # まず最大サイズのプリウィンドウを見て K(min) を見積もる
                pre_end = min(j + self.global_batch_size, N)
                window = idxs[j:pre_end]
                eff_per_rank = self.per_rank_batch_size
                if self.max_effective_pairs_per_rank is not None and len(window) > 0:
                    # このウィンドウ内の最小K（全サンプルが共通して持つneg数）
                    k_min = int(self._neg_len[window].min()) if len(window) > 0 else 0
                    # K の上限を考慮
                    if self.max_neg_per_sample is not None:
                        k_min = min(k_min, int(self.max_neg_per_sample))
                    C = 1 + max(0, k_min)
                    if C > 1:
                        eff_per_rank = max(1, min(self.per_rank_batch_size, self.max_effective_pairs_per_rank // C))
                group_size = eff_per_rank * self.world_size
                chunk = idxs[j : min(j + group_size, N)]
                if len(chunk) < group_size and self.drop_last:
                    break
                batches.append(chunk)
                j += len(chunk)
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
        # 先に per-sample 上限を適用
        if self.max_neg_per_sample is not None:
            for it in items:
                if len(it["neg"]) > int(self.max_neg_per_sample):
                    it["neg"] = it["neg"][: int(self.max_neg_per_sample)]
                    it["neg_features"] = it["neg_features"][: int(self.max_neg_per_sample)]

        K = min(len(it["neg"]) for it in items)

        # そろえる（K=0 ならそのまま）
        if K >= 0:
            for it in items:
                it["neg"] = it["neg"][:K]
                it["neg_features"] = it["neg_features"][:K]

        return items


class AncOnlyBatchDataset(Dataset):
    """
    Simple batch dataset for anc-only data (e.g., finweb) that contains
    only `text` and `emb_idx` (or possibly `anc` and `anc_emb_idx`).

    It groups records into global-sized chunks so that our collator can
    perform per-rank slicing consistently (DataLoader uses batch_size=1).
    """

    def __init__(
        self,
        hf_dataset,
        per_rank_batch_size: int,
        world_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.hf_dataset = hf_dataset
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.world_size = int(world_size)
        self.global_batch_size = self.per_rank_batch_size * self.world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._batches: list[list[int]] = []
        self.rebuild_batches(0)

    def rebuild_batches(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        idxs = list(range(len(self.hf_dataset)))
        if self.shuffle:
            rng.shuffle(idxs)
        self._batches = []
        j = 0
        N = len(idxs)
        group_size = self.global_batch_size
        while j < N:
            chunk = idxs[j : min(j + group_size, N)]
            if len(chunk) < group_size and self.drop_last:
                break
            self._batches.append(chunk)
            j += len(chunk)

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, bidx):
        idxs = self._batches[bidx]
        items = []
        for i in idxs:
            rec = self.hf_dataset[i]
            items.append(
                {
                    "anc": rec.get("anc", rec.get("text", "")),
                    "anc_emb_idx": rec.get("anc_emb_idx", rec.get("emb_idx")),
                    "subset": rec.get("subset", "finweb"),
                }
            )
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
        # Guard against absurdly large tokenizer.model_max_length sentinels (e.g., 1e30)
        try:
            safe_max_len = int(self.max_length)
        except Exception:
            safe_max_len = 4096
        if safe_max_len <= 0 or safe_max_len > 100_000:
            safe_max_len = 4096
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=safe_max_len,
            return_tensors="pt",
        )

    def __call__(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        # rank スライス（バッチ毎に可変長対応）
        rank = get_rank_safe()
        world_size = get_world_size_safe()
        dynamic_per_rank = max(1, len(samples) // max(1, world_size))
        start = rank * dynamic_per_rank
        end = start + dynamic_per_rank
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


@dataclass
class DataCollatorForAncOnly:
    """
    Collator for datasets that only have `text` and `emb_idx`.

    - Treats `text` as anchors (anc)
    - Looks up teacher features from `embeddings[emb_idx]`
    - Does not provide pos/neg (use with args.use_pos=False)
    """

    tokenizer: PreTrainedTokenizer
    embeddings: np.ndarray
    max_length: int = 4096
    num_chunk: int = 4
    per_rank_batch_size: int = 32

    def preprocess(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, samples):
        # Support both standard list[dict] and a pre-batched list[[dict,...]]
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        # Dynamic per-rank slicing (mirror contrastive collator behavior)
        rank = get_rank_safe()
        world_size = get_world_size_safe()
        dynamic_per_rank = max(1, len(samples) // max(1, world_size))
        start = rank * dynamic_per_rank
        end = start + dynamic_per_rank
        samples = samples[start:end]

        # Map minimal schema to anc-only
        anc_text = [s.get("anc", s.get("text", "")) for s in samples]
        # teacher features from emb_idx (or anc_emb_idx fallback)
        emb_indices = [s.get("anc_emb_idx", s.get("emb_idx")) for s in samples]
        anc_feats = [
            torch.as_tensor(self.embeddings[int(idx)], dtype=torch.float32)
            for idx in emb_indices
        ]
        subsets = [s.get("subset", "unknown") for s in samples]

        return Batch(
            anc=[self.preprocess(list(a)) for a in divide(self.num_chunk, anc_text)],
            pos=[],  # no positives
            teacher_features=anc_feats,
            pos_features=[],  # no positives
            subset=subsets,
            neg=[],  # no negatives
            neg_features=[],
            num_neg=0,
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
        chunk_parts: int = 4,
        max_effective_pairs_per_rank: int | None = None,
        max_neg_per_sample: int | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_num = data_num
        self.per_rank_batch_size = batch_size  # 意味を明確化
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.tokenizer = student_tokenizer
        self.max_length = max_length
        self.seed = seed
        self.chunk_parts = chunk_parts
        self.max_effective_pairs_per_rank = (
            int(max_effective_pairs_per_rank)
            if max_effective_pairs_per_rank and max_effective_pairs_per_rank > 0
            else None
        )
        self.max_neg_per_sample = (
            int(max_neg_per_sample) if max_neg_per_sample is not None and int(max_neg_per_sample) > 0 else None
        )
        self.label_group_subsets: set[str] = set()
        if "w_label_data" in str(self.data_dir):
            self.label_group_subsets = {"agnews", "yahoo_answers_title_answer"}
        self.label_column = "label"

        if ("triplet" in str(self.data_dir)) or ("gte" in str(self.data_dir)):
            self.collate_fn_train = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
                disable_instruction=("gte" in str(self.data_dir)),
                add_prefix=add_prefix,
                num_chunk=self.chunk_parts,
                per_rank_batch_size=self.per_rank_batch_size,
            )
            self.collate_fn_val = DataCollatorForContrastiveDistill(
                tokenizer=self.tokenizer,
                max_length=max_length,
                disable_instruction=("gte" in str(self.data_dir)),
                add_prefix=add_prefix,
                num_chunk=self.chunk_parts,
                per_rank_batch_size=self.eval_batch_size,  # ← val 用
            )
        elif ("finweb" in str(self.data_dir).lower()) or ("fineweb" in str(self.data_dir).lower()):
            # Instantiate after loading embeddings in setup()
            self.collate_fn_train = None
            self.collate_fn_val = None
            self._use_finweb_collator = True
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

        test_size = int(min(len(datasets) * 0.1, 1000))
        test_size = max(1, test_size) if len(datasets) > 1 else 1
        datasets = datasets.train_test_split(test_size=test_size, seed=42, shuffle=True)
        logger.info(f"Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, embeddings: {embeddings.shape}")

        world_size = get_world_size_safe()  # ← ここで取得
        logger.info(f"World size: {world_size}")

        # Choose dataset + collator based on schema/path
        if getattr(self, "_use_finweb_collator", False) or (
            set(datasets["train"].column_names) >= {"text", "emb_idx"}
        ):
            # Build anc-only collators now that we have embeddings
            self.collate_fn_train = DataCollatorForAncOnly(
                tokenizer=self.tokenizer,
                embeddings=embeddings,
                max_length=self.max_length,
                num_chunk=self.chunk_parts,
                per_rank_batch_size=self.per_rank_batch_size,
            )
            self.collate_fn_val = DataCollatorForAncOnly(
                tokenizer=self.tokenizer,
                embeddings=embeddings,
                max_length=self.max_length,
                num_chunk=self.chunk_parts,
                per_rank_batch_size=self.eval_batch_size,
            )

            self.train_batches_ds = AncOnlyBatchDataset(
                datasets["train"],
                per_rank_batch_size=self.per_rank_batch_size,
                world_size=world_size,
                drop_last=True,
                shuffle=True,
                seed=self.seed,
            )
            self.val_batches_ds = AncOnlyBatchDataset(
                datasets["test"],
                per_rank_batch_size=self.eval_batch_size,
                world_size=world_size,
                drop_last=True,
                shuffle=False,
                seed=self.seed,
            )
        else:
            # Default contrastive dataset path
            self.train_batches_ds = TaskBatchDataset(
                datasets["train"],
                embeddings,
                per_rank_batch_size=self.per_rank_batch_size,
                world_size=world_size,
                max_effective_pairs_per_rank=self.max_effective_pairs_per_rank,
                max_neg_per_sample=self.max_neg_per_sample,
                label_group_subsets=self.label_group_subsets,
                label_column=self.label_column,
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
                max_effective_pairs_per_rank=self.max_effective_pairs_per_rank,
                max_neg_per_sample=self.max_neg_per_sample,
                label_group_subsets=self.label_group_subsets,
                label_column=self.label_column,
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
            persistent_workers=True if self.num_workers and self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_batches_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_val,
            pin_memory=True,
            persistent_workers=True if self.num_workers and self.num_workers > 0 else False,
        )
