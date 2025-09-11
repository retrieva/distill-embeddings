from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabeledExample:
    text: str
    label: str | int


class LabeledTripletDataset(Dataset):
    """
    JSONL（{"text": str, "label": int|str}）から、
    各例を必ず anc として登場させ、同ラベル 1 件を pos、異ラベル 7 件を neg として返す Dataset。

    返す dict:
      {
        'anc': str,
        'pos': str,
        'neg': list[str],  # len == num_negs
        'label': int|str,  # anc のラベル
      }
    """

    def __init__(
        self,
        jsonl_path: str,
        num_negs: int = 7,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_negs = num_negs
        self._rng = random.Random(seed)

        self.examples: List[LabeledExample] = []
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = (obj.get("text") or "").strip()
                label = obj.get("label")
                self.examples.append(LabeledExample(text=text, label=label))

        # ラベル→インデックス群
        self.label_to_indices: dict[object, list[int]] = defaultdict(list)
        for i, ex in enumerate(self.examples):
            self.label_to_indices[ex.label].append(i)

        self.all_indices = list(range(len(self.examples)))

    def __len__(self) -> int:
        # anc に全サンプルが1回ずつ登場
        return len(self.examples)

    def _sample_pos(self, anc_idx: int) -> int:
        label = self.examples[anc_idx].label
        pool = self.label_to_indices[label]
        if len(pool) == 1:
            return anc_idx  # 代替がない場合は自分自身
        # 別の同ラベルをランダムに選ぶ
        while True:
            j = self._rng.choice(pool)
            if j != anc_idx:
                return j

    def _sample_negs(self, anc_idx: int) -> list[int]:
        label = self.examples[anc_idx].label
        # 異ラベルのプール
        # 高速化のためにラベルごとにまとめてから引く方法もあるが、単純に全体から引いてフィルタで十分
        negs: list[int] = []
        tries = 0
        while len(negs) < self.num_negs and tries < self.num_negs * 20:
            j = self._rng.randrange(len(self.examples))
            if j == anc_idx:
                tries += 1
                continue
            if self.examples[j].label == label:
                tries += 1
                continue
            # 重複を避ける（同じ負例を複数回返さない）
            if j in negs:
                tries += 1
                continue
            negs.append(j)
        # 足りない場合は補完（データが極端に小さいケース）
        while len(negs) < self.num_negs:
            negs.append(anc_idx if len(self.examples) == 1 else (anc_idx - 1) % len(self.examples))
        return negs

    def __getitem__(self, idx: int):
        anc = self.examples[idx]
        pos_idx = self._sample_pos(idx)
        neg_idx = self._sample_negs(idx)
        return {
            "anc": anc.text,
            "pos": self.examples[pos_idx].text,
            "neg": [self.examples[j].text for j in neg_idx],
            "label": anc.label,
        }

