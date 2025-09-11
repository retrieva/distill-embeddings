from __future__ import annotations

import argparse
import html as htmlmod
import os
import random
import re
import time

import pandas as pd
from datasets import Dataset


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    s = htmlmod.unescape(str(raw))
    # HTMLタグ除去（<a>は中身を残す）
    s = re.sub(r"(?is)<img[^>]*>", "", s)
    s = re.sub(r"(?is)<a\s+[^>]*>(.*?)</a>", r"\1", s)
    s = re.sub(r"(?is)<[^>]+>", "", s)
    # URL除去
    s = re.sub(r"(?i)https?://\S+", "", s)
    s = re.sub(r"(?i)\bwww\.\S+", "", s)
    s = re.sub(r"(?i)http___\S+", "", s)
    # タイムスタンプ除去
    s = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b", "", s)
    s = re.sub(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}(?::\d{2})?(?:\s+[A-Z]{2,4})?\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)\b", "", s, flags=re.IGNORECASE)
    # 連続空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_valid_text(s: str, min_alpha: int = 10, min_length: int = 20) -> bool:
    if not s:
        return False
    t = s.strip().lower()
    if not t or t == "none":
        return False
    if set(t) <= {"\\", "/", "-", "_"}:
        return False
    alpha = sum(ch.isalpha() for ch in t)
    return alpha >= min_alpha and len(t) >= min_length


def build_yahoo_triplets(
    csv_path: str,
    out_dir: str,
    num_negs: int = 7,
    joiner: str = "\n\n",
    min_alpha: int = 10,
    min_length: int = 20,
    limit: int | None = None,
    num_proc: int | None = None,
    seed: int = 42,
):
    t0 = time.time()
    print("[YA] Reading CSV via pandas ...")
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "q_title", "q_body", "answer"],
    )
    if limit is not None:
        df = df.iloc[:limit]
    print(f"[YA] Rows: {len(df)}")

    print("[YA] Converting to HF Dataset ...")
    ds = Dataset.from_pandas(df, preserve_index=False)

    np = num_proc or max(os.cpu_count() or 1, 1)

    print(f"[YA] Cleaning text (num_proc={np}) ...")

    def _clean(ex):
        qt = clean_text(ex.get("q_title", ""))
        qb = clean_text(ex.get("q_body", ""))
        ans = clean_text(ex.get("answer", ""))
        anc = f"{qt}{joiner}{qb}" if qb else qt
        return {"anc": anc, "pos": ans, "label": ex.get("label")}

    ds = ds.map(_clean, num_proc=np, remove_columns=ds.column_names, desc="clean_text")

    print("[YA] Filtering invalid anc/pos ...")
    ds = ds.filter(
        lambda ex: is_valid_text(ex["anc"], min_alpha=min_alpha, min_length=min_length)
        and is_valid_text(ex["pos"], min_alpha=min_alpha, min_length=min_length),
        num_proc=np,
        desc="filter_valid",
    )
    print(f"[YA] Kept rows: {len(ds)}")

    print("[YA] Building label->answers pools ...")
    answers_by_label: dict[object, list[str]] = {}
    labels: list[object] = []
    for ex in ds:
        lab = ex["label"]
        if lab not in answers_by_label:
            answers_by_label[lab] = []
            labels.append(lab)
        answers_by_label[lab].append(ex["pos"])  # cleaned answer

    print(f"[YA] Mapping negs (num_proc={np}) ...")

    def _add_negs(ex, idx):
        rng = random.Random((seed + int(idx)) & 0xFFFFFFFF)
        lab = ex["label"]
        pos = ex["pos"]
        other_labels = [lb for lb in labels if lb != lab and answers_by_label.get(lb)]
        seen = set()
        negs: list[str] = []
        if other_labels:
            while len(negs) < num_negs:
                lb = rng.choice(other_labels)
                pool = answers_by_label.get(lb) or []
                if not pool:
                    continue
                j = rng.randrange(len(pool))
                if (lb, j) in seen:
                    continue
                cand = pool[j]
                if cand and cand != pos:
                    negs.append(cand)
                    seen.add((lb, j))
        return {"anc": ex["anc"], "pos": pos, "neg": negs, "label": lab}

    ds_trip = ds.map(
        _add_negs,
        with_indices=True,
        num_proc=np,
        remove_columns=ds.column_names,
        desc="add_negs",
    )

    print(f"[YA] save_to_disk -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    ds_trip.save_to_disk(out_dir)
    print(f"[YA] Done in {(time.time() - t0) / 60:.1f} min. Examples: {len(ds_trip)}")


def main():
    p = argparse.ArgumentParser(description="Yahoo Answers を anc/pos/neg の datasets 形式へ")
    p.add_argument(
        "--input",
        default="data/w_label_data/yahoo_answers/yahoo_answers_csv/train.csv",
        help="CSVパス (label,q_title,q_body,answer)",
    )
    p.add_argument(
        "--out-dir",
        default="data/w_label_data/yahoo_answers/yahoo_triplets_hf",
        help="save_to_disk の出力先",
    )
    p.add_argument("--num-negs", type=int, default=7)
    p.add_argument("--min-length", type=int, default=20)
    p.add_argument("--min-alpha", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joiner", default="\n\n")
    args = p.parse_args()

    build_yahoo_triplets(
        csv_path=args.input,
        out_dir=args.out_dir,
        num_negs=args.num_negs,
        joiner=args.joiner,
        min_alpha=args.min_alpha,
        min_length=args.min_length,
        limit=args.limit,
        num_proc=args.num_proc,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
