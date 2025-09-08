import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# ============== 設定 ==============
BASE_DATASET_PATH = "data/gte_plus-eng/gte/Qwen_Qwen3-Embedding-4B_encoded/794554"
RAW_DATA_DIR = "data/gte_plus-eng/gte/raw"
OUTPUT_BASE_DIR = "data/gte_plus-eng/gte/with_neg_noencode/794554"  # 新しい出力先(必要なら変更)
NEG_MAX_PER_PAIR = 7  # 1ペアあたりの最大 neg 数を制限したい場合は整数 (例: 50)

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 既存データセット読込
base_dataset = load_from_disk(BASE_DATASET_PATH)

subsets: list[str] = sorted(set(base_dataset["subset"]))
print(f"[INFO] 対象 subset 数: {len(subsets)}")
import ast


def ensure_list(x):
    if isinstance(x, list):
        return x

    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        import pdb

        pdb.set_trace()


query_col = "query" if "query" in base_dataset.column_names else "anc"

subset_output_dir = Path(OUTPUT_BASE_DIR) / "subsets"
subset_output_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []

for subset in tqdm(subsets, desc="Process subsets"):
    raw_dir = Path(RAW_DATA_DIR) / subset
    if not raw_dir.exists():
        print(f"[WARN] raw ディレクトリ無し: {raw_dir} -> skip")
        continue

    origin_parts = []
    for fn in os.listdir(raw_dir):
        if fn.endswith(".jsonl"):
            origin_parts.append(pd.read_json(raw_dir / fn, lines=True, dtype=str))
    if not origin_parts:
        print(f"[WARN] origin_data 無し subset={subset}")
        continue

    origin_df = pd.concat(origin_parts, ignore_index=True)
    if "pos" not in origin_df.columns or "neg" not in origin_df.columns:
        print(f"[WARN] 必須列不足 subset={subset}")
        continue

    origin_df["pos"] = origin_df["pos"].map(ensure_list)
    origin_df["neg"] = origin_df["neg"].map(ensure_list)

    # query ごとに集約
    origin_group = {}
    for _, row in origin_df.iterrows():
        origin_group.setdefault(row["query"].strip(), []).append(row)

    subset_dataset = base_dataset.filter(lambda ex, s=subset: ex["subset"] == s)
    if len(subset_dataset) == 0:
        continue
    subset_df = subset_dataset.to_pandas()

    collected_neg_lists = []

    for _, r in subset_df.iterrows():
        q = r[query_col]
        if q.startswith("Instruct:"):
            q = q.split("Query:")[1].strip()
        p = r["pos"]
        neg_set = set()
        for cand in origin_group.get(q, []):
            for n in cand["neg"]:
                if n != p:
                    if n and n not in (q, p):
                        neg_set.add(n)
            if NEG_MAX_PER_PAIR is not None and len(neg_set) > NEG_MAX_PER_PAIR:
                break
        else:
            if len(neg_set) == NEG_MAX_PER_PAIR:
                pass
            else:
                import pdb

                pdb.set_trace()
        collected_neg_lists.append(list(neg_set)[:NEG_MAX_PER_PAIR] if NEG_MAX_PER_PAIR is not None else list(neg_set))

    out_df = subset_df.copy()
    out_df["neg"] = collected_neg_lists

    final_dataset = Dataset.from_pandas(out_df, preserve_index=False)
    out_dir = subset_output_dir / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(out_dir)

    summary_rows.append({
        "subset": subset,
        "rows": len(final_dataset),
        "max_neg_per_pair": max(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0,
        "min_neg_per_pair": min(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0,
        "pairs_with_neg": sum(1 for lst in collected_neg_lists if lst),
    })
    print(
        f"[FINAL] subset={subset} rows={len(final_dataset)} max_neg={max(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0} min_neg={min(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0} saved={out_dir}"
    )

# 集計サマリ
summary = {
    "total_subsets_processed": len(summary_rows),
    "details": summary_rows,
}
with open(Path(OUTPUT_BASE_DIR) / "neg_extraction_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[DONE] neg 抽出のみ完了")
