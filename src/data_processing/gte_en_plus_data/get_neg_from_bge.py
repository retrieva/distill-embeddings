import json
import os
from pathlib import Path

import numpy as np
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
# subsets = ["en_NLI_data"]
# subsets = ["ms-marco"]
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
        # バグ修正: literal_eval には str を渡す
        return ast.literal_eval(s)
    except Exception:
        # パースできない場合は空扱い（pdb には入らない）
        return []


query_col = "query" if "query" in base_dataset.column_names else "anc"

subset_output_dir = Path(OUTPUT_BASE_DIR) / "subsets"
subset_output_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []

for subset in tqdm(subsets, desc="Process subsets"):
    raw_dir = Path(RAW_DATA_DIR) / subset
    if not raw_dir.exists():
        print(f"[WARN] raw ディレクトリ無し: {raw_dir} -> skip")
        continue

    # まず対象 subset のデータフレームを取得
    subset_dataset = base_dataset.filter(lambda ex, s=subset: ex["subset"] == s)
    if len(subset_dataset) == 0:
        continue
    subset_df = subset_dataset.to_pandas()

    # subset 内で実際に使うクエリだけを正規化して集合化
    def normalize_query(q: str) -> str:
        if isinstance(q, str) and q.startswith("Instruct:"):
            parts = q.split("Query:")
            return parts[1].strip() if len(parts) > 1 else q.strip()
        return str(q).strip()

    target_queries = set()
    for _, r in subset_df.iterrows():
        target_queries.add(normalize_query(r[query_col]))

    # origin の JSONL を行ストリーミングで読み、必要なクエリだけ neg を集約
    # メモリ節約のため、query -> set(neg) のみ保持（pos は利用時に除外判定するので不要）
    origin_group: dict[str, set] = {q: set() for q in target_queries}

    jsonl_files = [fn for fn in os.listdir(raw_dir) if fn.endswith(".jsonl")]
    if not jsonl_files:
        print(f"[WARN] origin_data 無し subset={subset}")
        continue

    for fn in jsonl_files:
        fpath = raw_dir / fn
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                q_raw = obj.get("query", "")
                q = normalize_query(q_raw)
                if q not in origin_group:
                    continue
                neg_list = ensure_list(obj.get("neg"))
                # クエリ自身は neg 候補から除外
                for n in neg_list:
                    if n and n != q:
                        origin_group[q].add(n)

    # neg 抽出
    collected_neg_lists = []
    for _, r in subset_df.iterrows():
        q = normalize_query(r[query_col])
        p = r["pos"]
        neg_set = set()
        for n in origin_group.get(q, set()):
            if n and n not in (q, p):
                neg_set.add(n)
            if NEG_MAX_PER_PAIR is not None and len(neg_set) >= NEG_MAX_PER_PAIR:
                break
        collected_neg_lists.append(list(neg_set)[:NEG_MAX_PER_PAIR] if NEG_MAX_PER_PAIR is not None else list(neg_set))

    out_df = subset_df.copy()
    out_df["neg"] = collected_neg_lists
    out_df["neg_num"] = [len(lst) for lst in collected_neg_lists]
    neg_num_summary = out_df["neg_num"].value_counts(dropna=False).sort_index().to_dict()
    neg_num_summary["subset"] = subset

    final_dataset = Dataset.from_pandas(out_df, preserve_index=False)
    out_dir = subset_output_dir / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(out_dir)
    summary_rows.append(neg_num_summary)

    print(
        f"[FINAL] subset={subset} rows={len(final_dataset)} "
        f"max_neg={max(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0} "
        f"min_neg={min(len(lst) for lst in collected_neg_lists) if collected_neg_lists else 0} "
        f"saved={out_dir}"
    )

# 集計サマリ
summary = {
    "total_subsets_processed": len(summary_rows),
    "details": summary_rows,
}
with open(Path(OUTPUT_BASE_DIR) / "neg_extraction_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[DONE] neg 抽出のみ完了")
