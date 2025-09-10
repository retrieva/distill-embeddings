from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk

base_dir = Path("data/gte_plus-eng")
print("data/gte_en_plus_w_neg/Qwen_Qwen3-Embedding-4B_encoded/")
gte_plus_data = load_from_disk(base_dir / "Qwen_Qwen3-Embedding-4B_encoded/1794550")
neg_gte_data = load_from_disk(base_dir / "gte/with_neg")

gte_plus_embs = np.load(base_dir / "Qwen_Qwen3-Embedding-4B_encoded/1794550/emb.npy")
# neg_gte_embs = np.load(base_dir / "gte/with_neg/neg_embeddings.npz")["embeddings"]

print("loaded data")

# # Show sizes to diagnose possible OOM
# print("gte_plus_embs:", getattr(gte_plus_embs, "shape", None), f"{getattr(gte_plus_embs, 'nbytes', 0) / 1e9:.2f} GB")
# print("neg_gte_embs:", getattr(neg_gte_embs, "shape", None), f"{getattr(neg_gte_embs, 'nbytes', 0) / 1e9:.2f} GB")

# # Prepare safe sample indices for checks (only if neg exists)
# neg_indices = neg_gte_data["neg_emb_idx"]
# has_neg = [i for i, v in enumerate(neg_indices) if v and len(v) > 0]
# print("rows_with_neg:", len(has_neg))
# test_idx1 = has_neg[0] if len(has_neg) >= 1 else None
# test_idx2 = has_neg[min(3000, len(has_neg) - 1)] if len(has_neg) >= 2 else None
# test_gte_neg_emb_1 = neg_gte_embs[neg_indices[test_idx1][0]] if test_idx1 is not None else None
# test_gte_neg_emb_2 = neg_gte_embs[neg_indices[test_idx2][0]] if test_idx2 is not None else None

# Offset neg indices by base length
neg_gte_data = neg_gte_data.map(
    lambda x: {
        "neg_emb_idx": [idx + len(gte_plus_embs) for idx in x["neg_emb_idx"]] if len(x["neg_emb_idx"]) > 0 else [],
    },
    num_proc=4,
)

# # Prepare output dir
# output_dir = Path(
#     f"data/gte_en_plus_w_neg/Qwen_Qwen3-Embedding-4B_encoded/{str(len(gte_plus_data) + len(neg_gte_data))}"
# )
# output_dir.mkdir(parents=True, exist_ok=True)

# # Create concatenated embeddings via memmap to avoid OOM
# if gte_plus_embs.ndim != 2 or neg_gte_embs.ndim != 2 or gte_plus_embs.shape[1] != neg_gte_embs.shape[1]:
#     raise ValueError(f"Embedding dims mismatch: {gte_plus_embs.shape} vs {neg_gte_embs.shape}")

# total = len(gte_plus_embs) + len(neg_gte_embs)
# emb_path = output_dir / "emb.npy"
# mm = open_memmap(emb_path, mode="w+", dtype=gte_plus_embs.dtype, shape=(total, gte_plus_embs.shape[1]))
# mm[: len(gte_plus_embs)] = gte_plus_embs
# mm[len(gte_plus_embs) : len(gte_plus_embs) + len(neg_gte_embs)] = neg_gte_embs
# del mm  # flush to disk

# # Verify neg indices point to the same vectors (if applicable)
# gathered_embs = np.load(emb_path, mmap_mode="r")
# if test_idx1 is not None:
#     assert np.allclose(gathered_embs[neg_gte_data["neg_emb_idx"][test_idx1][0]], test_gte_neg_emb_1)
# if test_idx2 is not None:
#     assert np.allclose(gathered_embs[neg_gte_data["neg_emb_idx"][test_idx2][0]], test_gte_neg_emb_2)
# print("neg idx check passed")

df_base = gte_plus_data.to_pandas()
df_neg = neg_gte_data.to_pandas()

# 2) 使うカラムを絞る（キー + 追加したいカラム）
merge_keys = [
    "anc",
    "pos",
    "anc_emb_idx",
    "pos_emb_idx",
    "subset",
]
extra_cols = ["neg", "neg_emb_idx"]  # 追加したいカラムに合わせて変更
# df_neg = df_neg[merge_keys + extra_cols].drop_duplicates(subset=merge_keys)

# 3) 左結合（neg 側が重複する場合は validate を many_to_one に）
df_merged = pd.merge(df_base, df_neg, on=merge_keys, how="left", validate="one_to_one")

# Simpler: vectorized NaN -> [] per-row, without apply
for col in ["neg", "neg_emb_idx"]:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].astype(object)
        mask = df_merged[col].isna()
        if mask.any():
            df_merged.loc[mask, col] = pd.Series(
                [list() for _ in range(int(mask.sum()))],
                index=df_merged.index[mask],
                dtype=object,
            )

# 4) Dataset に戻す
gte_data = Dataset.from_pandas(df_merged, preserve_index=False)
print(gte_data)


print("Data size:", len(gte_data))
output_dir = Path(f"data/gte_en_plus_w_neg/Qwen_Qwen3-Embedding-4B_encoded/{str(len(gte_data))}")
output_dir.mkdir(parents=True, exist_ok=True)
print(output_dir)

subset_to_neg_count = {}
for subset in gte_data.unique("subset"):
    subset_to_neg_count[subset] = {}
    subset_ds = gte_data.filter(lambda x: x["subset"] == subset)
    subset_with_len = subset_ds.map(lambda x: {"neg_len": len(x["neg"])})
    for neg_count in subset_with_len.unique("neg_len"):
        count = subset_with_len.filter(lambda x: x["neg_len"] == neg_count).num_rows
        subset_to_neg_count[subset][int(neg_count)] = int(count)

with open(output_dir / "subset_to_neg_count.json", "w") as f:
    import json

    json.dump(subset_to_neg_count, f, indent=4)
gte_data.save_to_disk(output_dir)
