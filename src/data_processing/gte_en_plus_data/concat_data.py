from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk

base_dir = Path("data/gte_plus")
triplet_data = load_from_disk(base_dir / "triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000")
base_gte_data = load_from_disk(base_dir / "gte/Qwen_Qwen3-Embedding-4B_encoded/794554")

triplet_embs = np.load(base_dir / "triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000/emb.npy")
base_gte_embs = np.load(base_dir / "gte/Qwen_Qwen3-Embedding-4B_encoded/794554/emb.npy")

test_triplet_anc_emb = triplet_embs[triplet_data["anc_emb_idx"][0]]
test_triplet_pos_emb = triplet_embs[triplet_data["pos_emb_idx"][2000]]

gathered_embs = np.concatenate([base_gte_embs, triplet_embs], axis=0)
# print(gathered_embs.shape[0], base_gte_embs.shape[0])
assert len(gathered_embs) == len(base_gte_embs) + len(triplet_embs)

triplet_data = triplet_data.map(
    lambda x: {
        "anc_emb_idx": x["anc_emb_idx"] + len(base_gte_embs),
        "pos_emb_idx": x["pos_emb_idx"] + len(base_gte_embs),
    },
    num_proc=4,
)

assert np.allclose(gathered_embs[triplet_data["anc_emb_idx"][0]], test_triplet_anc_emb)
assert np.allclose(gathered_embs[triplet_data["pos_emb_idx"][2000]], test_triplet_pos_emb)

concat_data = concatenate_datasets([base_gte_data, triplet_data], axis=0)
output_dir = Path(f"data/gte_en_plus/{str(len(concat_data))}")
output_dir.mkdir(parents=True, exist_ok=True)
concat_data.save_to_disk(output_dir)
np.save(f"{output_dir}/emb.npy", gathered_embs)
