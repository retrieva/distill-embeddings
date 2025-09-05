import os

import pandas as pd
from datasets import load_from_disk

base_dataset = load_from_disk("data/gte_plus-eng/gte/Qwen_Qwen3-Embedding-4B_encoded/794554")


subsets = list(set(base_dataset["subset"]))

raw_data_dir = "data/gte_plus-eng/gte/raw"
for subset in subsets:
    origin_data = []
    for path in os.listdir(f"{raw_data_dir}/{subset}"):
        if path.endswith(".jsonl"):
            origin_data.append(pd.read_json(f"{raw_data_dir}/{subset}/{path}", lines=True, dtype=str))
    # query:str, pos:list[str], neg:list[str]
    origin_data = pd.concat(origin_data, ignore_index=True)
    # query:str, pos:str, subset:str, id:int,anc_emb_idx:int,pos_emb_idx:int
    subset_data = base_dataset.filter(lambda example, subset=subset: example["subset"] == subset)

    # subset:str, id:int, query:str, pos:str, neg:list[str], anc_emb_idx:int,pos_emb_idx:int, neg_emb_idx:list[int]
    final_data = None
