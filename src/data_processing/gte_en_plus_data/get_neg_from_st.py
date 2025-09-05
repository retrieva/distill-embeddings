import pandas as pd
from datasets import load_from_disk

base_dataset = load_from_disk("data/gte_plus-eng/triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000")


subsets = list(set(base_dataset["subset"]))

raw_data_dir = "data/gte_plus-eng/triplet/raw"
for subset in subsets:
    origin_data = pd.read_json(f"{raw_data_dir}/{subset}.jsonl", lines=True, dtype=str)
    subset_data = base_dataset.filter(lambda example, subset=subset: example["subset"] == subset)
